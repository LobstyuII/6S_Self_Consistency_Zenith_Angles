# ==================== sensitivity_analyzer.py ====================
"""
敏感性分析模块
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy import stats
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from config import ExperimentConfig
from utils import setup_logger

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


class SensitivityAnalyzer:
    """敏感性分析器"""

    def __init__(self, results: pd.DataFrame, logger=None):
        """
        初始化敏感性分析器

        Parameters:
        -----------
        results : pd.DataFrame
            模拟结果数据
        logger : logging.Logger, optional
            日志记录器
        """
        self.results = results
        self.logger = logger or setup_logger('SensitivityAnalyzer')

        # 计算大气质量相关参数
        if 'sza' in results.columns:
            self.results['secz_sza'] = 1.0 / np.cos(np.radians(self.results['sza']))
        if 'vza' in results.columns:
            self.results['secz_vza'] = 1.0 / np.cos(np.radians(self.results['vza']))
        if 'secz_sza' in self.results.columns and 'secz_vza' in self.results.columns:
            self.results['total_airmass'] = self.results['secz_sza'] + self.results['secz_vza']
            self.results['airmass_product'] = self.results['secz_sza'] * self.results['secz_vza']

    def calculate_sensitivity_indices(self, target_var: str = 'error_absolute') -> Dict[str, float]:
        """
        计算敏感性指标（增加错误处理）
        """
        sensitivity = {}

        # 定义需要分析的参数
        params_to_analyze = ['sza', 'vza', 'aod550', 'h2o', 'o3', 'rho_true', 'total_airmass']

        for param in params_to_analyze:
            if param in self.results.columns:
                # 计算参数与目标变量的相关系数
                valid_mask = self.results[[param, target_var]].notna().all(axis=1)
                valid_data = self.results.loc[valid_mask]

                # 检查数据是否有效
                if len(valid_data) > 10:  # 至少需要10个有效样本
                    # 检查参数值是否变化
                    unique_values = valid_data[param].unique()
                    if len(unique_values) < 2:
                        self.logger.warning(f"参数 {param} 的值没有变化，跳过线性回归")
                        continue

                    try:
                        x = valid_data[param].values
                        y = valid_data[target_var].values

                        # 回归分析
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                        # 计算标准化敏感性系数
                        std_x = np.std(x)
                        std_y = np.std(y)
                        normalized_slope = slope * (std_x / std_y) if std_y > 0 else np.nan

                        sensitivity[param] = {
                            'correlation': r_value,
                            'slope': slope,
                            'r_squared': r_value ** 2,
                            'p_value': p_value,
                            'normalized_slope': normalized_slope,
                            'std_x': std_x,
                            'std_y': std_y,
                            'n_samples': len(x),
                            'n_unique_values': len(unique_values)
                        }

                    except Exception as e:
                        self.logger.error(f"计算参数 {param} 敏感性时出错: {e}")
                        continue

        return sensitivity

    def plot_conditional_boxplots(self, save_path: Optional[Path] = None):
        """绘制条件箱线图，控制其他主要因素"""

        # 选择要分析的参数
        target_params = ['sza', 'vza', 'aod550', 'h2o', 'o3']

        # 创建图形
        n_params = len(target_params)
        fig, axes = plt.subplots(n_params, 2, figsize=(14, 4 * n_params))

        for idx, param in enumerate(target_params):
            # 左图：原始箱线图（当前做法）
            ax1 = axes[idx, 0]

            if param in self.results.columns:
                # 创建分组
                if param == 'sza':
                    bins = np.arange(0, 91, 15)  # 15度一组
                    labels = [f'{b}-{b + 14}°' for b in bins[:-1]]
                elif param == 'vza':
                    bins = np.arange(0, 76, 15)
                    labels = [f'{b}-{b + 14}°' for b in bins[:-1]]
                else:
                    # 其他参数使用分位数分组
                    unique_vals = np.sort(self.results[param].dropna().unique())
                    if len(unique_vals) > 5:
                        bins = np.percentile(self.results[param].dropna(), [0, 25, 50, 75, 100])
                    else:
                        bins = np.linspace(self.results[param].min(),
                                           self.results[param].max(), len(unique_vals) + 1)
                    labels = [f'Q{i + 1}' for i in range(len(bins) - 1)]

                # 确保有足够数据
                if len(bins) > 1:
                    self.results[f'{param}_group'] = pd.cut(self.results[param], bins=bins, labels=labels)

                    # 绘制原始箱线图
                    data = []
                    for group in labels:
                        group_data = self.results[self.results[f'{param}_group'] == group]
                        data.append(group_data['error_absolute'].dropna().values)

                    bp = ax1.boxplot(data, positions=range(len(labels)), widths=0.6)
                    ax1.set_xlabel(f'{param}分组')
                    ax1.set_ylabel('绝对误差')
                    ax1.set_title(f'{param} - 原始箱线图（混杂所有因素）')
                    ax1.set_xticks(range(len(labels)))
                    ax1.set_xticklabels(labels, rotation=45)
                    ax1.grid(True, alpha=0.3)

            # 右图：条件箱线图（控制其他因素）
            ax2 = axes[idx, 1]

            if param in self.results.columns:
                # 定义控制的条件范围（针对不同参数）
                control_conditions = {}
                if param != 'sza':
                    control_conditions['sza'] = (30, 45)  # 固定sza在30-45°
                if param != 'vza':
                    control_conditions['vza'] = (0, 15)  # 固定vza在0-15°
                if param != 'aod550':
                    control_conditions['aod550'] = (0.2, 0.4)
                if param != 'h2o':
                    control_conditions['h2o'] = (1.5, 2.5)
                if param != 'o3':
                    control_conditions['o3'] = (0.25, 0.35)
                if param != 'rho_true':
                    control_conditions['rho_true'] = (0.15, 0.25)

                # 为每个param_group创建条件筛选
                conditional_data = []
                positions = []

                for group_idx, group in enumerate(labels):
                    # 获取该组数据
                    group_mask = (self.results[f'{param}_group'] == group)
                    group_data = self.results[group_mask]

                    if len(group_data) > 0:
                        # 应用控制条件
                        control_mask = pd.Series(True, index=group_data.index)
                        for control_param, (low, high) in control_conditions.items():
                            if control_param in group_data.columns:
                                control_mask &= (group_data[control_param] >= low) & (group_data[control_param] <= high)

                        conditional_group_data = group_data[control_mask]

                        if len(conditional_group_data) > 5:  # 确保有足够数据
                            conditional_data.append(conditional_group_data['error_absolute'].dropna().values)
                            positions.append(group_idx)
                        else:
                            conditional_data.append([])

                if len(positions) > 0:
                    bp_cond = ax2.boxplot(conditional_data, positions=positions, widths=0.6)
                    ax2.set_xlabel(f'{param}分组')
                    ax2.set_ylabel('绝对误差')
                    ax2.set_title(f'{param} - 条件箱线图（控制其他因素）')
                    ax2.set_xticks(positions)
                    ax2.set_xticklabels([labels[i] for i in positions], rotation=45)
                    ax2.grid(True, alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, '条件数据不足', ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title(f'{param} - 条件箱线图（控制其他因素）')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"条件箱线图已保存: {save_path}")

        plt.show()

    # 在 sensitivity_analyzer.py 中，修改 plot_violin_with_color 函数的这部分代码：

    def plot_violin_with_color(self, save_path: Optional[Path] = None):
        """绘制小提琴图，用颜色编码另一主要因素"""

        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 参数配对：分析参数 vs 颜色编码参数
        param_pairs = [
            ('sza', 'vza'),  # 分析sza，颜色表示vza
            ('vza', 'sza'),  # 分析vza，颜色表示sza
            ('aod550', 'sza'),  # 分析aod，颜色表示sza
            ('h2o', 'vza'),  # 分析水汽，颜色表示vza
            ('o3', 'aod550'),  # 分析臭氧，颜色表示aod
            ('rho_true', 'sza')  # 分析反射率，颜色表示sza
        ]

        for idx, (target_param, color_param) in enumerate(param_pairs):
            ax = axes[idx // 3, idx % 3]

            if target_param in self.results.columns and color_param in self.results.columns:
                # 将颜色参数离散化
                color_data = self.results[color_param].dropna()
                if len(color_data) > 0:
                    # 处理颜色参数分箱，避免重复边界
                    unique_color_vals = np.sort(color_data.unique())
                    if len(unique_color_vals) <= 3:
                        # 如果唯一值很少，直接使用这些值
                        color_labels = [f'{v:.1f}' for v in unique_color_vals]
                        color_bins = np.concatenate([
                            [unique_color_vals[0] - 0.01],
                            unique_color_vals,
                            [unique_color_vals[-1] + 0.01]
                        ])
                        self.results[f'{color_param}_level'] = pd.cut(
                            self.results[color_param], bins=color_bins,
                            labels=color_labels, include_lowest=True
                        )
                    else:
                        # 使用分位数，但避免重复边界
                        try:
                            color_bins = np.percentile(color_data, [0, 33, 67, 100])
                            # 确保边界不重复
                            color_bins = np.unique(color_bins)
                            if len(color_bins) < 4:
                                # 如果有重复，使用等间距分箱
                                color_bins = np.linspace(color_data.min(), color_data.max(), 4)
                            color_labels = ['低', '中', '高']
                            self.results[f'{color_param}_level'] = pd.cut(
                                self.results[color_param], bins=color_bins,
                                labels=color_labels, include_lowest=True
                            )
                        except:
                            # 如果分位数失败，使用等间距分箱
                            color_bins = np.linspace(color_data.min(), color_data.max(), 4)
                            color_labels = ['低', '中', '高']
                            self.results[f'{color_param}_level'] = pd.cut(
                                self.results[color_param], bins=color_bins,
                                labels=color_labels, include_lowest=True
                            )

                    # 将目标参数分组
                    target_data = self.results[target_param].dropna()
                    if len(target_data) > 0:
                        # 处理目标参数分箱，避免重复边界
                        unique_target_vals = np.sort(target_data.unique())
                        if len(unique_target_vals) <= 5:
                            # 如果唯一值很少，直接使用这些值
                            target_bins = np.concatenate([
                                [unique_target_vals[0] - 0.01],
                                unique_target_vals,
                                [unique_target_vals[-1] + 0.01]
                            ])
                            target_labels = [f'{v:.1f}' for v in unique_target_vals]
                        else:
                            # 使用分位数，但避免重复边界
                            try:
                                target_bins = np.percentile(target_data, [0, 20, 40, 60, 80, 100])
                                # 确保边界不重复
                                target_bins = np.unique(target_bins)
                                if len(target_bins) < 6:
                                    # 如果有重复，使用等间距分箱
                                    target_bins = np.linspace(target_data.min(), target_data.max(), 6)
                            except:
                                # 如果分位数失败，使用等间距分箱
                                target_bins = np.linspace(target_data.min(), target_data.max(), 6)
                            target_labels = [f'Q{i + 1}' for i in range(len(target_bins) - 1)]

                        # 确保有至少2个边界
                        if len(target_bins) >= 2:
                            self.results[f'{target_param}_group'] = pd.cut(
                                self.results[target_param], bins=target_bins,
                                labels=target_labels, include_lowest=True
                            )

                            # 为每个颜色等级创建数据
                            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色

                            for color_idx, color_level in enumerate(color_labels):
                                mask = self.results[f'{color_param}_level'] == color_level
                                color_data_subset = self.results[mask]

                                if len(color_data_subset) > 0:
                                    # 按目标参数分组收集误差数据
                                    group_data = []
                                    positions = []

                                    groups = sorted(self.results[f'{target_param}_group'].dropna().unique())
                                    for group_idx, group in enumerate(groups):
                                        group_mask = color_data_subset[f'{target_param}_group'] == group
                                        errors = color_data_subset.loc[group_mask, 'error_absolute'].dropna()
                                        if len(errors) > 5:
                                            group_data.append(errors.values)
                                            positions.append(group_idx + color_idx * 0.25 - 0.25)  # 偏移位置

                                    if group_data:
                                        vp = ax.violinplot(group_data, positions=positions,
                                                           widths=0.2, showmeans=True, showmedians=False)
                                        for body in vp['bodies']:
                                            body.set_facecolor(colors[color_idx])
                                            body.set_alpha(0.7)
                                        vp['cmeans'].set_color(colors[color_idx])
                                        vp['cmeans'].set_linewidth(2)

                            ax.set_xlabel(f'{target_param}分组')
                            ax.set_ylabel('绝对误差')
                            ax.set_title(f'{target_param} vs 误差（按{color_param}着色）')

                            # 设置x轴标签
                            ax.set_xticks(range(len(groups)))
                            ax.set_xticklabels(groups, rotation=45)

                            # 添加图例
                            patches = [mpatches.Patch(color=colors[i], label=f'{color_param}: {color_labels[i]}')
                                       for i in range(len(color_labels))]
                            ax.legend(handles=patches, loc='upper right')

                            ax.grid(True, alpha=0.3)
                        else:
                            ax.text(0.5, 0.5, '目标参数分组失败', ha='center', va='center', transform=ax.transAxes)
                            ax.set_title(f'{target_param} vs 误差')
                    else:
                        ax.text(0.5, 0.5, '目标参数数据不足', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{target_param} vs 误差')
                else:
                    ax.text(0.5, 0.5, '颜色参数数据不足', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{target_param} vs 误差')

        plt.suptitle('参数敏感性分析（带条件着色）', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"彩色小提琴图已保存: {save_path}")

        plt.show()

    def plot_partial_regression(self, save_path: Optional[Path] = None):
        """绘制部分回归图，显示每个参数的净效应"""

        try:
            import statsmodels.api as sm
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            self.logger.warning("statsmodels 或 scikit-learn 未安装，跳过部分回归分析")
            return

        # 选择特征和目标
        features = ['sza', 'vza', 'aod550', 'h2o', 'o3', 'rho_true']
        available_features = [f for f in features if f in self.results.columns]

        # 准备数据
        data = self.results[available_features + ['error_absolute']].dropna()

        if len(data) < 50:
            self.logger.warning("数据不足，跳过部分回归分析")
            return

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data[available_features])
        y = data['error_absolute'].values

        # 多元线性回归
        X_with_const = sm.add_constant(X_scaled)
        model = sm.OLS(y, X_with_const).fit()

        # 计算每个特征的部分回归图
        n_features = len(available_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.ravel() if n_features > 1 else [axes]

        for i, feature in enumerate(available_features):
            ax = axes[i]

            # 计算部分残差
            # 1. 回归y到除当前特征外的所有特征
            other_features = [f for f in available_features if f != feature]
            if other_features:
                # 获取其他特征的列索引
                other_indices = [available_features.index(f) for f in other_features]
                X_other = X_scaled[:, other_indices]
                X_other_const = sm.add_constant(X_other)

                # 回归
                model_other = sm.OLS(y, X_other_const).fit()
                y_resid = y - model_other.predict(X_other_const)

                # 2. 回归当前特征到其他特征
                X_current = X_scaled[:, i].reshape(-1, 1)
                model_current = sm.OLS(X_current, X_other_const).fit()
                X_resid = X_scaled[:, i] - model_current.predict(X_other_const).flatten()
            else:
                # 如果只有一个特征
                y_resid = y
                X_resid = X_scaled[:, i]

            # 绘制部分回归图
            ax.scatter(X_resid, y_resid, alpha=0.5, s=10, color='blue')

            # 添加趋势线
            coeff, intercept = np.polyfit(X_resid, y_resid, 1)
            x_fit = np.linspace(X_resid.min(), X_resid.max(), 100)
            y_fit = coeff * x_fit + intercept

            ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'斜率: {coeff:.4f}')

            # 添加统计信息
            from scipy.stats import pearsonr
            corr, p_value = pearsonr(X_resid, y_resid)

            stats_text = (f'相关系数: {corr:.4f}\n'
                          f'P值: {p_value:.4f}\n'
                          f'标准化系数: {model.params[i + 1]:.4f}')

            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax.set_xlabel(f'{feature}的部分残差')
            ax.set_ylabel('误差的部分残差')
            ax.set_title(f'{feature}的净效应')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle('部分回归图（控制其他因素后的净效应）', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"部分回归图已保存: {save_path}")

        plt.show()

        # 打印回归摘要
        self.logger.info("多元回归摘要:")
        self.logger.info(model.summary().as_text())

    def plot_interaction_grid(self, save_path: Optional[Path] = None):
        """绘制交互效应网格图"""

        # 选择最重要的3个参数
        primary_params = ['sza', 'vza', 'aod550']

        # 创建3x3网格（对角线上是单因素效应，非对角线是交互效应）
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))

        for i, param1 in enumerate(primary_params):
            for j, param2 in enumerate(primary_params):
                ax = axes[i, j]

                if i == j:
                    # 对角线：单因素散点图
                    if param1 in self.results.columns:
                        ax.scatter(self.results[param1], self.results['error_absolute'],
                                   alpha=0.3, s=5, color='blue')
                        ax.set_xlabel(param1)
                        ax.set_ylabel('绝对误差' if j == 0 else '')
                        ax.set_title(f'{param1}单因素效应')
                        ax.grid(True, alpha=0.3)

                else:
                    # 非对角线：二维热力图
                    if param1 in self.results.columns and param2 in self.results.columns:
                        # 创建2D直方图/热力图
                        x = self.results[param1].values
                        y = self.results[param2].values
                        z = self.results['error_absolute'].values

                        # 移除NaN
                        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
                        x, y, z = x[mask], y[mask], z[mask]

                        if len(x) > 20:
                            # 使用hexbin
                            hb = ax.hexbin(x, y, C=z, gridsize=20, cmap='RdBu_r',
                                           reduce_C_function=np.mean)

                            ax.set_xlabel(param1)
                            ax.set_ylabel(param2 if i == 2 else '')
                            ax.set_title(f'{param1} × {param2}交互效应')

                            # 添加颜色条
                            if i == 0 and j == 2:
                                plt.colorbar(hb, ax=ax, label='平均绝对误差')

                # 如果参数不在数据中，清空子图
                if param1 not in self.results.columns or (i != j and param2 not in self.results.columns):
                    ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])

        plt.suptitle('参数交互效应网格分析', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"交互效应网格图已保存: {save_path}")

        plt.show()

    def plot_improved_sensitivity_analysis(self, save_dir: Optional[Path] = None,
                                           band_id: str = 'band3'):
        """
        改进的敏感性分析，生成多个图形

        Parameters:
        -----------
        save_dir : Path, optional
            保存目录
        band_id : str
            波段ID，用于文件名
        """
        if save_dir is None:
            from config import ExperimentConfig
            save_dir = ExperimentConfig.FIGURES_DIR

        save_dir.mkdir(exist_ok=True)

        # 1. 条件箱线图
        cond_boxplot_file = save_dir / f"sensitivity_conditional_boxplots_{band_id}.png"
        self.plot_conditional_boxplots(cond_boxplot_file)

        # 2. 彩色小提琴图（针对SZA和VZA）
        violin_file = save_dir / f"sensitivity_violin_sza_vza_{band_id}.png"
        self.plot_violin_with_color(violin_file)

        # 3. 部分回归图（如果安装了statsmodels）
        try:
            import statsmodels
            partial_reg_file = save_dir / f"sensitivity_partial_regression_{band_id}.png"
            self.plot_partial_regression(partial_reg_file)
        except ImportError:
            self.logger.warning("statsmodels未安装，跳过部分回归分析")

        # 4. 交互效应网格图
        interaction_grid_file = save_dir / f"sensitivity_interaction_grid_{band_id}.png"
        self.plot_interaction_grid(interaction_grid_file)

        self.logger.info(f"改进的敏感性分析完成，图形保存到: {save_dir}")

    # 以下为原始函数，保持不变
    def plot_single_factor_sensitivity(self, save_path: Optional[Path] = None):
        """绘制单因素敏感性分析图（增加错误处理）"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        # 定义参数及其标签
        params = {
            'sza': {'label': '太阳天顶角 (°)', 'unit': '°'},
            'vza': {'label': '观测天顶角 (°)', 'unit': '°'},
            'aod550': {'label': 'AOD550', 'unit': ''},
            'h2o': {'label': '水汽含量', 'unit': 'g/cm²'},
            'o3': {'label': '臭氧含量', 'unit': 'cm-atm'},
            'rho_true': {'label': '地表反射率', 'unit': ''}
        }

        # 获取参数敏感性指标
        sensitivity = self.calculate_sensitivity_indices()

        for idx, (param_name, param_info) in enumerate(params.items()):
            ax = axes[idx]

            if param_name in self.results.columns:
                # 检查参数是否有变化
                unique_values = self.results[param_name].unique()
                if len(unique_values) < 2:
                    ax.text(0.5, 0.5, f'{param_info["label"]}值无变化',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{param_info["label"]}敏感性分析')
                    continue

                # 绘制散点图
                ax.scatter(self.results[param_name], self.results['error_absolute'],
                           alpha=0.3, s=10, color='blue')

                # 添加趋势线
                valid_mask = self.results[[param_name, 'error_absolute']].notna().all(axis=1)
                if valid_mask.sum() > 2:
                    x = self.results.loc[valid_mask, param_name].values
                    y = self.results.loc[valid_mask, 'error_absolute'].values

                    # 检查x值是否有变化
                    if len(np.unique(x)) < 2:
                        ax.text(0.5, 0.5, '数据点无变化，无法拟合',
                                ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{param_info["label"]}敏感性分析')
                        continue

                    try:
                        # 线性拟合
                        coeffs = np.polyfit(x, y, 1)
                        poly = np.poly1d(coeffs)
                        x_fit = np.linspace(x.min(), x.max(), 100)
                        y_fit = poly(x_fit)

                        ax.plot(x_fit, y_fit, 'r-', linewidth=2)

                        # 计算R²
                        y_pred = poly(x)
                        ss_res = np.sum((y - y_pred) ** 2)
                        ss_tot = np.sum((y - np.mean(y)) ** 2)
                        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                        # 添加敏感性指标
                        sens_info = sensitivity.get(param_name, {})
                        corr = sens_info.get('correlation', np.nan)
                        slope = sens_info.get('slope', np.nan)

                        text_str = f'R² = {r2:.3f}\nCorr = {corr:.3f}\nSlope = {slope:.6f}'
                        ax.text(0.05, 0.95, text_str, transform=ax.transAxes,
                                verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                    except Exception as e:
                        self.logger.warning(f"参数 {param_name} 拟合失败: {e}")
                        ax.text(0.05, 0.95, '拟合失败', transform=ax.transAxes,
                                verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                ax.set_xlabel(f'{param_info["label"]} ({param_info["unit"]})')
                ax.set_ylabel('绝对误差')
                ax.set_title(f'{param_info["label"]}敏感性分析')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'无{param_info["label"]}数据',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{param_info["label"]}敏感性分析')

        plt.suptitle('单因素敏感性分析', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"单因素敏感性图已保存: {save_path}")
        else:
            plt.show()

    def plot_interaction_effects(self, save_path: Optional[Path] = None):
        """绘制交互效应图"""
        # 选择两个最重要的参数进行交互分析
        sensitivity = self.calculate_sensitivity_indices()

        # 按相关性绝对值排序
        sorted_params = sorted(sensitivity.items(),
                               key=lambda x: abs(x[1]['correlation']),
                               reverse=True)

        if len(sorted_params) >= 2:
            # 选择前两个参数
            param1, sens1 = sorted_params[0]
            param2, sens2 = sorted_params[1]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # 1. 参数1 vs 参数2的误差等高线
            ax1 = axes[0]

            # 创建网格数据
            x = self.results[param1].values
            y = self.results[param2].values
            z = self.results['error_absolute'].values

            # 移除NaN值
            valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            z_valid = z[valid_mask]

            if len(x_valid) > 10:
                # 插值到规则网格
                xi = np.linspace(x_valid.min(), x_valid.max(), 50)
                yi = np.linspace(y_valid.min(), y_valid.max(), 50)
                xi_grid, yi_grid = np.meshgrid(xi, yi)

                # 线性插值
                from scipy.interpolate import griddata
                zi = griddata((x_valid, y_valid), z_valid, (xi_grid, yi_grid), method='linear')

                # 绘制等高线
                contour = ax1.contourf(xi_grid, yi_grid, zi, levels=20, cmap='RdBu_r')
                ax1.contour(xi_grid, yi_grid, zi, levels=10, colors='k', linewidths=0.5, alpha=0.5)

                ax1.set_xlabel(param1)
                ax1.set_ylabel(param2)
                ax1.set_title(f'{param1} 和 {param2} 交互效应')
                plt.colorbar(contour, ax=ax1, label='绝对误差')
                ax1.grid(True, alpha=0.3)

            # 2. 分组误差分析
            ax2 = axes[1]

            # 按参数1分组，分析参数2的影响
            param1_median = np.nanmedian(self.results[param1])
            param2_median = np.nanmedian(self.results[param2])

            conditions = [
                (self.results[param1] <= param1_median) & (self.results[param2] <= param2_median),
                (self.results[param1] <= param1_median) & (self.results[param2] > param2_median),
                (self.results[param1] > param1_median) & (self.results[param2] <= param2_median),
                (self.results[param1] > param1_median) & (self.results[param2] > param2_median)
            ]

            group_names = ['低-低', '低-高', '高-低', '高-高']
            group_errors = []
            group_stds = []

            for cond in conditions:
                errors = self.results.loc[cond, 'error_absolute'].dropna()
                group_errors.append(errors.mean() if len(errors) > 0 else np.nan)
                group_stds.append(errors.std() if len(errors) > 0 else np.nan)

            x_pos = range(len(group_names))
            ax2.bar(x_pos, group_errors, yerr=group_stds, capsize=5,
                    color=['blue', 'green', 'orange', 'red'], alpha=0.7)

            ax2.set_xlabel('参数组合')
            ax2.set_ylabel('平均误差')
            ax2.set_title(f'{param1}和{param2}组合影响')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(group_names)
            ax2.grid(True, axis='y', alpha=0.3)

            plt.suptitle('参数交互效应分析', fontsize=14)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"交互效应图已保存: {save_path}")
            else:
                plt.show()

    def plot_airmass_breakdown(self, save_path: Optional[Path] = None):
        """绘制大气质量分解分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        # 1. 总大气质量与误差关系
        ax1 = axes[0]
        if 'total_airmass' in self.results.columns:
            valid_mask = self.results[['total_airmass', 'error_absolute']].notna().all(axis=1)
            ax1.scatter(self.results.loc[valid_mask, 'total_airmass'],
                        self.results.loc[valid_mask, 'error_absolute'],
                        alpha=0.5, s=10)
            ax1.set_xlabel('总大气质量 (secθ_SZA + secθ_VZA)')
            ax1.set_ylabel('绝对误差')
            ax1.set_title('总大气质量 vs 误差')
            ax1.grid(True, alpha=0.3)

            # 添加趋势线
            if valid_mask.sum() > 2:
                x = self.results.loc[valid_mask, 'total_airmass'].values
                y = self.results.loc[valid_mask, 'error_absolute'].values
                coeffs = np.polyfit(x, y, 1)
                poly = np.poly1d(coeffs)
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = poly(x_fit)
                ax1.plot(x_fit, y_fit, 'r-', linewidth=2)

        # 2. 大气质量分量贡献
        ax2 = axes[1]
        if all(col in self.results.columns for col in ['secz_sza', 'secz_vza']):
            valid_mask = self.results[['secz_sza', 'secz_vza', 'error_absolute']].notna().all(axis=1)

            # 计算贡献比例
            total_airmass = self.results.loc[valid_mask, 'secz_sza'] + self.results.loc[valid_mask, 'secz_vza']
            sza_contribution = self.results.loc[valid_mask, 'secz_sza'] / total_airmass
            vza_contribution = self.results.loc[valid_mask, 'secz_vza'] / total_airmass

            # 按SZA贡献分组
            bins = np.linspace(0, 1, 6)
            sza_contribution_binned = pd.cut(sza_contribution, bins)

            error_by_contribution = self.results.loc[valid_mask].groupby(sza_contribution_binned)['error_absolute'].agg(
                ['mean', 'std', 'count'])

            x_pos = range(len(error_by_contribution))
            ax2.errorbar(x_pos, error_by_contribution['mean'],
                         yerr=error_by_contribution['std'], fmt='o-', capsize=5)

            ax2.set_xlabel('SZA大气质量贡献比例')
            ax2.set_ylabel('平均误差')
            ax2.set_title('大气质量分量贡献分析')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([f'{b.left:.1f}-{b.right:.1f}' for b in error_by_contribution.index])
            ax2.grid(True, alpha=0.3)

        # 3. 不同大气参数下的大气质量-误差关系
        ax3 = axes[2]
        if all(col in self.results.columns for col in ['aod550', 'h2o', 'o3', 'total_airmass', 'error_absolute']):
            # 按AOD分组
            if 'aod550' in self.results.columns:
                aod_groups = pd.cut(self.results['aod550'], bins=3)
                colors = ['blue', 'green', 'red']

                for (aod_bin, group), color in zip(self.results.groupby(aod_groups), colors):
                    valid_mask = group[['total_airmass', 'error_absolute']].notna().all(axis=1)
                    if valid_mask.sum() > 5:
                        ax3.scatter(group.loc[valid_mask, 'total_airmass'],
                                    group.loc[valid_mask, 'error_absolute'],
                                    alpha=0.5, s=10, color=color,
                                    label=f'AOD: {aod_bin}')

                ax3.set_xlabel('总大气质量')
                ax3.set_ylabel('绝对误差')
                ax3.set_title('不同AOD下的大气质量-误差关系')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

        # 4. 误差分解（方差贡献）
        ax4 = axes[3]

        # 计算各参数的方差贡献
        params_to_analyze = ['sza', 'vza', 'aod550', 'h2o', 'o3']
        variance_contributions = {}

        for param in params_to_analyze:
            if param in self.results.columns:
                valid_mask = self.results[[param, 'error_absolute']].notna().all(axis=1)
                if valid_mask.sum() > 10:
                    # 计算该参数能解释的方差比例
                    x = self.results.loc[valid_mask, param].values
                    y = self.results.loc[valid_mask, 'error_absolute'].values

                    # 回归分析
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    variance_contributions[param] = r_value ** 2

        if variance_contributions:
            params = list(variance_contributions.keys())
            contributions = [variance_contributions[p] for p in params]

            # 排序
            sorted_idx = np.argsort(contributions)[::-1]
            params_sorted = [params[i] for i in sorted_idx]
            contributions_sorted = [contributions[i] for i in sorted_idx]

            ax4.bar(range(len(params_sorted)), contributions_sorted,
                    color=plt.cm.Set3(np.arange(len(params_sorted)) / len(params_sorted)))

            ax4.set_xlabel('参数')
            ax4.set_ylabel('方差解释比例 (R²)')
            ax4.set_title('各参数对误差方差的贡献')
            ax4.set_xticks(range(len(params_sorted)))
            ax4.set_xticklabels(params_sorted, rotation=45)
            ax4.grid(True, axis='y', alpha=0.3)

        plt.suptitle('大气质量分解分析', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"大气质量分解图已保存: {save_path}")
        else:
            plt.show()

    def generate_sensitivity_report(self, save_path: Optional[Path] = None) -> str:
        """生成敏感性分析报告"""

        # 检查必要的列是否存在
        required_columns = ['error_absolute']
        missing_columns = [col for col in required_columns if col not in self.results.columns]

        if missing_columns:
            error_msg = f"错误：数据中缺少必要的列: {missing_columns}\n可用列: {list(self.results.columns)}"
            self.logger.error(error_msg)

            if save_path:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(error_msg)
            return error_msg
        """生成敏感性分析报告"""
        sensitivity = self.calculate_sensitivity_indices()

        report = []
        report.append("=" * 60)
        report.append("大气参数敏感性分析报告")
        report.append(f"样本数: {len(self.results)}")
        report.append(f"有效误差数据: {self.results['error_absolute'].notna().sum()}")
        report.append("=" * 60)
        report.append("")

        # 总体统计
        error_stats = self.results['error_absolute'].describe()
        report.append("误差统计摘要:")
        report.append("-" * 30)
        for stat, value in error_stats.items():
            report.append(f"{stat}: {value:.6f}")
        report.append("")

        # 敏感性指标
        report.append("参数敏感性指标 (按相关性排序):")
        report.append("-" * 60)

        # 按相关性绝对值排序
        sorted_sensitivity = sorted(sensitivity.items(),
                                    key=lambda x: abs(x[1]['correlation']),
                                    reverse=True)

        for param, sens in sorted_sensitivity:
            report.append(f"\n{param}:")
            report.append(f"  相关系数: {sens['correlation']:.4f}")
            report.append(f"  R²: {sens['r_squared']:.4f}")
            report.append(f"  斜率: {sens['slope']:.6f}")
            report.append(f"  标准化斜率: {sens['normalized_slope']:.4f}")
            report.append(f"  P值: {sens['p_value']:.6f}")
            report.append(f"  样本数: {sens['n_samples']}")

        report.append("")

        # 大气质量分析
        if 'total_airmass' in self.results.columns:
            report.append("大气质量分析:")
            report.append("-" * 30)

            total_airmass = self.results['total_airmass'].dropna()
            if len(total_airmass) > 0:
                report.append(f"平均总大气质量: {total_airmass.mean():.2f}")
                report.append(f"最小总大气质量: {total_airmass.min():.2f}")
                report.append(f"最大总大气质量: {total_airmass.max():.2f}")

                # 大气质量与误差相关性
                valid_mask = self.results[['total_airmass', 'error_absolute']].notna().all(axis=1)
                if valid_mask.sum() > 10:
                    corr = np.corrcoef(
                        self.results.loc[valid_mask, 'total_airmass'],
                        self.results.loc[valid_mask, 'error_absolute']
                    )[0, 1]
                    report.append(f"总大气质量与误差相关系数: {corr:.4f}")

        # 关键发现
        report.append("\n关键发现:")
        report.append("-" * 30)

        if sorted_sensitivity:
            # 最敏感的参数
            most_sensitive = sorted_sensitivity[0]
            report.append(f"1. 最敏感参数: {most_sensitive[0]} (相关系数: {most_sensitive[1]['correlation']:.4f})")

            # 水汽和臭氧的影响
            if 'h2o' in sensitivity:
                h2o_sens = sensitivity['h2o']
                report.append(f"2. 水汽敏感性: 相关系数 = {h2o_sens['correlation']:.4f}, "
                              f"标准化斜率 = {h2o_sens['normalized_slope']:.4f}")

            if 'o3' in sensitivity:
                o3_sens = sensitivity['o3']
                report.append(f"3. 臭氧敏感性: 相关系数 = {o3_sens['correlation']:.4f}, "
                              f"标准化斜率 = {o3_sens['normalized_slope']:.4f}")

            # 协同效应
            report.append("\n4. 参数协同效应建议:")
            report.append("   - 高角度条件下，水汽和AOD的协同效应需重点关注")
            report.append("   - 臭氧在可见光波段影响较小，但在紫外波段可能更重要")
            report.append("   - 地表反射率与大气参数的相互作用需要考虑")

        report_str = "\n".join(report)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_str)
            self.logger.info(f"敏感性报告已保存: {save_path}")

        return report_str