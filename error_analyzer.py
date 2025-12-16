# ==================== error_analyzer.py ====================
"""
误差分析模块
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from pathlib import Path
from scipy import stats
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from config import ExperimentConfig
from utils import setup_logger, calculate_statistics
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False

class ErrorAnalyzer:
    """误差分析器"""

    def __init__(self, results: Dict[str, pd.DataFrame], logger=None):
        """
        初始化误差分析器

        Parameters:
        -----------
        results : dict
            各波段的模拟结果
        logger : logging.Logger, optional
            日志记录器
        """
        self.results = results
        self.logger = logger or setup_logger('ErrorAnalyzer')

        # 合并所有波段数据
        self.all_data = self._combine_bands()

        # 计算统计量
        self.stats = self.calculate_overall_statistics()

    def _combine_bands(self) -> pd.DataFrame:
        """合并所有波段数据"""
        dfs = []
        for band_id, df in self.results.items():
            df_band = df.copy()
            df_band['band'] = band_id
            df_band['wavelength'] = ExperimentConfig.BANDS[band_id]['wavelength']
            dfs.append(df_band)

        return pd.concat(dfs, ignore_index=True)

    def calculate_overall_statistics(self) -> Dict[str, Dict[str, float]]:
        """计算总体统计量"""
        stats_dict = {}

        try:
            # 按波段计算
            for band_id in self.results.keys():
                df_band = self.results[band_id]

                # 检查是否有 error_absolute 列
                if 'error_absolute' in df_band.columns:
                    errors = df_band['error_absolute'].dropna().values

                    if len(errors) > 0:
                        try:
                            stats_dict[band_id] = calculate_statistics(errors, prefix='')
                        except:
                            stats_dict[band_id] = {'error': '计算统计量失败'}

            # 所有波段合并
            all_errors = []
            for df in self.results.values():
                if 'error_absolute' in df.columns:
                    errors = df['error_absolute'].dropna().values
                    all_errors.extend(errors.tolist() if hasattr(errors, 'tolist') else errors)

            all_errors = np.array(all_errors)
            if len(all_errors) > 0:
                try:
                    stats_dict['all'] = calculate_statistics(all_errors, prefix='')
                except:
                    stats_dict['all'] = {'error': '计算总统计量失败'}

        except Exception as e:
            self.logger.error(f"计算统计量失败: {e}")
            stats_dict['error'] = str(e)

        return stats_dict

    def plot_error_distribution(self, save_path: Optional[Path] = None):
        """绘制误差分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        # 1. 误差直方图
        errors = self.all_data['error_absolute'].dropna()
        axes[0].hist(errors, bins=50, density=True, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('绝对误差')
        axes[0].set_ylabel('概率密度')
        axes[0].set_title('误差分布直方图')
        axes[0].grid(True, alpha=0.3)

        # 添加统计信息
        stats_text = (
            f"均值: {errors.mean():.4f}\n"
            f"标准差: {errors.std():.4f}\n"
            f"RMSE: {np.sqrt(np.mean(errors ** 2)):.4f}\n"
            f"样本数: {len(errors):,}"
        )
        axes[0].text(0.05, 0.95, stats_text, transform=axes[0].transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 2. 误差与SZA的关系
        valid_data = self.all_data.dropna(subset=['error_absolute', 'sza'])
        axes[1].scatter(valid_data['sza'], valid_data['error_absolute'],
                        alpha=0.5, s=1, c=valid_data['wavelength'], cmap='viridis')
        axes[1].set_xlabel('太阳天顶角 (度)')
        axes[1].set_ylabel('绝对误差')
        axes[1].set_title('误差 vs 太阳天顶角')
        axes[1].grid(True, alpha=0.3)

        # 3. 误差与VZA的关系
        valid_data = self.all_data.dropna(subset=['error_absolute', 'vza'])
        axes[2].scatter(valid_data['vza'], valid_data['error_absolute'],
                        alpha=0.5, s=1, c=valid_data['wavelength'], cmap='viridis')
        axes[2].set_xlabel('观测天顶角 (度)')
        axes[2].set_ylabel('绝对误差')
        axes[2].set_title('误差 vs 观测天顶角')
        axes[2].grid(True, alpha=0.3)

        # 4. 误差与sec(SZA)*sec(VZA)的关系
        valid_data = valid_data.copy()
        valid_data['airmass_product'] = valid_data['secz_sza'] * valid_data['secz_vza']

        axes[3].scatter(valid_data['airmass_product'], valid_data['error_absolute'],
                        alpha=0.5, s=1, c=valid_data['wavelength'], cmap='viridis')
        axes[3].set_xlabel('sec(SZA) × sec(VZA)')
        axes[3].set_ylabel('绝对误差')
        axes[3].set_title('误差 vs 大气质量乘积')
        axes[3].grid(True, alpha=0.3)

        # 添加线性拟合
        if len(valid_data) > 2:
            x = valid_data['airmass_product'].values
            y = valid_data['error_absolute'].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = slope * x_fit + intercept
            axes[3].plot(x_fit, y_fit, 'r-', linewidth=2,
                         label=f'拟合: y = {slope:.4f}x + {intercept:.4f}\nR2 = {r_value ** 2:.4f}')
            axes[3].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"误差分布图已保存: {save_path}")

        # plt.show()

    def plot_layered_contours(self,
                              primary_factors: List[str] = ['sza', 'vza'],
                              color_factor: str = 'error_absolute',
                              fixed_conditions: Dict[str, Any] = None,
                              n_subplots: int = 4,
                              save_path: Optional[Path] = None):
        """
        绘制分层Contour图，展示多因素影响

        Parameters:
        -----------
        primary_factors : list
            主坐标轴因素，如['sza', 'vza']
        color_factor : str
            颜色映射因素，如'error_absolute', 'aod550', 'rho_true'
        fixed_conditions : dict
            固定条件，如{'aod550': [0.1, 0.3, 0.5], 'rho_true': 0.2}
        n_subplots : int
            子图数量（根据可变因素数量）
        """
        from typing import Any

        if fixed_conditions is None:
            fixed_conditions = {'aod550': 0.3, 'rho_true': 0.2, 'band': 'band3'}

        # 创建子图网格
        fig, axes = plt.subplots(1, n_subplots, figsize=(5 * n_subplots, 4))
        if n_subplots == 1:
            axes = [axes]

        # 获取可变因素列表
        variable_factors = []
        for factor_name, factor_values in fixed_conditions.items():
            if isinstance(factor_values, (list, np.ndarray)) and len(factor_values) > 1:
                variable_factors.append((factor_name, factor_values))

        # 如果可变因素不足，使用固定值创建对比
        if len(variable_factors) < n_subplots:
            # 使用波段作为可变因素
            bands = list(self.results.keys())
            if len(bands) >= n_subplots:
                variable_factors = [('band', bands[:n_subplots])]

        # 绘制每个子图
        for i in range(n_subplots):
            if i < len(variable_factors):
                factor_name, factor_values = variable_factors[i]
                # 选择中间值作为代表
                if isinstance(factor_values, (list, np.ndarray)):
                    mid_value = factor_values[len(factor_values) // 2]
                else:
                    mid_value = factor_values

                # 筛选数据
                subset = self.all_data[self.all_data[factor_name] == mid_value].copy()
                ax = axes[i]

                # 调用内部绘图方法
                self._plot_single_contour(subset, ax, primary_factors, color_factor)
                ax.set_title(f'{factor_name} = {mid_value}', fontsize=10)
            else:
                axes[i].axis('off')

        plt.suptitle(f'Layered Contours: {color_factor} vs {primary_factors[0]} and {primary_factors[1]}',
                     fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"分层Contour图已保存: {save_path}")

    def _plot_single_contour(self, data: pd.DataFrame, ax,
                             primary_factors: List[str],
                             color_factor: str):
        """绘制单个Contour图（内部方法）"""
        if len(data) < 10:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center')
            return

        # 创建网格
        x_unique = np.sort(data[primary_factors[0]].unique())
        y_unique = np.sort(data[primary_factors[1]].unique())

        if len(x_unique) < 2 or len(y_unique) < 2:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center')
            return

        # 插值到规则网格
        x_grid, y_grid = np.meshgrid(x_unique, y_unique)
        z_grid = np.full_like(x_grid, np.nan)

        for i, x_val in enumerate(x_unique):
            for j, y_val in enumerate(y_unique):
                mask = (data[primary_factors[0]] == x_val) & (data[primary_factors[1]] == y_val)
                if mask.any():
                    z_grid[j, i] = data.loc[mask, color_factor].mean()

        # 绘制等高线
        contour = ax.contourf(x_grid, y_grid, z_grid, levels=20, cmap='RdBu_r')
        ax.contour(x_grid, y_grid, z_grid, levels=10, colors='k', linewidths=0.5, alpha=0.5)

        ax.set_xlabel(f'{primary_factors[0]} (deg)')
        ax.set_ylabel(f'{primary_factors[1]} (deg)')

        # 添加颜色条
        plt.colorbar(contour, ax=ax, label=color_factor)
        ax.grid(True, alpha=0.3)

    def plot_3d_error_surface(self,
                              fixed_conditions: Dict[str, float] = None,
                              save_path: Optional[Path] = None):
        """
        绘制3D误差曲面及其2D投影
        """
        # 筛选固定条件下的数据
        if fixed_conditions is None:
            fixed_conditions = {'aod550': 0.3, 'rho_true': 0.2, 'band': 'band3'}

        mask = pd.Series(True, index=self.all_data.index)
        for key, value in fixed_conditions.items():
            if key in self.all_data.columns:
                mask = mask & (self.all_data[key] == value)

        data = self.all_data[mask]

        if len(data) < 10:
            self.logger.warning("数据不足，无法绘制3D曲面")
            return

        # 创建3D图形
        fig = plt.figure(figsize=(15, 5))

        # 子图1：3D曲面
        ax1 = fig.add_subplot(131, projection='3d')

        # 三角化曲面
        ax1.plot_trisurf(data['sza'], data['vza'], data['error_absolute'],
                         cmap='viridis', alpha=0.8, edgecolor='none')
        ax1.set_xlabel('SZA (deg)')
        ax1.set_ylabel('VZA (deg)')
        ax1.set_zlabel('Absolute Error')
        ax1.set_title('3D Error Surface')

        # 子图2：XY投影（Contour）
        ax2 = fig.add_subplot(132)

        # 使用tricontourf处理不规则网格
        contour = ax2.tricontourf(data['sza'], data['vza'], data['error_absolute'],
                                  levels=20, cmap='RdBu_r')
        plt.colorbar(contour, ax=ax2, label='Absolute Error')
        ax2.set_xlabel('SZA (deg)')
        ax2.set_ylabel('VZA (deg)')
        ax2.set_title('2D Error Contour')
        ax2.grid(True, alpha=0.3)

        # 子图3：误差分布直方图
        ax3 = fig.add_subplot(133)
        ax3.hist(data['error_absolute'].dropna(), bins=50, density=True,
                 alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Absolute Error')
        ax3.set_ylabel('Density')
        ax3.set_title('Error Distribution')
        ax3.grid(True, alpha=0.3)

        # 添加正态分布拟合
        from scipy.stats import norm
        errors = data['error_absolute'].dropna()
        if len(errors) > 0:
            mu, std = norm.fit(errors)
            x = np.linspace(errors.min(), errors.max(), 100)
            p = norm.pdf(x, mu, std)
            ax3.plot(x, p, 'r-', linewidth=2, label=f'N({mu:.4f}, {std:.4f}²)')
            ax3.legend()

        conditions_str = ', '.join([f'{k}={v}' for k, v in fixed_conditions.items()])
        plt.suptitle(f'3D Error Analysis - Fixed: {conditions_str}', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"3D误差曲面图已保存: {save_path}")

    # =============== 新增方法：改进的单因素分析 ===============

    def plot_conditional_quantiles(self,
                                   factor: str,
                                   conditional_factors: List[str] = None,
                                   n_quantiles: int = 5,
                                   save_path: Optional[Path] = None):
        """
        绘制条件分位数回归图，展示在控制其他因素情况下的分布

        Parameters:
        -----------
        factor : str
            分析的因素（如'sza', 'vza'）
        conditional_factors : list
            需要控制的因素列表
        n_quantiles : int
            分位数数量
        """
        if conditional_factors is None:
            conditional_factors = ['aod550', 'rho_true', 'band']

        # 过滤有效数据
        data = self.all_data.copy()
        valid_conditional_factors = [f for f in conditional_factors if f in data.columns]

        # 使用seaborn的conditional plotting
        n_rows = len(valid_conditional_factors)
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows))

        if n_rows == 1:
            axes = [axes]

        for i, cond_factor in enumerate(valid_conditional_factors):
            ax = axes[i]

            # 对条件因素进行分箱（如果是连续变量）
            if data[cond_factor].dtype.kind in 'fi' and len(data[cond_factor].unique()) > 5:
                # 分成3组
                data[f'{cond_factor}_binned'] = pd.qcut(data[cond_factor], 3, duplicates='drop')
                hue_var = f'{cond_factor}_binned'
            else:
                hue_var = cond_factor

            # 绘制散点图
            sc = ax.scatter(data[factor], data['error_absolute'],
                            c=pd.factorize(data[hue_var])[0],
                            cmap='viridis', alpha=0.5, s=20)

            # 添加分位数回归线
            for q in np.linspace(0.1, 0.9, n_quantiles):
                # 对每个条件水平分别拟合
                unique_conditions = data[hue_var].unique()
                for cond_value in unique_conditions:
                    mask = data[hue_var] == cond_value
                    if mask.sum() > 10:  # 确保有足够的数据点
                        X = data.loc[mask, factor].values.reshape(-1, 1)
                        X_with_const = np.column_stack([np.ones(len(X)), X])
                        y = data.loc[mask, 'error_absolute'].values

                        try:
                            qr = QuantReg(y, X_with_const).fit(q=q)
                            x_range = np.linspace(data[factor].min(), data[factor].max(), 100)
                            y_pred = qr.params[0] + qr.params[1] * x_range

                            # 使用不同线型区分分位数
                            linestyle = '--' if q < 0.5 else '-'
                            alpha = 0.7 - abs(q - 0.5)
                            ax.plot(x_range, y_pred, linestyle, alpha=alpha, color='gray', lw=1)
                        except:
                            continue

            # 添加LOESS平滑曲线
            try:
                import statsmodels.api as sm
                lowess = sm.nonparametric.lowess
                z = lowess(data['error_absolute'], data[factor], frac=0.3)
                ax.plot(z[:, 0], z[:, 1], 'r-', lw=2, label='LOESS Fit')
            except:
                pass

            ax.set_xlabel(factor)
            ax.set_ylabel('Absolute Error')
            ax.set_title(f'Conditional on {cond_factor}')
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.suptitle(f'Conditional Quantile Regression: {factor}', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"条件分位数回归图已保存: {save_path}")

    def plot_violin_swarm_by_factor(self,
                                    factor: str,
                                    categorize_by: str = None,
                                    save_path: Optional[Path] = None):
        """
        绘制小提琴图与蜂群图的组合

        Parameters:
        -----------
        factor : str
            分析的因素（如'sza', 'vza'）
        categorize_by : str, optional
            按此变量分类显示子图
        save_path : Path, optional
            保存路径
        """
        # 如果因素连续，进行分箱
        data = self.all_data.copy()

        if data[factor].dtype.kind in 'fi':  # 数值型
            n_bins = min(8, len(data[factor].unique()))
            if n_bins > 1:
                bins = pd.qcut(data[factor], n_bins, duplicates='drop')
                data[f'{factor}_binned'] = bins
                x_var = f'{factor}_binned'
            else:
                x_var = factor
        else:
            x_var = factor

        # 如果有分类变量，使用子图
        if categorize_by and categorize_by in data.columns:
            categories = data[categorize_by].unique()[:4]  # 限制类别数量
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.ravel()

            for idx, category in enumerate(categories):
                if idx >= 4:
                    break

                cat_data = data[data[categorize_by] == category]
                self._plot_single_violin_swarm(cat_data, x_var, factor, axes[idx], None)
                axes[idx].set_title(f'{categorize_by} = {category}', fontsize=10)

            plt.suptitle(f'Error Distribution by {factor} (Grouped by {categorize_by})', fontsize=14)
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                           gridspec_kw={'height_ratios': [3, 1]})
            self._plot_single_violin_swarm(data, x_var, factor, ax1, ax2)
            plt.suptitle(f'Error Distribution by {factor}', fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"小提琴+蜂群图已保存: {save_path}")

    def _plot_single_violin_swarm(self, data: pd.DataFrame, x_var: str, factor: str,
                                  ax1, ax2=None):
        """
        绘制单个小提琴+蜂群图（内部方法）

        Parameters:
        -----------
        data : pd.DataFrame
            数据
        x_var : str
            x轴变量名（可能被分箱）
        factor : str
            原始因素名称
        ax1 : matplotlib.axes.Axes
            主坐标轴（小提琴图）
        ax2 : matplotlib.axes.Axes, optional
            次坐标轴（蜂群图）
        """
        # 子图1：小提琴图
        sns.violinplot(x=x_var, y='error_absolute', data=data,
                       inner='quartile', palette='muted', ax=ax1)

        # 添加中位数标记
        medians = data.groupby(x_var)['error_absolute'].median()
        ax1.scatter(range(len(medians)), medians.values,
                    color='red', s=100, zorder=5, label='Median')

        ax1.set_ylabel('Absolute Error')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_xlabel(factor if factor == x_var else f'{factor} (binned)')

        # 子图2：蜂群图（随机子样本避免过密）
        if ax2 is not None:
            sample_size = min(500, len(data))
            sample_data = data.sample(sample_size, random_state=42) if len(data) > sample_size else data

            sns.swarmplot(x=x_var, y='error_absolute', data=sample_data,
                          size=3, alpha=0.6, ax=ax2)

            # 添加线性趋势线
            try:
                x_numeric = pd.factorize(sample_data[x_var])[0]
                valid_idx = ~np.isnan(x_numeric) & ~np.isnan(sample_data['error_absolute'])
                x_numeric = x_numeric[valid_idx]
                y_values = sample_data['error_absolute'].values[valid_idx]

                if len(x_numeric) > 2:
                    coeffs = np.polyfit(x_numeric, y_values, 1)
                    poly = np.poly1d(coeffs)
                    x_range = np.linspace(0, len(sample_data[x_var].unique()) - 1, 100)
                    ax2.plot(x_range, poly(x_range), 'r--', lw=2, label='Trend')
            except:
                pass

            ax2.set_xlabel(factor if factor == x_var else f'{factor} (binned)')
            ax2.set_ylabel('Absolute Error')
            ax2.legend()
            ax2.tick_params(axis='x', rotation=45)

    def plot_partial_dependence(self,
                                factor: str,
                                model_type: str = 'random_forest',
                                fixed_conditions: Dict[str, float] = None,
                                save_path: Optional[Path] = None):
        """
        绘制部分依赖图，展示单个因素的影响
        """
        # 筛选数据
        data = self.all_data.copy()
        if fixed_conditions:
            for key, value in fixed_conditions.items():
                if key in data.columns:
                    data = data[data[key] == value]

        if len(data) < 50:
            self.logger.warning("数据不足，无法计算部分依赖")
            return

        # 准备特征和目标
        features = ['sza', 'vza', 'aod550', 'rho_true']
        features = [f for f in features if f in data.columns and f != factor]

        # 确保所有特征都是数值型
        X = data[features].values
        y = data['error_absolute'].values

        # 移除NaN
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 50:
            self.logger.warning("有效数据不足")
            return

        # 训练模型
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100,
                                          random_state=42,
                                          min_samples_split=5,
                                          max_depth=10)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(n_estimators=100,
                                              random_state=42,
                                              max_depth=5)
        else:
            self.logger.warning(f"不支持的模型类型: {model_type}")
            return

        model.fit(X, y)

        # 计算部分依赖
        fig, ax = plt.subplots(figsize=(10, 6))

        # 获取因素索引
        feature_names = features.copy()
        if factor not in feature_names:
            feature_names.append(factor)

        feature_idx = feature_names.index(factor)

        try:
            PartialDependenceDisplay.from_estimator(
                model, X, features=[feature_idx],
                feature_names=feature_names,
                grid_resolution=50, ax=ax
            )

            ax.set_xlabel(factor)
            ax.set_ylabel('Partial Dependence')
            ax.set_title(f'Partial Dependence of Error on {factor}')
            ax.grid(True, alpha=0.3)

            # 添加特征重要性
            importance = None
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_names[:len(model.feature_importances_)],
                                      model.feature_importances_))

            if importance:
                importance_text = "Feature Importance:\n" + "\n".join(
                    [f"{k}: {v:.3f}" for k, v in sorted(importance.items(),
                                                        key=lambda x: x[1],
                                                        reverse=True)[:3]]
                )
                ax.text(0.05, 0.95, importance_text, transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"部分依赖图已保存: {save_path}")

        except Exception as e:
            self.logger.error(f"计算部分依赖失败: {e}")

    def create_summary_figure(self,
                              fixed_conditions: Dict[str, float] = None,
                              band_id: str = 'band3',
                              save_path: Optional[Path] = None):
        """
        创建综合摘要图
        """
        if fixed_conditions is None:
            fixed_conditions = {'aod550': 0.3, 'rho_true': 0.2, 'band': band_id}

        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. 误差等高线图（最具代表性条件）
        ax1 = fig.add_subplot(gs[0, :2])
        data = self.all_data.copy()
        for key, value in fixed_conditions.items():
            if key in data.columns:
                data = data[data[key] == value]

        if len(data) > 10:
            self._plot_single_contour(data, ax1, ['sza', 'vza'], 'error_absolute')
            ax1.set_title('(a) Error Contour at Typical Conditions', fontsize=12)
        else:
            ax1.text(0.5, 0.5, '数据不足', ha='center', va='center')

        # 2. 3D误差曲面
        ax2 = fig.add_subplot(gs[0, 2], projection='3d')
        if len(data) > 10:
            ax2.plot_trisurf(data['sza'], data['vza'], data['error_absolute'],
                             cmap='viridis', alpha=0.8, edgecolor='none')
            ax2.set_xlabel('SZA')
            ax2.set_ylabel('VZA')
            ax2.set_zlabel('Error')
            ax2.set_title('(b) 3D Error Surface', fontsize=12)
        else:
            ax2.text(0.5, 0.5, 0.5, '数据不足', ha='center', va='center')

        # 3. SZA条件分位数回归（简化版）
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_simple_scatter_with_trend(data, 'sza', ax3)
        ax3.set_title('(c) SZA Effect', fontsize=12)

        # 4. VZA条件分位数回归（简化版）
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_simple_scatter_with_trend(data, 'vza', ax4)
        ax4.set_title('(d) VZA Effect', fontsize=12)

        # 5. 因素重要性（使用随机森林）
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_feature_importance(data, ax5)
        ax5.set_title('(e) Factor Importance', fontsize=12)

        # 6. 误差分布
        ax6 = fig.add_subplot(gs[2, 0])
        if 'error_absolute' in data.columns:
            errors = data['error_absolute'].dropna()
            ax6.hist(errors, bins=50, density=True, alpha=0.7, edgecolor='black')
            ax6.set_xlabel('Absolute Error')
            ax6.set_ylabel('Density')
            ax6.set_title('(f) Error Distribution', fontsize=12)
            ax6.grid(True, alpha=0.3)

        # 7. AOD影响
        ax7 = fig.add_subplot(gs[2, 1])
        if 'aod550' in data.columns:
            ax7.scatter(data['aod550'], data['error_absolute'], alpha=0.5, s=10)
            ax7.set_xlabel('AOD550')
            ax7.set_ylabel('Absolute Error')
            ax7.set_title('(g) AOD Effect', fontsize=12)
            ax7.grid(True, alpha=0.3)

        # 8. 地表反射率影响
        ax8 = fig.add_subplot(gs[2, 2])
        if 'rho_true' in data.columns:
            ax8.scatter(data['rho_true'], data['error_absolute'], alpha=0.5, s=10)
            ax8.set_xlabel('Surface Reflectance')
            ax8.set_ylabel('Absolute Error')
            ax8.set_title('(h) Reflectance Effect', fontsize=12)
            ax8.grid(True, alpha=0.3)

        conditions_str = ', '.join([f'{k}={v}' for k, v in fixed_conditions.items()])
        plt.suptitle(f'Summary of Geometric Error Analysis\nFixed Conditions: {conditions_str}',
                     fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"综合摘要图已保存: {save_path}")

    def _plot_simple_scatter_with_trend(self, data: pd.DataFrame,
                                        factor: str, ax):
        """绘制简单的散点图与趋势线（内部方法）"""
        ax.scatter(data[factor], data['error_absolute'],
                   alpha=0.3, s=10, color='blue')

        # 添加趋势线
        valid_mask = ~np.isnan(data[factor]) & ~np.isnan(data['error_absolute'])
        if valid_mask.sum() > 2:
            x = data.loc[valid_mask, factor].values
            y = data.loc[valid_mask, 'error_absolute'].values

            # 线性回归
            coeffs = np.polyfit(x, y, 1)
            poly = np.poly1d(coeffs)
            x_range = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_range, poly(x_range), 'r-', lw=2,
                    label=f'y={coeffs[0]:.4f}x+{coeffs[1]:.4f}')

            # 计算R²
            y_pred = poly(x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel(factor)
        ax.set_ylabel('Absolute Error')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_feature_importance(self, data: pd.DataFrame, ax):
        """绘制特征重要性图（内部方法）"""
        features = ['sza', 'vza', 'aod550', 'rho_true']
        features = [f for f in features if f in data.columns]

        X = data[features].values
        y = data['error_absolute'].values

        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) > 10:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1]

            ax.bar(range(len(features)), importance[indices], align='center')
            ax.set_xticks(range(len(features)))
            ax.set_xticklabels([features[i] for i in indices], rotation=45)
            ax.set_ylabel('Feature Importance')
            ax.set_title('Random Forest Feature Importance')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center')