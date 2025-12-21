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

        # 计算相对误差（如果不存在）
        if 'error_relative' not in self.results.columns and 'error_absolute' in self.results.columns and 'rho_true' in self.results.columns:
            self.results['error_relative'] = self.results['error_absolute'] / self.results['rho_true']
            self.logger.info("计算了相对误差 (error_absolute / rho_true)")

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

    def calculate_net_sensitivity_indices(self, target_var: str = 'error_absolute') -> Dict[str, float]:
        """
        计算净敏感性指标（控制其他因素）

        Parameters:
        -----------
        target_var : str
            目标变量

        Returns:
        --------
        dict
            净敏感性指标
        """
        from sklearn.linear_model import LinearRegression

        net_sensitivity = {}

        # 定义需要分析的参数（连续变量）
        params_to_analyze = ['sza', 'vza', 'aod550', 'h2o', 'o3', 'rho_true']

        # 只保留连续变量
        continuous_params = []
        for param in params_to_analyze:
            if param in self.results.columns:
                # 检查是否为连续变量（有足够多的不同值）
                unique_vals = self.results[param].dropna().unique()
                if len(unique_vals) >= 5:
                    continuous_params.append(param)

        if len(continuous_params) < 2:
            self.logger.warning("连续变量不足，无法计算净敏感性")
            return net_sensitivity

        # 准备数据：去除缺失值
        all_cols = continuous_params + [target_var]
        valid_mask = self.results[all_cols].notna().all(axis=1)
        valid_data = self.results.loc[valid_mask, all_cols]

        if len(valid_data) < 20:
            self.logger.warning(f"有效数据不足 ({len(valid_data)} < 20)，无法计算净敏感性")
            return net_sensitivity

        # 为每个参数计算净效应
        for param in continuous_params:
            try:
                # 控制变量：除了当前参数之外的所有其他参数
                control_params = [p for p in continuous_params if p != param]

                # 构建设计矩阵
                X_control = valid_data[control_params].values

                # 1. 用控制变量预测目标变量
                model_y = LinearRegression()
                model_y.fit(X_control, valid_data[target_var].values)
                y_residual = valid_data[target_var].values - model_y.predict(X_control)

                # 2. 用控制变量预测当前参数
                model_x = LinearRegression()
                model_x.fit(X_control, valid_data[param].values)
                x_residual = valid_data[param].values - model_x.predict(X_control)

                # 3. 计算残差之间的相关性（净效应）
                if np.std(x_residual) > 0 and np.std(y_residual) > 0:
                    # 计算偏相关系数
                    net_corr = np.corrcoef(x_residual, y_residual)[0, 1]

                    # 计算净回归系数
                    net_slope = net_corr * (np.std(y_residual) / np.std(x_residual))

                    # 计算净R²
                    net_r2 = net_corr ** 2

                    net_sensitivity[param] = {
                        'net_correlation': net_corr,
                        'net_slope': net_slope,
                        'net_r_squared': net_r2,
                        'n_samples': len(x_residual),
                        'y_residual_std': np.std(y_residual),
                        'x_residual_std': np.std(x_residual)
                    }

                    self.logger.info(f"参数 {param} 净效应: 偏相关系数={net_corr:.4f}, 净斜率={net_slope:.6f}")

            except Exception as e:
                self.logger.error(f"计算参数 {param} 净效应时出错: {e}")
                continue

        return net_sensitivity

    def plot_conditional_boxplots(self, save_path: Optional[Path] = None):
        """绘制条件箱线图，控制其他主要因素"""

        # 选择要分析的参数
        target_params = ['sza', 'vza', 'aod550', 'h2o', 'o3']

        # 创建图形 - 2行，每行2个子图（绝对误差和相对误差）
        n_params = len(target_params)
        fig, axes = plt.subplots(n_params, 2, figsize=(14, 4 * n_params))

        for idx, param in enumerate(target_params):
            # 左图：绝对误差的箱线图
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

                    # 绘制原始箱线图（绝对误差）
                    data = []
                    for group in labels:
                        group_data = self.results[self.results[f'{param}_group'] == group]
                        data.append(group_data['error_absolute'].dropna().values)

                    bp = ax1.boxplot(data, positions=range(len(labels)), widths=0.6)
                    ax1.set_xlabel(f'{param} group')
                    ax1.set_ylabel('Absolute error')
                    ax1.set_title(f'{param} - Absolute error boxplot')
                    ax1.set_xticks(range(len(labels)))
                    ax1.set_xticklabels(labels, rotation=45)
                    ax1.grid(True, alpha=0.3)

            # 右图：相对误差的箱线图
            ax2 = axes[idx, 1]

            if param in self.results.columns and 'error_relative' in self.results.columns:
                # 创建分组（与左图相同）
                if param == 'sza':
                    bins = np.arange(0, 91, 15)
                    labels = [f'{b}-{b + 14}°' for b in bins[:-1]]
                elif param == 'vza':
                    bins = np.arange(0, 76, 15)
                    labels = [f'{b}-{b + 14}°' for b in bins[:-1]]
                else:
                    unique_vals = np.sort(self.results[param].dropna().unique())
                    if len(unique_vals) > 5:
                        bins = np.percentile(self.results[param].dropna(), [0, 25, 50, 75, 100])
                    else:
                        bins = np.linspace(self.results[param].min(),
                                           self.results[param].max(), len(unique_vals) + 1)
                    labels = [f'Q{i + 1}' for i in range(len(bins) - 1)]

                # 绘制相对误差箱线图
                data_rel = []
                for group in labels:
                    group_data = self.results[self.results[f'{param}_group'] == group]
                    data_rel.append(group_data['error_relative'].dropna().values)

                bp_rel = ax2.boxplot(data_rel, positions=range(len(labels)), widths=0.6)
                ax2.set_xlabel(f'{param} group')
                ax2.set_ylabel('Relative error')
                ax2.set_title(f'{param} - Relative error boxplot')
                ax2.set_xticks(range(len(labels)))
                ax2.set_xticklabels(labels, rotation=45)
                ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"条件箱线图已保存: {save_path}")

        plt.show()

    def plot_single_factor_sensitivity(self, save_path: Optional[Path] = None):
        """绘制单因素敏感性分析图（增加相对误差分析）"""
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.ravel()

        # 定义参数及其标签
        params = {
            'sza': {'label': 'Solar zenith angle', 'unit': '°'},
            'vza': {'label': 'View zenith angle', 'unit': '°'},
            'aod550': {'label': 'AOD550', 'unit': ''},
            'h2o': {'label': 'Water vapor', 'unit': 'g/cm²'},
            'o3': {'label': 'Ozone', 'unit': 'cm-atm'},
            'rho_true': {'label': 'Surface reflectance', 'unit': ''}
        }

        # 获取参数敏感性指标（绝对误差）
        sensitivity_abs = self.calculate_sensitivity_indices('error_absolute')

        # 获取参数敏感性指标（相对误差）
        sensitivity_rel = {}
        if 'error_relative' in self.results.columns:
            sensitivity_rel = self.calculate_sensitivity_indices('error_relative')

        for idx, (param_name, param_info) in enumerate(params.items()):
            ax_abs = axes[idx * 2]  # 绝对误差子图
            ax_rel = axes[idx * 2 + 1]  # 相对误差子图

            # 绘制绝对误差敏感性
            if param_name in self.results.columns:
                # 检查参数是否有变化
                unique_values = self.results[param_name].unique()
                if len(unique_values) < 2:
                    ax_abs.text(0.5, 0.5, f'{param_info["label"]} values no variation',
                                ha='center', va='center', transform=ax_abs.transAxes)
                    ax_abs.set_title(f'{param_info["label"]} - Absolute')
                    continue

                # 绘制散点图（绝对误差）
                ax_abs.scatter(self.results[param_name], self.results['error_absolute'],
                               alpha=0.3, s=10, color='blue')

                # 添加趋势线
                valid_mask = self.results[[param_name, 'error_absolute']].notna().all(axis=1)
                if valid_mask.sum() > 2:
                    x = self.results.loc[valid_mask, param_name].values
                    y = self.results.loc[valid_mask, 'error_absolute'].values

                    # 检查x值是否有变化
                    if len(np.unique(x)) < 2:
                        ax_abs.text(0.5, 0.5, 'Data points no variation, cannot fit',
                                    ha='center', va='center', transform=ax_abs.transAxes)
                        ax_abs.set_title(f'{param_info["label"]} - Absolute')
                        continue

                    try:
                        # 线性拟合
                        coeffs = np.polyfit(x, y, 1)
                        poly = np.poly1d(coeffs)
                        x_fit = np.linspace(x.min(), x.max(), 100)
                        y_fit = poly(x_fit)

                        ax_abs.plot(x_fit, y_fit, 'r-', linewidth=2)

                        # 添加敏感性指标
                        sens_info = sensitivity_abs.get(param_name, {})
                        corr = sens_info.get('correlation', np.nan)
                        slope = sens_info.get('slope', np.nan)
                        r2 = sens_info.get('r_squared', np.nan)

                        text_str = f'R² = {r2:.3f}\nCorr = {corr:.3f}\nSlope = {slope:.6f}'
                        ax_abs.text(0.05, 0.95, text_str, transform=ax_abs.transAxes,
                                    verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                    except Exception as e:
                        self.logger.warning(f"参数 {param_name} 绝对误差拟合失败: {e}")

                ax_abs.set_xlabel(f'{param_info["label"]} ({param_info["unit"]})')
                ax_abs.set_ylabel('Absolute error')
                ax_abs.set_title(f'{param_info["label"]} - Absolute')
                ax_abs.grid(True, alpha=0.3)
            else:
                ax_abs.text(0.5, 0.5, f'No {param_info["label"]} data',
                            ha='center', va='center', transform=ax_abs.transAxes)
                ax_abs.set_title(f'{param_info["label"]} - Absolute')

            # 绘制相对误差敏感性
            if param_name in self.results.columns and 'error_relative' in self.results.columns:
                # 检查参数是否有变化
                unique_values = self.results[param_name].unique()
                if len(unique_values) < 2:
                    ax_rel.text(0.5, 0.5, f'{param_info["label"]} values no variation',
                                ha='center', va='center', transform=ax_rel.transAxes)
                    ax_rel.set_title(f'{param_info["label"]} - Relative')
                    continue

                # 绘制散点图（相对误差）
                ax_rel.scatter(self.results[param_name], self.results['error_relative'],
                               alpha=0.3, s=10, color='green')

                # 添加趋势线
                valid_mask = self.results[[param_name, 'error_relative']].notna().all(axis=1)
                if valid_mask.sum() > 2:
                    x = self.results.loc[valid_mask, param_name].values
                    y = self.results.loc[valid_mask, 'error_relative'].values

                    # 检查x值是否有变化
                    if len(np.unique(x)) < 2:
                        ax_rel.text(0.5, 0.5, 'Data points no variation, cannot fit',
                                    ha='center', va='center', transform=ax_rel.transAxes)
                        ax_rel.set_title(f'{param_info["label"]} - Relative')
                        continue

                    try:
                        # 线性拟合
                        coeffs = np.polyfit(x, y, 1)
                        poly = np.poly1d(coeffs)
                        x_fit = np.linspace(x.min(), x.max(), 100)
                        y_fit = poly(x_fit)

                        ax_rel.plot(x_fit, y_fit, 'r-', linewidth=2)

                        # 添加敏感性指标
                        sens_info = sensitivity_rel.get(param_name, {})
                        corr = sens_info.get('correlation', np.nan)
                        slope = sens_info.get('slope', np.nan)
                        r2 = sens_info.get('r_squared', np.nan)

                        text_str = f'R² = {r2:.3f}\nCorr = {corr:.3f}\nSlope = {slope:.6f}'
                        ax_rel.text(0.05, 0.95, text_str, transform=ax_rel.transAxes,
                                    verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

                    except Exception as e:
                        self.logger.warning(f"参数 {param_name} 相对误差拟合失败: {e}")

                ax_rel.set_xlabel(f'{param_info["label"]} ({param_info["unit"]})')
                ax_rel.set_ylabel('Relative error')
                ax_rel.set_title(f'{param_info["label"]} - Relative')
                ax_rel.grid(True, alpha=0.3)
            else:
                ax_rel.text(0.5, 0.5, 'No relative error data',
                            ha='center', va='center', transform=ax_rel.transAxes)
                ax_rel.set_title(f'{param_info["label"]} - Relative')

        plt.suptitle('Single factor sensitivity analysis (Left: Absolute error, Right: Relative error)', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"单因素敏感性图已保存: {save_path}")
        else:
            plt.show()

    def plot_net_sensitivity_analysis(self, save_path: Optional[Path] = None):
        """
        绘制净效应敏感性分析图

        净效应：控制其他因素不变，分析单个因素的影响
        """
        # 计算原始效应和净效应
        raw_sensitivity_abs = self.calculate_sensitivity_indices('error_absolute')
        net_sensitivity_abs = self.calculate_net_sensitivity_indices('error_absolute')

        # 计算相对误差的敏感性（如果存在）
        raw_sensitivity_rel = {}
        net_sensitivity_rel = {}
        if 'error_relative' in self.results.columns:
            raw_sensitivity_rel = self.calculate_sensitivity_indices('error_relative')
            net_sensitivity_rel = self.calculate_net_sensitivity_indices('error_relative')

        # 定义参数及其标签
        params = {
            'sza': {'label': 'Solar zenith angle', 'unit': '°'},
            'vza': {'label': 'View zenith angle', 'unit': '°'},
            'aod550': {'label': 'AOD550', 'unit': ''},
            'h2o': {'label': 'Water vapor', 'unit': 'g/cm²'},
            'o3': {'label': 'Ozone', 'unit': 'cm-atm'},
            'rho_true': {'label': 'Surface reflectance', 'unit': ''}
        }

        # 创建图形 - 2行，每行3个参数
        n_params = len(params)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        for idx, (param_name, param_info) in enumerate(params.items()):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # 准备数据
            data = []
            labels = []
            colors = []

            # 绝对误差的原始效应和净效应
            if param_name in raw_sensitivity_abs:
                raw_data = raw_sensitivity_abs[param_name]
                data.append(abs(raw_data['correlation']))  # 取绝对值便于比较
                labels.append('Raw Effect (Abs)')
                colors.append('skyblue')

                # 添加净效应（如果存在）
                if param_name in net_sensitivity_abs:
                    net_data = net_sensitivity_abs[param_name]
                    data.append(abs(net_data['net_correlation']))
                    labels.append('Net Effect (Abs)')
                    colors.append('steelblue')

            # 相对误差的原始效应和净效应（如果存在）
            if 'error_relative' in self.results.columns and param_name in raw_sensitivity_rel:
                raw_data_rel = raw_sensitivity_rel[param_name]
                data.append(abs(raw_data_rel['correlation']))
                labels.append('Raw Effect (Rel)')
                colors.append('lightgreen')

                if param_name in net_sensitivity_rel:
                    net_data_rel = net_sensitivity_rel[param_name]
                    data.append(abs(net_data_rel['net_correlation']))
                    labels.append('Net Effect (Rel)')
                    colors.append('forestgreen')

            if not data:
                ax.text(0.5, 0.5, f'No data for {param_name}',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(param_info['label'])
                continue

            # 绘制柱状图
            bars = ax.bar(range(len(data)), data, color=colors)

            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, data)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)

            # 设置图表属性
            ax.set_xlabel('Effect Type')
            ax.set_ylabel('|Correlation|')
            ax.set_title(f'{param_info["label"]}\nRaw vs Net Effect')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')

            # 设置y轴范围
            ax.set_ylim(0, max(data) * 1.2)

        # 隐藏多余的子图
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Net Sensitivity Analysis: Controlling for Other Factors\n(Comparing Raw Effects vs Net Effects)',
                     fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"净效应敏感性分析图已保存: {save_path}")
        else:
            plt.show()

    def plot_airmass_breakdown(self, save_path: Optional[Path] = None):
        """绘制大气质量分解分析图（增加相对误差分析）"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        # 1. 总大气质量与绝对误差关系
        ax1 = axes[0]
        if 'total_airmass' in self.results.columns:
            valid_mask = self.results[['total_airmass', 'error_absolute']].notna().all(axis=1)
            ax1.scatter(self.results.loc[valid_mask, 'total_airmass'],
                        self.results.loc[valid_mask, 'error_absolute'],
                        alpha=0.5, s=10, color='blue', label='Absolute')

            # 添加趋势线
            if valid_mask.sum() > 2:
                x = self.results.loc[valid_mask, 'total_airmass'].values
                y = self.results.loc[valid_mask, 'error_absolute'].values
                coeffs = np.polyfit(x, y, 1)
                poly = np.poly1d(coeffs)
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = poly(x_fit)
                ax1.plot(x_fit, y_fit, 'r-', linewidth=2)

            ax1.set_xlabel('Total airmass (secθ_SZA + secθ_VZA)')
            ax1.set_ylabel('Absolute error')
            ax1.set_title('Total airmass vs absolute error')
            ax1.grid(True, alpha=0.3)

        # 2. 总大气质量与相对误差关系
        ax2 = axes[1]
        if 'total_airmass' in self.results.columns and 'error_relative' in self.results.columns:
            valid_mask = self.results[['total_airmass', 'error_relative']].notna().all(axis=1)
            ax2.scatter(self.results.loc[valid_mask, 'total_airmass'],
                        self.results.loc[valid_mask, 'error_relative'],
                        alpha=0.5, s=10, color='green', label='Relative')

            # 添加趋势线
            if valid_mask.sum() > 2:
                x = self.results.loc[valid_mask, 'total_airmass'].values
                y = self.results.loc[valid_mask, 'error_relative'].values
                coeffs = np.polyfit(x, y, 1)
                poly = np.poly1d(coeffs)
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = poly(x_fit)
                ax2.plot(x_fit, y_fit, 'r-', linewidth=2)

            ax2.set_xlabel('Total airmass (secθ_SZA + secθ_VZA)')
            ax2.set_ylabel('Relative error')
            ax2.set_title('Total airmass vs relative error')
            ax2.grid(True, alpha=0.3)

        # 3. 大气质量分量贡献（绝对误差）
        ax3 = axes[2]
        if all(col in self.results.columns for col in ['secz_sza', 'secz_vza']):
            valid_mask = self.results[['secz_sza', 'secz_vza', 'error_absolute']].notna().all(axis=1)

            # 计算贡献比例
            total_airmass = self.results.loc[valid_mask, 'secz_sza'] + self.results.loc[valid_mask, 'secz_vza']
            sza_contribution = self.results.loc[valid_mask, 'secz_sza'] / total_airmass

            # 按SZA贡献分组
            bins = np.linspace(0, 1, 6)
            sza_contribution_binned = pd.cut(sza_contribution, bins)

            error_by_contribution = self.results.loc[valid_mask].groupby(sza_contribution_binned)['error_absolute'].agg(
                ['mean', 'std', 'count'])

            x_pos = range(len(error_by_contribution))
            ax3.errorbar(x_pos, error_by_contribution['mean'],
                         yerr=error_by_contribution['std'], fmt='o-', capsize=5, color='blue')

            ax3.set_xlabel('SZA airmass contribution ratio')
            ax3.set_ylabel('Mean absolute error')
            ax3.set_title('Airmass component contribution (Absolute)')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([f'{b.left:.1f}-{b.right:.1f}' for b in error_by_contribution.index])
            ax3.grid(True, alpha=0.3)

        # 4. 大气质量分量贡献（相对误差）
        ax4 = axes[3]
        if all(col in self.results.columns for col in ['secz_sza', 'secz_vza', 'error_relative']):
            valid_mask = self.results[['secz_sza', 'secz_vza', 'error_relative']].notna().all(axis=1)

            # 计算贡献比例
            total_airmass = self.results.loc[valid_mask, 'secz_sza'] + self.results.loc[valid_mask, 'secz_vza']
            sza_contribution = self.results.loc[valid_mask, 'secz_sza'] / total_airmass

            # 按SZA贡献分组
            bins = np.linspace(0, 1, 6)
            sza_contribution_binned = pd.cut(sza_contribution, bins)

            error_by_contribution = self.results.loc[valid_mask].groupby(sza_contribution_binned)['error_relative'].agg(
                ['mean', 'std', 'count'])

            x_pos = range(len(error_by_contribution))
            ax4.errorbar(x_pos, error_by_contribution['mean'],
                         yerr=error_by_contribution['std'], fmt='o-', capsize=5, color='green')

            ax4.set_xlabel('SZA airmass contribution ratio')
            ax4.set_ylabel('Mean relative error')
            ax4.set_title('Airmass component contribution (Relative)')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels([f'{b.left:.1f}-{b.right:.1f}' for b in error_by_contribution.index])
            ax4.grid(True, alpha=0.3)

        plt.suptitle('Airmass breakdown analysis (Left: Absolute error, Right: Relative error)', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"大气质量分解图已保存: {save_path}")
        else:
            plt.show()

    def generate_sensitivity_report(self, save_path: Optional[Path] = None) -> str:
        """生成敏感性分析报告（包含绝对和相对误差）"""

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

        sensitivity_abs = self.calculate_sensitivity_indices('error_absolute')

        # 计算净效应
        net_sensitivity_abs = self.calculate_net_sensitivity_indices('error_absolute')

        report = []
        report.append("=" * 60)
        report.append("Atmospheric Parameter Sensitivity Analysis Report")
        report.append(f"Number of samples: {len(self.results)}")
        report.append(f"Valid absolute error data: {self.results['error_absolute'].notna().sum()}")

        if 'error_relative' in self.results.columns:
            report.append(f"Valid relative error data: {self.results['error_relative'].notna().sum()}")
            sensitivity_rel = self.calculate_sensitivity_indices('error_relative')
            net_sensitivity_rel = self.calculate_net_sensitivity_indices('error_relative')
        else:
            report.append("No relative error data available")
            sensitivity_rel = {}
            net_sensitivity_rel = {}

        report.append("=" * 60)
        report.append("")

        # 总体统计
        error_stats_abs = self.results['error_absolute'].describe()
        report.append("Absolute error statistics summary:")
        report.append("-" * 30)
        for stat, value in error_stats_abs.items():
            report.append(f"{stat}: {value:.6f}")
        report.append("")

        if 'error_relative' in self.results.columns:
            error_stats_rel = self.results['error_relative'].describe()
            report.append("Relative error statistics summary:")
            report.append("-" * 30)
            for stat, value in error_stats_rel.items():
                report.append(f"{stat}: {value:.6f}")
            report.append("")

        # 原始敏感性指标 - 绝对误差
        report.append("RAW EFFECTS - Absolute error parameter sensitivity indices (sorted by correlation):")
        report.append("-" * 60)

        # 按相关性绝对值排序
        sorted_sensitivity_abs = sorted(sensitivity_abs.items(),
                                        key=lambda x: abs(x[1]['correlation']),
                                        reverse=True)

        for param, sens in sorted_sensitivity_abs:
            report.append(f"\n{param}:")
            report.append(f"  Correlation coefficient: {sens['correlation']:.4f}")
            report.append(f"  R²: {sens['r_squared']:.4f}")
            report.append(f"  Slope: {sens['slope']:.6f}")
            report.append(f"  Normalized slope: {sens['normalized_slope']:.4f}")
            report.append(f"  P-value: {sens['p_value']:.6f}")
            report.append(f"  Number of samples: {sens['n_samples']}")

        report.append("")

        # 净效应敏感性指标 - 绝对误差
        report.append("NET EFFECTS - Absolute error net sensitivity indices (controlling for other factors):")
        report.append("-" * 60)

        if net_sensitivity_abs:
            sorted_net_sensitivity_abs = sorted(net_sensitivity_abs.items(),
                                                key=lambda x: abs(x[1]['net_correlation']),
                                                reverse=True)

            for param, sens in sorted_net_sensitivity_abs:
                report.append(f"\n{param}:")
                report.append(f"  Net correlation coefficient: {sens['net_correlation']:.4f}")
                report.append(f"  Net R²: {sens['net_r_squared']:.4f}")
                report.append(f"  Net slope: {sens['net_slope']:.6f}")
                report.append(f"  Number of samples: {sens['n_samples']}")
        else:
            report.append("\nNo net effects calculated (insufficient data or no continuous variables)")

        report.append("")

        # 原始敏感性指标 - 相对误差
        if sensitivity_rel:
            report.append("RAW EFFECTS - Relative error parameter sensitivity indices (sorted by correlation):")
            report.append("-" * 60)

            sorted_sensitivity_rel = sorted(sensitivity_rel.items(),
                                            key=lambda x: abs(x[1]['correlation']),
                                            reverse=True)

            for param, sens in sorted_sensitivity_rel:
                report.append(f"\n{param}:")
                report.append(f"  Correlation coefficient: {sens['correlation']:.4f}")
                report.append(f"  R²: {sens['r_squared']:.4f}")
                report.append(f"  Slope: {sens['slope']:.6f}")
                report.append(f"  Normalized slope: {sens['normalized_slope']:.4f}")
                report.append(f"  P-value: {sens['p_value']:.6f}")
                report.append(f"  Number of samples: {sens['n_samples']}")

            report.append("")

            # 净效应敏感性指标 - 相对误差
            if net_sensitivity_rel:
                report.append("NET EFFECTS - Relative error net sensitivity indices (controlling for other factors):")
                report.append("-" * 60)

                sorted_net_sensitivity_rel = sorted(net_sensitivity_rel.items(),
                                                    key=lambda x: abs(x[1]['net_correlation']),
                                                    reverse=True)

                for param, sens in sorted_net_sensitivity_rel:
                    report.append(f"\n{param}:")
                    report.append(f"  Net correlation coefficient: {sens['net_correlation']:.4f}")
                    report.append(f"  Net R²: {sens['net_r_squared']:.4f}")
                    report.append(f"  Net slope: {sens['net_slope']:.6f}")
                    report.append(f"  Number of samples: {sens['n_samples']}")

        # 大气质量分析
        if 'total_airmass' in self.results.columns:
            report.append("\nAirmass analysis:")
            report.append("-" * 30)

            total_airmass = self.results['total_airmass'].dropna()
            if len(total_airmass) > 0:
                report.append(f"Mean total airmass: {total_airmass.mean():.2f}")
                report.append(f"Minimum total airmass: {total_airmass.min():.2f}")
                report.append(f"Maximum total airmass: {total_airmass.max():.2f}")

                # 大气质量与绝对误差相关性
                valid_mask = self.results[['total_airmass', 'error_absolute']].notna().all(axis=1)
                if valid_mask.sum() > 10:
                    corr_abs = np.corrcoef(
                        self.results.loc[valid_mask, 'total_airmass'],
                        self.results.loc[valid_mask, 'error_absolute']
                    )[0, 1]
                    report.append(f"Total airmass-absolute error correlation: {corr_abs:.4f}")

                # 大气质量与相对误差相关性
                if 'error_relative' in self.results.columns:
                    valid_mask = self.results[['total_airmass', 'error_relative']].notna().all(axis=1)
                    if valid_mask.sum() > 10:
                        corr_rel = np.corrcoef(
                            self.results.loc[valid_mask, 'total_airmass'],
                            self.results.loc[valid_mask, 'error_relative']
                        )[0, 1]
                        report.append(f"Total airmass-relative error correlation: {corr_rel:.4f}")

        # 关键发现 - 比较原始效应和净效应
        report.append("\nKey findings: Raw Effects vs Net Effects")
        report.append("-" * 50)

        if sorted_sensitivity_abs:
            # 比较原始效应和净效应
            report.append("1. Comparison of raw and net effects (absolute error):")

            for param in ['sza', 'vza', 'aod550', 'h2o', 'o3', 'rho_true']:
                raw_info = sensitivity_abs.get(param)
                net_info = net_sensitivity_abs.get(param)

                if raw_info and net_info:
                    raw_corr = raw_info['correlation']
                    net_corr = net_info['net_correlation']
                    diff = abs(net_corr) - abs(raw_corr)

                    report.append(f"   {param}: Raw |r|={abs(raw_corr):.4f}, Net |r|={abs(net_corr):.4f}, "
                                  f"Difference={diff:+.4f}")

                    if abs(diff) > 0.1:
                        if diff > 0:
                            report.append(f"     -> Net effect is STRONGER than raw effect (by {diff:.4f})")
                        else:
                            report.append(f"     -> Net effect is WEAKER than raw effect (by {abs(diff):.4f})")

            # 最敏感的参数（净效应）
            if net_sensitivity_abs:
                sorted_net = sorted(net_sensitivity_abs.items(),
                                    key=lambda x: abs(x[1]['net_correlation']),
                                    reverse=True)
                if sorted_net:
                    most_sensitive_net = sorted_net[0]
                    report.append(f"\n2. Most sensitive parameter (net effect): {most_sensitive_net[0]} "
                                  f"(net |correlation|: {abs(most_sensitive_net[1]['net_correlation']):.4f})")

            # 最敏感的参数（原始效应）
            most_sensitive_raw = sorted_sensitivity_abs[0]
            report.append(f"3. Most sensitive parameter (raw effect): {most_sensitive_raw[0]} "
                          f"(raw |correlation|: {abs(most_sensitive_raw[1]['correlation']):.4f})")

            # 净效应分析的意义
            report.append("\n4. Interpretation of net effects:")
            report.append("   - Raw effects show overall relationships including confounding")
            report.append("   - Net effects isolate the direct influence of each parameter")
            report.append("   - Large differences indicate strong confounding effects")
            report.append("   - Small differences suggest parameter acts independently")

            # 协同效应
            report.append("\n5. Parameter synergy and confounding:")

            # 识别可能的混杂因素
            confounding_candidates = []
            for param in ['sza', 'vza', 'aod550', 'h2o', 'o3']:
                if param in sensitivity_abs and param in net_sensitivity_abs:
                    raw_corr = abs(sensitivity_abs[param]['correlation'])
                    net_corr = abs(net_sensitivity_abs[param]['net_correlation'])
                    if abs(raw_corr - net_corr) > 0.15:  # 阈值
                        confounding_candidates.append((param, raw_corr - net_corr))

            if confounding_candidates:
                report.append("   Parameters with significant confounding:")
                for param, diff in sorted(confounding_candidates, key=lambda x: abs(x[1]), reverse=True):
                    report.append(f"     - {param}: raw vs net difference = {diff:+.4f}")
            else:
                report.append("   No strong confounding effects detected")

        report_str = "\n".join(report)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_str)
            self.logger.info(f"敏感性报告已保存: {save_path}")

        return report_str

    def plot_partial_dependence_analysis(self, target_var: str = 'error_absolute',
                                         save_path: Optional[Path] = None):
        """
        绘制部分依赖图（Partial Dependence Plots），展示单个参数的净影响

        Parameters:
        -----------
        target_var : str
            目标变量，如 'error_absolute' 或 'error_relative'
        save_path : Path, optional
            保存路径
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import partial_dependence

        # 定义分析参数
        params_to_analyze = ['sza', 'vza', 'aod550', 'h2o', 'o3', 'rho_true']

        # 只保留存在且为连续变量的参数
        continuous_params = []
        for param in params_to_analyze:
            if param in self.results.columns:
                # 检查是否为连续变量（有足够多的不同值）
                unique_vals = self.results[param].dropna().unique()
                if len(unique_vals) >= 5:
                    continuous_params.append(param)

        if len(continuous_params) < 2:
            self.logger.warning("连续变量不足，无法绘制部分依赖图")
            return

        # 准备数据
        all_cols = continuous_params + [target_var]
        valid_mask = self.results[all_cols].notna().all(axis=1)
        valid_data = self.results.loc[valid_mask, all_cols]

        if len(valid_data) < 20:
            self.logger.warning(f"有效数据不足 ({len(valid_data)} < 20)，无法绘制部分依赖图")
            return

        # 分割特征和目标变量
        X = valid_data[continuous_params].values
        y = valid_data[target_var].values

        # 训练随机森林模型（用于部分依赖分析）
        try:
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X, y)
            self.logger.info(f"随机森林模型训练完成，R²: {rf_model.score(X, y):.4f}")
        except Exception as e:
            self.logger.error(f"训练随机森林模型失败: {e}")
            return

        # 计算参数数量，确定子图布局
        n_params = len(continuous_params)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n_params == 1:
            axes = np.array([axes])
        axes = axes.ravel()

        # 为每个参数绘制部分依赖图
        for idx, param in enumerate(continuous_params):
            ax = axes[idx]

            try:
                # 计算部分依赖
                param_idx = continuous_params.index(param)
                pd_results = partial_dependence(
                    rf_model, X, features=[param_idx],
                    percentiles=(0.05, 0.95),
                    grid_resolution=20
                )

                # 提取结果
                pd_values = pd_results['average'][0]
                grid_points = pd_results['values'][0]

                # 绘制部分依赖曲线
                ax.plot(grid_points, pd_values, 'b-', linewidth=2, label='PDP')

                # 添加置信区间（使用模型的不确定性估计）
                tree_preds = []
                for tree in rf_model.estimators_:
                    tree_pd = partial_dependence(
                        tree, X, features=[param_idx],
                        percentiles=(0.05, 0.95),
                        grid_resolution=20
                    )
                    tree_preds.append(tree_pd['average'][0])

                tree_preds = np.array(tree_preds)
                mean_pred = np.mean(tree_preds, axis=0)
                std_pred = np.std(tree_preds, axis=0)

                # 绘制置信区间
                ax.fill_between(grid_points,
                                mean_pred - std_pred,
                                mean_pred + std_pred,
                                alpha=0.3, color='blue', label='±1 std')

                # 添加原始数据分布
                param_values = valid_data[param].values
                param_min, param_max = param_values.min(), param_values.max()

                # 创建第二个y轴显示数据分布
                ax_twin = ax.twinx()
                ax_twin.hist(param_values, bins=20, alpha=0.3,
                             color='gray', density=True)
                ax_twin.set_ylabel('Data Density', color='gray')
                ax_twin.tick_params(axis='y', labelcolor='gray')

                # 设置图表属性
                ax.set_xlabel(param)
                ax.set_ylabel(f'Partial Dependence\n({target_var})')
                ax.set_title(f'{param} - Partial Dependence Plot')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')

                # 计算并显示重要性指标
                importance = rf_model.feature_importances_[param_idx]
                ax.text(0.05, 0.95, f'Importance: {importance:.3f}',
                        transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            except Exception as e:
                self.logger.warning(f"计算参数 {param} 部分依赖失败: {e}")
                ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...',
                        ha='center', va='center')
                ax.set_title(f'{param} - Failed')

        # 隐藏多余的子图
        for i in range(len(continuous_params), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f'Partial Dependence Analysis for {target_var}\n'
                     f'Showing Net Effects (controlling for other factors)',
                     fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"部分依赖图已保存: {save_path}")
        else:
            plt.show()

    def plot_interaction_effects(self, save_path: Optional[Path] = None):
        """
        绘制交互效应图，展示两个参数之间的交互作用
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import partial_dependence

        # 主要交互对
        interaction_pairs = [
            ('sza', 'vza'),
            ('sza', 'aod550'),
            ('vza', 'aod550'),
            ('h2o', 'o3'),
            ('sza', 'rho_true')
        ]

        # 筛选存在的参数对
        valid_pairs = []
        for param1, param2 in interaction_pairs:
            if param1 in self.results.columns and param2 in self.results.columns:
                valid_pairs.append((param1, param2))

        if not valid_pairs:
            self.logger.warning("没有有效的参数对用于交互效应分析")
            return

        # 准备数据
        features = []
        for param1, param2 in valid_pairs:
            features.extend([param1, param2])
        features = list(set(features))  # 去重

        target_var = 'error_absolute'
        all_cols = features + [target_var]
        valid_mask = self.results[all_cols].notna().all(axis=1)
        valid_data = self.results.loc[valid_mask, all_cols]

        if len(valid_data) < 20:
            self.logger.warning(f"有效数据不足 ({len(valid_data)} < 20)，无法绘制交互效应图")
            return

        X = valid_data[features].values
        y = valid_data[target_var].values

        # 训练模型
        try:
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X, y)
        except Exception as e:
            self.logger.error(f"训练模型失败: {e}")
            return

        # 创建图形
        n_pairs = len(valid_pairs)
        n_cols = min(2, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_pairs == 1:
            axes = np.array([axes])
        axes = axes.ravel()

        # 为每个参数对绘制交互效应图
        for idx, (param1, param2) in enumerate(valid_pairs):
            if idx >= len(axes):
                break

            ax = axes[idx]

            try:
                # 获取参数索引
                idx1 = features.index(param1)
                idx2 = features.index(param2)

                # 计算二维部分依赖
                pd_results = partial_dependence(
                    rf_model, X, features=[idx1, idx2],
                    grid_resolution=15
                )

                # 提取结果
                pd_values = pd_results['average']
                x_grid = pd_results['values'][0]
                y_grid = pd_results['values'][1]

                # 创建网格
                X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

                # 绘制等高线图
                contour = ax.contourf(X_grid, Y_grid, pd_values.T,
                                      levels=20, cmap='RdBu_r')

                # 添加等高线
                ax.contour(X_grid, Y_grid, pd_values.T,
                           levels=10, colors='k', linewidths=0.5, alpha=0.5)

                # 设置图表属性
                ax.set_xlabel(param1)
                ax.set_ylabel(param2)
                ax.set_title(f'Interaction: {param1} × {param2}')
                ax.grid(True, alpha=0.3)

                # 添加颜色条
                plt.colorbar(contour, ax=ax, label='Partial Dependence')

                # 计算交互强度
                # 简单方法：计算二维表面的方差
                interaction_strength = np.var(pd_values)
                ax.text(0.05, 0.95, f'Interaction: {interaction_strength:.4f}',
                        transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            except Exception as e:
                self.logger.warning(f"计算交互效应 {param1} × {param2} 失败: {e}")
                ax.text(0.5, 0.5, 'Calculation failed',
                        ha='center', va='center')
                ax.set_title(f'{param1} × {param2} - Failed')

        # 隐藏多余的子图
        for i in range(len(valid_pairs), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Interaction Effects between Key Parameters\n'
                     '(Higher values indicate stronger interactions)',
                     fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"交互效应图已保存: {save_path}")
        else:
            plt.show()

    def generate_comprehensive_sensitivity_report(self, save_dir: Optional[Path] = None) -> str:
        """
        生成全面的敏感性分析报告，包括净效应和部分依赖分析

        Parameters:
        -----------
        save_dir : Path, optional
            保存目录，将保存多个图表和报告

        Returns:
        --------
        str
            报告内容
        """
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        report_parts = []

        # 1. 基本统计
        report_parts.append("=" * 70)
        report_parts.append("COMPREHENSIVE SENSITIVITY ANALYSIS REPORT")
        report_parts.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_parts.append(f"Total samples: {len(self.results)}")
        report_parts.append(f"Absolute error samples: {self.results['error_absolute'].notna().sum()}")

        if 'error_relative' in self.results.columns:
            report_parts.append(f"Relative error samples: {self.results['error_relative'].notna().sum()}")

        report_parts.append("=" * 70)
        report_parts.append("")

        # 2. 原始效应分析
        raw_sens_abs = self.calculate_sensitivity_indices('error_absolute')
        report_parts.append("1. RAW EFFECTS ANALYSIS (Absolute Error)")
        report_parts.append("-" * 50)

        if raw_sens_abs:
            sorted_raw = sorted(raw_sens_abs.items(),
                                key=lambda x: abs(x[1]['correlation']),
                                reverse=True)

            for param, sens in sorted_raw:
                report_parts.append(f"\n{param.upper()}:")
                report_parts.append(f"  Correlation: {sens['correlation']:+.4f}")
                report_parts.append(f"  R²: {sens['r_squared']:.4f}")
                report_parts.append(f"  Slope: {sens['slope']:+.6f}")
                report_parts.append(f"  Normalized slope: {sens['normalized_slope']:+.4f}")
                report_parts.append(f"  P-value: {sens['p_value']:.6e}")
                report_parts.append(f"  Samples: {sens['n_samples']}")
        else:
            report_parts.append("No raw effects calculated")

        report_parts.append("")

        # 3. 净效应分析
        net_sens_abs = self.calculate_net_sensitivity_indices('error_absolute')
        report_parts.append("2. NET EFFECTS ANALYSIS (Absolute Error)")
        report_parts.append("-" * 50)

        if net_sens_abs:
            sorted_net = sorted(net_sens_abs.items(),
                                key=lambda x: abs(x[1]['net_correlation']),
                                reverse=True)

            for param, sens in sorted_net:
                report_parts.append(f"\n{param.upper()}:")
                report_parts.append(f"  Net correlation: {sens['net_correlation']:+.4f}")
                report_parts.append(f"  Net R²: {sens['net_r_squared']:.4f}")
                report_parts.append(f"  Net slope: {sens['net_slope']:+.6f}")
                report_parts.append(f"  Samples: {sens['n_samples']}")

                # 比较原始效应和净效应
                raw_corr = raw_sens_abs.get(param, {}).get('correlation', 0)
                net_corr = sens['net_correlation']
                diff = abs(net_corr) - abs(raw_corr)

                if abs(diff) > 0.1:
                    if diff > 0:
                        report_parts.append(f"  → Net effect is STRONGER than raw effect (difference: {diff:+.3f})")
                        report_parts.append(f"    This suggests confounding effects from other variables")
                    else:
                        report_parts.append(f"  → Net effect is WEAKER than raw effect (difference: {diff:+.3f})")
                        report_parts.append(f"    This suggests correlation was partly due to other variables")
                else:
                    report_parts.append(f"  → Raw and net effects are similar (difference: {diff:+.3f})")
        else:
            report_parts.append("No net effects calculated")

        report_parts.append("")

        # 4. 交互效应分析
        report_parts.append("3. INTERACTION EFFECTS ANALYSIS")
        report_parts.append("-" * 50)

        # 计算简单的交互效应指标
        interaction_insights = []

        # 检查SZA和VZA的交互
        if all(p in self.results.columns for p in ['sza', 'vza', 'error_absolute']):
            # 分组分析
            data = self.results[['sza', 'vza', 'error_absolute']].dropna()

            if len(data) > 20:
                # 创建交互项
                data['sza_vza_interaction'] = data['sza'] * data['vza'] / 1000.0  # 缩放

                # 回归分析
                X = data[['sza', 'vza', 'sza_vza_interaction']].values
                y = data['error_absolute'].values

                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X, y)

                # 检查交互项的系数
                interaction_coef = model.coef_[2]
                if abs(interaction_coef) > 0.001:
                    interaction_insights.append(
                        f"SZA×VZA interaction coefficient: {interaction_coef:.6f}"
                    )
                    if interaction_coef > 0:
                        interaction_insights.append(
                            "  → Positive interaction: SZA and VZA amplify each other's effects")
                    else:
                        interaction_insights.append(
                            "  → Negative interaction: SZA and VZA compensate each other's effects")

        if interaction_insights:
            for insight in interaction_insights:
                report_parts.append(insight)
        else:
            report_parts.append("No significant interaction effects detected")

        report_parts.append("")

        # 5. 建议和结论
        report_parts.append("4. CONCLUSIONS AND RECOMMENDATIONS")
        report_parts.append("-" * 50)

        if raw_sens_abs and net_sens_abs:
            # 识别最重要的参数
            most_important_raw = max(raw_sens_abs.items(),
                                     key=lambda x: abs(x[1]['correlation']))[0]
            most_important_net = max(net_sens_abs.items(),
                                     key=lambda x: abs(x[1]['net_correlation']))[0]

            report_parts.append(f"Most important parameter (raw effect): {most_important_raw}")
            report_parts.append(f"Most important parameter (net effect): {most_important_net}")

            if most_important_raw != most_important_net:
                report_parts.append("  → Different parameters are most important in raw vs net analysis")
                report_parts.append("  → This indicates strong confounding effects")

            # 识别强混杂因素
            strong_confounding = []
            for param in ['sza', 'vza', 'aod550', 'h2o', 'o3']:
                if param in raw_sens_abs and param in net_sens_abs:
                    raw_corr = abs(raw_sens_abs[param]['correlation'])
                    net_corr = abs(net_sens_abs[param]['net_correlation'])
                    if abs(raw_corr - net_corr) > 0.2:
                        strong_confounding.append(param)

            if strong_confounding:
                report_parts.append("\nParameters with strong confounding (>0.2 difference):")
                for param in strong_confounding:
                    report_parts.append(f"  - {param}")
                report_parts.append("  → These parameters' effects are strongly influenced by other variables")

            # 建模建议
            report_parts.append("\nModeling recommendations:")
            report_parts.append("  1. Include these key parameters in correction models:")

            # 选择前3个最重要的净效应参数
            top_params = sorted(net_sens_abs.items(),
                                key=lambda x: abs(x[1]['net_correlation']),
                                reverse=True)[:3]

            for i, (param, sens) in enumerate(top_params, 1):
                report_parts.append(f"     {i}. {param} (net |r|={abs(sens['net_correlation']):.3f})")

            if strong_confounding:
                report_parts.append("  2. Consider interaction terms for strongly confounded parameters")

            report_parts.append("  3. Use net effects for interpreting individual parameter contributions")

        report_parts.append("")
        report_parts.append("=" * 70)

        # 组合报告
        report_str = "\n".join(report_parts)

        # 保存报告
        if save_dir:
            report_file = save_dir / "comprehensive_sensitivity_report.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_str)
            self.logger.info(f"全面敏感性分析报告已保存: {report_file}")

            # 保存图表
            try:
                # 净效应图
                net_effect_file = save_dir / "net_effect_analysis.png"
                self.plot_net_sensitivity_analysis(net_effect_file)

                # 部分依赖图（绝对误差）
                pdp_abs_file = save_dir / "partial_dependence_absolute.png"
                self.plot_partial_dependence_analysis('error_absolute', pdp_abs_file)

                # 部分依赖图（相对误差，如果存在）
                if 'error_relative' in self.results.columns:
                    pdp_rel_file = save_dir / "partial_dependence_relative.png"
                    self.plot_partial_dependence_analysis('error_relative', pdp_rel_file)

                # 交互效应图
                interaction_file = save_dir / "interaction_effects.png"
                self.plot_interaction_effects(interaction_file)

            except Exception as e:
                self.logger.error(f"保存图表失败: {e}")

        return report_str