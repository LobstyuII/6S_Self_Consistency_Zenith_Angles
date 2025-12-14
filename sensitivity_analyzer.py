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

    def calculate_sensitivity_indices(self, target_var: str = 'error_absolute') -> Dict[str, float]:
        """
        计算敏感性指标（增加错误处理）
        """
        sensitivity = {}

        # 定义需要分析的参数
        params_to_analyze = ['sza', 'vza', 'aod550', 'h2o', 'o3', 'rho_true']

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
        if all(col in self.results.columns for col in ['airmass_sza', 'airmass_vza']):
            valid_mask = self.results[['airmass_sza', 'airmass_vza', 'error_absolute']].notna().all(axis=1)

            # 计算贡献比例
            total_airmass = self.results.loc[valid_mask, 'airmass_sza'] + self.results.loc[valid_mask, 'airmass_vza']
            sza_contribution = self.results.loc[valid_mask, 'airmass_sza'] / total_airmass
            vza_contribution = self.results.loc[valid_mask, 'airmass_vza'] / total_airmass

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

    def generate_sensitivity_report(self, save_path: Optional[Path] = None) -> str:
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