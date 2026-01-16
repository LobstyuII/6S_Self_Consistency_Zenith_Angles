# ==================== sensitivity_analyzer.py ====================
"""
敏感性分析模块
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional
from pathlib import Path
from scipy import stats

from config import ExperimentConfig
from utils import setup_logger

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


class SensitivityAnalyzer:
    """敏感性分析器"""

    def __init__(self, results: pd.DataFrame, logger=None):
        self.results = results
        self.logger = logger or setup_logger('SensitivityAnalyzer')

        if 'sza' in results.columns:
            self.results['secz_sza'] = 1.0 / np.cos(np.radians(self.results['sza']))
        if 'vza' in results.columns:
            self.results['secz_vza'] = 1.0 / np.cos(np.radians(self.results['vza']))
        if 'secz_sza' in self.results.columns and 'secz_vza' in self.results.columns:
            self.results['total_airmass'] = self.results['secz_sza'] + self.results['secz_vza']

        if 'error_relative' not in self.results.columns and 'error_absolute' in self.results.columns and 'rho_true' in self.results.columns:
            self.results['error_relative'] = self.results['error_absolute'] / self.results['rho_true']
            self.logger.info("计算相对误差")

    def calculate_sensitivity_indices(self, target_var: str = 'error_absolute') -> Dict[str, float]:
        """计算敏感性指标"""
        sensitivity = {}
        params_to_analyze = ['sza', 'vza', 'aod550', 'h2o', 'o3', 'rho_true', 'total_airmass']

        for param in params_to_analyze:
            if param in self.results.columns:
                valid_mask = self.results[[param, target_var]].notna().all(axis=1)
                valid_data = self.results.loc[valid_mask]

                if len(valid_data) > 10:
                    unique_values = valid_data[param].unique()
                    if len(unique_values) < 2:
                        self.logger.warning(f"参数 {param} 的值没有变化，跳过线性回归")
                        continue

                    try:
                        x = valid_data[param].values
                        y = valid_data[target_var].values
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

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
        """绘制单因素敏感性分析图"""
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.ravel()

        params = {
            'sza': {'label': 'Solar zenith angle', 'unit': '°'},
            'vza': {'label': 'View zenith angle', 'unit': '°'},
            'aod550': {'label': 'AOD550', 'unit': ''},
            'h2o': {'label': 'Water vapor', 'unit': 'g/cm²'},
            'o3': {'label': 'Ozone', 'unit': 'cm-atm'},
            'rho_true': {'label': 'Surface reflectance', 'unit': ''}
        }

        sensitivity_abs = self.calculate_sensitivity_indices('error_absolute')
        sensitivity_rel = {}
        if 'error_relative' in self.results.columns:
            sensitivity_rel = self.calculate_sensitivity_indices('error_relative')

        for idx, (param_name, param_info) in enumerate(params.items()):
            ax_abs = axes[idx * 2]
            ax_rel = axes[idx * 2 + 1]

            # 绝对误差敏感性
            if param_name in self.results.columns:
                unique_values = self.results[param_name].unique()
                if len(unique_values) < 2:
                    ax_abs.text(0.5, 0.5, f'{param_info["label"]} values no variation',
                                ha='center', va='center', transform=ax_abs.transAxes)
                    ax_abs.set_title(f'{param_info["label"]} - Absolute')
                    continue

                ax_abs.scatter(self.results[param_name], self.results['error_absolute'],
                               alpha=0.3, s=10, color='blue')

                valid_mask = self.results[[param_name, 'error_absolute']].notna().all(axis=1)
                if valid_mask.sum() > 2:
                    x = self.results.loc[valid_mask, param_name].values
                    y = self.results.loc[valid_mask, 'error_absolute'].values

                    if len(np.unique(x)) < 2:
                        ax_abs.text(0.5, 0.5, 'Data points no variation, cannot fit',
                                    ha='center', va='center', transform=ax_abs.transAxes)
                        ax_abs.set_title(f'{param_info["label"]} - Absolute')
                        continue

                    try:
                        coeffs = np.polyfit(x, y, 1)
                        poly = np.poly1d(coeffs)
                        x_fit = np.linspace(x.min(), x.max(), 100)
                        y_fit = poly(x_fit)
                        ax_abs.plot(x_fit, y_fit, 'r-', linewidth=2)

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

            # 相对误差敏感性
            if param_name in self.results.columns and 'error_relative' in self.results.columns:
                unique_values = self.results[param_name].unique()
                if len(unique_values) < 2:
                    ax_rel.text(0.5, 0.5, f'{param_info["label"]} values no variation',
                                ha='center', va='center', transform=ax_rel.transAxes)
                    ax_rel.set_title(f'{param_info["label"]} - Relative')
                    continue

                ax_rel.scatter(self.results[param_name], self.results['error_relative'],
                               alpha=0.3, s=10, color='green')

                valid_mask = self.results[[param_name, 'error_relative']].notna().all(axis=1)
                if valid_mask.sum() > 2:
                    x = self.results.loc[valid_mask, param_name].values
                    y = self.results.loc[valid_mask, 'error_relative'].values

                    if len(np.unique(x)) < 2:
                        ax_rel.text(0.5, 0.5, 'Data points no variation, cannot fit',
                                    ha='center', va='center', transform=ax_rel.transAxes)
                        ax_rel.set_title(f'{param_info["label"]} - Relative')
                        continue

                    try:
                        coeffs = np.polyfit(x, y, 1)
                        poly = np.poly1d(coeffs)
                        x_fit = np.linspace(x.min(), x.max(), 100)
                        y_fit = poly(x_fit)
                        ax_rel.plot(x_fit, y_fit, 'r-', linewidth=2)

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

    def generate_sensitivity_report(self, save_path: Optional[Path] = None) -> str:
        """生成敏感性分析报告"""
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

        report = []
        report.append("=" * 60)
        report.append("Atmospheric Parameter Sensitivity Analysis Report")
        report.append(f"Number of samples: {len(self.results)}")
        report.append(f"Valid absolute error data: {self.results['error_absolute'].notna().sum()}")

        if 'error_relative' in self.results.columns:
            report.append(f"Valid relative error data: {self.results['error_relative'].notna().sum()}")
            sensitivity_rel = self.calculate_sensitivity_indices('error_relative')
        else:
            report.append("No relative error data available")
            sensitivity_rel = {}

        report.append("=" * 60)
        report.append("")

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

        report.append("RAW EFFECTS - Absolute error parameter sensitivity indices (sorted by correlation):")
        report.append("-" * 60)

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

        if 'total_airmass' in self.results.columns:
            report.append("\nAirmass analysis:")
            report.append("-" * 30)

            total_airmass = self.results['total_airmass'].dropna()
            if len(total_airmass) > 0:
                report.append(f"Mean total airmass: {total_airmass.mean():.2f}")
                report.append(f"Minimum total airmass: {total_airmass.min():.2f}")
                report.append(f"Maximum total airmass: {total_airmass.max():.2f}")

                valid_mask = self.results[['total_airmass', 'error_absolute']].notna().all(axis=1)
                if valid_mask.sum() > 10:
                    corr_abs = np.corrcoef(
                        self.results.loc[valid_mask, 'total_airmass'],
                        self.results.loc[valid_mask, 'error_absolute']
                    )[0, 1]
                    report.append(f"Total airmass-absolute error correlation: {corr_abs:.4f}")

                if 'error_relative' in self.results.columns:
                    valid_mask = self.results[['total_airmass', 'error_relative']].notna().all(axis=1)
                    if valid_mask.sum() > 10:
                        corr_rel = np.corrcoef(
                            self.results.loc[valid_mask, 'total_airmass'],
                            self.results.loc[valid_mask, 'error_relative']
                        )[0, 1]
                        report.append(f"Total airmass-relative error correlation: {corr_rel:.4f}")

        if sorted_sensitivity_abs:
            most_sensitive_raw = sorted_sensitivity_abs[0]
            report.append(f"\nMost sensitive parameter: {most_sensitive_raw[0]} "
                          f"(|correlation|: {abs(most_sensitive_raw[1]['correlation']):.4f})")

        report_str = "\n".join(report)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_str)
            self.logger.info(f"敏感性报告已保存: {save_path}")

        return report_str