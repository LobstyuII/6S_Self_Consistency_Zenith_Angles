# ==================== error_analyzer.py ====================
"""
误差分析模块
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

        # 按波段计算
        for band_id in self.results.keys():
            df_band = self.results[band_id]
            errors = df_band['error_absolute'].dropna().values

            if len(errors) > 0:
                stats_dict[band_id] = calculate_statistics(errors, prefix='')

        # 所有波段合并
        all_errors = self.all_data['error_absolute'].dropna().values
        if len(all_errors) > 0:
            stats_dict['all'] = calculate_statistics(all_errors, prefix='')

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

    def plot_error_contours(self, rho_true: float = 0.2, aod550: float = 0.3,
                            save_path: Optional[Path] = None):
        """绘制误差等高线图"""
        # 筛选特定条件的数据
        mask = (
                (self.all_data['rho_true'] == rho_true) &
                (self.all_data['aod550'] == aod550) &
                (self.all_data['raa'] == 0)  # 前向散射
        )
        subset = self.all_data[mask].copy()

        if len(subset) == 0:
            self.logger.warning(f"没有找到满足条件的数据: rho_true={rho_true}, aod550={aod550}")
            return

        # 按波段分组
        n_bands = len(ExperimentConfig.BANDS)
        fig, axes = plt.subplots(1, n_bands, figsize=(5 * n_bands, 4))

        if n_bands == 1:
            axes = [axes]

        for idx, (band_id, band_info) in enumerate(ExperimentConfig.BANDS.items()):
            ax = axes[idx]

            # 获取该波段数据
            band_data = subset[subset['band'] == band_id]

            if len(band_data) < 10:
                ax.text(0.5, 0.5, '数据不足', ha='center', va='center')
                continue

            # 创建网格
            sza_unique = np.sort(band_data['sza'].unique())
            vza_unique = np.sort(band_data['vza'].unique())

            if len(sza_unique) < 2 or len(vza_unique) < 2:
                ax.text(0.5, 0.5, '数据不足', ha='center', va='center')
                continue

            # 插值到规则网格
            sza_grid, vza_grid = np.meshgrid(sza_unique, vza_unique)
            error_grid = np.full_like(sza_grid, np.nan)

            for i, sza in enumerate(sza_unique):
                for j, vza in enumerate(vza_unique):
                    mask_point = (band_data['sza'] == sza) & (band_data['vza'] == vza)
                    if mask_point.any():
                        error_grid[j, i] = band_data.loc[mask_point, 'error_absolute'].mean()

            # 绘制等高线
            contour = ax.contourf(sza_grid, vza_grid, error_grid, levels=20, cmap='RdBu_r')
            ax.contour(sza_grid, vza_grid, error_grid, levels=10, colors='k', linewidths=0.5, alpha=0.5)

            ax.set_xlabel('太阳天顶角 (度)')
            ax.set_ylabel('观测天顶角 (度)')
            ax.set_title(f'{band_info["name"]}\n误差等高线')
            ax.grid(True, alpha=0.3)

            # 添加颜色条
            plt.colorbar(contour, ax=ax, label='绝对误差')

        plt.suptitle(f'地表反射率={rho_true}, AOD550={aod550}, RAA=0°', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"误差等高线图已保存: {save_path}")

        # plt.show()

    def plot_error_by_parameter(self, save_path: Optional[Path] = None):
        """按参数分组的误差分析"""
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)

        # 1. 按SZA分组
        ax1 = fig.add_subplot(gs[0, 0])
        sza_bins = np.arange(0, 91, 10)
        self.all_data['sza_bin'] = pd.cut(self.all_data['sza'], bins=sza_bins)
        error_by_sza = self.all_data.groupby('sza_bin')['error_absolute'].agg(['mean', 'std', 'count'])
        ax1.errorbar(range(len(error_by_sza)), error_by_sza['mean'],
                     yerr=error_by_sza['std'], fmt='o-', capsize=5)
        ax1.set_xlabel('SZA区间 (度)')
        ax1.set_ylabel('平均误差')
        ax1.set_title('不同SZA区间的误差')
        ax1.set_xticks(range(len(error_by_sza)))
        ax1.set_xticklabels([str(bin_) for bin_ in error_by_sza.index], rotation=45)
        ax1.grid(True, alpha=0.3)

        # 2. 按VZA分组
        ax2 = fig.add_subplot(gs[0, 1])
        vza_bins = np.arange(0, 81, 10)
        self.all_data['vza_bin'] = pd.cut(self.all_data['vza'], bins=vza_bins)
        error_by_vza = self.all_data.groupby('vza_bin')['error_absolute'].agg(['mean', 'std', 'count'])
        ax2.errorbar(range(len(error_by_vza)), error_by_vza['mean'],
                     yerr=error_by_vza['std'], fmt='o-', capsize=5)
        ax2.set_xlabel('VZA区间 (度)')
        ax2.set_ylabel('平均误差')
        ax2.set_title('不同VZA区间的误差')
        ax2.set_xticks(range(len(error_by_vza)))
        ax2.set_xticklabels([str(bin_) for bin_ in error_by_vza.index], rotation=45)
        ax2.grid(True, alpha=0.3)

        # 3. 按AOD分组
        ax3 = fig.add_subplot(gs[0, 2])
        error_by_aod = self.all_data.groupby('aod550')['error_absolute'].agg(['mean', 'std', 'count'])
        ax3.errorbar(range(len(error_by_aod)), error_by_aod['mean'],
                     yerr=error_by_aod['std'], fmt='o-', capsize=5)
        ax3.set_xlabel('AOD550')
        ax3.set_ylabel('平均误差')
        ax3.set_title('不同AOD的误差')
        ax3.set_xticks(range(len(error_by_aod)))
        ax3.set_xticklabels([str(aod) for aod in error_by_aod.index])
        ax3.grid(True, alpha=0.3)

        # 4. 按地表反射率分组
        ax4 = fig.add_subplot(gs[1, 0])
        error_by_rho = self.all_data.groupby('rho_true')['error_absolute'].agg(['mean', 'std', 'count'])
        ax4.errorbar(range(len(error_by_rho)), error_by_rho['mean'],
                     yerr=error_by_rho['std'], fmt='o-', capsize=5)
        ax4.set_xlabel('地表反射率')
        ax4.set_ylabel('平均误差')
        ax4.set_title('不同地表反射率的误差')
        ax4.set_xticks(range(len(error_by_rho)))
        ax4.set_xticklabels([str(rho) for rho in error_by_rho.index])
        ax4.grid(True, alpha=0.3)

        # 5. 按相对方位角分组
        ax5 = fig.add_subplot(gs[1, 1])
        error_by_raa = self.all_data.groupby('raa')['error_absolute'].agg(['mean', 'std', 'count'])
        ax5.errorbar(range(len(error_by_raa)), error_by_raa['mean'],
                     yerr=error_by_raa['std'], fmt='o-', capsize=5)
        ax5.set_xlabel('相对方位角 (度)')
        ax5.set_ylabel('平均误差')
        ax5.set_title('不同相对方位角的误差')
        ax5.set_xticks(range(len(error_by_raa)))
        ax5.set_xticklabels([str(raa) for raa in error_by_raa.index])
        ax5.grid(True, alpha=0.3)

        # 6. 按波段分组
        ax6 = fig.add_subplot(gs[1, 2])
        error_by_band = self.all_data.groupby('band')['error_absolute'].agg(['mean', 'std', 'count'])
        ax6.errorbar(range(len(error_by_band)), error_by_band['mean'],
                     yerr=error_by_band['std'], fmt='o-', capsize=5)
        ax6.set_xlabel('波段')
        ax6.set_ylabel('平均误差')
        ax6.set_title('不同波段的误差')
        ax6.set_xticks(range(len(error_by_band)))
        ax6.set_xticklabels([f"Band{i + 1}" for i in range(len(error_by_band))])
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"参数分组误差图已保存: {save_path}")

        # plt.show()

    def generate_error_report(self, save_path: Optional[Path] = None) -> str:
        """生成误差分析报告"""
        report = []
        report.append("=" * 60)
        report.append("6S几何误差分析报告")
        report.append(f"实验名称: {ExperimentConfig.EXP_NAME}")
        report.append(f"实验日期: {ExperimentConfig.EXP_DATE}")
        report.append("=" * 60)
        report.append("")

        # 总体统计
        report.append("总体统计:")
        report.append("-" * 30)
        if 'all' in self.stats:
            stats_all = self.stats['all']
            for key, value in stats_all.items():
                report.append(f"{key}: {value:.6f}")
        report.append("")

        # 各波段统计
        report.append("各波段统计:")
        report.append("-" * 30)
        for band_id in ExperimentConfig.BANDS.keys():
            if band_id in self.stats:
                report.append(f"\n{band_id}:")
                stats_band = self.stats[band_id]
                for key, value in stats_band.items():
                    report.append(f"  {key}: {value:.6f}")
        report.append("")

        # 误差特征
        report.append("误差特征分析:")
        report.append("-" * 30)

        # 检查误差与几何参数的相关性
        valid_data = self.all_data.dropna(subset=['error_absolute', 'sza', 'vza', 'secz_sza', 'secz_vza'])

        if len(valid_data) > 2:
            correlations = {
                'SZA': np.corrcoef(valid_data['sza'], valid_data['error_absolute'])[0, 1],
                'VZA': np.corrcoef(valid_data['vza'], valid_data['error_absolute'])[0, 1],
                'sec(SZA)': np.corrcoef(valid_data['secz_sza'], valid_data['error_absolute'])[0, 1],
                'sec(VZA)': np.corrcoef(valid_data['secz_vza'], valid_data['error_absolute'])[0, 1],
                'sec(SZA)*sec(VZA)': np.corrcoef(
                    valid_data['secz_sza'] * valid_data['secz_vza'],
                    valid_data['error_absolute']
                )[0, 1]
            }

            for param, corr in correlations.items():
                report.append(f"误差与{param}的相关系数: {corr:.4f}")

        # 极端几何条件下的误差
        report.append("\n极端几何条件误差分析:")
        report.append("-" * 30)

        # 定义极端条件
        extreme_mask = (
                (valid_data['sza'] > 60) |
                (valid_data['vza'] > 60) |
                (valid_data['secz_sza'] * valid_data['secz_vza'] > 4)
        )

        if extreme_mask.any():
            extreme_errors = valid_data.loc[extreme_mask, 'error_absolute']
            normal_errors = valid_data.loc[~extreme_mask, 'error_absolute']

            report.append(f"极端条件样本数: {len(extreme_errors):,}")
            report.append(f"正常条件样本数: {len(normal_errors):,}")
            report.append(f"极端条件平均误差: {extreme_errors.mean():.6f}")
            report.append(f"正常条件平均误差: {normal_errors.mean():.6f}")
            report.append(f"误差增加倍数: {extreme_errors.mean() / normal_errors.mean():.2f}")

        report_str = "\n".join(report)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_str)
            self.logger.info(f"误差报告已保存: {save_path}")

        return report_str
