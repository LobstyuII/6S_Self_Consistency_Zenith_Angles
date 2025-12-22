# ==================== error_analyzer.py ====================
"""
误差分析模块
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from scipy import stats
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

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
        """
        self.results = results
        self.logger = logger or setup_logger('ErrorAnalyzer')
        self.all_data = self._combine_bands()

        if 'error_relative' not in self.all_data.columns and 'error_absolute' in self.all_data.columns and 'rho_true' in self.all_data.columns:
            self.all_data['error_relative'] = self.all_data['error_absolute'] / self.all_data['rho_true']
            self.logger.info("计算相对误差")

        self.stats = self.calculate_overall_statistics()

    def _combine_bands(self) -> pd.DataFrame:
        """合并所有波段数据"""
        dfs = []
        for band_id, df in self.results.items():
            df_band = df.copy()
            df_band['band'] = band_id

            if band_id == 'all' and 'wavelength' in df_band.columns:
                pass
            elif band_id in ExperimentConfig.BANDS:
                df_band['wavelength'] = ExperimentConfig.BANDS[band_id]['wavelength']
            else:
                df_band['wavelength'] = np.nan
                self.logger.warning(f"波段 {band_id} 不在配置中")

            dfs.append(df_band)

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def calculate_overall_statistics(self) -> Dict[str, Dict[str, float]]:
        """计算总体统计量"""
        stats_dict = {}

        for band_id in self.results.keys():
            df_band = self.results[band_id]
            if 'error_absolute' in df_band.columns:
                errors = df_band['error_absolute'].dropna().values
                if len(errors) > 0:
                    try:
                        stats_dict[band_id] = calculate_statistics(errors, prefix='')
                    except:
                        stats_dict[band_id] = {'error': '计算统计量失败'}

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

        return stats_dict

    def plot_error_distribution(self, save_path: Optional[Path] = None):
        """绘制误差分布图"""
        if self.all_data.empty:
            self.logger.warning("没有数据")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        # 绝对误差直方图
        if 'error_absolute' in self.all_data.columns:
            errors = self.all_data['error_absolute'].dropna()
            if len(errors) > 0:
                axes[0].hist(errors, bins=50, density=True, alpha=0.7, edgecolor='black')
                axes[0].set_xlabel('Absolute error')
                axes[0].set_ylabel('Probability density')
                axes[0].set_title('Absolute error distribution')
                axes[0].grid(True, alpha=0.3)

                stats_text = (
                    f"Mean: {errors.mean():.4f}\n"
                    f"Std: {errors.std():.4f}\n"
                    f"RMSE: {np.sqrt(np.mean(errors ** 2)):.4f}\n"
                    f"Samples: {len(errors):,}"
                )
                axes[0].text(0.05, 0.95, stats_text, transform=axes[0].transAxes,
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 相对误差直方图
        if 'error_relative' in self.all_data.columns:
            rel_errors = self.all_data['error_relative'].dropna()
            if len(rel_errors) > 0:
                axes[1].hist(rel_errors, bins=50, density=True, alpha=0.7, edgecolor='black', color='green')
                axes[1].set_xlabel('Relative error')
                axes[1].set_ylabel('Probability density')
                axes[1].set_title('Relative error distribution')
                axes[1].grid(True, alpha=0.3)

                stats_text = (
                    f"Mean: {rel_errors.mean():.4f}\n"
                    f"Std: {rel_errors.std():.4f}\n"
                    f"Max: {rel_errors.max():.4f}\n"
                    f"Samples: {len(rel_errors):,}"
                )
                axes[1].text(0.05, 0.95, stats_text, transform=axes[1].transAxes,
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        # 误差与SZA的关系
        if 'sza' in self.all_data.columns:
            valid_data = self.all_data.dropna(subset=['error_absolute', 'sza'])
            if len(valid_data) > 0:
                axes[2].scatter(valid_data['sza'], valid_data['error_absolute'],
                                alpha=0.5, s=1, color='blue', label='Absolute')

            if 'error_relative' in self.all_data.columns:
                valid_data_rel = self.all_data.dropna(subset=['error_relative', 'sza'])
                if len(valid_data_rel) > 0:
                    axes[2].scatter(valid_data_rel['sza'], valid_data_rel['error_relative'],
                                    alpha=0.5, s=1, color='green', label='Relative')

            axes[2].set_xlabel('Solar zenith angle (deg)')
            axes[2].set_ylabel('Error')
            axes[2].set_title('Error vs solar zenith angle')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        # 误差与VZA的关系
        if 'vza' in self.all_data.columns:
            valid_data = self.all_data.dropna(subset=['error_absolute', 'vza'])
            if len(valid_data) > 0:
                axes[3].scatter(valid_data['vza'], valid_data['error_absolute'],
                                alpha=0.5, s=1, color='blue', label='Absolute')

            if 'error_relative' in self.all_data.columns:
                valid_data_rel = self.all_data.dropna(subset=['error_relative', 'vza'])
                if len(valid_data_rel) > 0:
                    axes[3].scatter(valid_data_rel['vza'], valid_data_rel['error_relative'],
                                    alpha=0.5, s=1, color='green', label='Relative')

            axes[3].set_xlabel('View zenith angle (deg)')
            axes[3].set_ylabel('Error')
            axes[3].set_title('Error vs view zenith angle')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"误差分布图已保存: {save_path}")

    def plot_3d_error_surface(self,
                              fixed_conditions: Dict[str, float] = None,
                              save_path: Optional[Path] = None):
        """
        绘制3D误差曲面
        """
        if fixed_conditions is None:
            fixed_conditions = {'aod550': 0.2, 'rho_true': 0.2, 'band': 'band3'}

        mask = pd.Series(True, index=self.all_data.index)
        for key, value in fixed_conditions.items():
            if key in self.all_data.columns:
                mask = mask & (self.all_data[key] == value)

        data = self.all_data[mask]

        if len(data) < 10:
            self.logger.warning("数据不足")
            return

        fig = plt.figure(figsize=(15, 5))

        # 3D曲面
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot_trisurf(data['sza'], data['vza'], data['error_absolute'],
                         cmap='viridis', alpha=0.8, edgecolor='none')
        ax1.set_xlabel('SZA (deg)')
        ax1.set_ylabel('VZA (deg)')
        ax1.set_zlabel('Absolute Error')
        ax1.set_title('3D Absolute Error Surface')

        # 2D投影
        ax2 = fig.add_subplot(132)
        contour = ax2.tricontourf(data['sza'], data['vza'], data['error_absolute'],
                                  levels=20, cmap='RdBu_r')
        plt.colorbar(contour, ax=ax2, label='Absolute Error')
        ax2.set_xlabel('SZA (deg)')
        ax2.set_ylabel('VZA (deg)')
        ax2.set_title('2D Absolute Error Contour')
        ax2.grid(True, alpha=0.3)

        # 相对误差分布
        ax3 = fig.add_subplot(133)
        if 'error_relative' in data.columns:
            rel_errors = data['error_relative'].dropna()
            if len(rel_errors) > 0:
                ax3.hist(rel_errors, bins=50, density=True,
                         alpha=0.7, edgecolor='black', color='green')
                ax3.set_xlabel('Relative Error')
                ax3.set_ylabel('Density')
                ax3.set_title('Relative Error Distribution')
                ax3.grid(True, alpha=0.3)

                mu, std = stats.norm.fit(rel_errors)
                x = np.linspace(rel_errors.min(), rel_errors.max(), 100)
                p = stats.norm.pdf(x, mu, std)
                ax3.plot(x, p, 'r-', linewidth=2, label=f'N({mu:.4f}, {std:.4f}²)')
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'No relative error data', ha='center', va='center')
        else:
            ax3.text(0.5, 0.5, 'No relative error column', ha='center', va='center')

        conditions_str = ', '.join([f'{k}={v}' for k, v in fixed_conditions.items()])
        plt.suptitle(f'3D Error Analysis - Fixed: {conditions_str}', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"3D误差曲面图已保存: {save_path}")

    def create_summary_figure(self,
                              fixed_conditions: Dict[str, float] = None,
                              band_id: str = 'band3',
                              save_path: Optional[Path] = None):
        """
        创建综合摘要图
        """
        if fixed_conditions is None:
            fixed_conditions = {'aod550': 0.2, 'rho_true': 0.2}

        if band_id == 'all':
            data = self.all_data.copy()
        elif band_id in self.results:
            data = self.results[band_id].copy()
        else:
            self.logger.warning(f"波段 {band_id} 不在结果中")
            return

        for key, value in fixed_conditions.items():
            if key in data.columns:
                data = data[data[key] == value]

        if 'error_absolute' not in data.columns or data['error_absolute'].dropna().empty:
            self.logger.warning(f"波段 {band_id} 没有误差数据")
            return

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 绝对误差等高线图
        ax1 = fig.add_subplot(gs[0, 0])
        if len(data) > 10:
            self._plot_single_contour(data, ax1, ['sza', 'vza'], 'error_absolute')
            ax1.set_title('(a) Absolute Error Contour', fontsize=12)
        else:
            ax1.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')

        # 相对误差等高线图
        ax2 = fig.add_subplot(gs[0, 1])
        if len(data) > 10 and 'error_relative' in data.columns:
            self._plot_single_contour(data, ax2, ['sza', 'vza'], 'error_relative')
            ax2.set_title('(b) Relative Error Contour', fontsize=12)
        else:
            ax2.text(0.5, 0.5, 'No relative error data', ha='center', va='center')

        # 3D绝对误差曲面
        ax3 = fig.add_subplot(gs[0, 2], projection='3d')
        if len(data) > 10:
            ax3.plot_trisurf(data['sza'], data['vza'], data['error_absolute'],
                             cmap='viridis', alpha=0.8, edgecolor='none')
            ax3.set_xlabel('SZA')
            ax3.set_ylabel('VZA')
            ax3.set_zlabel('Abs Error')
            ax3.set_title('(c) 3D Absolute Error Surface', fontsize=12)
        else:
            ax3.text(0.5, 0.5, 0.5, 'Insufficient data', ha='center', va='center')

        # SZA与误差关系
        ax4 = fig.add_subplot(gs[1, 0])
        if 'sza' in data.columns:
            valid_data = data.dropna(subset=['error_absolute', 'sza'])
            if len(valid_data) > 0:
                ax4.scatter(valid_data['sza'], valid_data['error_absolute'],
                            alpha=0.5, s=5, color='blue', label='Absolute')
                coeffs = np.polyfit(valid_data['sza'].values, valid_data['error_absolute'].values, 1)
                poly = np.poly1d(coeffs)
                x_fit = np.linspace(valid_data['sza'].min(), valid_data['sza'].max(), 100)
                ax4.plot(x_fit, poly(x_fit), 'b-', linewidth=2, alpha=0.8)

            if 'error_relative' in data.columns:
                valid_data_rel = data.dropna(subset=['error_relative', 'sza'])
                if len(valid_data_rel) > 0:
                    ax4.scatter(valid_data_rel['sza'], valid_data_rel['error_relative'],
                                alpha=0.5, s=5, color='green', label='Relative')

            ax4.set_xlabel('SZA (deg)')
            ax4.set_ylabel('Error')
            ax4.set_title('(d) Error vs SZA', fontsize=12)
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # VZA与误差关系
        ax5 = fig.add_subplot(gs[1, 1])
        if 'vza' in data.columns:
            valid_data = data.dropna(subset=['error_absolute', 'vza'])
            if len(valid_data) > 0:
                ax5.scatter(valid_data['vza'], valid_data['error_absolute'],
                            alpha=0.5, s=5, color='blue', label='Absolute')
                coeffs = np.polyfit(valid_data['vza'].values, valid_data['error_absolute'].values, 1)
                poly = np.poly1d(coeffs)
                x_fit = np.linspace(valid_data['vza'].min(), valid_data['vza'].max(), 100)
                ax5.plot(x_fit, poly(x_fit), 'b-', linewidth=2, alpha=0.8)

            if 'error_relative' in data.columns:
                valid_data_rel = data.dropna(subset=['error_relative', 'vza'])
                if len(valid_data_rel) > 0:
                    ax5.scatter(valid_data_rel['vza'], valid_data_rel['error_relative'],
                                alpha=0.5, s=5, color='green', label='Relative')

            ax5.set_xlabel('VZA (deg)')
            ax5.set_ylabel('Error')
            ax5.set_title('(e) Error vs VZA', fontsize=12)
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 相对误差分布直方图
        ax6 = fig.add_subplot(gs[1, 2])
        if 'error_relative' in data.columns:
            rel_errors = data['error_relative'].dropna()
            if len(rel_errors) > 0:
                ax6.hist(rel_errors, bins=50, density=True,
                         alpha=0.7, edgecolor='black', color='green')
                ax6.set_xlabel('Relative Error')
                ax6.set_ylabel('Density')
                ax6.set_title('(f) Relative Error Distribution', fontsize=12)
                ax6.grid(True, alpha=0.3)

                stats_text = (
                    f"Mean: {rel_errors.mean():.4f}\n"
                    f"Std: {rel_errors.std():.4f}\n"
                    f"Max: {rel_errors.max():.4f}\n"
                    f"N: {len(rel_errors):,}"
                )
                ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            else:
                ax6.text(0.5, 0.5, 'No relative error data', ha='center', va='center')
        else:
            ax6.text(0.5, 0.5, 'No relative error column', ha='center', va='center')

        conditions_str = ', '.join([f'{k}={v}' for k, v in fixed_conditions.items()])
        plt.suptitle(f'Summary of Geometric Error Analysis - Band: {band_id}\nFixed Conditions: {conditions_str}',
                     fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"综合摘要图已保存: {save_path}")

    def _plot_single_contour(self, data: pd.DataFrame, ax,
                             primary_factors: List[str],
                             color_factor: str):
        """绘制单个Contour图"""
        if len(data) < 10:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            return

        # 创建5度间隔的网格
        if len(data[primary_factors[0]].unique()) > 0:
            x_min = np.floor(data[primary_factors[0]].min() / 5) * 5
            x_max = np.ceil(data[primary_factors[0]].max() / 5) * 5
            x_grid_vals = np.arange(x_min, x_max + 5, 5)
        else:
            x_grid_vals = np.arange(0, 86, 5)

        if len(data[primary_factors[1]].unique()) > 0:
            y_min = np.floor(data[primary_factors[1]].min() / 5) * 5
            y_max = np.ceil(data[primary_factors[1]].max() / 5) * 5
            y_grid_vals = np.arange(y_min, y_max + 5, 5)
        else:
            y_grid_vals = np.arange(0, 76, 5)

        x_grid, y_grid = np.meshgrid(x_grid_vals, y_grid_vals)

        points = data[[primary_factors[0], primary_factors[1]]].values
        values = data[color_factor].values

        valid_mask = ~np.isnan(points).any(axis=1) & ~np.isnan(values)
        if np.sum(valid_mask) < 4:
            ax.text(0.5, 0.5, 'Insufficient data for interpolation', ha='center', va='center')
            return

        points_valid = points[valid_mask]
        values_valid = values[valid_mask]

        z_grid = griddata(points_valid, values_valid, (x_grid, y_grid), method='linear')

        contour = ax.contourf(x_grid, y_grid, z_grid, levels=20, cmap='RdBu_r')
        ax.contour(x_grid, y_grid, z_grid, levels=10, colors='k', linewidths=0.5, alpha=0.5)

        ax.set_xlabel(f'{primary_factors[0]} (deg)')
        ax.set_ylabel(f'{primary_factors[1]} (deg)')
        ax.set_xticks(x_grid_vals)
        ax.set_yticks(y_grid_vals)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.colorbar(contour, ax=ax, label=color_factor)

    # 新增：论文图表方法
    def create_paper_contour_figure(self, data: pd.DataFrame, error_type: str = 'absolute',
                                    save_path: Optional[Path] = None) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        创建论文所需的contour图 - 固定所有参数，只改变波段

        Args:
            data: 筛选后的数据
            error_type: 'absolute' 或 'relative'
            save_path: 保存路径

        Returns:
            fig: matplotlib图形对象
            axes: 子图坐标轴列表
        """
        config = ExperimentConfig.PAPER_FIGURES
        layout = config['contour_fixed_all']['layout']
        figsize = config['contour_fixed_all']['figsize']

        fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize,
                                 constrained_layout=True)

        # 展平axes数组以便遍历
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        bands = list(ExperimentConfig.BANDS.keys())

        for idx, band_id in enumerate(bands[:6]):  # 只取前6个波段
            if idx >= len(axes_flat):
                break

            ax = axes_flat[idx]
            band_data = data[data['band'] == band_id].copy()

            if len(band_data) < 10:
                ax.text(0.5, 0.5, f'No data for {band_id}',
                        ha='center', va='center')
                continue

            error_col = 'error_absolute' if error_type == 'absolute' else 'error_relative'

            # 使用固定参数筛选数据
            fixed_params = config['fixed_params']
            band_data_filtered = band_data.copy()

            for param, value in fixed_params.items():
                if param in band_data_filtered.columns:
                    band_data_filtered = band_data_filtered[band_data_filtered[param] == value]

            if len(band_data_filtered) < 10:
                ax.text(0.5, 0.5, f'Insufficient data\n{band_id}',
                        ha='center', va='center')
                continue

            self._plot_paper_contour(band_data_filtered, ax, ['sza', 'vza'], error_col)

            band_name = ExperimentConfig.BANDS[band_id]['name']

            # 构建标题，显示固定参数
            title_parts = [band_name]
            for param in ['aod550', 'rho_true', 'h2o', 'o3']:
                if param in fixed_params:
                    if param == 'aod550':
                        title_parts.append(f'AOD={fixed_params[param]}')
                    elif param == 'rho_true':
                        title_parts.append(f'ρ={fixed_params[param]}')
                    elif param == 'h2o':
                        title_parts.append(f'H2O={fixed_params[param]}g/cm²')
                    elif param == 'o3':
                        title_parts.append(f'O3={fixed_params[param]}cm-atm')

            ax.set_title('\n'.join(title_parts), fontsize=10)

        # 隐藏多余的子图
        for idx in range(len(bands), len(axes_flat)):
            axes_flat[idx].axis('off')

        fig.suptitle(f'{error_type.title()} Error Contours (Fixed Parameters)',
                     fontsize=14, y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"论文Contour图已保存: {save_path}")

        return fig, axes

    def _plot_paper_contour(self, data: pd.DataFrame, ax,
                            primary_factors: List[str], color_factor: str):
        """为论文绘制高质量的contour图"""
        if len(data) < 10:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            return

        # 获取数据范围
        x_data = data[primary_factors[0]]
        y_data = data[primary_factors[1]]
        z_data = data[color_factor]

        # 创建网格
        x_unique = np.sort(x_data.unique())
        y_unique = np.sort(y_data.unique())
        x_grid, y_grid = np.meshgrid(x_unique, y_unique)

        # 重新组织数据到网格
        z_grid = np.full_like(x_grid, np.nan, dtype=float)

        for i, x_val in enumerate(x_unique):
            for j, y_val in enumerate(y_unique):
                mask = (x_data == x_val) & (y_data == y_val)
                if mask.any():
                    z_grid[j, i] = z_data[mask].mean()

        # 插值填充NaN值
        valid_mask = ~np.isnan(z_grid)
        if valid_mask.sum() > 3:
            points = np.column_stack([x_grid[valid_mask], y_grid[valid_mask]])
            values = z_grid[valid_mask]
            try:
                z_grid_filled = griddata(points, values, (x_grid, y_grid), method='linear')
            except:
                z_grid_filled = z_grid
        else:
            z_grid_filled = z_grid

        # 绘制contour
        contour = ax.contourf(x_grid, y_grid, z_grid_filled,
                              levels=20, cmap='RdBu_r', extend='both')

        # 添加等高线
        ax.contour(x_grid, y_grid, z_grid_filled,
                   levels=10, colors='k', linewidths=0.5, alpha=0.5)

        # 添加数据点
        ax.scatter(x_data, y_data, c='k', s=10, alpha=0.3, marker='o')

        ax.set_xlabel('SZA (deg)', fontsize=9)
        ax.set_ylabel('VZA (deg)', fontsize=9)
        ax.set_xticks(x_unique)
        ax.set_yticks(y_unique)
        ax.grid(True, alpha=0.3, linestyle='--')

        # 添加颜色条
        plt.colorbar(contour, ax=ax, label=color_factor.replace('_', ' ').title())

    def create_varying_aod_figure(self, data: pd.DataFrame, error_type: str = 'absolute',
                                  save_path: Optional[Path] = None) -> Tuple[plt.Figure, List[plt.Axes]]:
        """创建不同AOD的contour图"""
        config = ExperimentConfig.PAPER_FIGURES
        layout = config['contour_varying_aod']['layout']
        figsize = config['contour_varying_aod']['figsize']
        aod_values = config['contour_varying_aod']['aod550_values']
        band_id = config['contour_varying_aod']['band']

        fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize,
                                 constrained_layout=True)

        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        band_data = data[data['band'] == band_id].copy()
        error_col = 'error_absolute' if error_type == 'absolute' else 'error_relative'

        for idx, aod_val in enumerate(aod_values):
            if idx >= len(axes_flat):
                break

            ax = axes_flat[idx]

            # 使用固定参数筛选数据
            filtered_data = band_data.copy()

            # 固定参数
            fixed_params = {
                'rho_true': config['contour_varying_aod']['rho_true'],
                'h2o': config['contour_varying_aod']['h2o'],
                'o3': config['contour_varying_aod']['o3']
            }

            for param, value in fixed_params.items():
                if param in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data[param] == value]

            # 变化参数：AOD
            filtered_data = filtered_data[filtered_data['aod550'] == aod_val]

            if len(filtered_data) < 10:
                ax.text(0.5, 0.5, f'Insufficient data\nAOD={aod_val}',
                        ha='center', va='center')
                continue

            self._plot_paper_contour(filtered_data, ax, ['sza', 'vza'], error_col)

            band_name = ExperimentConfig.BANDS[band_id]['name']

            # 构建标题
            title_parts = [band_name, f'AOD={aod_val}']
            for param, value in fixed_params.items():
                if param == 'rho_true':
                    title_parts.append(f'ρ={value}')
                elif param == 'h2o':
                    title_parts.append(f'H2O={value}g/cm²')
                elif param == 'o3':
                    title_parts.append(f'O3={value}cm-atm')

            ax.set_title('\n'.join(title_parts), fontsize=10)

        fig.suptitle(f'{error_type.title()} Error Contours - Varying AOD',
                     fontsize=14, y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"不同AOD Contour图已保存: {save_path}")

        return fig, axes

    def create_varying_rho_figure(self, data: pd.DataFrame, error_type: str = 'absolute',
                                  save_path: Optional[Path] = None) -> Tuple[plt.Figure, List[plt.Axes]]:
        """创建不同反射率的contour图"""
        config = ExperimentConfig.PAPER_FIGURES
        layout = config['contour_varying_rho']['layout']
        figsize = config['contour_varying_rho']['figsize']
        rho_values = config['contour_varying_rho']['rho_true_values']
        band_id = config['contour_varying_rho']['band']

        fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize,
                                 constrained_layout=True)

        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        band_data = data[data['band'] == band_id].copy()
        error_col = 'error_absolute' if error_type == 'absolute' else 'error_relative'

        for idx, rho_val in enumerate(rho_values):
            if idx >= len(axes_flat):
                break

            ax = axes_flat[idx]

            # 使用固定参数筛选数据
            filtered_data = band_data.copy()

            # 固定参数
            fixed_params = {
                'aod550': config['contour_varying_rho']['aod550'],
                'h2o': config['contour_varying_rho']['h2o'],
                'o3': config['contour_varying_rho']['o3']
            }

            for param, value in fixed_params.items():
                if param in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data[param] == value]

            # 变化参数：反射率
            filtered_data = filtered_data[filtered_data['rho_true'] == rho_val]

            if len(filtered_data) < 10:
                ax.text(0.5, 0.5, f'Insufficient data\nρ={rho_val}',
                        ha='center', va='center')
                continue

            self._plot_paper_contour(filtered_data, ax, ['sza', 'vza'], error_col)

            band_name = ExperimentConfig.BANDS[band_id]['name']

            # 构建标题
            title_parts = [band_name, f'ρ={rho_val}']
            for param, value in fixed_params.items():
                if param == 'aod550':
                    title_parts.append(f'AOD={value}')
                elif param == 'h2o':
                    title_parts.append(f'H2O={value}g/cm²')
                elif param == 'o3':
                    title_parts.append(f'O3={value}cm-atm')

            ax.set_title('\n'.join(title_parts), fontsize=10)

        fig.suptitle(f'{error_type.title()} Error Contours - Varying Reflectance',
                     fontsize=14, y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"不同反射率Contour图已保存: {save_path}")

        return fig, axes

    def create_varying_h2o_figure(self, data: pd.DataFrame, error_type: str = 'absolute',
                                  save_path: Optional[Path] = None) -> Tuple[plt.Figure, List[plt.Axes]]:
        """创建不同水汽含量的contour图"""
        config = ExperimentConfig.PAPER_FIGURES
        layout = config['contour_varying_h2o']['layout']
        figsize = config['contour_varying_h2o']['figsize']
        h2o_values = config['contour_varying_h2o']['h2o_values']
        band_id = config['contour_varying_h2o']['band']

        fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize,
                                 constrained_layout=True)

        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        band_data = data[data['band'] == band_id].copy()
        error_col = 'error_absolute' if error_type == 'absolute' else 'error_relative'

        for idx, h2o_val in enumerate(h2o_values):
            if idx >= len(axes_flat):
                break

            ax = axes_flat[idx]

            # 使用固定参数筛选数据
            filtered_data = band_data.copy()

            # 固定参数
            fixed_params = {
                'aod550': config['contour_varying_h2o']['aod550'],
                'rho_true': config['contour_varying_h2o']['rho_true'],
                'o3': config['contour_varying_h2o']['o3']
            }

            for param, value in fixed_params.items():
                if param in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data[param] == value]

            # 变化参数：水汽
            filtered_data = filtered_data[filtered_data['h2o'] == h2o_val]

            if len(filtered_data) < 10:
                ax.text(0.5, 0.5, f'Insufficient data\nH2O={h2o_val}g/cm²',
                        ha='center', va='center')
                continue

            self._plot_paper_contour(filtered_data, ax, ['sza', 'vza'], error_col)

            band_name = ExperimentConfig.BANDS[band_id]['name']

            # 构建标题
            title_parts = [band_name, f'H2O={h2o_val}g/cm²']
            for param, value in fixed_params.items():
                if param == 'aod550':
                    title_parts.append(f'AOD={value}')
                elif param == 'rho_true':
                    title_parts.append(f'ρ={value}')
                elif param == 'o3':
                    title_parts.append(f'O3={value}cm-atm')

            ax.set_title('\n'.join(title_parts), fontsize=10)

        fig.suptitle(f'{error_type.title()} Error Contours - Varying Water Vapor',
                     fontsize=14, y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"不同水汽Contour图已保存: {save_path}")

        return fig, axes

    def create_varying_o3_figure(self, data: pd.DataFrame, error_type: str = 'absolute',
                                 save_path: Optional[Path] = None) -> Tuple[plt.Figure, List[plt.Axes]]:
        """创建不同臭氧含量的contour图"""
        config = ExperimentConfig.PAPER_FIGURES
        layout = config['contour_varying_o3']['layout']
        figsize = config['contour_varying_o3']['figsize']
        o3_values = config['contour_varying_o3']['o3_values']
        band_id = config['contour_varying_o3']['band']

        fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize,
                                 constrained_layout=True)

        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        band_data = data[data['band'] == band_id].copy()
        error_col = 'error_absolute' if error_type == 'absolute' else 'error_relative'

        for idx, o3_val in enumerate(o3_values):
            if idx >= len(axes_flat):
                break

            ax = axes_flat[idx]

            # 使用固定参数筛选数据
            filtered_data = band_data.copy()

            # 固定参数
            fixed_params = {
                'aod550': config['contour_varying_o3']['aod550'],
                'rho_true': config['contour_varying_o3']['rho_true'],
                'h2o': config['contour_varying_o3']['h2o']
            }

            for param, value in fixed_params.items():
                if param in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data[param] == value]

            # 变化参数：臭氧
            filtered_data = filtered_data[filtered_data['o3'] == o3_val]

            if len(filtered_data) < 10:
                ax.text(0.5, 0.5, f'Insufficient data\nO3={o3_val}cm-atm',
                        ha='center', va='center')
                continue

            self._plot_paper_contour(filtered_data, ax, ['sza', 'vza'], error_col)

            band_name = ExperimentConfig.BANDS[band_id]['name']

            # 构建标题
            title_parts = [band_name, f'O3={o3_val}cm-atm']
            for param, value in fixed_params.items():
                if param == 'aod550':
                    title_parts.append(f'AOD={value}')
                elif param == 'rho_true':
                    title_parts.append(f'ρ={value}')
                elif param == 'h2o':
                    title_parts.append(f'H2O={value}g/cm²')

            ax.set_title('\n'.join(title_parts), fontsize=10)

        fig.suptitle(f'{error_type.title()} Error Contours - Varying Ozone',
                     fontsize=14, y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"不同臭氧Contour图已保存: {save_path}")

        return fig, axes

    def create_error_distribution_figure(self, data: pd.DataFrame, error_type: str = 'absolute',
                                         save_path: Optional[Path] = None) -> Tuple[plt.Figure, List[plt.Axes]]:
        """创建各波段误差分布子母图"""
        config = ExperimentConfig.PAPER_FIGURES
        layout = config['error_distribution']['layout']
        figsize = config['error_distribution']['figsize']

        fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize,
                                 constrained_layout=True)

        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        bands = list(ExperimentConfig.BANDS.keys())
        error_col = 'error_absolute' if error_type == 'absolute' else 'error_relative'

        for idx, band_id in enumerate(bands):
            if idx >= len(axes_flat):
                break

            ax = axes_flat[idx]
            band_data = data[data['band'] == band_id].copy()

            if error_col not in band_data.columns or band_data[error_col].dropna().empty:
                ax.text(0.5, 0.5, f'No {error_type} error data\n{band_id}',
                        ha='center', va='center')
                continue

            errors = band_data[error_col].dropna()

            # 绘制直方图
            ax.hist(errors, bins=30, density=True, alpha=0.7,
                    edgecolor='black', color='blue' if error_type == 'absolute' else 'green')

            # 拟合正态分布
            mu, std = stats.norm.fit(errors)
            x = np.linspace(errors.min(), errors.max(), 100)
            p = stats.norm.pdf(x, mu, std)
            ax.plot(x, p, 'r-', linewidth=2, label=f'N({mu:.4f}, {std:.4f}²)')

            # 添加统计信息
            stats_text = (f'Mean: {errors.mean():.4f}\n'
                          f'Std: {errors.std():.4f}\n'
                          f'N: {len(errors)}\n'
                          f'Band: {band_id}')
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax.set_xlabel(f'{error_type.title()} Error', fontsize=9)
            ax.set_ylabel('Probability Density', fontsize=9)
            ax.set_title(ExperimentConfig.BANDS[band_id]['name'], fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=8)

        fig.suptitle(f'{error_type.title()} Error Distribution by Band',
                     fontsize=14, y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"误差分布子母图已保存: {save_path}")

        return fig, axes