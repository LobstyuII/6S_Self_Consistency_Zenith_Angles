# ==================== error_analyzer.py ====================
"""
误差分析模块
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from scipy import stats
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
                        title_parts.append(f'H$_2$O={fixed_params[param]}g/cm$^2$')
                    elif param == 'o3':
                        title_parts.append(f'O$_3$={fixed_params[param]}cm-atm')

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
                    title_parts.append(f'H$_2$O={value}g/cm$^2$')
                elif param == 'o3':
                    title_parts.append(f'O$_3$={value}cm-atm')

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
                    title_parts.append(f'H$_2$O={value}g/cm$^2$')
                elif param == 'o3':
                    title_parts.append(f'O$_3$={value}cm-atm')

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
                ax.text(0.5, 0.5, f'Insufficient data\nH$_2$O={h2o_val}g/cm$^2$',
                        ha='center', va='center')
                continue

            self._plot_paper_contour(filtered_data, ax, ['sza', 'vza'], error_col)

            band_name = ExperimentConfig.BANDS[band_id]['name']

            # 构建标题
            title_parts = [band_name, f'H$_2$O={h2o_val}g/cm$^2$']
            for param, value in fixed_params.items():
                if param == 'aod550':
                    title_parts.append(f'AOD={value}')
                elif param == 'rho_true':
                    title_parts.append(f'ρ={value}')
                elif param == 'o3':
                    title_parts.append(f'O$_3$={value}cm-atm')

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
                ax.text(0.5, 0.5, f'Insufficient data\nO$_3$={o3_val}cm-atm',
                        ha='center', va='center')
                continue

            self._plot_paper_contour(filtered_data, ax, ['sza', 'vza'], error_col)

            band_name = ExperimentConfig.BANDS[band_id]['name']

            # 构建标题
            title_parts = [band_name, f'O$_3$O$_3$={o3_val}cm-atm']
            for param, value in fixed_params.items():
                if param == 'aod550':
                    title_parts.append(f'AOD={value}')
                elif param == 'rho_true':
                    title_parts.append(f'ρ={value}')
                elif param == 'h2o':
                    title_parts.append(f'H$_2$O={value}g/cm$^2$cm$^2$')

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