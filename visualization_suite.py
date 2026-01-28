# ==================== visualization_suite.py ====================
"""
可视化套件 - 集成原有所有的绘图逻辑
适配新的验证网格数据结构
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import xarray as xr
import warnings
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

warnings.filterwarnings('ignore')

from config import ExperimentConfig
from utils import setup_logger

# 设置专业科研字体（RSE期刊风格）
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 11,
    'figure.dpi': 300,
})


class VisualizationSuite:
    """
    集成可视化工具
    包含：
    1. 物理过程失真图 (LSR -> TOA -> Retrieved)
    2. 误差分布等高线图 (Contour Plots)
    """

    def __init__(self, data_dir: Optional[Path] = None, logger=None):
        self.logger = logger or setup_logger('VisualizationSuite')
        self.data_dir = data_dir or ExperimentConfig.DATA_DIR

        # 输出目录
        self.output_dir = ExperimentConfig.MANU_FIGURES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载数据
        self.all_data = self._load_validation_grid_data()

        # 定义标准大气条件 (适配 config.py 中的生成值: 0.05, 0.2, 0.5)
        # H2O 和 O3 固定为 config 中的单一值
        self.atmospheric_conditions = {
            'clean': {
                'name': 'Clean (AOD=0.05)',
                'aod550': 0.05,
                'h2o': 2.0,
                'o3': 0.3
            },
            'average': {
                'name': 'Average (AOD=0.2)',
                'aod550': 0.2,  # 对应 Config 中的 0.2
                'h2o': 2.0,
                'o3': 0.3
            },
            'polluted': {
                'name': 'Polluted (AOD=0.5)',
                'aod550': 0.5,
                'h2o': 2.0,
                'o3': 0.3
            }
        }

    def _load_validation_grid_data(self) -> pd.DataFrame:
        """加载验证网格数据 (validation_grid_*.nc)"""
        all_dfs = []
        self.logger.info("正在加载验证网格数据用于绘图...")

        for band_id in ExperimentConfig.BANDS.keys():
            # 修改：读取 validation_grid 文件
            data_file = self.data_dir / f"validation_grid_{band_id}.nc"

            if data_file.exists():
                try:
                    ds = xr.open_dataset(data_file)
                    df = ds.to_dataframe().reset_index()
                    ds.close()

                    # 确保必要的列存在
                    df['band'] = band_id
                    if 'wavelength' not in df.columns:
                        df['wavelength'] = ExperimentConfig.BANDS[band_id]['wavelength']

                    # 确保必要的计算列
                    if 'error_absolute' in df.columns and 'rho_true' in df.columns:
                        df['error_relative'] = df['error_absolute'] / df['rho_true']

                    # 确保 rho_retrieved 存在 (如果是从 NC 加载可能需要重新计算或确保已保存)
                    if 'rho_retrieved' not in df.columns and 'rho_toa' in df.columns:
                        # 这是一个 fallback，理想情况下应该在 simulate 阶段保存
                        # 如果没有反演值，这里暂时用 rho_toa 代替以防报错，但在日志中警告
                        self.logger.warning(f"波段 {band_id} 缺少 rho_retrieved，绘图可能不准确")
                        df['rho_retrieved'] = df.get('rho_true', 0) + df.get('error_absolute', 0)

                    all_dfs.append(df)
                    self.logger.info(f"  已加载 {band_id}: {len(df)} 条记录")

                except Exception as e:
                    self.logger.error(f"加载波段 {band_id} 失败: {e}")
            else:
                self.logger.warning(f"未找到文件: {data_file}，跳过该波段")

        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        else:
            self.logger.error("没有加载到任何验证数据！")
            return pd.DataFrame()

    def run_all_plots(self):
        """运行所有绘图任务"""
        if self.all_data.empty:
            self.logger.error("数据为空，无法绘图")
            return

        self.logger.info(">>> 开始生成物理过程失真图 (Plot 1)...")
        self.plot_distortion_process()

        self.logger.info(">>> 开始生成误差分布 Contour 图 (Plot 2)...")
        self.plot_error_contours(error_type='absolute')
        self.plot_error_contours(error_type='relative')

    # =========================================================================
    # Part 1: Weird Line Plots (Distortion Process)
    # =========================================================================
    def plot_distortion_process(self):
        """生成 LSR -> TOA -> LSR' 过程图"""
        # 1. 固定 SZA，变化 VZA (使用 Band 3)
        band_data = self.all_data[self.all_data['band'] == 'band3'].copy()
        if band_data.empty:
            self.logger.warning("Band 3 数据缺失，跳过 Plot 1")
            return

        # Config 中 SZA 有 [0, 20, 40, 60, 70, 75, 80, 85]
        # 我们选取几个典型的用于展示
        fixed_sza_values = [0, 40, 60, 75]
        self._create_fixed_sza_figure(band_data, fixed_sza_values)

        # 2. 固定 VZA，变化 SZA
        # Config 中 VZA 有 [0, 20, 40, 60, 70, 75]
        fixed_vza_values = [0, 20, 40, 60]
        self._create_fixed_vza_figure(band_data, fixed_vza_values)

    def _create_fixed_sza_figure(self, data: pd.DataFrame, fixed_sza_values: List):
        atm_order = ['clean', 'average', 'polluted']
        fig, axes = plt.subplots(3, 4, figsize=(12, 9), gridspec_kw={'hspace': 0.3, 'wspace': 0.25})

        # VZA 颜色映射
        colors_vza = [(0.9, 0.95, 1.0), (0.4, 0.6, 1.0), (0.1, 0.2, 0.8), (0.5, 0.1, 0.6)]
        cmap_vza = LinearSegmentedColormap.from_list('wide_gradient_vza', colors_vza, N=256)
        vza_min, vza_max = 0, 75

        for row_idx, condition_key in enumerate(atm_order):
            cond_params = self.atmospheric_conditions[condition_key]
            # 宽松过滤数据
            cond_data = data[
                (np.abs(data['aod550'] - cond_params['aod550']) < 0.05) &
                (np.abs(data['h2o'] - cond_params['h2o']) < 0.5)
                ]

            for col_idx, sza_val in enumerate(fixed_sza_values):
                ax = axes[row_idx, col_idx]
                sza_data = cond_data[np.abs(cond_data['sza'] - sza_val) < 2.0].copy()

                if sza_data.empty:
                    ax.axis('off')
                    continue

                vza_in_data = np.sort(sza_data['vza'].unique())

                for vza_val in vza_in_data:
                    point_data = sza_data[np.abs(sza_data['vza'] - vza_val) < 2.0]
                    if len(point_data) > 0:
                        vals = [
                            point_data['rho_true'].mean(),
                            point_data['rho_toa'].mean(),
                            point_data['rho_retrieved'].mean()
                        ]

                        color_norm = (vza_val - vza_min) / (vza_max - vza_min)
                        color = cmap_vza(color_norm)

                        ax.plot([0, 1, 2], vals, '-', linewidth=1.2, markersize=4, marker='o',
                                color=color, alpha=0.8)

                # 样式调整
                ax.set_xticks([0, 1, 2])
                ax.set_xticklabels([r'$\rho_{true}$', r'$\rho_{TOA}$', r'$\rho_{retr}$'], fontsize=8)

                if row_idx == 0: ax.set_title(rf'SZA ≈ {sza_val}°', fontsize=10, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel('Reflectance', fontsize=9)
                    ax.text(-0.3, 0.5, cond_params['name'], transform=ax.transAxes,
                            rotation=90, va='center', fontweight='bold', fontsize=9)

                ax.grid(True, alpha=0.15)

        # 颜色条
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = ScalarMappable(cmap=cmap_vza, norm=Normalize(vmin=vza_min, vmax=vza_max))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('View Zenith Angle (°)', fontsize=9)

        fig.suptitle('Reflectance Distortion Process (Fixed SZA)', fontsize=12, fontweight='bold', y=0.98)
        output_path = self.output_dir / "plot1_distortion_fixed_sza.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Plot 1 (Fixed SZA) 已保存: {output_path}")

    def _create_fixed_vza_figure(self, data: pd.DataFrame, fixed_vza_values: List):
        atm_order = ['clean', 'average', 'polluted']
        fig, axes = plt.subplots(3, 4, figsize=(12, 9), gridspec_kw={'hspace': 0.3, 'wspace': 0.25})

        colors_sza = [(1.0, 1.0, 0.9), (1.0, 0.6, 0.2), (0.5, 0.1, 0.1)]
        cmap_sza = LinearSegmentedColormap.from_list('wide_gradient_sza', colors_sza, N=256)
        sza_min, sza_max = 0, 85

        for row_idx, condition_key in enumerate(atm_order):
            cond_params = self.atmospheric_conditions[condition_key]
            cond_data = data[
                (np.abs(data['aod550'] - cond_params['aod550']) < 0.05) &
                (np.abs(data['h2o'] - cond_params['h2o']) < 0.5)
                ]

            for col_idx, vza_val in enumerate(fixed_vza_values):
                ax = axes[row_idx, col_idx]
                vza_data = cond_data[np.abs(cond_data['vza'] - vza_val) < 2.0].copy()

                if vza_data.empty:
                    ax.axis('off')
                    continue

                sza_in_data = np.sort(vza_data['sza'].unique())

                for sza_val in sza_in_data:
                    point_data = vza_data[np.abs(vza_data['sza'] - sza_val) < 2.0]
                    if len(point_data) > 0:
                        vals = [
                            point_data['rho_true'].mean(),
                            point_data['rho_toa'].mean(),
                            point_data['rho_retrieved'].mean()
                        ]

                        color_norm = (sza_val - sza_min) / (sza_max - sza_min)
                        color = cmap_sza(color_norm)

                        ax.plot([0, 1, 2], vals, '-', linewidth=1.2, markersize=4, marker='s',
                                color=color, alpha=0.8)

                # 样式
                ax.set_xticks([0, 1, 2])
                ax.set_xticklabels([r'$\rho_{true}$', r'$\rho_{TOA}$', r'$\rho_{retr}$'], fontsize=8)

                if row_idx == 0: ax.set_title(rf'VZA ≈ {vza_val}°', fontsize=10, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel('Reflectance', fontsize=9)
                    ax.text(-0.3, 0.5, cond_params['name'], transform=ax.transAxes,
                            rotation=90, va='center', fontweight='bold', fontsize=9)
                ax.grid(True, alpha=0.15)

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = ScalarMappable(cmap=cmap_sza, norm=Normalize(vmin=sza_min, vmax=sza_max))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Solar Zenith Angle (°)', fontsize=9)

        fig.suptitle('Reflectance Distortion Process (Fixed VZA)', fontsize=12, fontweight='bold', y=0.98)
        output_path = self.output_dir / "plot1_distortion_fixed_vza.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Plot 1 (Fixed VZA) 已保存: {output_path}")

    # =========================================================================
    # Part 2: Contour Plots
    # =========================================================================
    def plot_error_contours(self, error_type: str = 'absolute'):
        """生成 3(AOD) x 6(Band) 的 Contour 图"""
        atm_order = ['clean', 'average', 'polluted']
        bands = ['band1', 'band2', 'band3', 'band4', 'band5', 'band6']

        # 根据 Config 中的 VALIDATION_GRID 设置轴
        sza_grid = ExperimentConfig.VALIDATION_GRID['sza']
        vza_grid = ExperimentConfig.VALIDATION_GRID['vza']

        error_col = 'error_absolute' if error_type == 'absolute' else 'error_relative'

        # 确定 Color Scale
        all_vals = self.all_data[error_col].dropna().values
        if len(all_vals) == 0: return

        if error_type == 'absolute':
            vmin, vmax = -0.15, 0.02  # 略微宽松一点的范围
        else:
            vmin, vmax = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
            if vmax > 0: vmax = 0.05  # 允许稍微正一点的误差

        colors = [(0.0, 0.15, 0.7), (0.5, 0.7, 1.0), (0.95, 0.98, 1.0)]
        cmap = LinearSegmentedColormap.from_list('error_cmap', colors, N=256)
        norm = Normalize(vmin=vmin, vmax=vmax)

        fig, axes = plt.subplots(3, 6, figsize=(24, 12), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

        for row_idx, condition_key in enumerate(atm_order):
            cond_params = self.atmospheric_conditions[condition_key]
            cond_data = self.all_data[
                (np.abs(self.all_data['aod550'] - cond_params['aod550']) < 0.05) &
                (np.abs(self.all_data['h2o'] - cond_params['h2o']) < 0.5)
                ]

            for col_idx, band in enumerate(bands):
                ax = axes[row_idx, col_idx]
                band_data = cond_data[cond_data['band'] == band]

                if band_data.empty:
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                    continue

                # 构建网格矩阵
                matrix = np.zeros((len(vza_grid), len(sza_grid)))
                matrix.fill(np.nan)

                for i, vza in enumerate(vza_grid):
                    for j, sza in enumerate(sza_grid):
                        val = band_data[
                            (np.abs(band_data['sza'] - sza) < 1.0) &
                            (np.abs(band_data['vza'] - vza) < 1.0)
                            ][error_col].median()
                        matrix[i, j] = val

                if not np.all(np.isnan(matrix)):
                    X, Y = np.meshgrid(sza_grid, vza_grid)
                    contour = ax.contourf(X, Y, matrix, levels=15, cmap=cmap, norm=norm, extend='both')
                    # 只有当数据波动足够大时才画线
                    if np.nanstd(matrix) > 0.001:
                        ax.contour(X, Y, matrix, levels=5, colors='black', linewidths=0.5, alpha=0.5)

                # Labels
                if row_idx == 0:
                    wl = ExperimentConfig.BANDS[band]['wavelength']
                    ax.set_title(f'Band {col_idx + 1} ({wl}µm)', fontweight='bold')
                if col_idx == 0:
                    ax.text(-0.35, 0.5, cond_params['name'], transform=ax.transAxes,
                            rotation=90, va='center', fontweight='bold')

                ax.set_xlabel('SZA')
                ax.set_ylabel('VZA')
                ax.set_xticks(sza_grid[::2])  # 稀疏刻度
                ax.set_yticks(vza_grid[::2])

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(f'Median {error_type.capitalize()} Error')

        fig.suptitle(f'{error_type.capitalize()} Error Contour Plots', fontsize=14, fontweight='bold', y=0.98)

        output_path = self.output_dir / f"plot2_contour_{error_type}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Plot 2 (Contour {error_type}) 已保存: {output_path}")

