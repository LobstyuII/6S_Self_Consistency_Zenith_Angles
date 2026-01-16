# ==================== experimental_figures.py ====================
"""
实验性图表生成模块 - 使用模拟数据生成论文图表
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path
import xarray as xr
import warnings
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')

from config import ExperimentConfig
from utils import setup_logger, calculate_airmass

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


class ExperimentalFiguresGenerator:
    """实验性图表生成器"""

    def __init__(self, data_dir: Optional[Path] = None, logger=None):
        """
        初始化图表生成器

        Args:
            data_dir: 数据目录路径
            logger: 日志记录器
        """
        self.logger = logger or setup_logger('ExperimentalFiguresGenerator')
        self.data_dir = data_dir or ExperimentConfig.DATA_DIR

        # 加载所有波段数据
        self.all_data = self._load_all_bands_data()

        # 确保输出目录存在
        self.output_dir = ExperimentConfig.MANU_FIGURES_DIR / "experimental"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_all_bands_data(self) -> pd.DataFrame:
        """加载所有波段的数据"""
        all_dfs = []

        for band_id in ExperimentConfig.BANDS.keys():
            data_file = self.data_dir / f"simulation_results_{band_id}_parallel.nc"

            if data_file.exists():
                try:
                    # 加载NetCDF数据
                    ds = xr.open_dataset(data_file)
                    df = ds.to_dataframe().reset_index()
                    ds.close()

                    # 添加波段信息
                    df['band'] = band_id
                    df['wavelength'] = ExperimentConfig.BANDS[band_id]['wavelength']

                    # 计算空气质量
                    if 'sza' in df.columns:
                        df['secz_sza'] = 1.0 / np.cos(np.radians(df['sza']))
                    if 'vza' in df.columns:
                        df['secz_vza'] = 1.0 / np.cos(np.radians(df['vza']))
                    if 'secz_sza' in df.columns and 'secz_vza' in df.columns:
                        df['total_airmass'] = df['secz_sza'] + df['secz_vza']

                    # 计算相对误差
                    if 'error_absolute' in df.columns and 'rho_true' in df.columns:
                        df['error_relative'] = df['error_absolute'] / df['rho_true']

                    all_dfs.append(df)

                except Exception as e:
                    self.logger.error(f"加载波段 {band_id} 数据失败: {e}")
            else:
                self.logger.warning(f"文件不存在: {data_file}")

        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        else:
            return pd.DataFrame()

    def generate_all_figures(self):
        """生成所有实验性图表"""
        try:
            # 1. LSR → TOA → LSR' 过程折线图（固定SZA，变化VZA）
            self.generate_distortion_process_figure()

            # 2. 固定VZA，变化SZA（SZA上限85度）
            self._create_fixed_vza_figure()

        except Exception as e:
            self.logger.error(f"生成图表失败: {e}")

    def generate_distortion_process_figure(self):
        """LSR → TOA → LSR' 过程折线图"""
        if self.all_data.empty:
            self.logger.warning("没有数据可用于生成扭曲过程图")
            return

        # 定义大气环境（按clean->standard->polluted顺序）
        atmospheric_conditions = {
            'clean': {
                'name': 'Clean',
                'aod550': 0.1,
                'h2o': 1.0,
                'o3': 0.25
            },
            'standard': {
                'name': 'Standard',
                'aod550': 0.3,
                'h2o': 2.0,
                'o3': 0.3
            },
            'polluted': {
                'name': 'Polluted',
                'aod550': 0.5,
                'h2o': 3.0,
                'o3': 0.35
            }
        }

        # 使用band3的数据
        band_data = self.all_data[self.all_data['band'] == 'band3'].copy()

        # 固定的SZA值
        fixed_sza_values = [0, 30, 60, 75]

        # 创建固定SZA，变化VZA的图形
        self._create_fixed_sza_figure(band_data, atmospheric_conditions, fixed_sza_values)

    def _create_fixed_sza_figure(self, data: pd.DataFrame, atmospheric_conditions: Dict,
                                 fixed_sza_values: List):
        """创建固定SZA，变化VZA的图形（专业科研风格）"""

        # 设置大气环境顺序：clean -> standard -> polluted
        atm_order = ['clean', 'standard', 'polluted']

        # 创建图形
        fig, axes = plt.subplots(3, 4, figsize=(12, 9),
                                 gridspec_kw={'hspace': 0.3, 'wspace': 0.25})

        # 创建宽距渐变、梯度明显的颜色条 - 使用从浅蓝到深紫的渐变
        # 这种渐变在视觉上有明显的色阶差异
        colors_vza = [
            (0.9, 0.95, 1.0),      # 浅蓝
            (0.7, 0.8, 1.0),       # 淡蓝
            (0.4, 0.6, 1.0),       # 中蓝
            (0.2, 0.4, 0.9),       # 蓝色
            (0.1, 0.2, 0.8),       # 深蓝
            (0.3, 0.1, 0.7),       # 蓝紫
            (0.5, 0.1, 0.6),       # 紫色
        ]
        cmap_vza = LinearSegmentedColormap.from_list('wide_gradient_vza', colors_vza, N=256)

        # VZA范围严格限制在0-75度
        vza_min, vza_max = 0, 75

        # 创建离散但明显的颜色梯度 - 每15度一个明显的颜色变化
        # 这样可以在颜色条上清晰看到不同的VZA范围
        vza_bins = np.arange(vza_min, vza_max + 15, 15)

        # 收集所有数据用于统一y轴范围
        all_reflectance_data = []

        # 绘制图表
        for row_idx, condition_key in enumerate(atm_order):
            condition_params = atmospheric_conditions[condition_key]
            condition_name = condition_params['name']

            # 筛选当前大气环境的数据
            cond_data = data.copy()

            # 使用近似匹配
            aod_tolerance = 0.05
            h2o_tolerance = 0.5
            o3_tolerance = 0.05

            cond_data = cond_data[
                (np.abs(cond_data['aod550'] - condition_params['aod550']) < aod_tolerance) &
                (np.abs(cond_data['h2o'] - condition_params['h2o']) < h2o_tolerance) &
                (np.abs(cond_data['o3'] - condition_params['o3']) < o3_tolerance)
                ]

            # 对于每个固定的SZA值
            for col_idx, sza_val in enumerate(fixed_sza_values[:4]):
                ax = axes[row_idx, col_idx]

                # 筛选该SZA的数据
                sza_data = cond_data[np.abs(cond_data['sza'] - sza_val) < 1.0].copy()

                if len(sza_data) < 5:
                    ax.text(0.5, 0.5, f'No data\nSZA={sza_val}°',
                            ha='center', va='center', fontsize=9, style='italic')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                # 获取所有VZA值（排序）
                vza_in_data = np.sort(sza_data['vza'].unique())

                # 绘制每个VZA的折线
                for vza_val in vza_in_data:
                    # 筛选该VZA的数据
                    point_data = sza_data[np.abs(sza_data['vza'] - vza_val) < 1.0].copy()

                    if len(point_data) > 0:
                        # 计算平均值
                        avg_rho_true = point_data['rho_true'].mean()
                        avg_rho_toa = point_data['rho_toa'].mean()
                        avg_rho_retrieved = point_data['rho_retrieved'].mean()

                        # 收集数据用于统一y轴
                        all_reflectance_data.extend([avg_rho_true, avg_rho_toa, avg_rho_retrieved])

                        # 计算颜色（基于VZA归一化）
                        color_norm = (vza_val - vza_min) / (vza_max - vza_min)
                        color = cmap_vza(color_norm)

                        # 计算标记点外圈颜色（更深的同色系）
                        from matplotlib.colors import to_rgb
                        rgb = to_rgb(color)
                        edge_rgb = tuple(c * 0.7 for c in rgb)  # 加深颜色

                        # 绘制折线
                        ax.plot([0, 1, 2],
                                [avg_rho_true, avg_rho_toa, avg_rho_retrieved],
                                '-', linewidth=1.2, markersize=4, marker='o',
                                color=color, alpha=0.8,
                                markeredgecolor=edge_rgb, markeredgewidth=0.8)

                # 设置图表属性
                ax.set_xticks([0, 1, 2])
                ax.set_xticklabels([
                    r'$\rho_{\mathrm{true}}$',
                    r'$\rho_{\mathrm{TOA}}$',
                    r"$\rho_{\mathrm{retrieved}}$"
                ], fontsize=8)

                # 每个子图都显示y轴刻度
                ax.tick_params(axis='y', labelsize=8)

                # 仅第一列显示y轴标签
                if col_idx == 0:
                    ax.set_ylabel('Reflectance', fontsize=9, fontweight='normal')

                # 标题（仅第一行显示）
                if row_idx == 0:
                    ax.set_title(rf'SZA = {sza_val}°',
                                 fontsize=10, fontweight='bold', pad=8)

                # 添加大气条件信息（仅第一列显示）
                if col_idx == 0:
                    atm_label = condition_name
                    ax.text(-0.25, 0.5, atm_label, transform=ax.transAxes,
                            ha='right', va='center', fontsize=9, fontweight='bold',
                            rotation=90)

                # 添加浅灰色网格
                ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5, which='both')

                # 设置坐标轴边框样式
                for spine in ['top', 'right']:
                    ax.spines[spine].set_visible(False)

                for spine in ['bottom', 'left']:
                    ax.spines[spine].set_linewidth(0.75)
                    ax.spines[spine].set_color('black')

                # 设置刻度线样式
                ax.tick_params(axis='both', which='both', length=4, width=0.75,
                               direction='out', colors='black')

        # 设置统一的y轴范围（基于所有数据）
        if all_reflectance_data:
            y_min = max(0, min(all_reflectance_data) * 0.95)
            y_max = min(0.8, max(all_reflectance_data) * 1.05)
            for ax_row in axes:
                for ax in ax_row:
                    if hasattr(ax, 'get_visible') and ax.get_visible():
                        ax.set_ylim(y_min, y_max)

        # 添加全局颜色条 - 使用明显的梯度变化
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

        # 创建带明显梯度的颜色条
        norm = Normalize(vmin=vza_min, vmax=vza_max)
        sm = ScalarMappable(cmap=cmap_vza, norm=norm)
        sm.set_array([])

        # 添加颜色条
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label(r'View Zenith Angle (°)', fontsize=9, labelpad=10)
        cbar.ax.tick_params(labelsize=8)

        # 设置颜色条刻度 - 显示关键值，包括上限75
        vza_ticks = [0, 15, 30, 45, 60, 75]
        cbar.set_ticks(vza_ticks)
        cbar.set_ticklabels([f'{int(t)}' for t in vza_ticks])

        # 添加总标题
        fig.suptitle('Atmospheric Distortion in Surface Reflectance Retrieval',
                     fontsize=11, fontweight='bold', y=0.98)

        # 添加子标题
        fig.text(0.5, 0.94,
                 r'Fixed SZA with varying VZA (0-75°) | Band 3 (0.64 $\mu$m)',
                 ha='center', fontsize=10, style='italic')

        # 调整布局
        plt.tight_layout(rect=[0.02, 0.04, 0.90, 0.95])

        # 保存图像
        output_path = self.output_dir / "distortion_process_fixed_sza_journal.png"
        plt.savefig(output_path, dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        self.logger.info(f"固定SZA图已保存: {output_path}")

    def _create_fixed_vza_figure(self):
        """创建固定VZA，变化SZA的图形（SZA上限85度）"""

        # 加载band3的数据
        band_data = self.all_data[self.all_data['band'] == 'band3'].copy()

        # 定义大气环境顺序：clean -> standard -> polluted
        atmospheric_conditions = {
            'clean': {
                'name': 'Clean',
                'aod550': 0.1,
                'h2o': 1.0,
                'o3': 0.25
            },
            'standard': {
                'name': 'Standard',
                'aod550': 0.3,
                'h2o': 2.0,
                'o3': 0.3
            },
            'polluted': {
                'name': 'Polluted',
                'aod550': 0.5,
                'h2o': 3.0,
                'o3': 0.35
            }
        }
        atm_order = ['clean', 'standard', 'polluted']

        # 固定的VZA值
        fixed_vza_values = [0, 30, 60, 75]

        # 创建图形
        fig, axes = plt.subplots(3, 4, figsize=(12, 9),
                                 gridspec_kw={'hspace': 0.3, 'wspace': 0.25})

        # 创建宽距渐变、梯度明显的颜色条 - 使用从浅黄到深红的渐变
        # 这种渐变在视觉上有明显的色阶差异
        colors_sza = [
            (1.0, 1.0, 0.9),      # 浅黄
            (1.0, 0.95, 0.7),     # 淡黄
            (1.0, 0.8, 0.4),      # 橙色
            (1.0, 0.6, 0.2),      # 橙红
            (0.9, 0.4, 0.1),      # 红色
            (0.7, 0.2, 0.1),      # 深红
            (0.5, 0.1, 0.1),      # 暗红
        ]
        cmap_sza = LinearSegmentedColormap.from_list('wide_gradient_sza', colors_sza, N=256)

        # SZA范围严格限制在0-85度
        sza_min, sza_max = 0, 85

        # 收集所有数据用于统一y轴范围
        all_reflectance_data = []

        # 绘制图表
        for row_idx, condition_key in enumerate(atm_order):
            condition_params = atmospheric_conditions[condition_key]
            condition_name = condition_params['name']

            # 筛选当前大气环境的数据
            cond_data = band_data.copy()

            # 使用近似匹配
            aod_tolerance = 0.05
            h2o_tolerance = 0.5
            o3_tolerance = 0.05

            cond_data = cond_data[
                (np.abs(cond_data['aod550'] - condition_params['aod550']) < aod_tolerance) &
                (np.abs(cond_data['h2o'] - condition_params['h2o']) < h2o_tolerance) &
                (np.abs(cond_data['o3'] - condition_params['o3']) < o3_tolerance)
                ]

            # 对于每个固定的VZA值
            for col_idx, vza_val in enumerate(fixed_vza_values[:4]):
                ax = axes[row_idx, col_idx]

                # 筛选该VZA的数据
                vza_data = cond_data[np.abs(cond_data['vza'] - vza_val) < 1.0].copy()

                if len(vza_data) < 5:
                    ax.text(0.5, 0.5, f'No data\nVZA={vza_val}°',
                            ha='center', va='center', fontsize=9, style='italic')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                # 获取所有SZA值（排序，上限85度）
                sza_in_data = np.sort(vza_data['sza'].unique())
                sza_in_data = sza_in_data[sza_in_data <= 85]  # 限制SZA上限为85度

                # 绘制每个SZA的折线
                for sza_val in sza_in_data:
                    # 筛选该SZA的数据
                    point_data = vza_data[np.abs(vza_data['sza'] - sza_val) < 1.0].copy()

                    if len(point_data) > 0:
                        # 计算平均值
                        avg_rho_true = point_data['rho_true'].mean()
                        avg_rho_toa = point_data['rho_toa'].mean()
                        avg_rho_retrieved = point_data['rho_retrieved'].mean()

                        # 收集数据用于统一y轴
                        all_reflectance_data.extend([avg_rho_true, avg_rho_toa, avg_rho_retrieved])

                        # 计算颜色（基于SZA归一化）
                        color_norm = (sza_val - sza_min) / (sza_max - sza_min)
                        color = cmap_sza(color_norm)

                        # 计算标记点外圈颜色（更深的同色系）
                        from matplotlib.colors import to_rgb
                        rgb = to_rgb(color)
                        edge_rgb = tuple(c * 0.7 for c in rgb)  # 加深颜色

                        # 绘制折线
                        ax.plot([0, 1, 2],
                                [avg_rho_true, avg_rho_toa, avg_rho_retrieved],
                                '-', linewidth=1.2, markersize=4, marker='s',
                                color=color, alpha=0.8,
                                markeredgecolor=edge_rgb, markeredgewidth=0.8)

                # 设置图表属性
                ax.set_xticks([0, 1, 2])
                ax.set_xticklabels([
                    r'$\rho_{\mathrm{true}}$',
                    r'$\rho_{\mathrm{TOA}}$',
                    r"$\rho_{\mathrm{retrieved}}$"
                ], fontsize=8)

                # 每个子图都显示y轴刻度
                ax.tick_params(axis='y', labelsize=8)

                # 仅第一列显示y轴标签
                if col_idx == 0:
                    ax.set_ylabel('Reflectance', fontsize=9, fontweight='normal')

                # 标题（仅第一行显示）
                if row_idx == 0:
                    ax.set_title(rf'VZA = {vza_val}°',
                                 fontsize=10, fontweight='bold', pad=8)

                # 添加大气条件信息（仅第一列显示）
                if col_idx == 0:
                    atm_label = condition_name
                    ax.text(-0.25, 0.5, atm_label, transform=ax.transAxes,
                            ha='right', va='center', fontsize=9, fontweight='bold',
                            rotation=90)

                # 添加浅灰色网格
                ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5, which='both')

                # 设置坐标轴边框样式
                for spine in ['top', 'right']:
                    ax.spines[spine].set_visible(False)

                for spine in ['bottom', 'left']:
                    ax.spines[spine].set_linewidth(0.75)
                    ax.spines[spine].set_color('black')

                # 设置刻度线样式
                ax.tick_params(axis='both', which='both', length=4, width=0.75,
                               direction='out', colors='black')

        # 设置统一的y轴范围（基于所有数据）
        if all_reflectance_data:
            y_min = max(0, min(all_reflectance_data) * 0.95)
            y_max = min(0.8, max(all_reflectance_data) * 1.05)
            for ax_row in axes:
                for ax in ax_row:
                    if hasattr(ax, 'get_visible') and ax.get_visible():
                        ax.set_ylim(y_min, y_max)

        # 添加全局颜色条 - 使用明显的梯度变化
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

        # 创建带明显梯度的颜色条
        norm = Normalize(vmin=sza_min, vmax=sza_max)
        sm = ScalarMappable(cmap=cmap_sza, norm=norm)
        sm.set_array([])

        # 添加颜色条
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label(r'Solar Zenith Angle (°)', fontsize=9, labelpad=10)
        cbar.ax.tick_params(labelsize=8)

        # 设置颜色条刻度 - 显示关键值，包括上限85
        sza_ticks = [0, 17, 34, 51, 68, 85]
        cbar.set_ticks(sza_ticks)
        cbar.set_ticklabels([f'{int(t)}' for t in sza_ticks])

        # 添加总标题
        fig.suptitle('Atmospheric Distortion in Surface Reflectance Retrieval',
                     fontsize=11, fontweight='bold', y=0.98)

        # 添加子标题
        fig.text(0.5, 0.94,
                 r'Fixed VZA with varying SZA (0-85°) | Band 3 (0.64 $\mu$m)',
                 ha='center', fontsize=10, style='italic')

        # 调整布局
        plt.tight_layout(rect=[0.02, 0.04, 0.90, 0.95])

        # 保存图像
        output_path = self.output_dir / "distortion_process_fixed_vza_journal.png"
        plt.savefig(output_path, dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        self.logger.info(f"固定VZA图已保存: {output_path}")


def main():
    """主函数"""
    logger = setup_logger('ExperimentalFigures', level='INFO')

    try:
        generator = ExperimentalFiguresGenerator(logger=logger)

        if generator.all_data.empty:
            logger.error("没有加载到数据，请检查数据文件")
            return

        generator.generate_all_figures()

        logger.info(f"图表已保存到: {generator.output_dir}")

    except Exception as e:
        logger.error(f"程序失败: {e}")


if __name__ == "__main__":
    main()