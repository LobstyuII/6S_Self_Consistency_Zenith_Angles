# ==================== plot2_error_contour.py ====================
"""
生成3行6列的contour误差图
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


class PaperFiguresGeneratorV2:

    def __init__(self, data_dir: Optional[Path] = None, logger=None):
        """
        初始化图表生成器

        Args:
            data_dir: 数据目录路径
            logger: 日志记录器
        """
        self.logger = logger or setup_logger('PaperFiguresGeneratorV2')
        self.data_dir = data_dir or ExperimentConfig.DATA_DIR

        # 加载所有波段数据
        self.all_data = self._load_all_bands_data()

        # 确保输出目录存在
        self.output_dir = ExperimentConfig.MANU_FIGURES_DIR / "plot2"
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

    def _get_error_range(self, error_type: str, all_error_values: List[float]) -> Tuple[float, float]:
        """
        获取误差的颜色映射范围

        Args:
            error_type: 误差类型 ('absolute' 或 'relative')
            all_error_values: 所有误差值的列表

        Returns:
            (vmin, vmax): 颜色映射的最小值和最大值
        """
        if error_type == 'absolute':
            # 对于绝对误差，使用固定的范围0到-0.225
            return -0.225, 0
        else:
            # 对于相对误差
            if all_error_values and len(all_error_values) > 0:
                all_error_values = np.array(all_error_values)

                # 移除NaN和无限值
                all_error_values = all_error_values[np.isfinite(all_error_values)]

                if len(all_error_values) > 0:
                    # 使用1%和99%分位数来捕获更大的范围
                    vmax = np.percentile(all_error_values, 99)
                    vmin = np.percentile(all_error_values, 1)

                    # 确保范围合理
                    if vmax > 0:
                        vmax = 0  # 相对误差通常为负值

                    # 确保最小值为负，且范围足够大
                    if vmin > -0.01:  # 如果最小值接近0
                        vmin = -0.1  # 设置更宽的范围
                    elif vmax - vmin < 0.05:  # 如果范围太小
                        # 扩展范围
                        range_expand = 0.05 - (vmax - vmin)
                        vmin = vmin - range_expand / 2

                    # 限制范围在合理区间内
                    vmin = max(vmin, -1.0)  # 不超过-100%
                    vmax = min(vmax, 0.0)  # 不超过0

                    self.logger.info(
                        f"相对误差范围: {vmin:.4f} 到 {vmax:.4f}, 数据范围: {np.min(all_error_values):.4f} 到 {np.max(all_error_values):.4f}")
                    return vmin, vmax

            # 默认范围
            return -0.15, 0  # -15% 到 0%

    def generate_contour_figures(self, error_type: str = 'absolute'):
        """
        生成3行6列的contour误差图

        Args:
            error_type: 误差类型 ('absolute' 或 'relative')
        """
        if self.all_data.empty:
            self.logger.warning("没有数据可用于生成contour图")
            return

        # 定义大气环境（与LSR_TOA_processing.py一致）
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

        # 波段列表
        bands = ['band1', 'band2', 'band3', 'band4', 'band5', 'band6']

        # SZA和VZA的网格点
        sza_values = [0, 15, 30, 45, 60, 75]
        vza_values = [0, 15, 30, 45, 60, 75]

        # 误差列名
        error_col = 'error_absolute' if error_type == 'absolute' else 'error_relative'

        if error_col not in self.all_data.columns:
            self.logger.error(f"数据中不存在 {error_col} 列")
            return

        # 收集所有误差值以确定全局范围
        all_error_values = []

        for row_idx, (condition_key, condition_params) in enumerate(atmospheric_conditions.items()):
            # 筛选当前大气环境的数据
            cond_data = self.all_data.copy()

            # 使用近似匹配
            aod_tolerance = 0.05
            h2o_tolerance = 0.5
            o3_tolerance = 0.05

            cond_data = cond_data[
                (np.abs(cond_data['aod550'] - condition_params['aod550']) < aod_tolerance) &
                (np.abs(cond_data['h2o'] - condition_params['h2o']) < h2o_tolerance) &
                (np.abs(cond_data['o3'] - condition_params['o3']) < o3_tolerance)
                ]

            for col_idx, band in enumerate(bands):
                # 筛选当前波段的数据
                band_data = cond_data[cond_data['band'] == band].copy()

                if band_data.empty:
                    continue

                # 收集该波段的所有误差值
                error_values = band_data[error_col].dropna().values
                all_error_values.extend(error_values)

        # 设置颜色映射范围 - 使用新的方法
        vmin, vmax = self._get_error_range(error_type, all_error_values)

        # 创建颜色映射 - 从深蓝色到浅蓝色
        colors = [
            (0.0, 0.15, 0.7),  # 非常深的蓝色
            (0.1, 0.3, 0.8),  # 深蓝色
            (0.3, 0.5, 0.9),  # 蓝色
            (0.5, 0.7, 1.0),  # 中等蓝色
            (0.7, 0.85, 1.0),  # 浅蓝色
            (0.85, 0.92, 1.0),  # 浅蓝色
            (0.95, 0.98, 1.0),  # 非常浅的蓝色（接近白色）
        ]
        cmap = LinearSegmentedColormap.from_list('error_cmap_v2', colors, N=256)
        norm = Normalize(vmin=vmin, vmax=vmax)

        # 创建图形
        fig, axes = plt.subplots(3, 6, figsize=(24, 12),
                                 gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

        # 绘制每个子图
        for row_idx, (condition_key, condition_params) in enumerate(atmospheric_conditions.items()):
            condition_name = condition_params['name']

            # 筛选当前大气环境的数据
            cond_data = self.all_data.copy()

            # 使用近似匹配
            aod_tolerance = 0.05
            h2o_tolerance = 0.5
            o3_tolerance = 0.05

            cond_data = cond_data[
                (np.abs(cond_data['aod550'] - condition_params['aod550']) < aod_tolerance) &
                (np.abs(cond_data['h2o'] - condition_params['h2o']) < h2o_tolerance) &
                (np.abs(cond_data['o3'] - condition_params['o3']) < o3_tolerance)
                ]

            for col_idx, band in enumerate(bands):
                ax = axes[row_idx, col_idx]

                # 筛选当前波段的数据
                band_data = cond_data[cond_data['band'] == band].copy()

                if band_data.empty:
                    ax.text(0.5, 0.5, 'No Data',
                            ha='center', va='center', fontsize=9, style='italic')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                # 创建SZA-VZA网格的中位数矩阵
                median_matrix = np.zeros((len(vza_values), len(sza_values)))
                median_matrix.fill(np.nan)

                for i, vza in enumerate(vza_values):
                    for j, sza in enumerate(sza_values):
                        # 筛选该SZA-VZA组合的数据
                        combo_data = band_data[
                            (np.abs(band_data['sza'] - sza) < 1.0) &
                            (np.abs(band_data['vza'] - vza) < 1.0)
                            ]

                        if len(combo_data) > 0:
                            errors = combo_data[error_col].dropna().values
                            if len(errors) > 0:
                                median_matrix[i, j] = np.median(errors)

                # 绘制contour图
                if not np.all(np.isnan(median_matrix)):
                    # 创建网格
                    X, Y = np.meshgrid(sza_values, vza_values)

                    # 绘制填充等高线
                    contour = ax.contourf(X, Y, median_matrix, levels=20,
                                          cmap=cmap, norm=norm,
                                          vmin=vmin, vmax=vmax,
                                          alpha=0.8)

                    # 添加等高线标签
                    ax.contour(X, Y, median_matrix, levels=10,
                               colors='black', linewidths=0.5, alpha=0.7)

                    # 添加数值标签（可选，如果数据密度合适）
                    # 只在数值变化明显的位置添加标签
                    if np.nanstd(median_matrix) > 0.001:
                        contour_labels = ax.contour(X, Y, median_matrix, levels=5,
                                                    colors='darkred', linewidths=0.8)
                        ax.clabel(contour_labels, inline=True, fontsize=6, fmt='%.3f')

                # 设置坐标轴
                ax.set_xlabel('SZA (°)', fontsize=9, labelpad=3)
                ax.set_ylabel('VZA (°)', fontsize=9, labelpad=3)

                # 设置刻度
                ax.set_xticks(sza_values)
                ax.set_yticks(vza_values)
                ax.set_xticklabels([f'{sza}' for sza in sza_values], fontsize=8)
                ax.set_yticklabels([f'{vza}' for vza in vza_values], fontsize=8)

                # 添加网格（可选）
                ax.grid(True, alpha=0.2, linestyle='--')

                # 设置坐标轴边框样式
                for spine in ['top', 'right']:
                    ax.spines[spine].set_visible(False)

                for spine in ['bottom', 'left']:
                    ax.spines[spine].set_linewidth(0.75)
                    ax.spines[spine].set_color('black')

                # 设置刻度线样式
                ax.tick_params(axis='both', which='both', length=4, width=0.75,
                               direction='out', colors='black')

                # 添加波段信息（仅第一行）
                if row_idx == 0:
                    wavelength = ExperimentConfig.BANDS[band]['wavelength']
                    ax.set_title(f'Band {band[-1]}\n({wavelength} µm)',
                                 fontsize=10, fontweight='bold', pad=10)

                # 添加大气条件信息（仅第一列）
                if col_idx == 0:
                    ax.text(-0.25, 0.5, condition_name, transform=ax.transAxes,
                            ha='right', va='center', fontsize=9, fontweight='bold',
                            rotation=90)

        # 添加全局颜色条
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

        # 设置颜色条刻度
        if error_type == 'absolute':
            # 对于绝对误差，使用固定的刻度
            cbar_ticks = np.linspace(vmin, vmax, 6)
            cbar_ticklabels = [f'{tick:.3f}' for tick in cbar_ticks]
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_ticklabels)
            cbar_label = 'Median Absolute Error'
        else:
            # 对于相对误差，使用动态刻度
            cbar_ticks = np.linspace(vmin, vmax, 6)
            cbar_ticklabels = [f'{tick:.3f}' for tick in cbar_ticks]
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_ticklabels)
            cbar_label = 'Median Relative Error'

        cbar.set_label(cbar_label, fontsize=9, labelpad=10)
        cbar.ax.tick_params(labelsize=8)

        # 添加图例说明
        fig.text(0.5, 0.02,
                 f'Contour plot showing {error_type} error distribution across SZA/VZA combinations\n'
                 f'SZA grid: {sza_values}°, VZA grid: {vza_values}°',
                 ha='center', fontsize=9, style='italic')

        # 添加总标题
        title = f'{error_type.title()} Error Contour Plots Across Bands and Atmospheric Conditions'
        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)

        # 调整布局
        plt.tight_layout(rect=[0.02, 0.04, 0.90, 0.95])

        # 保存图像
        output_path = self.output_dir / f"contour_plot_{error_type}.png"
        plt.savefig(output_path, dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        self.logger.info(f"Contour图已保存: {output_path}")

        # 同时生成另一种风格的版本（等值线更稀疏）
        self._generate_simplified_contour(error_type, atmospheric_conditions,
                                          bands, sza_values, vza_values, vmin, vmax)

    def _generate_simplified_contour(self, error_type: str, atmospheric_conditions: Dict,
                                     bands: List, sza_values: List, vza_values: List,
                                     vmin: float, vmax: float):
        """生成简化的contour图（等值线更稀疏）"""
        error_col = 'error_absolute' if error_type == 'absolute' else 'error_relative'

        # 创建颜色映射
        colors = [
            (0.0, 0.15, 0.7),  # 非常深的蓝色
            (0.1, 0.3, 0.8),  # 深蓝色
            (0.3, 0.5, 0.9),  # 蓝色
            (0.5, 0.7, 1.0),  # 中等蓝色
            (0.7, 0.85, 1.0),  # 浅蓝色
            (0.85, 0.92, 1.0),  # 浅蓝色
            (0.95, 0.98, 1.0),  # 非常浅的蓝色（接近白色）
        ]
        cmap = LinearSegmentedColormap.from_list('simplified_cmap', colors, N=256)
        norm = Normalize(vmin=vmin, vmax=vmax)

        # 创建图形
        fig, axes = plt.subplots(3, 6, figsize=(24, 12),
                                 gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

        # 绘制每个子图
        for row_idx, (condition_key, condition_params) in enumerate(atmospheric_conditions.items()):
            condition_name = condition_params['name']

            # 筛选当前大气环境的数据
            cond_data = self.all_data.copy()

            # 使用近似匹配
            aod_tolerance = 0.05
            h2o_tolerance = 0.5
            o3_tolerance = 0.05

            cond_data = cond_data[
                (np.abs(cond_data['aod550'] - condition_params['aod550']) < aod_tolerance) &
                (np.abs(cond_data['h2o'] - condition_params['h2o']) < h2o_tolerance) &
                (np.abs(cond_data['o3'] - condition_params['o3']) < o3_tolerance)
                ]

            for col_idx, band in enumerate(bands):
                ax = axes[row_idx, col_idx]

                # 筛选当前波段的数据
                band_data = cond_data[cond_data['band'] == band].copy()

                if band_data.empty:
                    ax.text(0.5, 0.5, 'No Data',
                            ha='center', va='center', fontsize=9, style='italic')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                # 创建SZA-VZA网格的中位数矩阵
                median_matrix = np.zeros((len(vza_values), len(sza_values)))
                median_matrix.fill(np.nan)

                for i, vza in enumerate(vza_values):
                    for j, sza in enumerate(sza_values):
                        # 筛选该SZA-VZA组合的数据
                        combo_data = band_data[
                            (np.abs(band_data['sza'] - sza) < 1.0) &
                            (np.abs(band_data['vza'] - vza) < 1.0)
                            ]

                        if len(combo_data) > 0:
                            errors = combo_data[error_col].dropna().values
                            if len(errors) > 0:
                                median_matrix[i, j] = np.median(errors)

                # 绘制简化版的contour图（更少的等值线）
                if not np.all(np.isnan(median_matrix)):
                    # 创建网格
                    X, Y = np.meshgrid(sza_values, vza_values)

                    # 只绘制填充，不绘制等值线
                    contour = ax.contourf(X, Y, median_matrix, levels=20,
                                          cmap=cmap, norm=norm,
                                          vmin=vmin, vmax=vmax,
                                          alpha=0.8)

                    # 只在数值变化明显的地方添加等值线
                    if np.nanstd(median_matrix) > 0.005:
                        # 使用更少的等值线
                        contour_lines = ax.contour(X, Y, median_matrix, levels=5,
                                                   colors='darkred', linewidths=1.0)
                        # 添加标签
                        ax.clabel(contour_lines, inline=True, fontsize=7, fmt='%.3f')

                # 设置坐标轴
                ax.set_xlabel('SZA (°)', fontsize=9)
                ax.set_ylabel('VZA (°)', fontsize=9)
                ax.set_xticks(sza_values)
                ax.set_yticks(vza_values)
                ax.set_xticklabels([f'{sza}' for sza in sza_values], fontsize=8, rotation=45)
                ax.set_yticklabels([f'{vza}' for vza in vza_values], fontsize=8)

                # 添加网格
                ax.grid(True, alpha=0.2, linestyle='--')

                # 设置坐标轴边框样式
                for spine in ['top', 'right']:
                    ax.spines[spine].set_visible(False)

                for spine in ['bottom', 'left']:
                    ax.spines[spine].set_linewidth(0.75)
                    ax.spines[spine].set_color('black')

                # 添加波段信息（仅第一行）
                if row_idx == 0:
                    wavelength = ExperimentConfig.BANDS[band]['wavelength']
                    ax.set_title(f'Band {band[-1]}\n({wavelength} µm)',
                                 fontsize=10, fontweight='bold', pad=10)

                # 添加大气条件信息（仅第一列）
                if col_idx == 0:
                    ax.text(-0.25, 0.5, condition_name, transform=ax.transAxes,
                            ha='right', va='center', fontsize=9, fontweight='bold',
                            rotation=90)

        # 添加全局颜色条
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

        # 设置颜色条
        cbar_ticks = np.linspace(vmin, vmax, 6)
        cbar_ticklabels = [f'{tick:.3f}' for tick in cbar_ticks]
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticklabels)

        if error_type == 'absolute':
            cbar_label = 'Median Absolute Error'
        else:
            cbar_label = 'Median Relative Error'

        cbar.set_label(cbar_label, fontsize=9, labelpad=10)
        cbar.ax.tick_params(labelsize=8)

        # 添加总标题
        title = f'{error_type.title()} Error Simplified Contour Plots'
        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)

        # 调整布局
        plt.tight_layout(rect=[0.02, 0.04, 0.90, 0.95])

        # 保存图像
        output_path = self.output_dir / f"simplified_contour_{error_type}.png"
        plt.savefig(output_path, dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        self.logger.info(f"简化Contour图已保存: {output_path}")

    def generate_all_figures(self):
        """生成所有contour图表"""
        try:
            # 生成绝对误差的contour图
            self.logger.info("生成绝对误差contour图...")
            self.generate_contour_figures('absolute')

            # 生成相对误差的contour图
            self.logger.info("生成相对误差contour图...")
            self.generate_contour_figures('relative')

            self.logger.info("所有contour图表生成完成!")

        except Exception as e:
            self.logger.error(f"生成contour图表失败: {e}")


def main():
    """主函数"""
    logger = setup_logger('PaperFiguresGeneratorV2', level='INFO')

    try:
        generator = PaperFiguresGeneratorV2(logger=logger)

        if generator.all_data.empty:
            logger.error("没有加载到数据，请检查数据文件")
            return

        generator.generate_all_figures()

        logger.info(f"图表已保存到: {generator.output_dir}")

    except Exception as e:
        logger.error(f"程序失败: {e}")


if __name__ == "__main__":
    main()