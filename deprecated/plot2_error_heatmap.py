# ==================== boxplot_heatmap_generator_v2.py ====================
"""
箱线图热图生成模块 - 生成3行6列的误差箱线图热图
修改颜色映射：0（浅色）到-0.225（深色）渐变
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
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


class BoxplotHeatmapGeneratorV2:
    """箱线图热图生成器（版本2）"""

    def __init__(self, data_dir: Optional[Path] = None, logger=None):
        """
        初始化图表生成器

        Args:
            data_dir: 数据目录路径
            logger: 日志记录器
        """
        self.logger = logger or setup_logger('BoxplotHeatmapGeneratorV2')
        self.data_dir = data_dir or ExperimentConfig.DATA_DIR

        # 加载所有波段数据
        self.all_data = self._load_all_bands_data()

        # 确保输出目录存在
        self.output_dir = ExperimentConfig.MANU_FIGURES_DIR / "boxplot_heatmaps_v2"
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

    def generate_boxplot_heatmaps(self, error_type: str = 'absolute'):
        """
        生成箱线图热图

        Args:
            error_type: 误差类型 ('absolute' 或 'relative')
        """
        if self.all_data.empty:
            self.logger.warning("没有数据可用于生成箱线图热图")
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

        # 创建图形
        fig, axes = plt.subplots(3, 6, figsize=(24, 12),
                                 gridspec_kw={'hspace': 0.4, 'wspace': 0.3})

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

        # 设置固定的颜色映射范围
        if error_type == 'absolute':
            # 对于绝对误差，使用固定的范围0到-0.225
            vmin, vmax = -0.225, 0
        else:
            # 对于相对误差，使用实际数据的范围
            if all_error_values:
                all_error_values = np.array(all_error_values)
                # 使用百分位数来定义颜色映射范围，排除极端值
                vmax = np.percentile(all_error_values, 95)
                vmin = np.percentile(all_error_values, 5)
            else:
                vmin, vmax = -0.225, 0

        # 创建颜色映射 - 从0（浅色）到-0.225（深色）
        # 使用从浅蓝到深蓝的渐变
        colors = [
            (0.95, 0.98, 1.0),  # 非常浅的蓝色（接近白色）
            (0.85, 0.92, 1.0),  # 浅蓝色
            (0.7, 0.85, 1.0),  # 浅蓝色
            (0.5, 0.7, 1.0),  # 中等蓝色
            (0.3, 0.5, 0.9),  # 蓝色
            (0.1, 0.3, 0.8),  # 深蓝色
            (0.0, 0.15, 0.7),  # 非常深的蓝色
        ]
        cmap = LinearSegmentedColormap.from_list('error_cmap_v2', colors, N=256)

        # 注意：由于我们的误差值是负数，vmin < vmax，而颜色映射从浅色到深色
        # 所以vmax（0）对应浅色，vmin（-0.225）对应深色
        norm = Normalize(vmin=vmin, vmax=vmax)

        # 绘制图表
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

                # 为每个SZA-VZA组合准备数据
                boxplot_data = []
                positions = []
                labels = []
                median_values = []

                pos = 1
                for sza in sza_values:
                    for vza in vza_values:
                        # 筛选该SZA-VZA组合的数据
                        combo_data = band_data[
                            (np.abs(band_data['sza'] - sza) < 1.0) &
                            (np.abs(band_data['vza'] - vza) < 1.0)
                            ]

                        if len(combo_data) > 0:
                            errors = combo_data[error_col].dropna().values
                            if len(errors) > 0:
                                boxplot_data.append(errors)
                                positions.append(pos)
                                median_values.append(np.median(errors))
                                pos += 1

                if not boxplot_data:
                    ax.text(0.5, 0.5, 'No Data',
                            ha='center', va='center', fontsize=9, style='italic')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                # 创建箱线图
                boxplot = ax.boxplot(boxplot_data, positions=positions,
                                     widths=0.6, patch_artist=True,
                                     showfliers=False,  # 不显示异常值
                                     medianprops={'color': 'black', 'linewidth': 1.5},
                                     whiskerprops={'color': 'gray', 'linewidth': 1},
                                     capprops={'color': 'gray', 'linewidth': 1})

                # 为每个箱线图着色（基于中位数）
                for i, box in enumerate(boxplot['boxes']):
                    # 获取中位数
                    median_val = median_values[i]
                    # 归一化到颜色映射
                    # 注意：由于误差值是负数，我们需要确保颜色映射正确
                    color_val = norm(median_val)
                    box.set_facecolor(cmap(color_val))
                    box.set_alpha(0.8)
                    box.set_edgecolor('black')
                    box.set_linewidth(0.5)

                # 设置x轴标签
                # 只显示每个SZA组的第一个位置
                if positions:
                    n_vza = len(vza_values)
                    xtick_positions = [positions[i * n_vza] for i in range(len(sza_values)) if
                                       i * n_vza < len(positions)]
                    ax.set_xticks(xtick_positions)
                    ax.set_xticklabels([f'SZA={sza}°' for sza in sza_values[:len(xtick_positions)]],
                                       fontsize=8, rotation=45)

                # 设置y轴标签
                if error_type == 'absolute':
                    ylabel = 'Absolute Error'
                    ylim = (-0.25, 0.05)  # 设置y轴范围
                else:
                    ylabel = 'Relative Error'
                    # 对于相对误差，使用数据范围
                    if median_values:
                        ylim = (min(median_values) * 1.1, max(median_values) * 1.1)
                    else:
                        ylim = (-0.25, 0.05)

                ax.set_ylabel(ylabel, fontsize=9)
                ax.set_ylim(ylim)

                # 添加零线参考
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

                # 添加网格
                ax.grid(True, alpha=0.2, linestyle='--', axis='y')

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

        # 设置颜色条刻度 - 确保0的位置正确
        if error_type == 'absolute':
            # 对于绝对误差，使用固定的刻度
            cbar_ticks = np.linspace(vmin, vmax, 6)  # -0.225, -0.18, -0.135, -0.09, -0.045, 0
            cbar_ticklabels = [f'{tick:.3f}' for tick in cbar_ticks]
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_ticklabels)
            cbar_label = 'Median Absolute Error'
        else:
            # 对于相对误差，使用动态刻度但确保包含0
            cbar_ticks = np.linspace(vmin, vmax, 6)
            cbar_ticklabels = [f'{tick:.3f}' for tick in cbar_ticks]
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_ticklabels)
            cbar_label = 'Median Relative Error'

        cbar.set_label(cbar_label, fontsize=9, labelpad=10)
        cbar.ax.tick_params(labelsize=8)

        # 添加图例说明
        fig.text(0.5, 0.02,
                 'Each box shows error distribution for specific SZA/VZA combination\n'
                 f'SZA grid: {sza_values}°, VZA grid: {vza_values}°',
                 ha='center', fontsize=9, style='italic')

        # 添加总标题
        title = f'{error_type.title()} Error Distribution Across Bands and Atmospheric Conditions'
        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)

        # 添加子标题（颜色映射说明）
        # fig.text(0.5, 0.94,
        #          f'Color scale: {vmax:.3f} (light) to {vmin:.3f} (dark)',
        #          ha='center', fontsize=10, style='italic')

        # 调整布局
        plt.tight_layout(rect=[0.02, 0.04, 0.90, 0.95])

        # 保存图像
        output_path = self.output_dir / f"boxplot_heatmap_{error_type}_v2_journal.png"
        plt.savefig(output_path, dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        self.logger.info(f"箱线图热图已保存: {output_path}")

        # 同时生成简化的热图版本
        self._generate_simplified_heatmap(error_type, atmospheric_conditions,
                                          bands, sza_values, vza_values, vmin, vmax)

    def _generate_simplified_heatmap(self, error_type: str, atmospheric_conditions: Dict,
                                     bands: List, sza_values: List, vza_values: List,
                                     vmin: float, vmax: float):
        """生成简化的热图（每个单元格显示中位数）"""
        error_col = 'error_absolute' if error_type == 'absolute' else 'error_relative'

        # 创建图形
        fig, axes = plt.subplots(3, 6, figsize=(24, 12),
                                 gridspec_kw={'hspace': 0.4, 'wspace': 0.3})

        # 创建颜色映射 - 与主图一致、

        colors = [
            (0.95, 0.98, 1.0),  # 非常浅的蓝色（接近白色）
            (0.85, 0.92, 1.0),  # 浅蓝色
            (0.7, 0.85, 1.0),  # 浅蓝色
            (0.5, 0.7, 1.0),  # 中等蓝色
            (0.3, 0.5, 0.9),  # 蓝色
            (0.1, 0.3, 0.8),  # 深蓝色
            (0.0, 0.15, 0.7),  # 非常深的蓝色
        ]
        colors = [
            (0.0, 0.15, 0.7),  # 非常深的蓝色
            (0.1, 0.3, 0.8),  # 深蓝色
            (0.3, 0.5, 0.9),  # 蓝色
            (0.5, 0.7, 1.0),  # 中等蓝色
            (0.7, 0.85, 1.0),  # 浅蓝色
            (0.85, 0.92, 1.0),  # 浅蓝色
            (0.95, 0.98, 1.0),  # 非常浅的蓝色（接近白色）
        ]
        cmap = LinearSegmentedColormap.from_list('median_cmap_v2', colors, N=256)
        norm = Normalize(vmin=vmin, vmax=vmax)

        # 绘制简化的热图
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

                # 绘制热图
                im = ax.imshow(median_matrix, cmap=cmap, norm=norm,
                               aspect='auto', origin='lower',
                               extent=[-0.5, len(sza_values) - 0.5, -0.5, len(vza_values) - 0.5])

                # 添加文本显示具体数值
                for i in range(len(vza_values)):
                    for j in range(len(sza_values)):
                        val = median_matrix[i, j]
                        if not np.isnan(val):
                            # 根据背景色调整文本颜色
                            cell_color = cmap(norm(val))
                            # 计算亮度
                            brightness = 0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
                            text_color = 'white' if brightness < 0.6 else 'black'

                            # 格式化数值显示
                            if abs(val) < 0.001:
                                text = f'{val:.2e}'
                            else:
                                text = f'{val:.3f}'

                            ax.text(j, i, text,
                                    ha='center', va='center',
                                    color=text_color, fontsize=7)

                # 设置刻度
                ax.set_xticks(range(len(sza_values)))
                ax.set_yticks(range(len(vza_values)))
                ax.set_xticklabels([f'{sza}°' for sza in sza_values], fontsize=7, rotation=45)
                ax.set_yticklabels([f'{vza}°' for vza in vza_values], fontsize=7)

                # 添加网格
                ax.set_xticks(np.arange(-0.5, len(sza_values), 1), minor=True)
                ax.set_yticks(np.arange(-0.5, len(vza_values), 1), minor=True)
                ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

                # 设置坐标轴标签
                if row_idx == 2:  # 最后一行显示x轴标签
                    ax.set_xlabel('SZA (°)', fontsize=9)

                if col_idx == 0:  # 第一列显示y轴标签
                    ax.set_ylabel('VZA (°)', fontsize=9)

                # 添加波段信息（仅第一行）
                if row_idx == 0:
                    wavelength = ExperimentConfig.BANDS[band]['wavelength']
                    ax.set_title(f'Band {band[-1]}\n({wavelength} µm)',
                                 fontsize=10, fontweight='bold', pad=10)

                # 添加大气条件信息（仅第一列）
                if col_idx == 0:
                    ax.text(-0.35, 0.5, condition_name, transform=ax.transAxes,
                            ha='right', va='center', fontsize=9, fontweight='bold',
                            rotation=90)

        # 添加全局颜色条
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

        # 设置颜色条刻度
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
        title = f'{error_type.title()} Error Median Values Across Bands and Atmospheric Conditions'
        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)

        # 添加子标题
        fig.text(0.5, 0.94,
                 f'Each cell shows median error for SZA/VZA grid | Color: {vmax:.3f} (light) to {vmin:.3f} (dark)',
                 ha='center', fontsize=10, style='italic')

        # 调整布局
        plt.tight_layout(rect=[0.02, 0.04, 0.90, 0.95])

        # 保存图像
        output_path = self.output_dir / f"simplified_heatmap_{error_type}_v2_journal.png"
        plt.savefig(output_path, dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        self.logger.info(f"简化热图已保存: {output_path}")

    def generate_all_figures(self):
        """生成所有图表"""
        try:
            # 生成绝对误差的箱线图热图
            self.logger.info("生成绝对误差箱线图热图...")
            self.generate_boxplot_heatmaps('absolute')

            # 生成相对误差的箱线图热图
            self.logger.info("生成相对误差箱线图热图...")
            self.generate_boxplot_heatmaps('relative')

            self.logger.info("所有图表生成完成!")

        except Exception as e:
            self.logger.error(f"生成图表失败: {e}")


def main():
    """主函数"""
    logger = setup_logger('BoxplotHeatmapGeneratorV2', level='INFO')

    try:
        generator = BoxplotHeatmapGeneratorV2(logger=logger)

        if generator.all_data.empty:
            logger.error("没有加载到数据，请检查数据文件")
            return

        generator.generate_all_figures()

        logger.info(f"图表已保存到: {generator.output_dir}")

    except Exception as e:
        logger.error(f"程序失败: {e}")


if __name__ == "__main__":
    main()