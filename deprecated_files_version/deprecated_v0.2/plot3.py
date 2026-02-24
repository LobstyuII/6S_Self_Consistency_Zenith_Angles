# ==================== plot3_error_geometric_index.py ====================
"""
生成误差随综合几何指数变化的散点图
横轴：综合几何指数 (secθ_s + secθ_v)
纵轴：绝对误差
三种大气污染条件对比
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path
import xarray as xr
from scipy import stats
import warnings
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

from config import ExperimentConfig
from utils import setup_logger

# 设置专业科研字体（RSE期刊风格）- 与plot2保持一致
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


class GeometricIndexFigureGenerator:
    """几何指数误差图生成器"""

    def __init__(self, data_dir: Optional[Path] = None, logger=None):
        """
        初始化图表生成器

        Args:
            data_dir: 数据目录路径
            logger: 日志记录器
        """
        self.logger = logger or setup_logger('GeometricIndexFigureGenerator')
        self.data_dir = data_dir or ExperimentConfig.DATA_DIR

        # 加载所有波段数据
        self.all_data = self._load_all_bands_data()

        # 确保输出目录存在
        self.output_dir = ExperimentConfig.MANU_FIGURES_DIR / "plot3"
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

                    # 计算综合几何指数
                    if 'sza' in df.columns and 'vza' in df.columns:
                        df['sec_sza'] = 1.0 / np.cos(np.radians(df['sza']))
                        df['sec_vza'] = 1.0 / np.cos(np.radians(df['vza']))
                        df['total_geometric_index'] = df['sec_sza'] + df['sec_vza']

                    # 计算相对误差
                    if 'error_absolute' in df.columns and 'rho_true' in df.columns:
                        df['error_relative'] = df['error_absolute'] / df['rho_true']

                    all_dfs.append(df)
                    self.logger.info(f"波段 {band_id}: 加载 {len(df)} 条数据")

                except Exception as e:
                    self.logger.error(f"加载波段 {band_id} 数据失败: {e}")
            else:
                self.logger.warning(f"文件不存在: {data_file}")

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            self.logger.info(f"总共加载 {len(combined_df)} 条数据")
            return combined_df
        else:
            return pd.DataFrame()

    def _get_atmospheric_conditions(self) -> Dict:
        """获取大气条件定义（与plot2一致）"""
        return {
            'clean': {
                'name': 'Clean',
                'color': (0.3, 0.5, 0.9),      # 蓝色 - 与plot2的颜色方案一致
                'edgecolor': (0.1, 0.2, 0.7),   # 边缘颜色稍深
                'aod550': 0.1,
                'h2o': 1.0,
                'o3': 0.25,
                'tolerance': {'aod': 0.05, 'h2o': 0.5, 'o3': 0.05}
            },
            'standard': {
                'name': 'Standard',
                'color': (0.5, 0.7, 1.0),      # 中等蓝色
                'edgecolor': (0.3, 0.5, 0.8),   # 边缘颜色稍深
                'aod550': 0.3,
                'h2o': 2.0,
                'o3': 0.3,
                'tolerance': {'aod': 0.05, 'h2o': 0.5, 'o3': 0.05}
            },
            'polluted': {
                'name': 'Polluted',
                'color': (0.0, 0.15, 0.7),     # 深蓝色
                'edgecolor': (0.0, 0.05, 0.5),  # 边缘颜色稍深
                'aod550': 0.5,
                'h2o': 3.0,
                'o3': 0.35,
                'tolerance': {'aod': 0.05, 'h2o': 0.5, 'o3': 0.05}
            }
        }

    def _filter_by_atmospheric_condition(self, df: pd.DataFrame, condition: Dict) -> pd.DataFrame:
        """根据大气条件筛选数据"""
        filtered_df = df.copy()

        aod_tolerance = condition['tolerance']['aod']
        h2o_tolerance = condition['tolerance']['h2o']
        o3_tolerance = condition['tolerance']['o3']

        filtered_df = filtered_df[
            (np.abs(filtered_df['aod550'] - condition['aod550']) < aod_tolerance) &
            (np.abs(filtered_df['h2o'] - condition['h2o']) < h2o_tolerance) &
            (np.abs(filtered_df['o3'] - condition['o3']) < o3_tolerance)
            ]

        return filtered_df

    def _perform_linear_regression(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """执行线性回归分析"""
        if len(x) < 2:
            return {'slope': np.nan, 'intercept': np.nan, 'r_squared': np.nan}

        try:
            # 线性回归
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # 计算R²
            r_squared = r_value ** 2

            # 计算预测值和残差
            y_pred = slope * x + intercept
            residuals = y - y_pred

            # 计算置信区间
            n = len(x)
            if n > 2:
                std_residuals = np.std(residuals)
                t_critical = stats.t.ppf(0.975, n - 2)  # 95%置信区间
                ci_width = t_critical * std_residuals * np.sqrt(1/n + (x - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
            else:
                ci_width = np.full_like(y_pred, np.nan)

            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared,
                'p_value': p_value,
                'std_err': std_err,
                'y_pred': y_pred,
                'residuals': residuals,
                'ci_upper': y_pred + ci_width,
                'ci_lower': y_pred - ci_width
            }
        except Exception as e:
            self.logger.warning(f"线性回归失败: {e}")
            return {'slope': np.nan, 'intercept': np.nan, 'r_squared': np.nan}

    def generate_geometric_index_scatter(self):
        """生成误差随综合几何指数变化的散点图"""
        if self.all_data.empty:
            self.logger.warning("没有数据可用于生成几何指数散点图")
            return

        # 获取大气条件定义
        atmospheric_conditions = self._get_atmospheric_conditions()

        # 波段列表
        bands = list(ExperimentConfig.BANDS.keys())

        # 创建图形 - 2行3列
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # 为每个波段创建子图
        for idx, band_id in enumerate(bands):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # 筛选当前波段的数据
            band_data = self.all_data[self.all_data['band'] == band_id].copy()

            if band_data.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                        fontsize=9, style='italic')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # 检查是否有几何指数数据
            if 'total_geometric_index' not in band_data.columns:
                ax.text(0.5, 0.5, 'No Geometric Index Data', ha='center', va='center',
                        fontsize=9, style='italic')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # 为每种大气条件绘制散点
            regression_results = {}

            for condition_key, condition_params in atmospheric_conditions.items():
                # 筛选当前大气条件的数据
                cond_data = self._filter_by_atmospheric_condition(band_data, condition_params)

                # 检查是否有足够的数据
                if len(cond_data) < 5:
                    self.logger.warning(f"波段 {band_id} 条件 {condition_key} 数据不足 ({len(cond_data)} 条)")
                    continue

                # 提取几何指数和误差数据
                valid_data = cond_data[['total_geometric_index', 'error_absolute']].dropna()

                if len(valid_data) < 5:
                    continue

                x = valid_data['total_geometric_index'].values
                y = valid_data['error_absolute'].values

                # 绘制散点
                scatter = ax.scatter(
                    x, y,
                    alpha=0.5,  # 透明度
                    s=20,       # 点大小
                    c=[condition_params['color']] * len(x),  # 颜色
                    edgecolors=condition_params['edgecolor'],  # 边缘颜色
                    linewidths=0.5,  # 边缘线宽
                    label=condition_params['name'],  # 图例标签
                    zorder=2  # 绘制顺序
                )

                # 执行线性回归
                regression = self._perform_linear_regression(x, y)
                regression_results[condition_key] = regression

                # 绘制回归线（如果有有效结果）
                if not np.isnan(regression['slope']) and not np.isnan(regression['intercept']):
                    # 创建x值范围
                    x_range = np.linspace(x.min(), x.max(), 100)
                    y_pred_range = regression['slope'] * x_range + regression['intercept']

                    # 绘制回归线
                    ax.plot(x_range, y_pred_range,
                            color=condition_params['color'],
                            linewidth=2,
                            linestyle='--',
                            alpha=0.8,
                            zorder=3)

            # 设置坐标轴标签
            ax.set_xlabel('综合几何指数 (secθ_s + secθ_v)', fontsize=9, labelpad=5)
            ax.set_ylabel('绝对误差', fontsize=9, labelpad=5)

            # 设置标题
            wavelength = ExperimentConfig.BANDS[band_id]['wavelength']
            ax.set_title(f'Band {band_id[-1]}\n({wavelength} µm)',
                         fontsize=10, fontweight='bold', pad=10)

            # 添加网格
            ax.grid(True, alpha=0.2, linestyle='--', zorder=1)

            # 设置坐标轴边框样式（与plot2一致）
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

            for spine in ['bottom', 'left']:
                ax.spines[spine].set_linewidth(0.75)
                ax.spines[spine].set_color('black')

            # 设置刻度线样式
            ax.tick_params(axis='both', which='both', length=4, width=0.75,
                           direction='out', colors='black')

            # 设置坐标轴范围
            ax.set_xlim(1.5, 7)  # 综合几何指数的典型范围
            # y轴范围自动调整

            # 添加回归统计信息
            stats_text = ''
            for condition_key, regression in regression_results.items():
                if not np.isnan(regression.get('r_squared', np.nan)):
                    condition_name = atmospheric_conditions[condition_key]['name']
                    r2 = regression['r_squared']
                    slope = regression['slope']
                    stats_text += f'{condition_name[:3]}: R²={r2:.3f}, slope={slope:.4f}\n'

            if stats_text:
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                        fontsize=7, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3))

        # 添加全局图例（放在第一个子图）
        if len(bands) > 0:
            # 获取图例句柄和标签
            handles, labels = axes[0].get_legend_handles_labels()

            # 只保留唯一的图例项
            by_label = dict(zip(labels, handles))

            if by_label:
                # 将图例放在图形底部
                fig.legend(by_label.values(), by_label.keys(),
                           loc='lower center', ncol=3, fontsize=9,
                           frameon=True, fancybox=True, shadow=False,
                           framealpha=0.9, edgecolor='black')

        # 调整子图间距
        plt.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.92,
                            wspace=0.25, hspace=0.3)

        # 添加总标题
        fig.suptitle('误差随综合几何指数变化的散点图（三种大气条件对比）',
                     fontsize=12, fontweight='bold', y=0.98)

        # 添加脚注
        fig.text(0.5, 0.01,
                 '综合几何指数 = sec(SZA) + sec(VZA)，其中sec(θ) = 1/cos(θ)\n'
                 '散点颜色表示不同大气条件：Clean (AOD=0.1), Standard (AOD=0.3), Polluted (AOD=0.5)',
                 ha='center', fontsize=8, style='italic')

        # 保存图像
        output_path = self.output_dir / "geometric_index_error_scatter.png"
        plt.savefig(output_path, dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        self.logger.info(f"几何指数散点图已保存: {output_path}")

        # 生成增强版本（添加统计信息）
        self._generate_enhanced_version(atmospheric_conditions, bands)

    def _generate_enhanced_version(self, atmospheric_conditions: Dict, bands: List):
        """生成增强版本的几何指数散点图（添加更多统计信息）"""
        if self.all_data.empty:
            return

        # 创建更复杂的布局
        fig = plt.figure(figsize=(20, 14))

        # 创建网格布局：6个散点图 + 1个汇总统计图
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

        # 创建散点图（前6个子图）
        axes = []
        for i in range(6):
            if i < len(bands):
                ax = fig.add_subplot(gs[i // 3, i % 3])
                axes.append(ax)
            else:
                break

        # 汇总统计图
        stats_ax = fig.add_subplot(gs[:, 3])

        # 收集所有统计信息
        all_stats = []

        # 为每个波段创建散点图
        for idx, (ax, band_id) in enumerate(zip(axes, bands)):
            # 筛选当前波段的数据
            band_data = self.all_data[self.all_data['band'] == band_id].copy()

            if band_data.empty or 'total_geometric_index' not in band_data.columns:
                continue

            # 为每种大气条件绘制散点
            for condition_key, condition_params in atmospheric_conditions.items():
                # 筛选当前大气条件的数据
                cond_data = self._filter_by_atmospheric_condition(band_data, condition_params)

                if len(cond_data) < 5:
                    continue

                # 提取几何指数和误差数据
                valid_data = cond_data[['total_geometric_index', 'error_absolute']].dropna()

                if len(valid_data) < 5:
                    continue

                x = valid_data['total_geometric_index'].values
                y = valid_data['error_absolute'].values

                # 绘制散点
                ax.scatter(
                    x, y,
                    alpha=0.4,
                    s=15,
                    c=[condition_params['color']] * len(x),
                    edgecolors=condition_params['edgecolor'],
                    linewidths=0.3,
                    label=condition_params['name'] if idx == 0 else None,
                    zorder=2
                )

                # 执行线性回归
                regression = self._perform_linear_regression(x, y)

                # 保存统计信息
                if not np.isnan(regression.get('r_squared', np.nan)):
                    all_stats.append({
                        'band': band_id,
                        'condition': condition_key,
                        'condition_name': condition_params['name'],
                        'slope': regression['slope'],
                        'r_squared': regression['r_squared'],
                        'color': condition_params['color']
                    })

                # 绘制回归线
                if not np.isnan(regression['slope']) and not np.isnan(regression['intercept']):
                    x_range = np.linspace(x.min(), x.max(), 50)
                    y_pred_range = regression['slope'] * x_range + regression['intercept']

                    ax.plot(x_range, y_pred_range,
                            color=condition_params['color'],
                            linewidth=1.5,
                            linestyle='--',
                            alpha=0.7,
                            zorder=3)

            # 设置子图
            wavelength = ExperimentConfig.BANDS[band_id]['wavelength']
            ax.set_title(f'Band {band_id[-1]} ({wavelength} µm)', fontsize=10, fontweight='bold')
            ax.set_xlabel('综合几何指数', fontsize=9)
            ax.set_ylabel('绝对误差', fontsize=9)
            ax.grid(True, alpha=0.2, linestyle='--', zorder=1)

            # 设置坐标轴边框样式
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        # 创建汇总统计图
        if all_stats:
            stats_df = pd.DataFrame(all_stats)

            # 按波段和条件分组
            for condition_key in atmospheric_conditions.keys():
                cond_data = stats_df[stats_df['condition'] == condition_key]

                if len(cond_data) > 0:
                    # 获取颜色
                    color = atmospheric_conditions[condition_key]['color']

                    # 绘制斜率变化
                    bands_order = ['band1', 'band2', 'band3', 'band4', 'band5', 'band6']
                    slopes = []
                    for band in bands_order:
                        band_slope = cond_data[cond_data['band'] == band]['slope']
                        slopes.append(band_slope.values[0] if len(band_slope) > 0 else np.nan)

                    # 绘制条形图
                    x_pos = np.arange(len(bands_order))
                    bars = stats_ax.bar(x_pos + (list(atmospheric_conditions.keys()).index(condition_key) - 1) * 0.25,
                                        slopes, 0.25,
                                        color=color, alpha=0.7,
                                        label=atmospheric_conditions[condition_key]['name'],
                                        edgecolor='black', linewidth=0.5)

            # 设置汇总统计图
            stats_ax.set_xlabel('波段', fontsize=10)
            stats_ax.set_ylabel('回归斜率', fontsize=10)
            stats_ax.set_title('回归斜率随波段变化', fontsize=11, fontweight='bold')
            stats_ax.set_xticks(np.arange(len(bands_order)))
            stats_ax.set_xticklabels([f'B{i+1}' for i in range(len(bands_order))])
            stats_ax.grid(True, alpha=0.2, linestyle='--', axis='y')
            stats_ax.legend(fontsize=9)

            # 添加统计摘要
            summary_text = '统计摘要:\n'
            for condition_key, condition_params in atmospheric_conditions.items():
                cond_data = stats_df[stats_df['condition'] == condition_key]
                if len(cond_data) > 0:
                    avg_slope = cond_data['slope'].mean()
                    avg_r2 = cond_data['r_squared'].mean()
                    summary_text += f"{condition_params['name']}: 平均斜率={avg_slope:.4f}, 平均R²={avg_r2:.3f}\n"

            stats_ax.text(0.02, 0.98, summary_text, transform=stats_ax.transAxes,
                          fontsize=8, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.3))

        # 添加全局图例
        handles, labels = axes[0].get_legend_handles_labels()
        if handles and labels:
            fig.legend(handles, labels, loc='upper center', ncol=3,
                       fontsize=9, frameon=True, fancybox=True, shadow=False,
                       framealpha=0.9, edgecolor='black', bbox_to_anchor=(0.5, 0.02))

        # 添加总标题
        fig.suptitle('误差随综合几何指数变化分析（增强版）',
                     fontsize=14, fontweight='bold', y=0.98)

        # 调整布局
        plt.subplots_adjust(left=0.06, right=0.96, bottom=0.1, top=0.94)

        # 保存图像
        output_path = self.output_dir / "geometric_index_error_enhanced.png"
        plt.savefig(output_path, dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        self.logger.info(f"增强版几何指数散点图已保存: {output_path}")

    def generate_all_figures(self):
        """生成所有图表"""
        try:
            self.logger.info("开始生成几何指数误差图...")

            # 生成主要散点图
            self.generate_geometric_index_scatter()

            self.logger.info("几何指数误差图生成完成!")

        except Exception as e:
            self.logger.error(f"生成图表失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())


def main():
    """主函数"""
    logger = setup_logger('GeometricIndexFigureGenerator', level='INFO')

    try:
        generator = GeometricIndexFigureGenerator(logger=logger)

        if generator.all_data.empty:
            logger.error("没有加载到数据，请检查数据文件")
            return

        generator.generate_all_figures()

        logger.info(f"图表已保存到: {generator.output_dir}")

    except Exception as e:
        logger.error(f"程序失败: {e}")


if __name__ == "__main__":
    main()