# ==================== paper_figures.py ====================
"""
论文图表生成模块
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from pathlib import Path
from scipy import stats

from config import ExperimentConfig
from utils import setup_logger
from error_analyzer import ErrorAnalyzer
from deprecated.sensitivity_analyzer import SensitivityAnalyzer
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False


class PaperFiguresGenerator:
    """论文图表生成器"""

    def __init__(self, results: Dict[str, pd.DataFrame], logger=None):
        """
        初始化论文图表生成器

        Args:
            results: 各波段的模拟结果字典
            logger: 日志记录器
        """
        self.results = results
        self.logger = logger or setup_logger('PaperFiguresGenerator')

        # 创建误差分析器
        self.error_analyzer = ErrorAnalyzer(results, logger)

        # 合并所有数据
        self.all_data = self.error_analyzer.all_data

        # 确保Manu_figures目录存在
        ExperimentConfig.MANU_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    def generate_all_figures(self):
        """生成所有论文图表"""
        self.logger.info("开始生成论文图表...")

        try:
            # 1. 生成固定所有参数的contour图（波段变化）
            self.logger.info("生成固定所有参数的contour图（波段变化）...")
            self.generate_contour_figures()

            # 2. 生成单因素变化contour图
            self.logger.info("生成单因素变化contour图...")
            self.generate_single_factor_contour_figures()

            # 3. 生成单因素敏感性分析图
            self.logger.info("生成单因素敏感性分析图...")
            self.generate_sensitivity_figures()

            # 4. 生成误差分布图
            self.logger.info("生成误差分布图...")
            self.generate_error_distribution_figures()

            self.logger.info("所有论文图表生成完成!")

        except Exception as e:
            self.logger.error(f"生成论文图表失败: {e}")
            raise

    def generate_contour_figures(self):
        """生成固定所有参数的contour图（只改变波段）"""
        # 图1: 固定所有参数，所有波段的2×3子母图
        for error_type in ['absolute', 'relative']:
            fig1, _ = self.error_analyzer.create_paper_contour_figure(
                self.all_data, error_type,
                save_path=ExperimentConfig.MANU_FIGURES_DIR /
                          f"contour_fixed_all_{error_type}.png"
            )
            plt.close(fig1)

    def generate_single_factor_contour_figures(self):
        """生成单因素变化contour图"""
        # 图2: band3，固定其他参数，改变AOD的1×3子母图
        for error_type in ['absolute', 'relative']:
            fig2, _ = self.error_analyzer.create_varying_aod_figure(
                self.all_data, error_type,
                save_path=ExperimentConfig.MANU_FIGURES_DIR /
                          f"contour_varying_aod_{error_type}.png"
            )
            plt.close(fig2)

        # 图3: band3，固定其他参数，改变反射率的1×3子母图
        for error_type in ['absolute', 'relative']:
            fig3, _ = self.error_analyzer.create_varying_rho_figure(
                self.all_data, error_type,
                save_path=ExperimentConfig.MANU_FIGURES_DIR /
                          f"contour_varying_rho_{error_type}.png"
            )
            plt.close(fig3)

        # 图4: band3，固定其他参数，改变水汽的1×2子母图
        for error_type in ['absolute', 'relative']:
            fig4, _ = self.error_analyzer.create_varying_h2o_figure(
                self.all_data, error_type,
                save_path=ExperimentConfig.MANU_FIGURES_DIR /
                          f"contour_varying_h2o_{error_type}.png"
            )
            plt.close(fig4)

        # 图5: band3，固定其他参数，改变臭氧的1×2子母图
        for error_type in ['absolute', 'relative']:
            fig5, _ = self.error_analyzer.create_varying_o3_figure(
                self.all_data, error_type,
                save_path=ExperimentConfig.MANU_FIGURES_DIR /
                          f"contour_varying_o3_{error_type}.png"
            )
            plt.close(fig5)

    def generate_sensitivity_figures(self):
        """生成单因素敏感性分析图"""
        # 加载single_factor数据
        single_factor_data = self._load_single_factor_data()

        if single_factor_data.empty:
            self.logger.warning("没有single_factor数据，跳过敏感性分析图")
            return

        # 创建敏感性分析器
        sensitivity_analyzer = SensitivityAnalyzer(single_factor_data, self.logger)

        # 生成绝对误差的敏感性分析图
        self._create_sensitivity_subplot(single_factor_data, 'error_absolute',
                                         'Absolute Error Sensitivity Analysis',
                                         ExperimentConfig.MANU_FIGURES_DIR /
                                         'sensitivity_analysis_absolute.png')

        # 生成相对误差的敏感性分析图
        if 'error_relative' in single_factor_data.columns:
            self._create_sensitivity_subplot(single_factor_data, 'error_relative',
                                             'Relative Error Sensitivity Analysis',
                                             ExperimentConfig.MANU_FIGURES_DIR /
                                             'sensitivity_analysis_relative.png')

    def _load_single_factor_data(self) -> pd.DataFrame:
        """加载single_factor数据"""
        single_factor_data = []

        for band_id, df in self.results.items():
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                # 检查是否是single_factor数据
                # single_factor数据的特点是每个参数变化时其他参数固定
                # 我们可以通过检查每个参数的唯一值数量来判断
                param_cols = ['sza', 'vza', 'aod550', 'h2o', 'o3', 'rho_true']

                # 计算每个参数的非重复值数量
                unique_counts = {}
                for col in param_cols:
                    if col in df.columns:
                        unique_counts[col] = df[col].nunique()

                # 如果大部分参数只有1-2个唯一值，而一个参数有多个唯一值，可能是single_factor数据
                # 这里我们简单判断：如果数据量适中且包含误差数据，就使用
                if 'error_absolute' in df.columns and len(df) > 50:
                    df_band = df.copy()
                    df_band['band'] = band_id
                    single_factor_data.append(df_band)

        if single_factor_data:
            return pd.concat(single_factor_data, ignore_index=True)
        else:
            # 如果没有找到合适的single_factor数据，使用全部数据
            self.logger.info("使用全部数据进行敏感性分析")
            return self.all_data.copy()

    def _create_sensitivity_subplot(self, data: pd.DataFrame, error_col: str,
                                    title: str, save_path: Path):
        """创建敏感性分析子母图"""
        config = ExperimentConfig.PAPER_FIGURES
        layout = config['single_factor_sensitivity']['layout']
        figsize = config['single_factor_sensitivity']['figsize']

        # 要分析的参数
        params_to_analyze = ['sza', 'vza', 'aod550', 'h2o', 'o3', 'rho_true']

        fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize,
                                 constrained_layout=True)

        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        for idx, param in enumerate(params_to_analyze):
            if idx >= len(axes_flat):
                break

            ax = axes_flat[idx]

            if param not in data.columns or error_col not in data.columns:
                ax.text(0.5, 0.5, f'No data for {param}',
                        ha='center', va='center')
                continue

            # 过滤有效数据
            valid_data = data.dropna(subset=[param, error_col])

            if len(valid_data) < 10:
                ax.text(0.5, 0.5, f'Insufficient data\n{param}',
                        ha='center', va='center')
                continue

            # 绘制散点图
            scatter = ax.scatter(valid_data[param], valid_data[error_col],
                                 alpha=0.5, s=10, c='blue', edgecolors='none')

            # 计算线性回归
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    valid_data[param], valid_data[error_col]
                )

                # 绘制回归线
                x_range = np.linspace(valid_data[param].min(), valid_data[param].max(), 100)
                y_pred = slope * x_range + intercept
                ax.plot(x_range, y_pred, 'r-', linewidth=2,
                        label=f'Slope: {slope:.4f}\nR²: {r_value ** 2:.4f}')

                ax.legend(fontsize=8)

            except Exception as e:
                self.logger.warning(f"参数 {param} 线性回归失败: {e}")

            # 设置坐标轴标签
            param_label = {
                'sza': 'Solar Zenith Angle (°)',
                'vza': 'View Zenith Angle (°)',
                'aod550': 'AOD550',
                'h2o': 'Water Vapor (g/cm²)',
                'o3': 'Ozone (cm-atm)',
                'rho_true': 'Surface Reflectance'
            }.get(param, param)

            ax.set_xlabel(param_label, fontsize=9)
            ax.set_ylabel(error_col.replace('_', ' ').title(), fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')

        # 隐藏多余的子图
        for idx in range(len(params_to_analyze), len(axes_flat)):
            axes_flat[idx].axis('off')

        fig.suptitle(title, fontsize=14, y=1.02)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"敏感性分析图已保存: {save_path}")
        plt.close(fig)

    def generate_error_distribution_figures(self):
        """生成误差分布图"""
        # 生成绝对误差分布图
        fig_abs, _ = self.error_analyzer.create_error_distribution_figure(
            self.all_data, 'absolute',
            save_path=ExperimentConfig.MANU_FIGURES_DIR /
                      'error_distribution_absolute.png'
        )
        plt.close(fig_abs)

        # 生成相对误差分布图
        fig_rel, _ = self.error_analyzer.create_error_distribution_figure(
            self.all_data, 'relative',
            save_path=ExperimentConfig.MANU_FIGURES_DIR /
                      'error_distribution_relative.png'
        )
        plt.close(fig_rel)


def generate_paper_figures_from_files():
    """从文件生成论文图表"""
    logger = setup_logger('PaperFiguresFromFiles')

    try:
        # 加载所有波段数据
        results = {}
        for band_id in ExperimentConfig.BANDS.keys():
            data_file = ExperimentConfig.DATA_DIR / f"simulation_results_{band_id}_paper_figures.nc"

            if not data_file.exists():
                # 尝试其他模式的文件
                alt_files = list(ExperimentConfig.DATA_DIR.glob(f"simulation_results_{band_id}_*.nc"))
                if alt_files:
                    data_file = alt_files[0]
                    logger.info(f"波段 {band_id} 使用替代数据文件: {data_file}")
                else:
                    logger.warning(f"找不到波段 {band_id} 的数据文件，跳过")
                    continue

            try:
                from utils import load_dataset
                data_dict = load_dataset(data_file)
                if data_dict:
                    df = pd.DataFrame(data_dict)
                    if len(df) > 0:
                        results[band_id] = df
                        logger.info(f"波段 {band_id} 数据加载成功，形状: {df.shape}")
            except Exception as e:
                logger.error(f"加载波段 {band_id} 数据失败: {e}")

        if not results:
            logger.error("没有加载到任何数据")
            return

        # 创建论文图表生成器
        generator = PaperFiguresGenerator(results, logger)

        # 生成所有图表
        generator.generate_all_figures()

        logger.info("论文图表生成完成!")

    except Exception as e:
        logger.error(f"生成论文图表失败: {e}")
        raise


if __name__ == "__main__":
    generate_paper_figures_from_files()