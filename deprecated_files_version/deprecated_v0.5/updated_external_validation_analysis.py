# ==================== updated_external_validation_analysis.py ====================
"""
更新后的外部验证分析模块
使用特征对齐进行校正
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.patches import Circle

from config import ExperimentConfig
from utils import setup_logger
from external_feature_alignment import ExternalFeatureAlignment

warnings.filterwarnings('ignore')


class UpdatedExternalValidationAnalysis:
    """
    更新后的外部验证分析类
    使用特征对齐进行校正
    """

    def __init__(self, config: ExperimentConfig = None, logger=None):
        self.config = config or ExperimentConfig
        self.logger = logger or setup_logger('UpdatedExternalValidationAnalysis')

        # 特征对齐器
        self.feature_aligner = ExternalFeatureAlignment(config, logger)

        # 存储数据
        self.validation_data = pd.DataFrame()
        self.correction_results = pd.DataFrame()

        # 颜色配置
        self.colors = {
            'band1': '#1f77b4',
            'band2': '#ff7f0e',
            'corrected': '#2ca02c',
            'uncorrected': '#d62728',
            'error': '#9467bd'
        }

    def load_validation_data(self, data_path: Path) -> bool:
        """
        加载验证数据
        """
        try:
            if data_path.suffix == '.parquet':
                self.validation_data = pd.read_parquet(data_path)
            elif data_path.suffix == '.csv':
                self.validation_data = pd.read_csv(data_path)
            else:
                # 尝试其他格式
                try:
                    self.validation_data = pd.read_parquet(data_path)
                except:
                    self.validation_data = pd.read_csv(data_path)

            self.logger.info(f"验证数据加载成功，形状: {self.validation_data.shape}")
            self.logger.info(f"数据列: {list(self.validation_data.columns)}")

            # 确保必要的列存在
            required_cols = ['SOZ', 'SAZ', 'SOA', 'SAA']
            for col in required_cols:
                if col not in self.validation_data.columns:
                    self.logger.warning(f"缺少必需列: {col}")

            # 检查波段数据
            for band in ['03', '04']:
                toa_col = f'TOA_Albedo_{band}'
                if toa_col not in self.validation_data.columns:
                    self.logger.warning(f"缺少波段{band}的TOA反射率: {toa_col}")

            return True

        except Exception as e:
            self.logger.error(f"加载验证数据失败: {str(e)}")
            return False

        # ==================== updated_external_validation_analysis.py 关键修复 ====================
        def apply_model_correction_aligned(self, model_path: Path) -> pd.DataFrame:
            """
            使用特征对齐应用模型校正 - 修复版本
            """
            try:
                if self.validation_data.empty:
                    self.logger.error("验证数据为空")
                    return pd.DataFrame()

                # 使用特征对齐器预测所有波段的校正量
                predictions = self.feature_aligner.predict_correction_for_all_bands(
                    data=self.validation_data,
                    model_path=model_path
                )

                if predictions.empty:
                    self.logger.error("预测失败，没有生成任何校正量")
                    return pd.DataFrame()

                # 将校正量应用到原始数据
                self.correction_results = self.feature_aligner.apply_correction_to_data(
                    original_data=self.validation_data,
                    predictions=predictions
                )

                if self.correction_results.empty:
                    self.logger.error("校正应用失败")
                    return pd.DataFrame()

                self.logger.info("\n校正完成统计:")
                for band in ['03', '04']:
                    corr_col = f'correction_{band}'
                    if corr_col in self.correction_results.columns:
                        corr_data = self.correction_results[corr_col].dropna()
                        if len(corr_data) > 0:
                            stats = {
                                'mean': corr_data.mean(),
                                'std': corr_data.std(),
                                'min': corr_data.min(),
                                'max': corr_data.max(),
                                'pos_ratio': (corr_data > 0).sum() / len(corr_data) * 100
                            }
                            self.logger.info(f"  波段 {band}:")
                            self.logger.info(f"    样本数: {len(corr_data)}")
                            self.logger.info(f"    均值: {stats['mean']:.6f}")
                            self.logger.info(f"    标准差: {stats['std']:.6f}")
                            self.logger.info(f"    范围: [{stats['min']:.6f}, {stats['max']:.6f}]")
                            self.logger.info(f"    正校正比例: {stats['pos_ratio']:.1f}%")

                return self.correction_results

            except Exception as e:
                self.logger.error(f"应用模型校正失败: {str(e)}")
                import traceback
                traceback.print_exc()
                return pd.DataFrame()

    def apply_lut_correction(self, lut_path: Path) -> pd.DataFrame:
        """
        使用LUT进行校正（简化版本）
        注：需要先完成LUT插值功能
        """
        self.logger.warning("LUT校正功能需要完整的LUT插值实现")
        return self.correction_results

    def run_complete_analysis(self,
                              data_path: Path,
                              model_path: Optional[Path] = None,
                              lut_path: Optional[Path] = None,
                              output_dir: Optional[Path] = None) -> bool:
        """
        运行完整的分析流程（使用特征对齐）
        """
        try:
            # 创建输出目录
            if output_dir is None:
                output_dir = self.config.RESULTS_DIR / "external_validation_aligned"
            output_dir.mkdir(parents=True, exist_ok=True)

            # 1. 加载验证数据
            self.logger.info("步骤1: 加载验证数据")
            if not self.load_validation_data(data_path):
                return False

            # 2. 应用校正
            if model_path and model_path.exists():
                self.logger.info("步骤2: 应用模型校正（特征对齐）")
                self.apply_model_correction_aligned(model_path)
            elif lut_path and lut_path.exists():
                self.logger.info("步骤2: 应用LUT校正")
                self.apply_lut_correction(lut_path)
            else:
                self.logger.error("没有提供有效的模型或LUT路径")
                return False

            if self.correction_results.empty:
                self.logger.error("校正结果为空")
                return False

            # 3. 保存结果
            self.logger.info("步骤3: 保存校正结果")

            # 保存详细结果
            detailed_path = output_dir / "detailed_correction_results.parquet"
            self.correction_results.to_parquet(detailed_path, index=False)

            # 保存汇总结果
            summary_path = output_dir / "correction_summary.csv"
            summary_data = []

            for band in ['03', '04']:
                corr_col = f'correction_{band}'
                if corr_col in self.correction_results.columns:
                    corr_data = self.correction_results[corr_col].dropna()
                    if len(corr_data) > 0:
                        summary = {
                            'band': band,
                            'n_samples': len(corr_data),
                            'mean_correction': corr_data.mean(),
                            'std_correction': corr_data.std(),
                            'min_correction': corr_data.min(),
                            'max_correction': corr_data.max(),
                            'positive_ratio': (corr_data > 0).sum() / len(corr_data) * 100
                        }
                        summary_data.append(summary)

            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(summary_path, index=False)
                self.logger.info(f"校正汇总已保存: {summary_path}")

            # 4. 生成可视化（可选）
            self.logger.info("步骤4: 生成可视化")
            self.generate_visualizations(output_dir)

            # 5. 生成报告
            self.logger.info("步骤5: 生成分析报告")
            self.generate_analysis_report(output_dir)

            self.logger.info(f"\n外部验证分析完成！")
            self.logger.info(f"结果保存在: {output_dir}")

            return True

        except Exception as e:
            self.logger.error(f"运行完整分析失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def generate_visualizations(self, output_dir: Path):
        """
        生成可视化图表
        """
        try:
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            if self.correction_results.empty:
                self.logger.warning("没有校正结果，跳过可视化")
                return

            # 1. 校正量分布图
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            for idx, band in enumerate(['03', '04']):
                corr_col = f'correction_{band}'
                if corr_col in self.correction_results.columns:
                    corr_data = self.correction_results[corr_col].dropna()

                    # 直方图
                    axes[idx, 0].hist(corr_data, bins=30, alpha=0.7, color=self.colors[f'band{idx + 1}'])
                    axes[idx, 0].set_title(f'Band {band} - Correction Distribution')
                    axes[idx, 0].set_xlabel('Correction Value')
                    axes[idx, 0].set_ylabel('Frequency')
                    axes[idx, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)

                    # 箱线图
                    axes[idx, 1].boxplot(corr_data, vert=True, patch_artist=True)
                    axes[idx, 1].set_title(f'Band {band} - Correction Statistics')
                    axes[idx, 1].set_ylabel('Correction Value')

            plt.tight_layout()
            plot_path = plots_dir / "correction_distribution.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"生成分布图: {plot_path}")

            # 2. SZA vs 校正量散点图
            if 'SOZ' in self.correction_results.columns:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                for idx, band in enumerate(['03', '04']):
                    corr_col = f'correction_{band}'
                    if corr_col in self.correction_results.columns:
                        mask = self.correction_results[corr_col].notna()
                        x = self.correction_results.loc[mask, 'SOZ']
                        y = self.correction_results.loc[mask, corr_col]

                        scatter = axes[idx].scatter(x, y, alpha=0.6, s=20,
                                                    c=self.correction_results.loc[mask, 'AOD550']
                                                    if 'AOD550' in self.correction_results.columns else 'blue',
                                                    cmap='viridis')
                        axes[idx].set_xlabel('Solar Zenith Angle (°)')
                        axes[idx].set_ylabel(f'Band {band} Correction')
                        axes[idx].set_title(f'Band {band}: SZA vs Correction')
                        axes[idx].grid(True, alpha=0.3)

                        if 'AOD550' in self.correction_results.columns:
                            plt.colorbar(scatter, ax=axes[idx], label='AOD550')

                plt.tight_layout()
                plot_path = plots_dir / "sza_vs_correction.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"生成SZA散点图: {plot_path}")

        except Exception as e:
            self.logger.error(f"生成可视化失败: {str(e)}")

    def generate_analysis_report(self, output_dir: Path):
        """
        生成分析报告
        """
        try:
            report_path = output_dir / "analysis_report.txt"

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("外部验证分析报告（特征对齐版本）\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"数据源: {len(self.validation_data)} 条记录\n")
                f.write(f"测站数量: {self.validation_data['station'].nunique()}\n")

                if not self.validation_data.empty and 'datetime' in self.validation_data.columns:
                    f.write(
                        f"时间范围: {self.validation_data['datetime'].min()} 到 {self.validation_data['datetime'].max()}\n")

                f.write("\n校正结果统计:\n")
                f.write("-" * 40 + "\n")

                for band in ['03', '04']:
                    corr_col = f'correction_{band}'
                    if corr_col in self.correction_results.columns:
                        corr_data = self.correction_results[corr_col].dropna()
                        if len(corr_data) > 0:
                            f.write(f"\n波段 {band}:\n")
                            f.write(f"  样本数: {len(corr_data)}\n")
                            f.write(f"  平均值: {corr_data.mean():.6f}\n")
                            f.write(f"  标准差: {corr_data.std():.6f}\n")
                            f.write(f"  最小值: {corr_data.min():.6f}\n")
                            f.write(f"  最大值: {corr_data.max():.6f}\n")
                            f.write(f"  正校正比例: {(corr_data > 0).sum() / len(corr_data) * 100:.1f}%\n")

                f.write("\n\n分析结论:\n")
                f.write("-" * 40 + "\n")
                f.write("1. 使用特征对齐方法成功将外部数据转换为模型需要的特征格式\n")
                f.write("2. 校正量分布反映了模型在不同条件下的预测行为\n")
                f.write("3. 建议进一步分析校正量与大气参数、几何参数的关系\n")
                f.write("4. 考虑在不同季节和地区进行更全面的验证\n")

            self.logger.info(f"分析报告已生成: {report_path}")

        except Exception as e:
            self.logger.error(f"生成分析报告失败: {str(e)}")