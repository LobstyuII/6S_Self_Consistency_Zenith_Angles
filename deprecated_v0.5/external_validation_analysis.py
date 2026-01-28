# ==================== external_validation_analysis.py（修复特征部分）====================
"""
外部验证分析模块 - 修复特征匹配问题
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

warnings.filterwarnings('ignore')


class ExternalValidationAnalysis:
    """
    外部验证分析类 - 修复版本
    """

    def __init__(self, config: ExperimentConfig = None, logger=None):
        self.config = config or ExperimentConfig
        self.logger = logger or setup_logger('ExternalValidationAnalysis')

        # 存储验证数据
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

        # 特征映射：外部验证数据列 -> 模型特征名
        self.feature_mapping = {
            'sza': 'SOZ',
            'vza': 'SAZ',
            'raa': lambda row: abs(row['SOA'] - row['SAA']),
            'aod550': 'AOD550',
            'h2o': 'water',  # water vapor
            'o3': 'ozone',
            'wavelength': None,  # 需要根据波段设置
            'rho_toa': None,  # 需要根据波段设置（TOA反射率）
            'rho_retrieved': None,  # 我们无法获取这个，用TOA反射率近似
            'cos_sza': lambda row: np.cos(np.radians(row['SOZ'])),
            'sin_sza': lambda row: np.sin(np.radians(row['SOZ'])),
            'cos_vza': lambda row: np.cos(np.radians(row['SAZ'])),
            'sin_vza': lambda row: np.sin(np.radians(row['SAZ'])),
            'cos_raa': lambda row: np.cos(np.radians(abs(row['SOA'] - row['SAA']))),
            'sin_raa': lambda row: np.sin(np.radians(abs(row['SOA'] - row['SAA']))),
        }

    def prepare_features_for_model(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        为模型准备特征 - 修复版本，与模型训练特征匹配
        """
        if data is None:
            data = self.validation_data.copy()

        if data.empty:
            self.logger.warning("数据为空，无法准备特征")
            return pd.DataFrame()

        # 创建每个波段的数据副本
        features_list = []

        for band_idx, band_name in enumerate(['03', '04']):
            band_data = data.copy()

            # 设置波长
            if band_name == '03':
                wavelength = 0.64
            else:
                wavelength = 0.86

            # TOA反射率列名
            toa_col = f'TOA_Albedo_{band_name}'

            if toa_col not in band_data.columns:
                self.logger.warning(f"列 {toa_col} 不存在")
                continue

            # 准备特征
            features = pd.DataFrame(index=band_data.index)

            # 1. 基本几何特征
            features['sza'] = band_data['SOZ']
            features['vza'] = band_data['SAZ']
            features['raa'] = abs(band_data['SOA'] - band_data['SAA'])

            # 2. 大气参数
            features['aod550'] = band_data['AOD550']
            features['h2o'] = band_data['water']
            features['o3'] = band_data['ozone']

            # 3. 波长（固定值）
            features['wavelength'] = wavelength

            # 4. 反射率特征
            features['rho_toa'] = band_data[toa_col]
            features['rho_retrieved'] = band_data[toa_col]  # 用TOA反射率近似

            # 5. 三角函数特征
            features['cos_sza'] = np.cos(np.radians(features['sza']))
            features['sin_sza'] = np.sin(np.radians(features['sza']))
            features['cos_vza'] = np.cos(np.radians(features['vza']))
            features['sin_vza'] = np.sin(np.radians(features['vza']))
            features['cos_raa'] = np.cos(np.radians(features['raa']))
            features['sin_raa'] = np.sin(np.radians(features['raa']))

            # 6. 散射角
            cos_scat = -features['cos_sza'] * features['cos_vza'] + \
                       np.sin(np.radians(features['sza'])) * \
                       np.sin(np.radians(features['vza'])) * \
                       np.cos(np.radians(features['raa']))
            cos_scat = np.clip(cos_scat, -1.0, 1.0)
            features['scattering_angle'] = np.degrees(np.arccos(cos_scat))

            # 7. 大气质量
            cos_sza_safe = features['cos_sza'].clip(lower=0.001)
            cos_vza_safe = features['cos_vza'].clip(lower=0.001)
            features['airmass_sza'] = 1.0 / cos_sza_safe
            features['airmass_vza'] = 1.0 / cos_vza_safe
            features['total_airmass'] = features['airmass_sza'] + features['airmass_vza']

            # 8. 派生几何特征
            features['vza_sza_ratio'] = features['vza'] / (features['sza'] + 1e-10)
            features['vza_minus_sza'] = features['vza'] - features['sza']
            features['vza_plus_sza'] = features['vza'] + features['sza']

            # 9. 气溶胶光学厚度与大气质量的相互作用
            features['aod_airmass'] = features['aod550'] * features['total_airmass']
            features['aod_wavelength'] = features['aod550'] / wavelength

            # 10. 反射率比率和差异
            features['rho_ratio'] = features['rho_toa'] / (features['rho_retrieved'] + 1e-10)
            features['rho_diff'] = features['rho_toa'] - features['rho_retrieved']
            features['rho_product'] = features['rho_toa'] * features['rho_retrieved']

            # 11. 波长与几何的相互作用
            features['wavelength_cos_sza'] = wavelength * features['cos_sza']
            features['wavelength_cos_vza'] = wavelength * features['cos_vza']

            # 12. 添加波段标识
            features['band'] = band_name

            # 13. 保留原始数据的索引和关键信息
            features['original_index'] = band_data.index
            features['station'] = band_data['station']
            features['datetime'] = band_data['datetime']
            features['TOA_reflectance'] = band_data[toa_col]

            features_list.append(features)

        if features_list:
            all_features = pd.concat(features_list, ignore_index=True)
            self.logger.info(f"准备的特征数据形状: {all_features.shape}")
            self.logger.info(f"特征列: {list(all_features.columns)}")
            return all_features
        else:
            self.logger.error("无法准备任何特征")
            return pd.DataFrame()

    def apply_model_correction(self, model_path: Path) -> pd.DataFrame:
        """
        应用训练好的模型进行校正 - 修复版本
        """
        try:
            # 加载模型
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            self.logger.info(f"加载模型: {model_path.name}")
            self.logger.info(f"模型类型: {type(model).__name__}")

            # 检查模型是否有feature_names_in_属性
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
                self.logger.info(f"模型期望的特征数: {len(expected_features)}")
                self.logger.info(f"前10个特征: {expected_features[:10]}")
            else:
                self.logger.warning("模型没有feature_names_in_属性")
                # 假设模型训练时的特征顺序
                expected_features = [
                    'sza', 'vza', 'raa', 'aod550', 'h2o', 'o3', 'wavelength',
                    'rho_toa', 'rho_retrieved', 'cos_sza', 'sin_sza', 'cos_vza',
                    'sin_vza', 'cos_raa', 'sin_raa', 'scattering_angle',
                    'airmass_sza', 'airmass_vza', 'total_airmass', 'vza_sza_ratio',
                    'vza_minus_sza', 'vza_plus_sza', 'aod_airmass', 'aod_wavelength',
                    'rho_ratio', 'rho_diff', 'rho_product', 'wavelength_cos_sza',
                    'wavelength_cos_vza'
                ]

            # 准备特征
            features_df = self.prepare_features_for_model()

            if features_df.empty:
                self.logger.error("特征数据为空")
                return pd.DataFrame()

            # 确保特征顺序与模型期望一致
            missing_features = []
            for feat in expected_features:
                if feat not in features_df.columns:
                    missing_features.append(feat)
                    features_df[feat] = 0.0  # 用0填充缺失特征

            if missing_features:
                self.logger.warning(f"填充缺失特征: {missing_features}")

            # 重新排序特征
            features_ordered = features_df[expected_features].copy()

            # 预测校正量
            self.logger.info("开始预测校正量...")

            try:
                corrections = model.predict(features_ordered)
                self.logger.info(f"预测完成，得到 {len(corrections)} 个校正值")

                # 将校正量添加回特征数据框
                features_df['predicted_correction'] = corrections

                # 分离波段3和波段4的结果
                self.correction_results = self.validation_data.copy()
                self.correction_results['correction_03'] = np.nan
                self.correction_results['correction_04'] = np.nan
                self.correction_results['corrected_03'] = np.nan
                self.correction_results['corrected_04'] = np.nan

                # 将校正结果映射回原始数据
                for band_name in ['03', '04']:
                    band_mask = features_df['band'] == band_name
                    band_features = features_df[band_mask]

                    if not band_features.empty:
                        # 获取原始索引
                        original_indices = band_features['original_index'].values
                        corrections_band = band_features['predicted_correction'].values

                        # 映射回原始数据
                        for idx, corr in zip(original_indices, corrections_band):
                            if idx < len(self.correction_results):
                                toa_col = f'TOA_Albedo_{band_name}'
                                if toa_col in self.correction_results.columns:
                                    self.correction_results.at[idx, f'correction_{band_name}'] = corr
                                    self.correction_results.at[idx, f'corrected_{band_name}'] = (
                                            self.correction_results.at[idx, toa_col] - corr
                                    )

                self.logger.info("校正应用完成")
                self.logger.info(f"校正结果统计 - 波段3: {self.correction_results['correction_03'].describe()}")
                self.logger.info(f"校正结果统计 - 波段4: {self.correction_results['correction_04'].describe()}")

                return self.correction_results

            except Exception as e:
                self.logger.error(f"预测失败: {str(e)}")
                import traceback
                traceback.print_exc()
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"应用模型校正失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def apply_lut_correction_simple(self, lut_path: Path) -> pd.DataFrame:
        """
        简化的LUT校正方法
        """
        try:
            self.correction_results = self.validation_data.copy()

            # 初始化结果列
            self.correction_results['correction_03'] = np.nan
            self.correction_results['correction_04'] = np.nan
            self.correction_results['corrected_03'] = np.nan
            self.correction_results['corrected_04'] = np.nan

            # 对于每个样本，计算简单的校正（基于经验公式）
            for idx, row in self.correction_results.iterrows():
                try:
                    # 计算相对方位角
                    raa = abs(row['SOA'] - row['SAA'])

                    # 简化的校正公式（需要根据实际情况调整）
                    # 这里使用一个简单的经验公式作为示例
                    sza_rad = np.radians(row['SOZ'])
                    vza_rad = np.radians(row['SAZ'])
                    raa_rad = np.radians(raa)

                    # 波段3（0.64μm）校正
                    if not np.isnan(row['TOA_Albedo_03']):
                        # 基于角度和气溶胶的简单校正
                        correction_03 = (
                                row['TOA_Albedo_03'] * 0.1 * np.sin(sza_rad) +
                                row['AOD550'] * 0.05 * np.cos(vza_rad)
                        )
                        self.correction_results.at[idx, 'correction_03'] = correction_03
                        self.correction_results.at[idx, 'corrected_03'] = (
                                row['TOA_Albedo_03'] - correction_03
                        )

                    # 波段4（0.86μm）校正
                    if not np.isnan(row['TOA_Albedo_04']):
                        # 波段4的校正可能不同
                        correction_04 = (
                                row['TOA_Albedo_04'] * 0.08 * np.sin(sza_rad) +
                                row['AOD550'] * 0.03 * np.cos(vza_rad)
                        )
                        self.correction_results.at[idx, 'correction_04'] = correction_04
                        self.correction_results.at[idx, 'corrected_04'] = (
                                row['TOA_Albedo_04'] - correction_04
                        )

                except Exception as e:
                    continue

            self.logger.info(f"简单校正完成，处理了 {self.correction_results['correction_03'].notna().sum()} 个样本")
            return self.correction_results

        except Exception as e:
            self.logger.error(f"简单LUT校正失败: {str(e)}")
            return pd.DataFrame()


    def prepare_features_for_model(self) -> pd.DataFrame:
        """
        为模型准备特征（需要根据实际模型调整）
        """
        features = pd.DataFrame()

        # 基本几何特征
        features['sza'] = self.validation_data['SOZ']
        features['vza'] = self.validation_data['SAZ']
        features['raa'] = abs(self.validation_data['SOA'] - self.validation_data['SAA'])

        # 三角函数特征
        features['cos_sza'] = np.cos(np.radians(features['sza']))
        features['cos_vza'] = np.cos(np.radians(features['vza']))
        features['cos_sza_safe'] = features['cos_sza'].clip(lower=0.001)
        features['cos_vza_safe'] = features['cos_vza'].clip(lower=0.001)

        # 大气质量
        features['airmass_sza'] = 1.0 / features['cos_sza_safe']
        features['airmass_vza'] = 1.0 / features['cos_vza_safe']
        features['total_airmass'] = features['airmass_sza'] + features['airmass_vza']

        # 散射角
        cos_scat = -features['cos_sza'] * features['cos_vza'] + \
                   np.sin(np.radians(features['sza'])) * \
                   np.sin(np.radians(features['vza'])) * \
                   np.cos(np.radians(features['raa']))
        cos_scat = cos_scat.clip(-1.0, 1.0)
        features['scattering_angle'] = np.degrees(np.arccos(cos_scat))

        # 大气参数
        features['aod550'] = self.validation_data['AOD550']
        features['ozone'] = self.validation_data['ozone']
        features['water'] = self.validation_data['water']

        # 反射率特征
        features['rho_03'] = self.validation_data['TOA_Albedo_03']
        features['rho_04'] = self.validation_data['TOA_Albedo_04']

        # 时间特征
        if 'datetime' in self.validation_data.columns:
            dt_series = pd.to_datetime(self.validation_data['datetime'])
            features['hour'] = dt_series.dt.hour
            features['doy_sin'] = np.sin(2 * np.pi * dt_series.dt.dayofyear / 365)
            features['doy_cos'] = np.cos(2 * np.pi * dt_series.dt.dayofyear / 365)

        # 地理特征
        if 'lat' in self.validation_data.columns:
            features['lat'] = self.validation_data['lat']
            features['lon'] = self.validation_data['lon']

        return features

    def plot_diurnal_curve_single_station(self,
                                          station_id: str,
                                          target_date: str,
                                          output_dir: Path):
        """
        绘制单个测站的日内曲线图（图1）
        """
        try:
            # 筛选指定测站和日期的数据
            mask = (
                    (self.correction_results['station'] == station_id) &
                    (self.correction_results['datetime'].dt.strftime('%Y-%m-%d') == target_date)
            )

            station_data = self.correction_results[mask].copy()

            if len(station_data) < 3:
                self.logger.warning(f"测站 {station_id} 在 {target_date} 的数据不足")
                return

            # 按时间排序
            station_data = station_data.sort_values('datetime')

            # 创建图形
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                           gridspec_kw={'height_ratios': [3, 1]})

            # 获取测站坐标
            lat = station_data['lat'].iloc[0]
            lon = station_data['lon'].iloc[0]

            # 主图：反射率日内变化
            hours = station_data['datetime'].dt.hour + station_data['datetime'].dt.minute / 60

            # 波段3
            ax1.plot(hours, station_data['TOA_Albedo_03'],
                     'o-', color=self.colors['uncorrected'], alpha=0.7,
                     label='TOA Band 3 (0.64μm)', markersize=8)
            ax1.plot(hours, station_data['corrected_03'],
                     's-', color=self.colors['corrected'], alpha=0.9,
                     label='Corrected Band 3', markersize=8, linewidth=2)

            # 波段4
            ax1.plot(hours, station_data['TOA_Albedo_04'],
                     'o-', color=self.colors['uncorrected'], alpha=0.5,
                     label='TOA Band 4 (0.86μm)', markersize=6)
            ax1.plot(hours, station_data['corrected_04'],
                     's-', color=self.colors['corrected'], alpha=0.7,
                     label='Corrected Band 4', markersize=6, linewidth=2)

            ax1.set_xlabel('Local Hour', fontsize=12)
            ax1.set_ylabel('Reflectance', fontsize=12)
            ax1.set_title(
                f'Diurnal Reflectance Curve - Station {station_id}\n'
                f'Date: {target_date}, Location: ({lat:.2f}°, {lon:.2f}°)',
                fontsize=14, fontweight='bold'
            )
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(6, 18)

            # 子图：误差箱线图
            error_data = []
            hour_labels = []

            for hour in sorted(station_data['hour'].unique()):
                hour_mask = station_data['hour'] == hour
                hour_data = station_data[hour_mask]

                # 计算校正误差（绝对值）
                errors = []
                for band in ['03', '04']:
                    if f'correction_{band}' in hour_data.columns:
                        errors.extend(hour_data[f'correction_{band}'].dropna().abs().tolist())

                if errors:
                    error_data.append(errors)
                    hour_labels.append(f'{hour:02d}:00')

            # 绘制箱线图
            bp = ax2.boxplot(error_data, labels=hour_labels, patch_artist=True)

            # 设置箱线图颜色
            for patch in bp['boxes']:
                patch.set_facecolor(self.colors['error'])
                patch.set_alpha(0.6)

            ax2.set_xlabel('Hour', fontsize=12)
            ax2.set_ylabel('Absolute Correction', fontsize=12)
            ax2.set_title('Hourly Correction Magnitude', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

            # 调整布局
            plt.tight_layout()

            # 保存图形
            output_file = output_dir / f'diurnal_curve_{station_id}_{target_date.replace("-", "")}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"日内曲线图已保存: {output_file}")

        except Exception as e:
            self.logger.error(f"绘制日内曲线图失败: {str(e)}")

    def plot_scatter_distribution(self,
                                  output_dir: Path,
                                  n_days: int = 7):
        """
        绘制散点分布图（图2）
        显示多个自然观测日内，不同SZA情况下的误差分布
        """
        try:
            if self.correction_results.empty:
                self.logger.error("没有校正结果数据")
                return

            # 选择最近n天的数据
            unique_dates = sorted(self.correction_results['datetime'].dt.date.unique())
            if len(unique_dates) > n_days:
                selected_dates = unique_dates[-n_days:]
            else:
                selected_dates = unique_dates

            mask = self.correction_results['datetime'].dt.date.isin(selected_dates)
            plot_data = self.correction_results[mask].copy()

            if len(plot_data) < 10:
                self.logger.warning("数据量不足")
                return

            # 创建图形
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            axes = axes.flatten()

            # SZA分箱
            sza_bins = [0, 30, 45, 60, 85]
            sza_labels = ['0-30°', '30-45°', '45-60°', '>60°']

            # 波段3和波段4分开处理
            for band_idx, band in enumerate(['03', '04']):
                ax_scatter = axes[band_idx * 2]
                ax_vza = axes[band_idx * 2 + 1]

                # 准备数据
                band_data = plot_data.copy()
                band_data = band_data.dropna(subset=[f'correction_{band}', 'SOZ', 'SAZ'])

                # 计算每个测站的平均误差
                station_stats = band_data.groupby('station').agg({
                    f'correction_{band}': 'mean',
                    'SAZ': 'mean',
                    'SOZ': 'mean',
                    'lat': 'first',
                    'lon': 'first'
                }).reset_index()

                station_stats = station_stats.rename(columns={
                    f'correction_{band}': 'mean_correction',
                    'SAZ': 'mean_VZA',
                    'SOZ': 'mean_SZA'
                })

                # 散点图：平均SZA vs 平均误差，用VZA大小表示
                scatter = ax_scatter.scatter(
                    station_stats['mean_SZA'],
                    station_stats['mean_correction'],
                    s=station_stats['mean_VZA'] * 10,  # VZA越大，点越大
                    c=station_stats['mean_correction'],
                    cmap='RdYlBu_r',
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=0.5
                )

                ax_scatter.set_xlabel('Mean Solar Zenith Angle (°)', fontsize=12)
                ax_scatter.set_ylabel('Mean Correction', fontsize=12)
                ax_scatter.set_title(
                    f'Band {band} (0.{"64" if band == "03" else "86"}μm): Station-wise Statistics\n'
                    f'Point size = VZA, Color = Correction magnitude',
                    fontsize=13, fontweight='bold'
                )
                ax_scatter.grid(True, alpha=0.3)

                # 添加颜色条
                cbar = plt.colorbar(scatter, ax=ax_scatter)
                cbar.set_label('Correction Magnitude', fontsize=10)

                # VZA分布图
                for sza_range, label in zip(zip(sza_bins[:-1], sza_bins[1:]), sza_labels):
                    # 筛选SZA范围内的数据
                    sza_mask = (
                            (station_stats['mean_SZA'] >= sza_range[0]) &
                            (station_stats['mean_SZA'] < sza_range[1])
                    )
                    vza_values = station_stats.loc[sza_mask, 'mean_VZA'].dropna()

                    if len(vza_values) > 0:
                        # 创建VZA圆圈
                        for vza in vza_values:
                            circle = Circle(
                                (sza_labels.index(label), 0),
                                radius=vza / 100,  # 缩小比例以便显示
                                alpha=0.3,
                                color=cm.RdYlBu_r(
                                    (vza - vza_values.min()) /
                                    (vza_values.max() - vza_values.min() + 1e-10)
                                )
                            )
                            ax_vza.add_patch(circle)

                ax_vza.set_xlabel('SZA Range', fontsize=12)
                ax_vza.set_ylabel('VZA Distribution', fontsize=12)
                ax_vza.set_title(
                    f'Band {band}: VZA Distribution by SZA Range\n'
                    f'Circle radius = VZA magnitude',
                    fontsize=13, fontweight='bold'
                )
                ax_vza.set_xticks(range(len(sza_labels)))
                ax_vza.set_xticklabels(sza_labels)
                ax_vza.set_ylim(-0.5, 0.5)
                ax_vza.grid(True, alpha=0.3)

            plt.tight_layout()

            # 保存图形
            output_file = output_dir / f'scatter_distribution_{len(selected_dates)}days.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"散点分布图已保存: {output_file}")

        except Exception as e:
            self.logger.error(f"绘制散点分布图失败: {str(e)}")

    def plot_influence_mechanisms(self, output_dir: Path):
        """
        绘制影响机制分析图（图3）
        分析不同因素对校正效果的影响
        """
        try:
            if self.correction_results.empty:
                self.logger.error("没有校正结果数据")
                return

            # 准备数据
            plot_data = self.correction_results.copy()

            # 计算绝对校正量
            for band in ['03', '04']:
                if f'correction_{band}' in plot_data.columns:
                    plot_data[f'abs_correction_{band}'] = plot_data[f'correction_{band}'].abs()

            # 创建图形
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # 1. AOD影响
            if 'AOD550' in plot_data.columns:
                ax = axes[0, 0]

                # 按AOD分箱
                aod_bins = [0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
                aod_labels = ['<0.1', '0.1-0.3', '0.3-0.5', '0.5-1.0', '1.0-2.0', '>2.0']

                plot_data['AOD_bin'] = pd.cut(
                    plot_data['AOD550'],
                    bins=aod_bins,
                    labels=aod_labels,
                    include_lowest=True
                )

                # 按波段统计
                for band_idx, band in enumerate(['03', '04']):
                    band_key = f'abs_correction_{band}'
                    if band_key in plot_data.columns:
                        stats = plot_data.groupby('AOD_bin')[band_key].agg(['mean', 'std']).reset_index()

                        ax.errorbar(
                            range(len(stats)),
                            stats['mean'],
                            yerr=stats['std'],
                            marker='o' if band_idx == 0 else 's',
                            label=f'Band {band}',
                            color=self.colors[f'band{band_idx + 1}'],
                            capsize=5
                        )

                ax.set_xlabel('AOD550 Range', fontsize=12)
                ax.set_ylabel('Absolute Correction', fontsize=12)
                ax.set_title('Effect of Aerosol Optical Depth', fontsize=13, fontweight='bold')
                ax.set_xticks(range(len(aod_labels)))
                ax.set_xticklabels(aod_labels, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)

            # 2. 水汽影响
            if 'water' in plot_data.columns:
                ax = axes[0, 1]

                # 按水汽含量分箱
                water_bins = [0, 1, 2, 3, 4, 5, 10]
                water_labels = ['<1', '1-2', '2-3', '3-4', '4-5', '>5']

                plot_data['water_bin'] = pd.cut(
                    plot_data['water'],
                    bins=water_bins,
                    labels=water_labels,
                    include_lowest=True
                )

                # 按波段统计
                for band_idx, band in enumerate(['03', '04']):
                    band_key = f'abs_correction_{band}'
                    if band_key in plot_data.columns:
                        stats = plot_data.groupby('water_bin')[band_key].agg(['mean', 'std']).reset_index()

                        ax.errorbar(
                            range(len(stats)),
                            stats['mean'],
                            yerr=stats['std'],
                            marker='o' if band_idx == 0 else 's',
                            label=f'Band {band}',
                            color=self.colors[f'band{band_idx + 1}'],
                            capsize=5
                        )

                ax.set_xlabel('Water Vapor (g/cm²)', fontsize=12)
                ax.set_ylabel('Absolute Correction', fontsize=12)
                ax.set_title('Effect of Water Vapor', fontsize=13, fontweight='bold')
                ax.set_xticks(range(len(water_labels)))
                ax.set_xticklabels(water_labels, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)

            # 3. 周中/周末影响
            if 'weekday' in plot_data.columns:
                ax = axes[0, 2]

                weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

                # 按波段统计
                for band_idx, band in enumerate(['03', '04']):
                    band_key = f'abs_correction_{band}'
                    if band_key in plot_data.columns:
                        stats = plot_data.groupby('weekday')[band_key].agg(['mean', 'std']).reset_index()

                        ax.errorbar(
                            stats['weekday'],
                            stats['mean'],
                            yerr=stats['std'],
                            marker='o' if band_idx == 0 else 's',
                            label=f'Band {band}',
                            color=self.colors[f'band{band_idx + 1}'],
                            capsize=5
                        )

                ax.set_xlabel('Weekday', fontsize=12)
                ax.set_ylabel('Absolute Correction', fontsize=12)
                ax.set_title('Effect of Weekday', fontsize=13, fontweight='bold')
                ax.set_xticks(range(7))
                ax.set_xticklabels(weekdays)
                ax.legend()
                ax.grid(True, alpha=0.3)

            # 4. 季节影响
            if 'month' in plot_data.columns:
                ax = axes[1, 0]

                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

                # 按波段统计
                for band_idx, band in enumerate(['03', '04']):
                    band_key = f'abs_correction_{band}'
                    if band_key in plot_data.columns:
                        stats = plot_data.groupby('month')[band_key].agg(['mean', 'std']).reset_index()

                        ax.errorbar(
                            stats['month'] - 1,  # 调整为0-based索引
                            stats['mean'],
                            yerr=stats['std'],
                            marker='o' if band_idx == 0 else 's',
                            label=f'Band {band}',
                            color=self.colors[f'band{band_idx + 1}'],
                            capsize=5
                        )

                ax.set_xlabel('Month', fontsize=12)
                ax.set_ylabel('Absolute Correction', fontsize=12)
                ax.set_title('Seasonal Effect', fontsize=13, fontweight='bold')
                ax.set_xticks(range(12))
                ax.set_xticklabels(months, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)

            # 5. 地表覆盖影响
            if 'land_cover' in plot_data.columns:
                ax = axes[1, 1]

                # 地表覆盖类型映射（IGBP分类）
                lc_types = {
                    1: 'Evergreen\nNeedleleaf',
                    2: 'Evergreen\nBroadleaf',
                    3: 'Deciduous\nNeedleleaf',
                    4: 'Deciduous\nBroadleaf',
                    5: 'Mixed\nForest',
                    6: 'Closed\nShrublands',
                    7: 'Open\nShrublands',
                    8: 'Woody\nSavannas',
                    9: 'Savannas',
                    10: 'Grasslands',
                    11: 'Permanent\nWetlands',
                    12: 'Croplands',
                    13: 'Urban',
                    14: 'Crop/Natural\nMosaic',
                    15: 'Snow/Ice',
                    16: 'Barren',
                    17: 'Water'
                }

                # 只显示数据中存在的类型
                present_types = plot_data['land_cover'].unique()
                valid_types = [t for t in present_types if t in lc_types]

                # 按波段统计
                for band_idx, band in enumerate(['03', '04']):
                    band_key = f'abs_correction_{band}'
                    if band_key in plot_data.columns:
                        stats = []
                        for lc_type in valid_types:
                            lc_data = plot_data[plot_data['land_cover'] == lc_type]
                            if not lc_data.empty:
                                mean_corr = lc_data[band_key].mean()
                                std_corr = lc_data[band_key].std()
                                stats.append({
                                    'lc_type': lc_type,
                                    'mean': mean_corr,
                                    'std': std_corr
                                })

                        if stats:
                            stats_df = pd.DataFrame(stats)
                            ax.errorbar(
                                range(len(stats_df)),
                                stats_df['mean'],
                                yerr=stats_df['std'],
                                marker='o' if band_idx == 0 else 's',
                                label=f'Band {band}',
                                color=self.colors[f'band{band_idx + 1}'],
                                capsize=5
                            )

                ax.set_xlabel('Land Cover Type', fontsize=12)
                ax.set_ylabel('Absolute Correction', fontsize=12)
                ax.set_title('Effect of Land Cover', fontsize=13, fontweight='bold')
                ax.set_xticks(range(len(valid_types)))
                ax.set_xticklabels([lc_types[t] for t in valid_types], rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)

            # 6. 时间（小时）影响
            if 'hour' in plot_data.columns:
                ax = axes[1, 2]

                # 按波段统计
                for band_idx, band in enumerate(['03', '04']):
                    band_key = f'abs_correction_{band}'
                    if band_key in plot_data.columns:
                        stats = plot_data.groupby('hour')[band_key].agg(['mean', 'std']).reset_index()

                        ax.errorbar(
                            stats['hour'],
                            stats['mean'],
                            yerr=stats['std'],
                            marker='o' if band_idx == 0 else 's',
                            label=f'Band {band}',
                            color=self.colors[f'band{band_idx + 1}'],
                            capsize=5
                        )

                ax.set_xlabel('Hour of Day', fontsize=12)
                ax.set_ylabel('Absolute Correction', fontsize=12)
                ax.set_title('Diurnal Variation', fontsize=13, fontweight='bold')
                ax.set_xlim(6, 18)
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.suptitle('Analysis of Main Influence Mechanisms on Correction Performance',
                         fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()

            # 保存图形
            output_file = output_dir / 'influence_mechanisms.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"影响机制分析图已保存: {output_file}")

        except Exception as e:
            self.logger.error(f"绘制影响机制分析图失败: {str(e)}")

    def run_complete_analysis(self,
                              data_path: Path,
                              model_path: Optional[Path] = None,
                              lut_path: Optional[Path] = None,
                              output_dir: Optional[Path] = None):
        """
        运行完整的分析流程
        """
        try:
            # 创建输出目录
            if output_dir is None:
                output_dir = self.config.RESULTS_DIR / "external_validation"
            output_dir.mkdir(parents=True, exist_ok=True)

            # 1. 加载数据
            if not self.load_validation_data(data_path):
                return False

            # 2. 应用校正
            if model_path and model_path.exists():
                self.logger.info("使用模型进行校正...")
                self.apply_model_correction(model_path)
            elif lut_path and lut_path.exists():
                self.logger.info("使用LUT进行校正...")
                self.apply_lut_correction(lut_path)
            else:
                self.logger.error("没有提供可用的模型或LUT路径")
                return False

            # 3. 生成图表
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            # 图1：日内曲线（选择第一个测站和第一个日期）
            if not self.correction_results.empty:
                sample_station = self.correction_results['station'].iloc[0]
                sample_date = self.correction_results['datetime'].iloc[0].strftime('%Y-%m-%d')

                self.plot_diurnal_curve_single_station(
                    sample_station, sample_date, plots_dir
                )

                # 图2：散点分布图
                self.plot_scatter_distribution(plots_dir)

                # 图3：影响机制分析
                self.plot_influence_mechanisms(plots_dir)

                # 4. 保存分析结果
                results_path = output_dir / "correction_results.csv"
                self.correction_results.to_csv(results_path, index=False)
                self.logger.info(f"校正结果已保存: {results_path}")

                # 生成分析报告
                self.generate_analysis_report(output_dir)

                return True
            else:
                self.logger.error("校正结果为空")
                return False

        except Exception as e:
            self.logger.error(f"运行完整分析失败: {str(e)}")
            return False

    def generate_analysis_report(self, output_dir: Path):
        """
        生成分析报告
        """
        try:
            report_path = output_dir / "analysis_report.txt"

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("外部验证分析报告\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"数据记录数: {len(self.validation_data)}\n")
                f.write(f"测站数量: {self.validation_data['station'].nunique()}\n")
                f.write(
                    f"时间范围: {self.validation_data['datetime'].min()} 到 {self.validation_data['datetime'].max()}\n\n")

                if not self.correction_results.empty:
                    f.write("校正结果统计:\n")
                    f.write("-" * 40 + "\n")

                    for band in ['03', '04']:
                        corr_key = f'correction_{band}'
                        if corr_key in self.correction_results.columns:
                            corr_data = self.correction_results[corr_key].dropna()
                            if len(corr_data) > 0:
                                f.write(f"波段 {band}:\n")
                                f.write(f"  平均值: {corr_data.mean():.6f}\n")
                                f.write(f"  标准差: {corr_data.std():.6f}\n")
                                f.write(f"  最小值: {corr_data.min():.6f}\n")
                                f.write(f"  最大值: {corr_data.max():.6f}\n")
                                f.write(f"  正校正比例: {(corr_data > 0).sum() / len(corr_data) * 100:.1f}%\n\n")

                f.write("已生成的图表:\n")
                f.write("-" * 40 + "\n")
                f.write("1. 日内曲线图（单个测站）\n")
                f.write("2. 散点分布图（多日SZA-VZA分析）\n")
                f.write("3. 影响机制分析图（AOD、水汽、周中/周末等）\n\n")

                f.write("结论和建议:\n")
                f.write("-" * 40 + "\n")
                f.write("1. 检查校正效果在不同条件下的稳定性\n")
                f.write("2. 分析主要影响因素的贡献程度\n")
                f.write("3. 针对特定条件优化模型参数\n")
                f.write("4. 考虑增加更多特征以提高校正精度\n")

            self.logger.info(f"分析报告已生成: {report_path}")

        except Exception as e:
            self.logger.error(f"生成分析报告失败: {str(e)}")