# ==================== external_feature_alignment.py ====================
"""
外部数据特征对齐模块 - 修复版本
将Himawari真实数据转换为与模型训练时相同的特征格式
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
from datetime import datetime
import pickle

from config import ExperimentConfig
from utils import setup_logger
from enhanced_feature_engineering import EnhancedFeatureEngineering

warnings.filterwarnings('ignore')


class ExternalFeatureAlignment:
    """
    外部数据特征对齐器 - 修复版本
    将真实Himawari数据转换为模型训练时的特征格式
    """

    def __init__(self, config: ExperimentConfig = None, logger=None):
        self.config = config or ExperimentConfig
        self.logger = logger or setup_logger('ExternalFeatureAlignment')

        # 特征映射：真实数据列名 -> 模型特征名
        self.feature_mapping = {
            'SOZ': 'sza',
            'SAZ': 'vza',
            'SAA': 'saa',
            'SOA': 'soa',
            'AOD550': 'aod550',
            'water': 'h2o',
            'ozone': 'o3',
            'TOA_Albedo_03': 'rho_toa_03',
            'TOA_Albedo_04': 'rho_toa_04'
        }

        # 波段配置
        self.band_config = {
            '03': {'wavelength': 0.64, 'name': 'band3'},
            '04': {'wavelength': 0.86, 'name': 'band4'}
        }

        # 特征工程
        self.feature_engineer = EnhancedFeatureEngineering(
            use_physical_features=True,
            use_interaction_terms=True
        )

        # 存储模型特征信息
        self.model_expected_features = None

    def prepare_closed_experiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        准备闭合实验特征（模拟训练数据的生成过程）
        为每个波段生成rho_retrieved（模拟反演反射率）
        """
        # 复制数据避免修改原始数据
        df = data.copy()

        # 计算相对方位角
        df['raa'] = np.abs(df['SOA'] - df['SAA'])

        # 为每个波段准备特征
        all_features = []

        for band_id, band_info in self.band_config.items():
            band_features = pd.DataFrame(index=df.index)

            # 1. 基本几何特征
            band_features['sza'] = df['SOZ']
            band_features['vza'] = df['SAZ']
            band_features['raa'] = df['raa']

            # 2. 大气参数
            band_features['aod550'] = df['AOD550'].fillna(0.1)
            band_features['h2o'] = df['water'].fillna(2.0)
            band_features['o3'] = df['ozone'].fillna(0.3)

            # 3. 波长
            band_features['wavelength'] = band_info['wavelength']

            # 4. 反射率特征（关键部分）
            toa_col = f'TOA_Albedo_{band_id}'

            if toa_col in df.columns:
                # TOA反射率
                band_features['rho_toa'] = df[toa_col].clip(0, 1)

                # 模拟反演反射率：在TOA基础上添加随机噪声（模拟反演误差）
                np.random.seed(self.config.RANDOM_SEED)
                noise_scale = 0.02  # 2%的噪声，模拟反演误差
                rho_toa_values = df[toa_col].values
                noise = np.random.normal(0, noise_scale * np.abs(rho_toa_values), size=len(rho_toa_values))
                band_features['rho_retrieved'] = rho_toa_values + noise
                band_features['rho_retrieved'] = band_features['rho_retrieved'].clip(0, 1)

                # 计算模拟的"真实值"和"误差"（用于训练，但外部验证时我们不知道真实值）
                band_features['rho_true'] = df[toa_col].clip(0, 1)
                band_features['error_absolute'] = band_features['rho_retrieved'] - band_features['rho_true']

                # 添加波段标识
                band_features['band'] = band_id
                band_features['band_name'] = band_info['name']

                # 添加原始索引和关键信息
                band_features['original_index'] = df.index
                band_features['station'] = df['station'] if 'station' in df.columns else 'unknown'
                band_features['datetime'] = df['datetime'] if 'datetime' in df.columns else pd.NaT
                band_features['TOA_reflectance'] = df[toa_col]

                # 计算物理特征
                physical_features = self.feature_engineer.calculate_physical_features(band_features)
                interaction_features = self.feature_engineer.calculate_interaction_features(
                    band_features, physical_features
                )

                # 合并所有特征
                band_all_features = pd.concat([band_features, physical_features, interaction_features], axis=1)
                all_features.append(band_all_features)

        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            self.logger.info(f"准备闭合实验特征完成，形状: {combined_features.shape}")
            return combined_features
        else:
            self.logger.error("未能准备任何闭合实验特征")
            return pd.DataFrame()

    def align_with_model_features(self,
                                  data: pd.DataFrame,
                                  expected_features: Union[List[str], np.ndarray, None] = None) -> pd.DataFrame:
        """
        将特征与模型期望的特征对齐 - 修复版本
        """
        # 使用特征工程生成所有特征
        if 'rho_true' not in data.columns or 'error_absolute' not in data.columns:
            # 如果缺少闭合实验的关键列，先准备闭合实验特征
            data = self.prepare_closed_experiment_features(data)

        if data.empty:
            self.logger.error("数据为空，无法对齐特征")
            return pd.DataFrame()

        # 生成所有特征（使用增强特征工程）
        features_df = self.feature_engineer.create_all_features(data)

        # 添加波段信息
        if 'band' in data.columns:
            features_df['band'] = data['band']

        # 添加原始索引
        if 'original_index' in data.columns:
            features_df['original_index'] = data['original_index']

        # 如果提供了期望的特征列表，确保顺序一致
        if expected_features is not None:
            # 如果是numpy数组，转换为列表
            if isinstance(expected_features, np.ndarray):
                expected_features = expected_features.tolist()

            # 检查缺失的特征
            missing_features = []
            for feat in expected_features:
                if feat not in features_df.columns:
                    missing_features.append(feat)

            if missing_features:
                self.logger.warning(f"特征缺失 {len(missing_features)} 个，将填充默认值")
                self.logger.debug(f"缺失特征: {missing_features[:10]}...")

                # 填充缺失特征
                for feat in missing_features:
                    if feat in ['h2o', 'o3', 'aod550']:
                        features_df[feat] = 0.0
                    elif 'rho' in feat:
                        features_df[feat] = 0.1
                    elif 'cos' in feat or 'sin' in feat:
                        features_df[feat] = 0.0
                    elif 'airmass' in feat:
                        features_df[feat] = 1.0
                    else:
                        features_df[feat] = 0.0

            # 重新排序特征
            available_features = [f for f in expected_features if f in features_df.columns]
            features_df = features_df[available_features]

        self.logger.info(f"特征对齐完成，形状: {features_df.shape}")
        self.logger.info(f"特征列数: {len(features_df.columns)}")

        return features_df

    def load_model_and_get_features(self, model_path: Path) -> Tuple[object, List[str]]:
        """
        加载模型并获取期望的特征列表
        """
        try:
            self.logger.info(f"加载模型: {model_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            # 获取模型期望的特征
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
                self.logger.info(f"模型期望特征数: {len(expected_features)}")
                self.logger.info(f"前10个特征: {expected_features[:10]}")
            else:
                self.logger.warning("模型没有feature_names_in_属性，使用默认特征")
                # 根据模型训练器代码推断特征
                expected_features = [
                    'sza', 'vza', 'raa', 'aod550', 'h2o', 'o3', 'wavelength',
                    'rho_toa', 'rho_retrieved', 'cos_sza', 'sin_sza', 'cos_vza',
                    'sin_vza', 'cos_raa', 'sin_raa', 'scattering_angle',
                    'airmass_sza', 'airmass_vza', 'total_airmass', 'vza_sza_ratio',
                    'vza_minus_sza', 'vza_plus_sza', 'aod_airmass', 'aod_wavelength',
                    'rho_ratio', 'rho_diff', 'rho_product', 'wavelength_cos_sza',
                    'wavelength_cos_vza'
                ]

            self.model_expected_features = expected_features
            return model, expected_features

        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            return None, None

    def predict_model_correction(self,
                                 data: pd.DataFrame,
                                 model_path: Path,
                                 band: str = None) -> pd.DataFrame:
        """
        使用训练好的模型预测校正量 - 修复版本
        """
        try:
            # 加载模型
            model, expected_features = self.load_model_and_get_features(model_path)
            if model is None:
                return pd.DataFrame()

            # 过滤指定波段的数据
            if band and 'band' in data.columns:
                band_mask = data['band'] == band
                band_data = data[band_mask].copy()
                self.logger.info(f"处理波段 {band}，样本数: {len(band_data)}")
            else:
                band_data = data.copy()

            if band_data.empty:
                self.logger.warning(f"波段 {band} 数据为空")
                return pd.DataFrame()

            # 准备特征
            features = self.align_with_model_features(band_data, expected_features)

            if features.empty:
                self.logger.error("特征准备失败")
                return pd.DataFrame()

            # 移除非特征列
            feature_cols = [col for col in features.columns
                            if col not in ['original_index', 'band', 'station', 'datetime', 'TOA_reflectance']]

            # 确保特征顺序
            if expected_features is not None:
                # 只保留模型期望的特征
                available_features = [f for f in expected_features if f in features.columns]
                features_for_pred = features[available_features]

                # 填充缺失特征为0
                missing_features = [f for f in expected_features if f not in features.columns]
                if missing_features:
                    self.logger.warning(f"填充 {len(missing_features)} 个缺失特征为0")
                    for feat in missing_features:
                        features_for_pred[feat] = 0.0

                # 重新排序
                features_for_pred = features_for_pred[expected_features]
            else:
                features_for_pred = features[feature_cols]

            # 预测校正量
            self.logger.info(f"开始预测，输入特征形状: {features_for_pred.shape}")
            predictions = model.predict(features_for_pred)

            # 创建结果DataFrame
            results = pd.DataFrame({
                'original_index': features['original_index'] if 'original_index' in features.columns else range(
                    len(predictions)),
                'band': features['band'] if 'band' in features.columns else band,
                'predicted_correction': predictions
            })

            # 添加其他信息
            if 'station' in features.columns:
                results['station'] = features['station']
            if 'datetime' in features.columns:
                results['datetime'] = features['datetime']
            if 'TOA_reflectance' in features.columns:
                results['TOA_reflectance'] = features['TOA_reflectance']

            self.logger.info(f"预测完成，校正量统计:")
            self.logger.info(f"  均值: {results['predicted_correction'].mean():.6f}")
            self.logger.info(f"  标准差: {results['predicted_correction'].std():.6f}")
            self.logger.info(
                f"  范围: [{results['predicted_correction'].min():.6f}, {results['predicted_correction'].max():.6f}]")

            return results

        except Exception as e:
            self.logger.error(f"预测校正量失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def predict_correction_for_all_bands(self,
                                         data: pd.DataFrame,
                                         model_path: Path) -> pd.DataFrame:
        """
        为所有波段预测校正量
        """
        all_predictions = []

        for band_id in ['03', '04']:
            self.logger.info(f"\n处理波段 {band_id}")

            # 检查该波段是否有数据
            toa_col = f'TOA_Albedo_{band_id}'
            if toa_col not in data.columns:
                self.logger.warning(f"跳过波段 {band_id}，缺少数据列: {toa_col}")
                continue

            # 为当前波段创建数据副本
            band_data = data.copy()
            band_data['target_band'] = band_id

            # 预测校正量
            band_predictions = self.predict_model_correction(
                data=band_data,
                model_path=model_path,
                band=band_id
            )

            if not band_predictions.empty:
                all_predictions.append(band_predictions)

        if all_predictions:
            predictions_df = pd.concat(all_predictions, ignore_index=True)
            self.logger.info(f"\n所有波段预测完成，总样本数: {len(predictions_df)}")
            return predictions_df
        else:
            self.logger.error("所有波段的预测都失败")
            return pd.DataFrame()

    def apply_correction_to_data(self,
                                 original_data: pd.DataFrame,
                                 predictions: pd.DataFrame) -> pd.DataFrame:
        """
        将校正量应用到原始数据
        """
        try:
            corrected_data = original_data.copy()

            # 初始化校正结果列
            for band_id in ['03', '04']:
                corrected_data[f'correction_{band_id}'] = np.nan
                corrected_data[f'corrected_{band_id}'] = np.nan

            # 将校正量映射回原始数据
            for _, pred_row in predictions.iterrows():
                orig_idx = int(pred_row['original_index']) if 'original_index' in pred_row else None
                band_id = pred_row['band'] if 'band' in pred_row else None

                if orig_idx is not None and band_id and orig_idx < len(corrected_data):
                    correction = pred_row['predicted_correction']
                    toa_col = f'TOA_Albedo_{band_id}'

                    if toa_col in corrected_data.columns:
                        corrected_data.at[orig_idx, f'correction_{band_id}'] = correction
                        corrected_data.at[orig_idx, f'corrected_{band_id}'] = (
                                corrected_data.at[orig_idx, toa_col] - correction
                        )

            # 统计校正结果
            self.logger.info("\n校正应用统计:")
            for band_id in ['03', '04']:
                corr_col = f'correction_{band_id}'
                if corr_col in corrected_data.columns:
                    corr_data = corrected_data[corr_col].dropna()
                    if len(corr_data) > 0:
                        pos_ratio = (corr_data > 0).sum() / len(corr_data) * 100
                        self.logger.info(f"  波段 {band_id}: {len(corr_data)}个样本，{pos_ratio:.1f}%为正校正")

            return corrected_data

        except Exception as e:
            self.logger.error(f"应用校正失败: {str(e)}")
            return original_data

    def save_aligned_features(self,
                              original_data: pd.DataFrame,
                              predictions: pd.DataFrame,
                              output_path: Path) -> pd.DataFrame:
        """
        保存对齐后的特征和校正结果
        """
        try:
            # 应用校正到原始数据
            corrected_data = self.apply_correction_to_data(original_data, predictions)

            # 保存结果
            corrected_data.to_parquet(output_path, index=False)
            self.logger.info(f"校正结果已保存: {output_path}")

            # 同时保存CSV版本便于查看
            csv_path = output_path.with_suffix('.csv')
            corrected_data.to_csv(csv_path, index=False)
            self.logger.info(f"CSV版本已保存: {csv_path}")

            # 保存预测详情
            if not predictions.empty:
                pred_path = output_path.parent / "predictions_details.parquet"
                predictions.to_parquet(pred_path, index=False)
                self.logger.info(f"预测详情已保存: {pred_path}")

            return corrected_data

        except Exception as e:
            self.logger.error(f"保存对齐特征失败: {str(e)}")
            return pd.DataFrame()