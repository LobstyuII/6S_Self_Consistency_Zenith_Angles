# ==================== external_data_processor.py ====================
"""
外部验证数据处理器 - 完全重构版本
目标：将外部数据转换为与模型训练一致的特征格式
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
import warnings
from datetime import datetime
from dataclasses import dataclass

from config import ExperimentConfig
from utils import setup_logger
from enhanced_feature_engineering import EnhancedFeatureEngineering

warnings.filterwarnings('ignore')


@dataclass
class FeatureMapping:
    """特征映射配置"""
    # 原始数据列 -> 训练特征列
    sza: str = 'SOZ'  # 太阳天顶角
    vza: str = 'SAZ'  # 观测天顶角
    saa: str = 'SAA'  # 太阳方位角
    vaa: str = 'SOA'  # 观测方位角
    aod550: str = 'AOD550'  # 气溶胶光学厚度
    h2o: str = 'water'  # 水汽含量
    o3: str = 'ozone'  # 臭氧含量
    # 反射率列
    toa_band1: str = 'TOA_Albedo_03'
    toa_band2: str = 'TOA_Albedo_04'
    # 其他
    station: str = 'station'
    datetime: str = 'datetime'
    lat: str = 'lat'
    lon: str = 'lon'


class ExternalDataProcessor:
    """
    外部数据处理器 - 将外部数据转换为模型可用的特征格式
    """

    def __init__(self, config: ExperimentConfig = None, logger=None):
        self.config = config or ExperimentConfig
        self.logger = logger or setup_logger('ExternalDataProcessor')
        self.feature_mapping = FeatureMapping()
        self.feature_engineering = EnhancedFeatureEngineering(
            use_physical_features=True,
            use_interaction_terms=True
        )

        # 波段配置
        self.band_config = {
            '03': {'wavelength': 0.64, 'name': 'band03'},
            '04': {'wavelength': 0.86, 'name': 'band04'}
        }

    def load_external_data(self, data_path: Path) -> pd.DataFrame:
        """
        加载外部验证数据
        """
        self.logger.info(f"加载外部验证数据: {data_path}")

        if data_path.suffix == '.parquet':
            df = pd.read_parquet(data_path)
        elif data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
        elif data_path.suffix == '.nc':
            ds = xr.open_dataset(data_path)
            df = ds.to_dataframe().reset_index()
        else:
            raise ValueError(f"不支持的文件格式: {data_path.suffix}")

        self.logger.info(f"数据形状: {df.shape}")
        self.logger.info(f"数据列: {list(df.columns)}")

        return df

    def preprocess_external_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理外部数据
        """
        df_processed = df.copy()

        # 1. 重命名列以匹配训练特征名
        column_mapping = {
            self.feature_mapping.sza: 'SOZ',
            self.feature_mapping.vza: 'SAZ',
            self.feature_mapping.saa: 'SAA',
            self.feature_mapping.vaa: 'SOA',
            self.feature_mapping.aod550: 'AOD550',
            self.feature_mapping.h2o: 'water',
            self.feature_mapping.o3: 'ozone',
            self.feature_mapping.toa_band1: 'TOA_Albedo_03',
            self.feature_mapping.toa_band2: 'TOA_Albedo_04',
        }

        # 应用反向映射（训练特征名 -> 外部数据列名）
        reverse_mapping = {v: k for k, v in column_mapping.items()}
        df_processed = df_processed.rename(columns=reverse_mapping)

        # 2. 计算相对方位角 (RAA)
        if 'SAA' in df_processed.columns and 'SOA' in df_processed.columns:
            df_processed['RAA'] = abs(df_processed['SOA'] - df_processed['SAA'])
            # 确保RAA在0-180度之间
            df_processed['RAA'] = df_processed['RAA'].apply(lambda x: x if x <= 180 else 360 - x)

        # 3. 数据清洗
        # 移除无效值
        df_processed = df_processed.replace(-9999.0, np.nan)

        # 角度限制
        df_processed = df_processed[
            (df_processed['SOZ'] >= 0) & (df_processed['SOZ'] <= 85) &
            (df_processed['SAZ'] >= 0) & (df_processed['SAZ'] <= 85) &
            (df_processed['RAA'] >= 0) & (df_processed['RAA'] <= 180)
            ].copy()

        # 反射率范围限制 (0-1)
        for band in ['03', '04']:
            toa_col = f'TOA_Albedo_{band}'
            if toa_col in df_processed.columns:
                df_processed[toa_col] = df_processed[toa_col].clip(0, 1)

        self.logger.info(f"预处理后数据形状: {df_processed.shape}")

        return df_processed

    def prepare_training_features(self, df_processed: pd.DataFrame) -> pd.DataFrame:
        """
        准备与训练数据一致的特征
        关键：模拟闭合实验，生成rho_retrieved
        """
        # 创建每个波段的独立数据行
        features_list = []

        for band_idx, band_name in enumerate(['03', '04']):
            band_data = df_processed.copy()

            # 设置波段特定参数
            wavelength = self.band_config[band_name]['wavelength']
            toa_col = f'TOA_Albedo_{band_name}'

            if toa_col not in band_data.columns:
                self.logger.warning(f"列 {toa_col} 不存在，跳过波段 {band_name}")
                continue

            # 创建基础特征DataFrame
            features = pd.DataFrame(index=band_data.index)

            # 1. 基本几何特征 (与训练时完全一致)
            features['sza'] = band_data['SOZ']
            features['vza'] = band_data['SAZ']
            features['raa'] = band_data.get('RAA', abs(band_data['SOA'] - band_data['SAA']))

            # 2. 大气参数
            features['aod550'] = band_data['AOD550'].fillna(0.1)  # 默认值
            features['h2o'] = band_data['water'].fillna(2.0)  # 默认值 g/cm²
            features['o3'] = band_data['ozone'].fillna(0.3)  # 默认值 cm-atm

            # 3. 波长 (固定值)
            features['wavelength'] = wavelength

            # 4. 反射率特征 (关键部分！)
            # 在训练数据中，rho_toa是TOA反射率
            features['rho_toa'] = band_data[toa_col].clip(0, 1)

            # 模拟闭合实验：rho_retrieved = rho_true + error
            # 在外部数据中，我们没有rho_true，所以需要模拟
            # 假设：rho_retrieved = rho_toa * 0.8 + 0.05 (经验公式)
            # 这是关键区别：训练时有rho_true，外部数据没有
            features['rho_retrieved'] = features['rho_toa'] * 0.8 + 0.05
            features['rho_retrieved'] = features['rho_retrieved'].clip(0, 1)

            # 5. 使用增强特征工程生成所有特征
            all_features = self.feature_engineering.create_all_features(features)

            # 6. 添加元数据
            all_features['band'] = band_name
            all_features['original_index'] = band_data.index
            all_features['station'] = band_data.get('station', 'unknown')
            all_features['datetime'] = band_data.get('datetime', pd.NaT)
            all_features['lat'] = band_data.get('lat', np.nan)
            all_features['lon'] = band_data.get('lon', np.nan)

            features_list.append(all_features)

        if not features_list:
            self.logger.error("没有生成任何特征")
            return pd.DataFrame()

        # 合并所有波段
        all_features = pd.concat(features_list, ignore_index=True)

        self.logger.info(f"生成的特征数据形状: {all_features.shape}")
        self.logger.info(f"特征列数: {len(all_features.columns)}")
        self.logger.info(f"前10个特征列: {list(all_features.columns)[:10]}")

        return all_features

    def align_features_with_model(self, features_df: pd.DataFrame,
                                  model_features: List[str]) -> pd.DataFrame:
        """
        将特征与模型期望的特征对齐
        """
        # 找出缺失的特征
        missing_in_features = set(model_features) - set(features_df.columns)
        missing_in_model = set(features_df.columns) - set(model_features)

        self.logger.info(f"特征对齐:")
        self.logger.info(f"  模型期望特征数: {len(model_features)}")
        self.logger.info(f"  数据特征数: {len(features_df.columns)}")
        self.logger.info(f"  数据中缺少的特征: {len(missing_in_features)}")
        self.logger.info(f"  模型中缺少的特征: {len(missing_in_model)}")

        if missing_in_features:
            self.logger.warning(f"填充缺失特征: {list(missing_in_features)[:5]}...")
            for feat in missing_in_features:
                # 根据特征类型填充默认值
                if 'rho' in feat:
                    features_df[feat] = 0.2  # 反射率默认值
                elif 'aod' in feat:
                    features_df[feat] = 0.1  # AOD默认值
                elif 'water' in feat or 'h2o' in feat:
                    features_df[feat] = 2.0  # 水汽默认值
                elif 'o3' in feat or 'ozone' in feat:
                    features_df[feat] = 0.3  # 臭氧默认值
                else:
                    features_df[feat] = 0.0

        # 确保特征顺序与模型一致
        aligned_features = pd.DataFrame(index=features_df.index)

        for feat in model_features:
            if feat in features_df.columns:
                aligned_features[feat] = features_df[feat]
            else:
                self.logger.warning(f"特征 {feat} 仍然缺失，用0填充")
                aligned_features[feat] = 0.0

        # 保留元数据
        for meta_col in ['band', 'original_index', 'station', 'datetime', 'lat', 'lon']:
            if meta_col in features_df.columns:
                aligned_features[meta_col] = features_df[meta_col]

        return aligned_features

    def save_processed_data(self, features_df: pd.DataFrame, output_path: Path):
        """
        保存处理后的数据
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == '.parquet':
            features_df.to_parquet(output_path, index=False)
        else:
            features_df.to_csv(output_path, index=False)

        self.logger.info(f"处理后的数据已保存: {output_path}")

        # 同时保存统计信息
        stats_path = output_path.with_suffix('.stats.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("处理后的特征统计\n")
            f.write("=" * 80 + "\n")
            f.write(f"数据形状: {features_df.shape}\n")
            f.write(
                f"特征列: {len([c for c in features_df.columns if c not in ['band', 'original_index', 'station', 'datetime', 'lat', 'lon']])}\n")
            f.write(f"波段分布: {features_df['band'].value_counts().to_dict()}\n")

            # 数值特征统计
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            numeric_cols = [c for c in numeric_cols if c not in ['original_index']]

            f.write("\n数值特征统计:\n")
            for col in numeric_cols[:10]:  # 只显示前10个
                f.write(f"{col:30s}: mean={features_df[col].mean():.4f}, std={features_df[col].std():.4f}\n")

        self.logger.info(f"统计信息已保存: {stats_path}")