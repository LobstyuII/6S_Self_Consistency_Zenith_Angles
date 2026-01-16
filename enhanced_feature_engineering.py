# ==================== enhanced_feature_engineering.py ====================
"""
增强的特征工程模块
包含散射角、大气质量因子等物理特征
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class EnhancedFeatureEngineering:
    """
    增强的特征工程类
    添加散射角、大气质量因子等物理特征
    """

    def __init__(self, use_physical_features: bool = True,
                 use_interaction_terms: bool = True):
        self.use_physical_features = use_physical_features
        self.use_interaction_terms = use_interaction_terms

        self.feature_descriptions = {
            'geometry': ['sza', 'vza', 'raa'],
            'atmosphere': ['aod550', 'h2o', 'o3'],
            'spectral': ['wavelength'],
            'reflectance': ['rho_toa', 'rho_retrieved'],
            'physical_derived': [],  # 物理派生特征
            'interaction': []  # 交互特征
        }

    def calculate_physical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算物理派生特征
        """
        features = pd.DataFrame(index=df.index)

        # 1. 三角函数特征
        sza_rad = np.radians(df['sza'])
        vza_rad = np.radians(df['vza'])
        raa_rad = np.radians(df['raa'])

        features['cos_sza'] = np.cos(sza_rad)
        features['sin_sza'] = np.sin(sza_rad)
        features['cos_vza'] = np.cos(vza_rad)
        features['sin_vza'] = np.sin(vza_rad)
        features['cos_raa'] = np.cos(raa_rad)
        features['sin_raa'] = np.sin(raa_rad)

        # 防止除零
        cos_sza_safe = np.clip(features['cos_sza'], 0.001, 1.0)
        cos_vza_safe = np.clip(features['cos_vza'], 0.001, 1.0)

        # 2. 大气质量因子
        features['airmass_sza'] = 1.0 / cos_sza_safe
        features['airmass_vza'] = 1.0 / cos_vza_safe
        features['total_airmass'] = features['airmass_sza'] + features['airmass_vza']

        # 3. 散射角 (Scattering Angle)
        cos_scat = -np.cos(sza_rad) * np.cos(vza_rad) + \
                   np.sin(sza_rad) * np.sin(vza_rad) * np.cos(raa_rad)
        cos_scat = np.clip(cos_scat, -1.0, 1.0)
        features['scattering_angle'] = np.degrees(np.arccos(cos_scat))

        # 4. 相对方位角的归一化版本
        features['raa_norm'] = df['raa'] / 180.0

        # 5. 角度组合特征
        features['vza_sza_ratio'] = df['vza'] / (df['sza'] + 1e-6)
        features['vza_minus_sza'] = df['vza'] - df['sza']
        features['vza_plus_sza'] = df['vza'] + df['sza']

        # 6. 极端角度标识
        features['is_extreme_sza'] = (df['sza'] > 60).astype(float)
        features['is_extreme_vza'] = (df['vza'] > 60).astype(float)
        features['is_extreme_geometry'] = ((df['sza'] > 60) | (df['vza'] > 60)).astype(float)

        # 更新特征描述
        self.feature_descriptions['physical_derived'].extend([
            'cos_sza', 'sin_sza', 'cos_vza', 'sin_vza', 'cos_raa', 'sin_raa',
            'airmass_sza', 'airmass_vza', 'total_airmass',
            'scattering_angle', 'raa_norm',
            'vza_sza_ratio', 'vza_minus_sza', 'vza_plus_sza',
            'is_extreme_sza', 'is_extreme_vza', 'is_extreme_geometry'
        ])

        return features

    def calculate_interaction_features(self, df: pd.DataFrame,
                                       physical_features: pd.DataFrame) -> pd.DataFrame:
        """
        计算交互特征
        """
        features = pd.DataFrame(index=df.index)

        # 1. 大气-几何交互
        features['aod_airmass'] = df['aod550'] * physical_features['total_airmass']
        features['h2o_airmass'] = df['h2o'] * physical_features['total_airmass']
        features['o3_airmass'] = df['o3'] * physical_features['total_airmass']

        # 2. 波长相关交互
        features['aod_wavelength'] = df['aod550'] * df['wavelength']
        features['wavelength_airmass'] = df['wavelength'] * physical_features['total_airmass']

        # 3. 反射率相关交互
        if 'rho_toa' in df.columns and 'rho_retrieved' in df.columns:
            features['rho_ratio'] = df['rho_retrieved'] / (df['rho_toa'] + 1e-6)
            features['rho_diff'] = df['rho_toa'] - df['rho_retrieved']
            features['rho_product'] = df['rho_toa'] * df['rho_retrieved']

        # 4. 角度-波长交互
        features['wavelength_cos_sza'] = df['wavelength'] * physical_features['cos_sza']
        features['wavelength_cos_vza'] = df['wavelength'] * physical_features['cos_vza']
        features['wavelength_scattering_angle'] = df['wavelength'] * physical_features['scattering_angle']

        # 5. 气溶胶-散射角交互
        features['aod_scattering_angle'] = df['aod550'] * physical_features['scattering_angle']

        # 更新特征描述
        self.feature_descriptions['interaction'].extend(list(features.columns))

        return features

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建所有特征
        """
        # 基础特征
        features = pd.DataFrame()

        # 几何参数
        features['sza'] = df['sza']
        features['vza'] = df['vza']
        features['raa'] = df['raa']

        # 大气参数
        features['aod550'] = df['aod550']
        features['h2o'] = df.get('h2o', 2.0)
        features['o3'] = df.get('o3', 0.3)

        # 光谱参数
        features['wavelength'] = df['wavelength']

        # 反射率参数
        features['rho_toa'] = df.get('rho_toa', 0.2)

        # 计算反演反射率
        if 'rho_true' in df.columns and 'error_absolute' in df.columns:
            features['rho_retrieved'] = df['rho_true'] + df['error_absolute']
        elif 'rho_retrieved' in df.columns:
            features['rho_retrieved'] = df['rho_retrieved']
        else:
            features['rho_retrieved'] = df.get('rho_true', 0.2)

        # 波段信息
        if 'band' in df.columns:
            features['band'] = df['band']

        # 物理派生特征
        if self.use_physical_features:
            physical_features = self.calculate_physical_features(df)
            features = pd.concat([features, physical_features], axis=1)

        # 交互特征
        if self.use_interaction_terms and self.use_physical_features:
            interaction_features = self.calculate_interaction_features(df, physical_features)
            features = pd.concat([features, interaction_features], axis=1)

        return features

    def get_feature_categories(self) -> Dict[str, List[str]]:
        """获取特征分类"""
        return self.feature_descriptions

    def analyze_feature_importance(self, features: pd.DataFrame, target: pd.Series,
                                   method: str = 'correlation', top_n: int = 20):
        """
        分析特征重要性
        """
        if method == 'correlation':
            # 计算与目标变量的相关系数
            correlations = {}
            for col in features.columns:
                if pd.api.types.is_numeric_dtype(features[col]):
                    corr = np.corrcoef(features[col].fillna(0), target)[0, 1]
                    correlations[col] = abs(corr)

            # 排序
            sorted_corrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

            print(f"Top {top_n} 特征相关性 (绝对值):")
            for i, (feat, corr) in enumerate(sorted_corrs[:top_n]):
                print(f"  {i + 1:2d}. {feat:30s}: {corr:.4f}")

            return dict(sorted_corrs[:top_n])

        else:
            raise ValueError(f"不支持的方法: {method}")
