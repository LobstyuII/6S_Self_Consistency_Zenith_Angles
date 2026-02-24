# ==================== data_loader.py ====================
"""
数据加载和特征工程模块
负责加载数据、特征工程、数据准备
目标变量：delta_toa (ρ_TOA^SA - ρ_TOA^PPA)
"""
import numpy as np
import pandas as pd
import xarray as xr
import warnings
from typing import Dict, Tuple, List, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression

from config import ExperimentConfig
from utils import setup_logger

warnings.filterwarnings('ignore')


class AdvancedFeatureEngineering:
    """高级特征工程类 - 包含完整的特征创建方法"""

    def __init__(self, use_interactions=True, use_trigonometric=True,
                 use_derived=True, use_ratios=True):
        self.use_interactions = use_interactions
        self.use_trigonometric = use_trigonometric
        self.use_derived = use_derived
        self.use_ratios = use_ratios

        # 特征映射字典
        self.feature_descriptions = {
            'geometry': ['sza', 'vza', 'raa'],
            'atmosphere': ['aod550', 'h2o', 'o3'],
            'spectral': ['wavelength'],
            'reflectance': ['rho_toa', 'rho_retrieved'],
            'derived': []  # 将在工程过程中填充
        }

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建所有特征
        参数:
            df: 原始数据框，应包含至少 sza, vza, raa, aod550, wavelength, rho_toa 等列
        返回:
            特征数据框
        """
        features = pd.DataFrame(index=df.index)

        # 1. 基础特征（几何、大气、光谱）
        features['sza'] = df['sza']
        features['vza'] = df['vza']
        features['raa'] = df['raa']
        features['aod550'] = df['aod550']
        features['h2o'] = df.get('h2o', 2.0)
        features['o3'] = df.get('o3', 0.3)
        features['wavelength'] = df['wavelength']

        # 2. 表观反射率 - 优先使用 rho_toa_sa（真实观测），若无则回退到 rho_toa
        if 'rho_toa_sa' in df.columns:
            features['rho_toa'] = df['rho_toa_sa']
        else:
            features['rho_toa'] = df.get('rho_toa', 0.2)

        # 3. 反演反射率（新框架可能不需要，但保留用于兼容旧数据）
        if 'rho_retrieved' in df.columns:
            features['rho_retrieved'] = df['rho_retrieved']
        elif 'rho_true' in df.columns and 'delta_toa' in df.columns:
            # 近似：假设 rho_retrieved = rho_true? 但新框架不依赖此列，设为0
            features['rho_retrieved'] = 0.0
        else:
            features['rho_retrieved'] = 0.0

        # 4. 三角函数特征
        if self.use_trigonometric:
            sza_rad = np.radians(df['sza'])
            vza_rad = np.radians(df['vza'])
            raa_rad = np.radians(df['raa'])

            features['cos_sza'] = np.cos(sza_rad)
            features['sin_sza'] = np.sin(sza_rad)
            features['cos_vza'] = np.cos(vza_rad)
            features['sin_vza'] = np.sin(vza_rad)
            features['cos_raa'] = np.cos(raa_rad)
            features['sin_raa'] = np.sin(raa_rad)

            # 散射角
            cos_scat = -np.cos(sza_rad) * np.cos(vza_rad) + \
                       np.sin(sza_rad) * np.sin(vza_rad) * np.cos(raa_rad)
            cos_scat = np.clip(cos_scat, -1.0, 1.0)
            features['scattering_angle'] = np.degrees(np.arccos(cos_scat))

            self.feature_descriptions['derived'].extend([
                'cos_sza', 'sin_sza', 'cos_vza', 'sin_vza',
                'cos_raa', 'sin_raa', 'scattering_angle'
            ])

        # 5. 大气质量因子
        if self.use_derived:
            # 防止除零
            cos_sza_safe = np.clip(features['cos_sza'] if 'cos_sza' in features else np.cos(np.radians(df['sza'])), 0.001, 1.0)
            cos_vza_safe = np.clip(features['cos_vza'] if 'cos_vza' in features else np.cos(np.radians(df['vza'])), 0.001, 1.0)

            features['airmass_sza'] = 1.0 / cos_sza_safe
            features['airmass_vza'] = 1.0 / cos_vza_safe
            features['total_airmass'] = features['airmass_sza'] + features['airmass_vza']

            self.feature_descriptions['derived'].extend([
                'airmass_sza', 'airmass_vza', 'total_airmass'
            ])

        # 6. 角度关系
        if self.use_ratios:
            features['vza_sza_ratio'] = df['vza'] / (df['sza'] + 1e-6)
            features['vza_minus_sza'] = df['vza'] - df['sza']
            features['vza_plus_sza'] = df['vza'] + df['sza']

            self.feature_descriptions['derived'].extend([
                'vza_sza_ratio', 'vza_minus_sza', 'vza_plus_sza'
            ])

        # 7. 交互特征
        if self.use_interactions:
            # 确保需要的派生特征存在
            if 'total_airmass' not in features:
                cos_sza_tmp = np.clip(np.cos(np.radians(df['sza'])), 0.001, 1.0)
                cos_vza_tmp = np.clip(np.cos(np.radians(df['vza'])), 0.001, 1.0)
                airmass_sza_tmp = 1.0 / cos_sza_tmp
                airmass_vza_tmp = 1.0 / cos_vza_tmp
                total_airmass_tmp = airmass_sza_tmp + airmass_vza_tmp
            else:
                total_airmass_tmp = features['total_airmass']

            features['aod_airmass'] = df['aod550'] * total_airmass_tmp
            features['aod_wavelength'] = df['aod550'] * df['wavelength']

            features['rho_ratio'] = features['rho_retrieved'] / (features['rho_toa'] + 1e-6)
            features['rho_diff'] = features['rho_toa'] - features['rho_retrieved']
            features['rho_product'] = features['rho_toa'] * features['rho_retrieved']

            if 'cos_sza' in features:
                features['wavelength_cos_sza'] = df['wavelength'] * features['cos_sza']
                features['wavelength_cos_vza'] = df['wavelength'] * features['cos_vza']
            else:
                features['wavelength_cos_sza'] = df['wavelength'] * np.cos(np.radians(df['sza']))
                features['wavelength_cos_vza'] = df['wavelength'] * np.cos(np.radians(df['vza']))

            self.feature_descriptions['derived'].extend([
                'aod_airmass', 'aod_wavelength', 'rho_ratio',
                'rho_diff', 'rho_product', 'wavelength_cos_sza',
                'wavelength_cos_vza'
            ])

        return features

    def select_best_features(self, features: pd.DataFrame, target: pd.Series,
                             method: str = 'mutual_info', k: int = 20) -> pd.DataFrame:
        """选择最佳特征"""
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_regression, k=min(k, features.shape[1]))
        elif method == 'f_regression':
            selector = SelectKBest(f_regression, k=min(k, features.shape[1]))
        else:
            return features

        selector.fit(features, target)
        selected_mask = selector.get_support()
        selected_features = features.columns[selected_mask]

        print(f"选择了 {len(selected_features)} 个最佳特征:")
        for i, feat in enumerate(selected_features):
            print(f"  {i+1}. {feat}")

        return features[selected_features]

    def analyze_collinearity(self, features: pd.DataFrame, threshold: float = 0.9):
        """分析特征共线性"""
        corr_matrix = features.corr().abs()

        # 找出高度相关的特征对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    col_i = corr_matrix.columns[i]
                    col_j = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    high_corr_pairs.append((col_i, col_j, corr_value))

        return high_corr_pairs, corr_matrix


class DataLoader:
    """数据加载器类"""

    def __init__(self, config: ExperimentConfig = None, logger=None):
        self.config = config or ExperimentConfig
        self.logger = logger or setup_logger('DataLoader')
        self.feature_engineer = AdvancedFeatureEngineering()

    def load_data(self, sample_fraction: float = 1.0,
                  filter_extreme_delta: bool = True,
                  delta_threshold: float = 100) -> pd.DataFrame:
        """
        加载完整数据集，包含 delta_toa 列
        参数:
            sample_fraction: 采样比例（用于快速测试）
            filter_extreme_delta: 是否过滤异常的 delta_toa 值
            delta_threshold: 异常阈值（绝对值超过该值视为异常，默认10）
        """
        print("=" * 70)
        print("📂 加载数据...")

        data_files = list(self.config.DATA_DIR.glob("training_data_*.nc"))

        if not data_files:
            raise FileNotFoundError(f"在目录 {self.config.DATA_DIR} 中找不到数据文件")

        print(f"找到 {len(data_files)} 个数据文件")

        if sample_fraction < 1.0:
            n_files = max(1, int(len(data_files) * sample_fraction))
            data_files = data_files[:n_files]
            print(f"采样: 使用 {n_files} 个文件")

        all_data = []
        total_samples = 0

        for i, file in enumerate(data_files):
            try:
                print(f"  加载文件 {i+1}/{len(data_files)}: {file.name}")

                ds = xr.open_dataset(file)
                df = ds.to_dataframe().reset_index(drop=True)

                # 添加波段信息
                band_name = file.stem.replace("training_data_", "").replace("_parallel", "")
                df['band'] = band_name

                if band_name in self.config.BANDS:
                    df['wavelength'] = self.config.BANDS[band_name]['wavelength']

                # 过滤成功样本
                if 'success' in df.columns:
                    df = df[df['success'] == 1]
                if 'closed_loop_success' in df.columns:
                    df = df[df['closed_loop_success'] == 1]

                # 确保目标列存在
                if 'delta_toa' not in df.columns:
                    # 尝试从 rho_toa_sa 和 rho_toa_ppa 计算
                    if 'rho_toa_sa' in df.columns and 'rho_toa_ppa' in df.columns:
                        df['delta_toa'] = df['rho_toa_sa'] - df['rho_toa_ppa']
                        self.logger.info(f"文件 {file.name}: 从 rho_toa_sa/ppa 计算 delta_toa")
                    else:
                        self.logger.warning(f"文件 {file.name} 缺少 delta_toa 且无法计算，跳过")
                        ds.close()
                        continue

                all_data.append(df)
                total_samples += len(df)
                ds.close()

            except Exception as e:
                self.logger.warning(f"加载文件 {file} 时出错: {e}")

        if not all_data:
            raise ValueError("没有成功加载任何数据")

        data = pd.concat(all_data, ignore_index=True)

        # 将无穷值替换为 NaN
        data = data.replace([np.inf, -np.inf], np.nan)

        # 过滤目标变量中的 NaN
        initial_count = len(data)
        data = data.dropna(subset=['delta_toa'])
        if initial_count > len(data):
            self.logger.warning(f"删除了 {initial_count - len(data)} 个包含 NaN 的目标变量样本")

        # 可选：过滤超出物理范围的 delta_toa
        if filter_extreme_delta:
            extreme_mask = np.abs(data['delta_toa']) > delta_threshold
            n_extreme = extreme_mask.sum()
            if n_extreme > 0:
                data = data[~extreme_mask]
                self.logger.warning(f"删除了 {n_extreme} 个 |ΔTOA| > {delta_threshold} 的异常样本")

        # 重置索引，确保索引连续
        data = data.reset_index(drop=True)

        # 打印 delta_toa 统计
        self.logger.info(f"delta_toa 统计: 均值={data['delta_toa'].mean():.6f}, "
                         f"标准差={data['delta_toa'].std():.6f}, "
                         f"范围=[{data['delta_toa'].min():.6f}, {data['delta_toa'].max():.6f}]")

        print(f"✅ 数据加载完成")
        print(f"   总样本数: {len(data):,}")
        print(f"   特征数: {data.shape[1]}")
        print(f"   内存使用: {data.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")

        return data

    def prepare_data(self, data: pd.DataFrame, test_size: float = 0.2,
                     use_feature_engineering: bool = True,
                     target_column: str = 'delta_toa') -> Dict:
        """
        准备训练和测试数据
        参数:
            data: 输入DataFrame
            test_size: 测试集比例
            use_feature_engineering: 是否使用特征工程
            target_column: 目标变量列名
        返回:
            包含 X_train, X_test, y_train, y_test, X_raw, feature_names, scaler 的字典
        """
        print("=" * 70)
        print("🔧 准备数据...")

        # 目标变量
        if target_column not in data.columns:
            raise ValueError(f"目标列 '{target_column}' 不存在于数据中")
        y = data[target_column]

        print(f"   目标变量统计:")
        print(f"     均值: {y.mean():.6f}")
        print(f"     标准差: {y.std():.6f}")
        print(f"     最小值: {y.min():.6f}")
        print(f"     最大值: {y.max():.6f}")
        print(f"     中位数: {y.median():.6f}")

        # 创建特征
        if use_feature_engineering:
            X = self.feature_engineer.create_features(data)
            print(f"   特征工程生成 {X.shape[1]} 个特征")
        else:
            # 基础特征：几何、大气、波长、以及 rho_toa_sa（若无则用 rho_toa）
            base_features = ['sza', 'vza', 'raa', 'aod550', 'h2o', 'o3', 'wavelength']
            X = data[base_features].copy()

            # 添加表观反射率
            if 'rho_toa_sa' in data.columns:
                X['rho_toa'] = data['rho_toa_sa']
            elif 'rho_toa' in data.columns:
                X['rho_toa'] = data['rho_toa']
            else:
                X['rho_toa'] = 0.2  # 默认值
                self.logger.warning("未找到表观反射率列，使用默认值 0.2")

            # 添加反演反射率（新框架可能不需要，但保留）
            if 'rho_retrieved' in data.columns:
                X['rho_retrieved'] = data['rho_retrieved']
            else:
                X['rho_retrieved'] = 0.0

            print(f"   使用基础特征: {list(X.columns)}")

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config.RANDOM_SEED
        )

        print(f"   数据集划分:")
        print(f"     训练集: {X_train.shape[0]:,} 样本")
        print(f"     测试集: {X_test.shape[0]:,} 样本")

        # 标准化
        print("   标准化特征...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'X_raw': X,
            'feature_names': list(X.columns),
            'scaler': scaler
        }

    def load_and_prepare(self, sample_fraction: float = 1.0,
                         test_size: float = 0.2,
                         use_feature_engineering: bool = True) -> Dict:
        """加载并准备数据的一站式方法"""
        data = self.load_data(sample_fraction=sample_fraction,
                              filter_extreme_delta=True,  # 默认过滤异常值
                              delta_threshold=100)
        prepared_data = self.prepare_data(
            data, test_size=test_size,
            use_feature_engineering=use_feature_engineering,
            target_column='delta_toa'
        )
        return prepared_data


if __name__ == "__main__":
    # 测试数据加载器
    dl = DataLoader()
    data = dl.load_data(sample_fraction=0.1)
    print(f"数据形状: {data.shape}")