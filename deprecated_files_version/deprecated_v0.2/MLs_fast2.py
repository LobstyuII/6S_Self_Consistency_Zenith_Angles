# ==================== ML_fast_corrected_v2.py ====================
"""
修正的机器学习模型训练与评估模块（版本2）
解决共线性问题，正确定义特征
"""
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import json
import time
from typing import Dict, List, Tuple
import warnings
import sys

warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet

# 高级模型
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("警告: XGBoost未安装，将跳过XGBoost模型")

try:
    import lightgbm as lgb

    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("警告: LightGBM未安装，将跳过LightGBM模型")

# 可视化

from config import ExperimentConfig
from utils import setup_logger


class AdvancedFeatureEngineer:
    """高级特征工程类，处理共线性和创建有物理意义的特征"""

    @staticmethod
    def create_physically_meaningful_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        创建有物理意义的特征，避免共线性
        返回新的特征DataFrame
        """
        features = pd.DataFrame()

        # 1. 角度特征（原始几何）
        features['sza_rad'] = np.radians(df['sza'])
        features['vza_rad'] = np.radians(df['vza'])
        features['raa_rad'] = np.radians(df['raa'])

        # 2. 三角函数特征（避免直接使用airmass的倒数关系）
        features['cos_sza'] = np.cos(features['sza_rad'])
        features['cos_vza'] = np.cos(features['vza_rad'])
        features['sin_sza'] = np.sin(features['sza_rad'])
        features['sin_vza'] = np.sin(features['vza_rad'])
        features['cos_raa'] = np.cos(features['raa_rad'])

        # 3. 几何相互作用特征
        features['scattering_angle'] = np.degrees(
            np.arccos(-np.cos(features['sza_rad']) * np.cos(features['vza_rad']) +
                      np.sin(features['sza_rad']) * np.sin(features['vza_rad']) * np.cos(features['raa_rad']))
        )

        features['total_secant'] = 1.0 / features['cos_sza'] + 1.0 / features['cos_vza']
        features['geometry_factor'] = features['cos_sza'] * features['cos_vza']

        # 4. 大气质量数的变体（避免与角度完全共线）
        # 使用更真实的球面大气质量近似
        features['airmass_sza_approx'] = AdvancedFeatureEngineer.spherical_airmass_series(df['sza'])
        features['airmass_vza_approx'] = AdvancedFeatureEngineer.spherical_airmass_series(df['vza'])
        features['total_airmass_approx'] = features['airmass_sza_approx'] + features['airmass_vza_approx']

        # 5. 相对几何特征
        features['vza_sza_ratio'] = df['vza'] / (df['sza'] + 1e-6)  # 避免除零
        features['vza_minus_sza'] = df['vza'] - df['sza']
        features['vza_plus_sza'] = df['vza'] + df['sza']

        # 6. 大气参数
        features['aod550'] = df['aod550']
        features['h2o'] = df.get('h2o', 2.0)
        features['o3'] = df.get('o3', 0.3)

        # 7. 波段特征
        features['wavelength'] = df['wavelength']
        features['wavelength_norm'] = (df['wavelength'] - 0.5) / 1.0  # 归一化到约[-0.5, 1.5]

        # 8. 反射率特征 - 关键修正！
        # 在实际应用中，我们不知道rho_true，只有：
        # a) TOA反射率（卫星观测）
        # b) 6S反演结果（初步地表反射率估计）

        if 'rho_toa' in df.columns:
            features['rho_toa'] = df['rho_toa']
        else:
            # 如果没有TOA数据，估算一个
            features['rho_toa'] = 0.3

        # 6S反演结果（闭合实验中：rho_retrieved = rho_true + error）
        if 'rho_true' in df.columns and 'error_absolute' in df.columns:
            features['rho_retrieved'] = df['rho_true'] + df['error_absolute']
        else:
            # 如果没有误差信息，假设6S反演就是rho_true
            features['rho_retrieved'] = df.get('rho_true', 0.2)

        # 9. 反射率相关衍生特征
        features['rho_ratio'] = features['rho_retrieved'] / (features['rho_toa'] + 1e-6)
        features['rho_diff'] = features['rho_toa'] - features['rho_retrieved']

        # 10. 波长与角度的交互特征
        features['wavelength_cos_sza'] = features['wavelength'] * features['cos_sza']
        features['wavelength_cos_vza'] = features['wavelength'] * features['cos_vza']

        # 11. AOD与角度的交互特征
        features['aod_total_airmass'] = features['aod550'] * features['total_airmass_approx']
        features['aod_cos_sza'] = features['aod550'] * features['cos_sza']

        return features

    @staticmethod
    def spherical_airmass(theta_deg: float) -> float:
        """
        计算球面大气的近似大气质量（单个值）
        比简单的sec(theta)更接近真实值，尤其在角度大时
        """
        theta_rad = np.radians(theta_deg)

        # 使用修正的Kasten公式
        if theta_deg < 85:
            am = 1.0 / (np.cos(theta_rad) + 0.50572 * (96.07995 - theta_deg) ** -1.6364)
        else:
            # 大角度近似
            # 地球半径R≈6371km，大气标高H≈8.5km
            R_over_H = 6371 / 8.5
            am = R_over_H * np.sqrt(np.pi / 2)

        return am

    @staticmethod
    def spherical_airmass_series(theta_series: pd.Series) -> pd.Series:
        """
        计算球面大气的近似大气质量（适用于pandas Series）
        使用向量化操作提高效率
        """
        # 将角度转换为弧度
        theta_rad = np.radians(theta_series)

        # 创建结果数组
        am = np.zeros_like(theta_series, dtype=np.float32)

        # 对小角度使用修正的Kasten公式
        small_angle_mask = theta_series < 85
        theta_small = theta_series[small_angle_mask]
        theta_rad_small = theta_rad[small_angle_mask]

        if len(theta_small) > 0:
            am[small_angle_mask] = 1.0 / (
                    np.cos(theta_rad_small) +
                    0.50572 * (96.07995 - theta_small) ** -1.6364
            )

        # 对大角度使用近似公式
        large_angle_mask = ~small_angle_mask
        if large_angle_mask.any():
            R_over_H = 6371 / 8.5
            am[large_angle_mask] = R_over_H * np.sqrt(np.pi / 2)

        return pd.Series(am, index=theta_series.index)

    @staticmethod
    def analyze_multicollinearity(features: pd.DataFrame, threshold: float = 0.95):
        """
        分析并处理多重共线性
        """
        # 计算相关系数矩阵
        corr_matrix = features.corr().abs()

        # 找出高度相关的特征对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))

        return high_corr_pairs


class CorrectedMachineLearningModelsV2:
    """修正的机器学习模型训练与评估类（版本2）- 解决共线性问题"""

    def __init__(self, config: ExperimentConfig = None, logger=None,
                 show_progress: bool = True, use_gpu: bool = True,
                 sample_fraction: float = None,
                 use_feature_engineering: bool = True):
        self.config = config or ExperimentConfig
        self.logger = logger or setup_logger('MLModelsV2')
        self.show_progress = show_progress
        self.use_gpu = use_gpu
        self.sample_fraction = sample_fraction
        self.use_feature_engineering = use_feature_engineering

        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.results = {}

        # 特征工程器
        self.feature_engineer = AdvancedFeatureEngineer() if use_feature_engineering else None

        # 基础特征（原始数据中的列名）
        self.base_columns = [
            'sza', 'vza', 'raa',  # 观测几何
            'aod550', 'h2o', 'o3',  # 大气参数
            'wavelength',  # 波段信息
            'rho_true',  # 真实地表反射率（仅在闭合实验中已知）
            'rho_toa'  # TOA反射率（卫星观测）
        ]

        # 目标列名
        self.target_column = 'error_absolute'

        # 实际可用的特征列（在特征工程后确定）
        self.feature_columns = []  # 将在prepare_features中确定

        # 模型配置
        self.model_configs = self._initialize_optimized_configs()

        # 输出目录
        self.output_dir = self.config.MODELS_DIR / "ml_models_corrected_v2"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"V2版机器学习模型初始化完成")
        self.logger.info(f"使用特征工程: {use_feature_engineering}")

    def _initialize_optimized_configs(self) -> Dict[str, Dict]:
        """初始化优化的模型配置"""
        configs = {
            'RandomForest': {
                'model_class': RandomForestRegressor,
                'params': {
                    'n_estimators': [100, 150],
                    'max_depth': [15, 25, None],
                    'min_samples_split': [5, 10],
                    'min_samples_leaf': [2, 4],
                    'max_features': ['sqrt', 0.8],
                    'random_state': [self.config.RANDOM_SEED],
                    'n_jobs': [-1]
                },
                'search_method': 'randomized',
                'n_iter': 15,
                'description': '随机森林回归',
                'show_progress': True
            },
            'XGBoost': {
                'model_class': xgb.XGBRegressor if XGB_AVAILABLE else None,
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 7, 9],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9],
                    'gamma': [0, 0.1, 0.2],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [1, 1.5, 2],
                    'random_state': [self.config.RANDOM_SEED],
                    'tree_method': ['gpu_hist'] if self.use_gpu else ['hist'],
                    'predictor': ['gpu_predictor'] if self.use_gpu else ['cpu_predictor']
                } if XGB_AVAILABLE else {},
                'search_method': 'randomized',
                'n_iter': 20,
                'description': '极端梯度提升',
                'show_progress': True,
                'requires_gpu': False  # XGBoost GPU加速可选
            } if XGB_AVAILABLE else None,
            'LightGBM': {
                'model_class': lgb.LGBMRegressor if LGB_AVAILABLE else None,
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10, -1],
                    'learning_rate': [0.05, 0.1],
                    'num_leaves': [31, 50, 70],
                    'min_child_samples': [20, 50],
                    'reg_alpha': [0, 0.1],
                    'reg_lambda': [0, 0.1],
                    'random_state': [self.config.RANDOM_SEED],
                    'device': ['gpu'] if self.use_gpu else ['cpu'],
                    'n_jobs': [-1]
                } if LGB_AVAILABLE else {},
                'search_method': 'randomized',
                'n_iter': 15,
                'description': 'LightGBM',
                'show_progress': True,
                'requires_gpu': False
            } if LGB_AVAILABLE else None,
            'Ridge': {
                'model_class': Ridge,
                'params': {
                    'alpha': np.logspace(-3, 3, 7),  # 10^-3 到 10^3
                    'random_state': [self.config.RANDOM_SEED],
                    'solver': ['auto', 'svd', 'cholesky']
                },
                'search_method': 'grid',
                'description': '岭回归',
                'show_progress': True
            },
            'ElasticNet': {
                'model_class': ElasticNet,
                'params': {
                    'alpha': np.logspace(-3, 1, 5),
                    'l1_ratio': [0.1, 0.5, 0.7, 0.9, 1.0],
                    'random_state': [self.config.RANDOM_SEED],
                    'max_iter': [5000]
                },
                'search_method': 'randomized',
                'n_iter': 10,
                'description': '弹性网络回归',
                'show_progress': True
            }
        }

        # 移除不可用的模型
        return {k: v for k, v in configs.items() if v is not None and v['model_class'] is not None}

    def load_and_prepare_data(self) -> pd.DataFrame:
        """加载数据并准备基础特征"""
        if self.show_progress:
            print("📂 加载数据...")

        data_files = list(self.config.DATA_DIR.glob("simulation_results_*.nc"))

        if not data_files:
            raise FileNotFoundError(f"在目录 {self.config.DATA_DIR} 中找不到数据文件")

        self.logger.info(f"找到 {len(data_files)} 个数据文件")

        # 采样文件
        if self.sample_fraction and self.sample_fraction < 0.3:
            n_files = max(1, int(len(data_files) * self.sample_fraction * 3))
            data_files = data_files[:n_files]
            self.logger.info(f"采样: 使用 {n_files} 个文件")

        all_data = []

        for i, file in enumerate(data_files):
            try:
                if self.show_progress and i % 5 == 0:
                    print(f"  加载 {i + 1}/{len(data_files)}: {file.name}")

                ds = xr.open_dataset(file)
                df = ds.to_dataframe().reset_index(drop=True)

                # 添加波段信息
                band_name = file.stem.replace("simulation_results_", "").replace("_parallel", "")
                df['band'] = band_name

                # 添加波长
                if band_name in self.config.BANDS:
                    df['wavelength'] = self.config.BANDS[band_name]['wavelength']

                # 过滤成功样本
                if 'success' in df.columns:
                    df = df[df['success'] == 1]
                if 'closed_loop_success' in df.columns:
                    df = df[df['closed_loop_success'] == 1]

                # ==================== 关键：准备必需的数据列 ====================

                # 1. 确保有相对方位角
                if 'raa' not in df.columns:
                    if 'phi' in df.columns:
                        df['raa'] = df['phi']
                    else:
                        df['raa'] = 0.0

                # 2. 确保有TOA反射率（闭合实验中模拟的TOA）
                if 'rho_toa' not in df.columns:
                    # 如果没有TOA列，我们需要计算一个近似值
                    # 在闭合实验中，TOA可以通过6S正向模拟得到
                    # 这里我们使用简化公式估算
                    df['rho_toa'] = df.get('rho_true', 0.2) * 0.7 + 0.05

                # 3. 确保有大气参数
                for param in ['aod550', 'h2o', 'o3']:
                    if param not in df.columns:
                        if param == 'aod550':
                            df[param] = 0.2
                        elif param == 'h2o':
                            df[param] = 2.0
                        elif param == 'o3':
                            df[param] = 0.3

                # 4. 数据采样
                if self.sample_fraction and len(df) > 10000:
                    sample_size = int(len(df) * self.sample_fraction)
                    df = df.sample(n=min(sample_size, 50000),
                                   random_state=self.config.RANDOM_SEED)

                all_data.append(df)
                ds.close()

            except Exception as e:
                self.logger.warning(f"加载文件 {file} 时出错: {e}")
                continue

        if not all_data:
            raise ValueError("没有成功加载任何数据")

        # 合并数据
        combined_data = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"合并后的数据集: {len(combined_data):,} 个样本")

        # 内存优化
        for col in combined_data.columns:
            if combined_data[col].dtype == 'float64':
                combined_data[col] = combined_data[col].astype('float32')

        return combined_data

    def prepare_features(self, data: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备特征和目标变量
        关键：仅使用实际可获取的特征
        """
        if self.use_feature_engineering:
            # 使用高级特征工程
            features = self.feature_engineer.create_physically_meaningful_features(data)
            self.feature_columns = list(features.columns)

            if self.show_progress:
                print(f"  特征工程生成 {len(self.feature_columns)} 个特征")
                print(f"  特征列表: {self.feature_columns}")

                # 分析共线性
                high_corr_pairs = self.feature_engineer.analyze_multicollinearity(features)
                if high_corr_pairs:
                    print(f"  发现 {len(high_corr_pairs)} 对高度相关特征 (>0.95)")
                    for feat1, feat2, corr in high_corr_pairs[:5]:  # 只显示前5对
                        print(f"    {feat1} - {feat2}: {corr:.3f}")

        else:
            # 使用基础特征（简单版本）
            # 注意：在实际应用中，我们不能使用rho_true！
            # 这里我们使用rho_retrieved作为替代
            features = pd.DataFrame()

            # 基础几何特征
            features['sza'] = data['sza']
            features['vza'] = data['vza']
            features['raa'] = data['raa']

            # 三角函数特征（避免直接使用sec）
            features['cos_sza'] = np.cos(np.radians(data['sza']))
            features['cos_vza'] = np.cos(np.radians(data['vza']))

            # 球面大气质量（比sec(theta)更合理）
            features['airmass_sza'] = AdvancedFeatureEngineer.spherical_airmass_series(data['sza'])
            features['airmass_vza'] = AdvancedFeatureEngineer.spherical_airmass_series(data['vza'])
            features['total_airmass'] = features['airmass_sza'] + features['airmass_vza']

            # 大气参数
            features['aod550'] = data['aod550']
            features['h2o'] = data.get('h2o', 2.0)
            features['o3'] = data.get('o3', 0.3)

            # 波段信息
            features['wavelength'] = data['wavelength']

            # 反射率特征 - 关键修正！
            # 在实际应用中，我们只有：
            # 1. TOA反射率（卫星观测）
            features['rho_toa'] = data['rho_toa']

            # 2. 6S反演结果（闭合实验中：rho_retrieved = rho_true + error）
            if 'rho_true' in data.columns and 'error_absolute' in data.columns:
                features['rho_retrieved'] = data['rho_true'] + data['error_absolute']
            else:
                # 如果没有误差信息，假设6S反演就是rho_true
                features['rho_retrieved'] = data.get('rho_true', 0.2)

            self.feature_columns = list(features.columns)

        # 目标变量
        if 'error_absolute' in data.columns:
            target = data['error_absolute']
        else:
            # 如果没有误差数据，我们无法训练
            raise ValueError("数据中没有error_absolute列，无法训练模型")

        # 移除常数特征
        constant_features = []
        for col in features.columns:
            if features[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            self.logger.warning(f"移除常数特征: {constant_features}")
            features = features.drop(columns=constant_features)
            self.feature_columns = [col for col in self.feature_columns if col not in constant_features]

        return features, target

    def train_and_evaluate(self, cv_folds: int = 5, models_to_train: List[str] = None):
        """完整的训练和评估流程"""

        # 1. 加载数据
        print("\n1. 📂 加载数据...")
        data = self.load_and_prepare_data()

        # 2. 准备特征
        print("2. 🔧 准备特征...")
        X, y = self.prepare_features(data, is_training=True)

        print(f"   特征形状: {X.shape}")
        print(f"   目标变量统计: 均值={y.mean():.4f}, 标准差={y.std():.4f}")
        print(f"   目标变量范围: [{y.min():.4f}, {y.max():.4f}]")

        # 3. 划分数据集
        print("3. 📊 划分数据集...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.config.RANDOM_SEED,
            shuffle=True
        )

        # 4. 标准化
        print("4. 📈 标准化特征...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_columns)

        print(f"   训练集: {X_train.shape[0]:,} 样本")
        print(f"   测试集: {X_test.shape[0]:,} 样本")

        # 5. 确定要训练的模型
        if models_to_train is None:
            models_to_train = ['RandomForest', 'XGBoost', 'LightGBM', 'Ridge', 'ElasticNet']

        # 6. 训练模型
        print(f"\n5. 🏋️ 训练 {len(models_to_train)} 个模型...")
        training_results = {}

        for model_name in models_to_train:
            if model_name not in self.model_configs:
                print(f"  ⚠️  跳过未知模型: {model_name}")
                continue

            print(f"\n  🔄 训练模型: {model_name}")
            print(f"    描述: {self.model_configs[model_name]['description']}")

            try:
                # 训练模型
                result = self._train_single_model(
                    model_name, X_train_scaled, y_train,
                    cv_folds=cv_folds
                )

                training_results[model_name] = result

                # 保存模型
                model_path = self.output_dir / f"{model_name}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(self.models[model_name], f)

                print(f"    ✓ 训练完成，模型已保存")

            except Exception as e:
                print(f"    ✗ 训练失败: {e}")
                continue

        # 7. 评估模型
        print(f"\n6. 📈 评估模型性能...")
        evaluation_results = self._evaluate_all_models(X_test_scaled, y_test)

        # 8. 保存结果
        print(f"\n7. 💾 保存所有结果...")
        self._save_results(training_results, evaluation_results, X_train_scaled)

        return evaluation_results

    def _train_single_model(self, model_name: str, X_train: pd.DataFrame,
                            y_train: pd.Series, cv_folds: int = 5) -> Dict:
        """训练单个模型"""
        config = self.model_configs[model_name]

        # 创建基础模型
        model_class = config['model_class']
        base_model = model_class()

        # 参数搜索设置
        search_method = config.get('search_method', 'grid')
        n_iter = config.get('n_iter', 10)

        # 创建交叉验证
        cv = KFold(n_splits=cv_folds, shuffle=True,
                   random_state=self.config.RANDOM_SEED)

        # 创建搜索对象
        if search_method == 'randomized':
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=config['params'],
                n_iter=n_iter,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0,
                random_state=self.config.RANDOM_SEED
            )
        else:
            search = GridSearchCV(
                estimator=base_model,
                param_grid=config['params'],
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )

        # 训练
        start_time = time.time()
        search.fit(X_train, y_train)
        training_time = time.time() - start_time

        # 获取最佳模型
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_rmse = np.sqrt(-search.best_score_)

        # 特征重要性
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            # 保存特征重要性
            self.feature_importance[model_name] = feature_importance

        # 保存模型
        self.models[model_name] = best_model

        # 返回结果
        result = {
            'model': best_model,
            'best_params': best_params,
            'best_rmse': best_rmse,
            'training_time': training_time,
            'cv_folds': cv_folds,
            'feature_importance': feature_importance,
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0]
        }

        print(f"    最佳RMSE (CV): {best_rmse:.6f}")
        print(f"    训练时间: {training_time:.1f}秒")
        print(f"    最佳参数: {best_params}")

        if feature_importance is not None:
            top_features = feature_importance.head(3)
            print(f"    最重要特征:")
            for _, row in top_features.iterrows():
                print(f"      {row['feature']}: {row['importance']:.4f}")

        return result

    def _evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """评估所有训练好的模型"""
        results = []

        for model_name, model in self.models.items():
            try:
                # 预测
                y_pred = model.predict(X_test)

                # 计算指标
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # 计算相对误差
                y_nonzero = y_test != 0
                if y_nonzero.any():
                    mre = np.mean(np.abs((y_test[y_nonzero] - y_pred[y_nonzero]) / y_test[y_nonzero]))
                else:
                    mre = 0.0

                # 偏差（平均误差）
                bias = np.mean(y_pred - y_test)

                results.append({
                    'Model': model_name,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'MRE': mre,
                    'Bias': bias,
                    'Description': self.model_configs[model_name]['description']
                })

                print(f"  {model_name:<15} RMSE: {rmse:.6f}  MAE: {mae:.6f}  R²: {r2:.4f}  Bias: {bias:.6f}")

            except Exception as e:
                print(f"  {model_name:<15} 评估失败: {e}")

        # 创建结果DataFrame
        results_df = pd.DataFrame(results)

        # 按RMSE排序
        if not results_df.empty:
            results_df = results_df.sort_values('RMSE')

            # 保存结果
            results_path = self.output_dir / "model_evaluation_results.csv"
            results_df.to_csv(results_path, index=False)

            # 打印最佳模型
            best_model = results_df.iloc[0]
            print(f"\n🏆 最佳模型: {best_model['Model']}")
            print(f"   RMSE: {best_model['RMSE']:.6f}")
            print(f"   R²: {best_model['R2']:.4f}")
            print(f"   Bias: {best_model['Bias']:.6f}")

        return results_df

    def _save_results(self, training_results: Dict, evaluation_results: pd.DataFrame,
                      X_train: pd.DataFrame):
        """保存所有结果"""

        # 1. 保存训练结果
        train_summary = {}
        for model_name, result in training_results.items():
            train_summary[model_name] = {
                'best_rmse': float(result['best_rmse']),
                'training_time': float(result['training_time']),
                'n_features': int(result['n_features']),
                'n_samples': int(result['n_samples']),
                'cv_folds': int(result['cv_folds'])
            }

        # 2. 保存特征信息
        feature_info = {
            'feature_columns': self.feature_columns,
            'n_features': len(self.feature_columns),
            'target_column': self.target_column,
            'use_feature_engineering': self.use_feature_engineering
        }

        # 3. 保存配置
        config_info = {
            'sample_fraction': self.sample_fraction,
            'use_gpu': self.use_gpu,
            'random_seed': self.config.RANDOM_SEED,
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # 合并所有信息
        all_results = {
            'training_summary': train_summary,
            'evaluation_results': evaluation_results.to_dict('records'),
            'feature_info': feature_info,
            'config': config_info
        }

        # 保存为JSON
        results_path = self.output_dir / "training_summary.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # 保存特征重要性
        for model_name, importance in self.feature_importance.items():
            if importance is not None:
                imp_path = self.output_dir / f"{model_name}_feature_importance.csv"
                importance.to_csv(imp_path, index=False)

        print(f"✓ 所有结果已保存到: {self.output_dir}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='V2版机器学习训练系统')
    parser.add_argument('--sample', type=float, default=0.2,
                        help='数据采样比例 (0.01-1.0)')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='使用GPU加速')
    parser.add_argument('--models', type=str, default='all',
                        help='要训练的模型: all/fast/模型列表')
    parser.add_argument('--no_feature_engineering', action='store_true',
                        help='不使用高级特征工程')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='交叉验证折数')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🧠 V2版机器学习训练系统")
    print("=" * 60)
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"采样比例: {args.sample * 100}%")
    print(f"GPU加速: {'启用' if args.gpu else '禁用'}")
    print(f"特征工程: {'禁用' if args.no_feature_engineering else '启用'}")
    print(f"交叉验证: {args.cv_folds}折")
    print("=" * 60)

    # 创建模型实例
    ml = CorrectedMachineLearningModelsV2(
        show_progress=True,
        use_gpu=args.gpu,
        sample_fraction=args.sample,
        use_feature_engineering=not args.no_feature_engineering
    )

    try:
        # 确定要训练的模型
        if args.models == 'all':
            models_to_train = list(ml.model_configs.keys())
        elif args.models == 'fast':
            models_to_train = ['RandomForest', 'XGBoost', 'LightGBM']
        else:
            models_to_train = [m.strip() for m in args.models.split(',')]

        # 训练和评估
        results = ml.train_and_evaluate(
            cv_folds=args.cv_folds,
            models_to_train=models_to_train
        )

        print("\n" + "=" * 60)
        print("🎉 训练完成!")
        print("=" * 60)
        print(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        if not results.empty:
            print("\n📊 模型排名:")
            for i, (_, row) in enumerate(results.iterrows()):
                print(f"  {i + 1}. {row['Model']:<15} RMSE: {row['RMSE']:.6f}  R²: {row['R2']:.4f}")

        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
