# ==================== MLs_enhanced.py ====================
"""
增强版机器学习模型训练与评估模块
支持完整数据训练、SHAP分析、多维可视化
"""
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import pickle
import json
import time
import psutil
import sys
import math
import warnings
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, explained_variance_score, mean_absolute_percentage_error)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.covariance import EllipticEnvelope
from sklearn.pipeline import Pipeline

# 统计库
from scipy import stats

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

# 高级分析工具
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("警告: SHAP未安装，将跳过SHAP分析")
    print("安装: pip install shap")

# 可视化库
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 项目模块
from config import ExperimentConfig
from utils import setup_logger, calculate_statistics

# 设置绘图样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class AdvancedFeatureEngineering:
    """高级特征工程类"""

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
        """创建所有特征"""
        features = pd.DataFrame()

        # 1. 基础特征
        features['sza'] = df['sza']
        features['vza'] = df['vza']
        features['raa'] = df['raa']
        features['aod550'] = df['aod550']
        features['h2o'] = df.get('h2o', 2.0)
        features['o3'] = df.get('o3', 0.3)
        features['wavelength'] = df['wavelength']
        features['rho_toa'] = df.get('rho_toa', 0.2)

        # 计算反演反射率（闭合实验中已知）
        if 'rho_true' in df.columns and 'error_absolute' in df.columns:
            features['rho_retrieved'] = df['rho_true'] + df['error_absolute']
        else:
            features['rho_retrieved'] = df.get('rho_true', 0.2)

        # 2. 三角函数特征
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
            features['scattering_angle'] = np.degrees(np.arccos(cos_scat))

            self.feature_descriptions['derived'].extend([
                'cos_sza', 'sin_sza', 'cos_vza', 'sin_vza',
                'cos_raa', 'sin_raa', 'scattering_angle'
            ])

        # 3. 大气质量数
        if self.use_derived:
            features['airmass_sza'] = 1.0 / np.cos(np.radians(df['sza']))
            features['airmass_vza'] = 1.0 / np.cos(np.radians(df['vza']))
            features['total_airmass'] = features['airmass_sza'] + features['airmass_vza']

            self.feature_descriptions['derived'].extend([
                'airmass_sza', 'airmass_vza', 'total_airmass'
            ])

        # 4. 角度关系
        if self.use_ratios:
            features['vza_sza_ratio'] = df['vza'] / (df['sza'] + 1e-6)
            features['vza_minus_sza'] = df['vza'] - df['sza']
            features['vza_plus_sza'] = df['vza'] + df['sza']

            self.feature_descriptions['derived'].extend([
                'vza_sza_ratio', 'vza_minus_sza', 'vza_plus_sza'
            ])

        # 5. 交互特征
        if self.use_interactions:
            # 大气-几何交互
            features['aod_airmass'] = features['aod550'] * features['total_airmass']
            features['aod_wavelength'] = features['aod550'] * features['wavelength']

            # 反射率关系
            features['rho_ratio'] = features['rho_retrieved'] / (features['rho_toa'] + 1e-6)
            features['rho_diff'] = features['rho_toa'] - features['rho_retrieved']
            features['rho_product'] = features['rho_toa'] * features['rho_retrieved']

            # 波长-角度交互
            features['wavelength_cos_sza'] = features['wavelength'] * features['cos_sza']
            features['wavelength_cos_vza'] = features['wavelength'] * features['cos_vza']

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
            print(f"  {i + 1}. {feat}")

        return features[selected_features]

    def analyze_collinearity(self, features: pd.DataFrame, threshold: float = 0.9):
        """分析特征共线性"""
        corr_matrix = features.corr().abs()

        # 找出高度相关的特征对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    col_i = corr_matrix.columns[i]
                    col_j = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    high_corr_pairs.append((col_i, col_j, corr_value))

        return high_corr_pairs, corr_matrix


class EnhancedMLModels:
    """增强版机器学习模型训练与评估类"""

    def __init__(self, config: ExperimentConfig = None, logger=None,
                 use_feature_engineering: bool = True,
                 use_shap: bool = True,
                 use_advanced_plots: bool = True,
                 n_jobs: int = -1):

        self.config = config or ExperimentConfig
        self.logger = logger or setup_logger('EnhancedML')
        self.use_feature_engineering = use_feature_engineering
        self.use_shap = use_shap and SHAP_AVAILABLE
        self.use_advanced_plots = use_advanced_plots
        self.n_jobs = n_jobs

        # 初始化组件
        self.feature_engineer = AdvancedFeatureEngineering()
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_importance = {}
        self.shap_values = {}

        # 输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.config.MODELS_DIR / f"ml_enhanced_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        self.plots_dir = self.output_dir / "plots"
        self.models_dir = self.output_dir / "models"
        self.tables_dir = self.output_dir / "tables"
        for d in [self.plots_dir, self.models_dir, self.tables_dir]:
            d.mkdir(exist_ok=True)

        # 特征和目标列
        self.target_column = 'error_absolute'

        # 初始化模型配置
        self.model_configs = self._initialize_enhanced_configs()

        self.logger.info(f"增强版ML模型初始化完成")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info(f"特征工程: {use_feature_engineering}")
        self.logger.info(f"SHAP分析: {self.use_shap}")

    def _initialize_enhanced_configs(self) -> Dict[str, Dict]:
        """初始化增强的模型配置"""
        configs = {
            'RandomForest': {
                'model_class': RandomForestRegressor,
                'params': {
                    'n_estimators': [50, 100],  # 减少树的数量
                    'max_depth': [10, 15, 20],  # 限制深度
                    'min_samples_split': [10, 20, 50],  # 增加分裂所需最小样本数
                    'min_samples_leaf': [5, 10, 20],  # 增加叶节点最小样本数
                    'max_features': ['sqrt', 0.5],  # 减少特征比例
                    'bootstrap': [True],
                    'random_state': [self.config.RANDOM_SEED],
                    'n_jobs': [1],  # 强制单线程避免内存爆炸
                    'max_samples': [0.5, 0.7]  # 添加样本抽样
                },
                'search_method': 'randomized',
                'n_iter': 8,  # 减少参数组合
                'description': '随机森林回归(优化版)',
                'color': '#1f77b4'
            },
            'XGBoost': {
                'model_class': xgb.XGBRegressor if XGB_AVAILABLE else None,
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'gamma': [0, 0.1, 0.2],
                    'reg_alpha': [0, 0.01, 0.1],
                    'reg_lambda': [1, 1.5, 2],
                    'random_state': [self.config.RANDOM_SEED],
                    'n_jobs': [self.n_jobs],
                    'tree_method': ['hist']  # 使用直方图算法更快
                } if XGB_AVAILABLE else {},
                'search_method': 'randomized',
                'n_iter': 15,
                'description': 'XGBoost回归',
                'color': '#ff7f0e'
            } if XGB_AVAILABLE else None,
            'LightGBM': {
                'model_class': lgb.LGBMRegressor if LGB_AVAILABLE else None,
                'params': {
                    'n_estimators': [100, 200],
                    'num_leaves': [31, 63],
                    'max_depth': [5, 10, -1],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'reg_alpha': [0, 0.01, 0.1],
                    'reg_lambda': [0, 0.01, 0.1],
                    'min_child_samples': [20, 50],
                    'random_state': [self.config.RANDOM_SEED],
                    'n_jobs': [self.n_jobs],
                    'verbose': [-1]
                } if LGB_AVAILABLE else {},
                'search_method': 'randomized',
                'n_iter': 15,
                'description': 'LightGBM回归',
                'color': '#2ca02c'
            } if LGB_AVAILABLE else None,
            'GradientBoosting': {
                'model_class': GradientBoostingRegressor,
                'params': {
                    'n_estimators': [50, 100],  # 大幅减少
                    'learning_rate': [0.05, 0.1, 0.2],  # 增大学习率
                    'max_depth': [3, 4],  # 减小深度
                    'min_samples_split': [20, 50],  # 增大
                    'min_samples_leaf': [10, 20],  # 增大
                    'subsample': [0.7, 0.8],  # 添加样本抽样
                    'max_features': [0.5, 0.7],  # 添加特征抽样
                    'random_state': [self.config.RANDOM_SEED]
                },
                'search_method': 'randomized',
                'n_iter': 8,  # 大幅减少参数组合
                'description': '梯度提升回归树(优化版)',
                'color': '#d62728'
            },
            'SVR_RBF': {
                'model_class': SVR,
                'params': {
                    'kernel': ['rbf'],
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto'] + list(np.logspace(-3, 0, 4)),
                    'epsilon': [0.01, 0.1, 0.2]
                },
                'search_method': 'grid',
                'description': '支持向量回归(RBF核)',
                'color': '#9467bd'
            },
            'MLP': {
                'model_class': MLPRegressor,
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive'],
                    'learning_rate_init': [0.001, 0.01],
                    'max_iter': [300, 500],  # 减少迭代次数
                    'early_stopping': [True],
                    'random_state': [self.config.RANDOM_SEED],
                    'batch_size': [128, 256]  # 添加批处理大小
                },
                'search_method': 'randomized',
                'n_iter': 10,
                'description': '多层感知器',
                'color': '#8c564b'
            },
            'Ridge': {
                'model_class': Ridge,
                'params': {
                    'alpha': np.logspace(-3, 3, 7),
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr'],
                    'random_state': [self.config.RANDOM_SEED]
                },
                'search_method': 'randomized',
                'n_iter': 8,
                'description': '岭回归',
                'color': '#e377c2'
            },
            'ElasticNet': {
                'model_class': ElasticNet,
                'params': {
                    'alpha': np.logspace(-3, 1, 5),
                    'l1_ratio': [0.1, 0.5, 0.9],
                    'max_iter': [1000],
                    'random_state': [self.config.RANDOM_SEED]
                },
                'search_method': 'randomized',
                'n_iter': 10,
                'description': '弹性网络回归',
                'color': '#7f7f7f'
            }
        }

        # 移除不可用的模型
        return {k: v for k, v in configs.items() if v is not None and v['model_class'] is not None}

    def load_data(self, sample_fraction: float = 1.0) -> pd.DataFrame:
        """加载完整数据集"""
        print("=" * 70)
        print("📂 加载数据...")

        data_files = list(self.config.DATA_DIR.glob("simulation_results_*.nc"))

        if not data_files:
            raise FileNotFoundError(f"在目录 {self.config.DATA_DIR} 中找不到数据文件")

        print(f"找到 {len(data_files)} 个数据文件")

        # 如果需要采样
        if sample_fraction < 1.0:
            n_files = max(1, int(len(data_files) * sample_fraction))
            data_files = data_files[:n_files]
            print(f"采样: 使用 {n_files} 个文件")

        all_data = []
        total_samples = 0

        for i, file in enumerate(data_files):
            try:
                print(f"  加载文件 {i + 1}/{len(data_files)}: {file.name}")

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

                # 确保有必需的特征
                if 'raa' not in df.columns:
                    df['raa'] = df.get('phi', 0.0)

                if 'rho_toa' not in df.columns:
                    df['rho_toa'] = df.get('rho_true', 0.2) * 0.7 + 0.05

                all_data.append(df)
                total_samples += len(df)
                ds.close()

            except Exception as e:
                self.logger.warning(f"加载文件 {file} 时出错: {e}")

        if not all_data:
            raise ValueError("没有成功加载任何数据")

        # 合并数据
        data = pd.concat(all_data, ignore_index=True)

        print(f"✅ 数据加载完成")
        print(f"   总样本数: {total_samples:,}")
        print(f"   特征数: {data.shape[1]}")
        print(f"   内存使用: {data.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")

        return data

    def prepare_data(self, data: pd.DataFrame, test_size: float = 0.2) -> Tuple:
        """准备训练和测试数据"""
        print("=" * 70)
        print("🔧 准备数据...")

        # 创建特征
        if self.use_feature_engineering:
            X = self.feature_engineer.create_features(data)
            print(f"   特征工程生成 {X.shape[1]} 个特征")
        else:
            # 使用基础特征
            base_features = ['sza', 'vza', 'raa', 'aod550', 'h2o', 'o3',
                             'wavelength', 'rho_toa']
            X = data[base_features].copy()

            # 添加反演反射率
            if 'rho_true' in data.columns and 'error_absolute' in data.columns:
                X['rho_retrieved'] = data['rho_true'] + data['error_absolute']
            else:
                X['rho_retrieved'] = data.get('rho_true', 0.2)

        # 目标变量
        y = data[self.target_column]

        print(f"   目标变量统计:")
        print(f"     均值: {y.mean():.6f}")
        print(f"     标准差: {y.std():.6f}")
        print(f"     最小值: {y.min():.6f}")
        print(f"     最大值: {y.max():.6f}")
        print(f"     中位数: {y.median():.6f}")

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config.RANDOM_SEED
        )

        print(f"   数据集划分:")
        print(f"     训练集: {X_train.shape[0]:,} 样本")
        print(f"     测试集: {X_test.shape[0]:,} 样本")

        # 标准化
        print("   标准化特征...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

        # 保存特征信息
        self.feature_names = list(X.columns)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                     models_to_train: List[str] = None, cv_folds: int = 5):
        """训练所有指定的模型"""
        print("=" * 70)
        print("🚀 训练模型...")

        if models_to_train is None:
            models_to_train = list(self.model_configs.keys())

        print(f"   将训练 {len(models_to_train)} 个模型:")
        for i, model_name in enumerate(models_to_train):
            config = self.model_configs[model_name]
            print(f"     {i + 1}. {model_name} - {config['description']}")

        training_results = {}

        for model_name in models_to_train:
            try:
                print(f"\n   🔄 训练 {model_name}...")
                start_time = time.time()

                result = self._train_single_model(
                    model_name, X_train, y_train, cv_folds
                )

                training_time = time.time() - start_time
                result['training_time'] = training_time

                training_results[model_name] = result

                print(f"   ✅ {model_name} 训练完成")
                print(f"     最佳RMSE (CV): {result['best_rmse']:.6f}")
                print(f"     训练时间: {training_time:.1f}秒")

                # 保存模型
                model_path = self.models_dir / f"{model_name}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(self.models[model_name], f)

            except Exception as e:
                print(f"   ❌ {model_name} 训练失败: {e}")
                import traceback
                traceback.print_exc()

        return training_results

    def _train_single_model(self, model_name: str, X_train: pd.DataFrame,
                            y_train: pd.Series, cv_folds: int = 5) -> Dict:
        """训练单个模型（带内存优化）"""
        config = self.model_configs[model_name]
        model_class = config['model_class']
        params = config['params']

        # 内存监控
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        print(f"   内存使用前: {memory_before:.1f} MB")

        # 对于内存密集型模型，减少数据量
        if model_name in ['RandomForest', 'GradientBoosting', 'ExtraTrees']:
            if len(X_train) > 10000:
                print(f"   采样数据以节省内存...")
                sample_idx = np.random.choice(len(X_train), 10000, replace=False)
                X_train_sampled = X_train.iloc[sample_idx]
                y_train_sampled = y_train.iloc[sample_idx]
            else:
                X_train_sampled = X_train
                y_train_sampled = y_train
        else:
            X_train_sampled = X_train
            y_train_sampled = y_train

        # 调整并行度避免内存爆炸
        if model_name in ['RandomForest']:
            if 'n_jobs' in params:
                params['n_jobs'] = [1]  # 强制单线程
            current_n_jobs = 1
        else:
            current_n_jobs = self.n_jobs

        # 创建基础模型
        if model_name in ['XGBoost', 'LightGBM']:
            base_model = model_class(random_state=self.config.RANDOM_SEED,
                                     n_jobs=current_n_jobs, verbose=0)
        else:
            base_model = model_class()

        # 创建搜索策略
        search_method = config.get('search_method', 'grid')
        n_iter = config.get('n_iter', 10)

        cv = KFold(n_splits=cv_folds, shuffle=True,
                   random_state=self.config.RANDOM_SEED)

        if search_method == 'randomized':
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=params,
                n_iter=n_iter,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=1 if model_name in ['RandomForest'] else self.n_jobs,  # RandomForest使用单线程
                random_state=self.config.RANDOM_SEED,
                verbose=0
            )
        else:
            search = GridSearchCV(
                estimator=base_model,
                param_grid=params,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=1 if model_name in ['RandomForest'] else self.n_jobs,
                verbose=0
            )

        # 训练
        search.fit(X_train_sampled, y_train_sampled)

        # 获取结果
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = -search.best_score_  # 转换为正MSE
        best_rmse = np.sqrt(best_score)

        # 保存模型
        self.models[model_name] = best_model

        # 计算特征重要性
        feature_importance = self._compute_feature_importance(best_model, X_train_sampled)
        if feature_importance is not None:
            self.feature_importance[model_name] = feature_importance

        # 内存使用后
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        print(f"   内存使用后: {memory_after:.1f} MB")
        print(f"   内存增量: {memory_after - memory_before:.1f} MB")

        return {
            'model': best_model,
            'best_params': best_params,
            'best_rmse': best_rmse,
            'best_score': best_score,
            'cv_scores': search.cv_results_,
            'feature_importance': feature_importance,
            'n_samples_used': len(X_train_sampled)
        }

    def _compute_feature_importance(self, model, X_train: pd.DataFrame):
        """计算特征重要性"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            return feature_importance
        elif hasattr(model, 'coef_'):
            # 对于线性模型
            importances = np.abs(model.coef_)
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            return feature_importance
        else:
            return None

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """评估所有模型"""
        print("=" * 70)
        print("📊 评估模型...")

        results = []

        for model_name, model in self.models.items():
            try:
                print(f"   评估 {model_name}...")

                # 预测
                y_pred = model.predict(X_test)

                # 计算指标
                metrics = self._compute_metrics(y_test, y_pred, model_name)
                self.results[model_name] = metrics

                # 保存结果
                result_row = {
                    'Model': model_name,
                    'Description': self.model_configs[model_name]['description'],
                    **metrics
                }
                results.append(result_row)

                print(f"     RMSE: {metrics['RMSE']:.6f}")
                print(f"     MAE: {metrics['MAE']:.6f}")
                print(f"     R²: {metrics['R2']:.4f}")

            except Exception as e:
                print(f"   ❌ {model_name} 评估失败: {e}")

        # 创建结果DataFrame
        results_df = pd.DataFrame(results)

        if not results_df.empty:
            # 按RMSE排序
            results_df = results_df.sort_values('RMSE')

            # 保存结果
            results_path = self.tables_dir / "model_evaluation_results.csv"
            results_df.to_csv(results_path, index=False)

            # 打印排名
            print(f"\n🏆 模型性能排名:")
            print("-" * 70)
            print(f"{'排名':<4} {'模型':<15} {'RMSE':<10} {'R²':<8} {'MAE':<10}")
            print("-" * 70)
            for i, (_, row) in enumerate(results_df.iterrows()):
                print(f"{i + 1:<4} {row['Model']:<15} {row['RMSE']:.6f}  {row['R2']:.4f}  {row['MAE']:.6f}")

        return results_df

    def _compute_metrics(self, y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> Dict:
        """计算综合评估指标"""
        metrics = {}

        # 基础指标
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['R2'] = r2_score(y_true, y_pred)
        metrics['Explained_Variance'] = explained_variance_score(y_true, y_pred)

        # 相对误差
        nonzero_mask = y_true != 0
        if nonzero_mask.any():
            relative_errors = np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])
            metrics['MAPE'] = np.mean(relative_errors) * 100  # 百分比
            metrics['Median_APE'] = np.median(relative_errors) * 100
        else:
            metrics['MAPE'] = np.nan
            metrics['Median_APE'] = np.nan

        # 偏差和精度
        residuals = y_true - y_pred
        metrics['Bias'] = np.mean(residuals)
        metrics['Std_Residuals'] = np.std(residuals)

        # 分位数误差
        metrics['Q10_Error'] = np.percentile(np.abs(residuals), 10)
        metrics['Q90_Error'] = np.percentile(np.abs(residuals), 90)

        # 保存预测值
        metrics['y_true'] = y_true.values
        metrics['y_pred'] = y_pred

        return metrics

    def generate_visualizations(self, X_test: pd.DataFrame, y_test: pd.Series,
                                results_df: pd.DataFrame):
        """生成所有可视化图表"""
        print("=" * 70)
        print("🎨 生成可视化图表...")

        # 1. 模型比较图
        self._plot_model_comparison(results_df)

        # 2. 散点图矩阵
        self._plot_scatter_matrix(X_test, y_test)

        # 3. 残差分析图
        self._plot_residual_analysis(X_test, y_test)

        # 4. 特征重要性图
        self._plot_feature_importance()

        # 5. 预测误差分布图
        self._plot_error_distribution()

        # 6. SHAP分析（如果可用）
        if self.use_shap:
            self._perform_shap_analysis(X_test)

        # 7. 学习曲线
        self._plot_learning_curves()

        # 8. 交互式3D图
        if self.use_advanced_plots:
            self._plot_interactive_3d()

        print(f"✅ 所有图表已保存到: {self.plots_dir}")

    def _plot_model_comparison(self, results_df: pd.DataFrame):
        """绘制模型比较图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. RMSE和MAE比较
        ax = axes[0, 0]
        x = np.arange(len(results_df))
        width = 0.35

        ax.bar(x - width / 2, results_df['RMSE'], width, label='RMSE', alpha=0.8)
        ax.bar(x + width / 2, results_df['MAE'], width, label='MAE', alpha=0.8)
        ax.set_xlabel('模型')
        ax.set_ylabel('误差')
        ax.set_title('模型误差比较')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. R²比较
        ax = axes[0, 1]
        colors = []
        for m in results_df['Model']:
            if m in self.model_configs:
                colors.append(self.model_configs[m]['color'])
            else:
                colors.append('#1f77b4')  # 默认颜色

        bars = ax.bar(results_df['Model'], results_df['R2'], color=colors, alpha=0.8)
        ax.set_xlabel('模型')
        ax.set_ylabel('R²')
        ax.set_title('模型决定系数(R²)比较')
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        # 添加数值标签
        for bar, value in zip(bars, results_df['R2']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        # 3. 预测偏差
        ax = axes[0, 2]
        ax.bar(results_df['Model'], results_df['Bias'], alpha=0.8)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('模型')
        ax.set_ylabel('预测偏差')
        ax.set_title('模型预测偏差')
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        # 4. 训练时间比较
        ax = axes[1, 0]
        training_times = [self.results.get(m, {}).get('training_time', 0)
                          for m in results_df['Model']]
        ax.bar(results_df['Model'], training_times, alpha=0.8)
        ax.set_xlabel('模型')
        ax.set_ylabel('训练时间(秒)')
        ax.set_title('模型训练时间比较')
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        # 5. MAPE比较
        ax = axes[1, 1]
        ax.bar(results_df['Model'], results_df['MAPE'], alpha=0.8)
        ax.set_xlabel('模型')
        ax.set_ylabel('MAPE(%)')
        ax.set_title('平均绝对百分比误差(MAPE)')
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        # 6. 解释方差
        ax = axes[1, 2]
        ax.bar(results_df['Model'], results_df['Explained_Variance'], alpha=0.8)
        ax.set_xlabel('模型')
        ax.set_ylabel('解释方差')
        ax.set_title('模型解释方差')
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_scatter_matrix(self, X_test: pd.DataFrame, y_test: pd.Series):
        """绘制散点图矩阵"""
        # 为每个模型创建散点图
        n_models = len(self.models)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (model_name, model) in enumerate(list(self.models.items())[:len(axes)]):
            ax = axes[idx]
            y_pred = model.predict(X_test)

            # 散点图
            scatter = ax.scatter(y_test, y_pred, alpha=0.6, s=10,
                                 c=np.abs(y_test - y_pred), cmap='viridis')

            # 对角线
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            # 统计信息
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            ax.text(0.05, 0.95, f'RMSE = {rmse:.4f}\nR² = {r2:.3f}\nMAE = {mae:.4f}',
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax.set_xlabel('实际值')
            ax.set_ylabel('预测值')
            ax.set_title(f'{model_name} - 预测vs实际')
            ax.grid(True, alpha=0.3)

            # 添加颜色条
            if idx == 0:
                plt.colorbar(scatter, ax=ax, label='绝对误差')

        # 隐藏多余的子图
        for idx in range(len(self.models), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "prediction_scatter_plots.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_residual_analysis(self, X_test: pd.DataFrame, y_test: pd.Series):
        """绘制残差分析图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 获取最佳模型
        best_model_name = list(self.results.keys())[0] if self.results else None
        if not best_model_name:
            return

        model = self.models[best_model_name]
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        # 1. 残差分布
        ax = axes[0, 0]
        ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('残差')
        ax.set_ylabel('频数')
        ax.set_title('残差分布')
        ax.grid(True, alpha=0.3)

        # 2. 残差vs预测值
        ax = axes[0, 1]
        ax.scatter(y_pred, residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('预测值')
        ax.set_ylabel('残差')
        ax.set_title('残差vs预测值')
        ax.grid(True, alpha=0.3)

        # 3. QQ图
        ax = axes[0, 2]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('残差QQ图')
        ax.grid(True, alpha=0.3)

        # 4. 残差vs特征（最重要的特征）
        if best_model_name in self.feature_importance:
            top_feature = self.feature_importance[best_model_name].iloc[0]['feature']
            ax = axes[1, 0]
            ax.scatter(X_test[top_feature], residuals, alpha=0.5, s=10)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
            ax.set_xlabel(top_feature)
            ax.set_ylabel('残差')
            ax.set_title(f'残差vs {top_feature}')
            ax.grid(True, alpha=0.3)

        # 5. 累计残差
        ax = axes[1, 1]
        cum_residuals = np.cumsum(residuals)
        ax.plot(cum_residuals)
        ax.set_xlabel('样本序号')
        ax.set_ylabel('累计残差')
        ax.set_title('累计残差图')
        ax.grid(True, alpha=0.3)

        # 6. 误差vs目标值
        ax = axes[1, 2]
        ax.scatter(y_test, np.abs(residuals), alpha=0.5, s=10)
        ax.set_xlabel('实际值')
        ax.set_ylabel('绝对误差')
        ax.set_title('误差vs实际值')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "residual_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_feature_importance(self):
        """绘制特征重要性图"""
        if not self.feature_importance:
            return

        n_models = len(self.feature_importance)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (model_name, importance_df) in enumerate(list(self.feature_importance.items())[:len(axes)]):
            ax = axes[idx]
            top_n = min(10, len(importance_df))
            top_features = importance_df.head(top_n)

            y_pos = np.arange(len(top_features))
            ax.barh(y_pos, top_features['importance'])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('重要性')
            ax.set_title(f'{model_name} - 特征重要性(Top {top_n})')
            ax.grid(True, alpha=0.3, axis='x')

        # 隐藏多余的子图
        for idx in range(len(self.feature_importance), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 保存特征重要性数据
        for model_name, importance_df in self.feature_importance.items():
            importance_path = self.tables_dir / f"{model_name}_feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)

    def _plot_error_distribution(self):
        """绘制误差分布图"""
        if not self.results:
            return

        n_models = len(self.results)
        n_cols = min(2, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (model_name, metrics) in enumerate(list(self.results.items())[:len(axes)]):
            ax = axes[idx]

            y_true = metrics['y_true']
            y_pred = metrics['y_pred']
            errors = y_true - y_pred

            # 误差直方图
            ax.hist(errors, bins=50, alpha=0.7, density=True, edgecolor='black')

            # 拟合正态分布
            mu, sigma = stats.norm.fit(errors)
            x = np.linspace(errors.min(), errors.max(), 100)
            p = stats.norm.pdf(x, mu, sigma)
            ax.plot(x, p, 'r-', linewidth=2, label=f'正态分布拟合\nμ={mu:.4f}, σ={sigma:.4f}')

            ax.set_xlabel('误差')
            ax.set_ylabel('密度')
            ax.set_title(f'{model_name} - 误差分布')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(len(self.results), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "error_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _perform_shap_analysis(self, X_test: pd.DataFrame):
        """执行SHAP分析"""
        if not self.use_shap or not self.models:
            return

        print("   执行SHAP分析...")

        # 选择最佳模型进行SHAP分析
        best_model_name = list(self.results.keys())[0]
        model = self.models[best_model_name]

        # 采样以减少计算时间
        if len(X_test) > 1000:
            X_sample = X_test.sample(n=1000, random_state=self.config.RANDOM_SEED)
        else:
            X_sample = X_test

        try:
            # 创建SHAP解释器
            if best_model_name in ['RandomForest', 'GradientBoosting', 'ExtraTrees']:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            elif best_model_name in ['XGBoost', 'LightGBM']:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            else:
                # 对于其他模型，使用KernelExplainer
                explainer = shap.KernelExplainer(model.predict, X_sample[:100])
                shap_values = explainer.shap_values(X_sample)

            # 保存SHAP值
            self.shap_values[best_model_name] = shap_values

            # 1. 特征重要性摘要图
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.title(f'{best_model_name} - SHAP特征重要性')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"{best_model_name}_shap_summary.png", dpi=300, bbox_inches='tight')
            plt.close()

            # 2. 条形图
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            plt.title(f'{best_model_name} - SHAP特征重要性(条形图)')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"{best_model_name}_shap_bar.png", dpi=300, bbox_inches='tight')
            plt.close()

            # 3. 依赖图（最重要的3个特征）
            if hasattr(shap_values, '__len__'):
                shap_values_array = np.array(shap_values)
                if len(shap_values_array.shape) > 1:
                    mean_abs_shap = np.mean(np.abs(shap_values_array), axis=0)
                    top_features_idx = np.argsort(mean_abs_shap)[-3:][::-1]

                    for i, feature_idx in enumerate(top_features_idx):
                        feature_name = X_sample.columns[feature_idx]
                        plt.figure(figsize=(8, 6))
                        shap.dependence_plot(feature_idx, shap_values_array, X_sample,
                                             show=False, interaction_index=None)
                        plt.title(f'{best_model_name} - SHAP依赖图: {feature_name}')
                        plt.tight_layout()
                        plt.savefig(self.plots_dir / f"{best_model_name}_shap_dependence_{feature_name}.png",
                                    dpi=300, bbox_inches='tight')
                        plt.close()

            # 4. 瀑布图（随机样本）
            for i in range(min(3, len(X_sample))):
                plt.figure(figsize=(10, 6))
                shap.waterfall_plot(shap.Explanation(values=shap_values[i],
                                                     base_values=explainer.expected_value,
                                                     data=X_sample.iloc[i],
                                                     feature_names=X_sample.columns.tolist()),
                                    show=False)
                plt.title(f'{best_model_name} - SHAP瀑布图(样本{i + 1})')
                plt.tight_layout()
                plt.savefig(self.plots_dir / f"{best_model_name}_shap_waterfall_{i + 1}.png",
                            dpi=300, bbox_inches='tight')
                plt.close()

            print("   ✅ SHAP分析完成")

        except Exception as e:
            print(f"   ⚠️ SHAP分析失败: {e}")

    def _plot_learning_curves(self):
        """绘制学习曲线"""
        # 这里可以添加学习曲线绘制的代码
        # 由于需要额外的训练过程，暂时省略
        pass

    def _plot_interactive_3d(self):
        """绘制交互式3D图"""
        try:
            import plotly.offline as pyo

            # 创建3D散点图
            if self.results:
                best_model_name = list(self.results.keys())[0]
                metrics = self.results[best_model_name]

                fig = go.Figure()

                # 添加预测vs实际散点
                fig.add_trace(go.Scatter3d(
                    x=metrics['y_true'],
                    y=metrics['y_pred'],
                    z=np.abs(metrics['y_true'] - metrics['y_pred']),
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=np.abs(metrics['y_true'] - metrics['y_pred']),
                        colorscale='Viridis',
                        opacity=0.6,
                        colorbar=dict(title="绝对误差")
                    ),
                    name='预测点'
                ))

                # 添加理想预测平面
                x_range = np.linspace(min(metrics['y_true']), max(metrics['y_true']), 10)
                y_range = np.linspace(min(metrics['y_pred']), max(metrics['y_pred']), 10)
                X, Y = np.meshgrid(x_range, y_range)
                Z = np.zeros_like(X)

                fig.add_trace(go.Surface(
                    x=X, y=Y, z=Z,
                    colorscale='Greys',
                    opacity=0.2,
                    showscale=False,
                    name='理想预测平面'
                ))

                fig.update_layout(
                    title=f'{best_model_name} - 3D预测可视化',
                    scene=dict(
                        xaxis_title='实际值',
                        yaxis_title='预测值',
                        zaxis_title='绝对误差'
                    ),
                    width=900,
                    height=700
                )

                # 保存为HTML文件
                plot_path = self.plots_dir / f"{best_model_name}_3d_visualization.html"
                fig.write_html(str(plot_path))

                print(f"   ✅ 3D交互图已保存: {plot_path}")

        except Exception as e:
            print(f"   ⚠️ 3D图生成失败: {e}")

    def save_summary_report(self, results_df: pd.DataFrame, training_results: Dict):
        """保存总结报告"""
        print("=" * 70)
        print("📄 生成总结报告...")

        report = {
            'training_summary': {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_models': len(self.models),
                'features_used': self.feature_names,
                'target_column': self.target_column,
                'feature_engineering': self.use_feature_engineering,
                'shap_analysis': self.use_shap
            },
            'model_performance': results_df.to_dict('records'),
            'best_model': {
                'name': results_df.iloc[0]['Model'] if not results_df.empty else None,
                'rmse': results_df.iloc[0]['RMSE'] if not results_df.empty else None,
                'r2': results_df.iloc[0]['R2'] if not results_df.empty else None
            },
            'training_details': {}
        }

        # 添加训练细节
        for model_name, result in training_results.items():
            report['training_details'][model_name] = {
                'best_params': result.get('best_params', {}),
                'best_rmse': float(result.get('best_rmse', 0)),
                'training_time': float(result.get('training_time', 0)),
                'cv_folds': result.get('cv_folds', 5),
                'n_samples_used': result.get('n_samples_used', 0)
            }

        # 保存JSON报告
        report_path = self.output_dir / "training_summary.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 生成文本报告
        text_report = self._generate_text_report(results_df, training_results)
        text_path = self.output_dir / "training_report.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_report)

        print(f"✅ 总结报告已保存:")
        print(f"   JSON报告: {report_path}")
        print(f"   文本报告: {text_path}")

    def _generate_text_report(self, results_df: pd.DataFrame, training_results: Dict) -> str:
        """生成文本格式的报告"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("机器学习模型训练总结报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"特征数: {len(self.feature_names)}")
        report_lines.append(f"特征工程: {'启用' if self.use_feature_engineering else '禁用'}")
        report_lines.append(f"SHAP分析: {'启用' if self.use_shap else '禁用'}")
        report_lines.append("")

        report_lines.append("模型性能排名:")
        report_lines.append("-" * 80)
        report_lines.append(f"{'排名':<4} {'模型':<15} {'RMSE':<12} {'R²':<10} {'MAE':<12} {'训练时间':<12}")
        report_lines.append("-" * 80)

        for i, (_, row) in enumerate(results_df.iterrows()):
            model_name = row['Model']
            training_time = training_results.get(model_name, {}).get('training_time', 0)
            report_lines.append(f"{i + 1:<4} {model_name:<15} {row['RMSE']:<12.6f} "
                                f"{row['R2']:<10.4f} {row['MAE']:<12.6f} {training_time:<12.1f}")

        report_lines.append("")
        report_lines.append("最佳模型详细信息:")
        report_lines.append("-" * 80)
        if not results_df.empty:
            best_model = results_df.iloc[0]
            report_lines.append(f"模型名称: {best_model['Model']}")
            report_lines.append(f"描述: {best_model['Description']}")
            report_lines.append(f"RMSE: {best_model['RMSE']:.6f}")
            report_lines.append(f"MAE: {best_model['MAE']:.6f}")
            report_lines.append(f"R²: {best_model['R2']:.4f}")
            report_lines.append(f"MAPE: {best_model['MAPE']:.2f}%")
            report_lines.append(f"偏差: {best_model['Bias']:.6f}")

        report_lines.append("")
        report_lines.append("特征重要性总结:")
        report_lines.append("-" * 80)
        for model_name, importance_df in self.feature_importance.items():
            if importance_df is not None and not importance_df.empty:
                report_lines.append(f"{model_name} - 前5个重要特征:")
                for j, (_, row) in enumerate(importance_df.head(5).iterrows()):
                    report_lines.append(f"  {j + 1}. {row['feature']}: {row['importance']:.4f}")
                report_lines.append("")

        report_lines.append("=" * 80)

        return "\n".join(report_lines)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='增强版机器学习训练系统')
    parser.add_argument('--sample', type=float, default=1.0,
                        help='数据采样比例 (0.01-1.0)')
    parser.add_argument('--models', type=str, default='all',
                        help='要训练的模型: all/fast/模型列表')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='交叉验证折数')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='并行作业数')
    parser.add_argument('--no_feature_engineering', action='store_true',
                        help='禁用特征工程')
    parser.add_argument('--no_shap', action='store_true',
                        help='禁用SHAP分析')
    parser.add_argument('--no_advanced_plots', action='store_true',
                        help='禁用高级图表')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='自定义输出目录')
    parser.add_argument('--fast_mode', action='store_true',
                        help='快速模式：只训练最快模型')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🚀 增强版机器学习训练系统")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据采样: {args.sample * 100:.1f}%")
    print(f"测试集比例: {args.test_size * 100:.0f}%")
    print(f"交叉验证: {args.cv_folds}折")
    print(f"并行作业: {args.n_jobs}")
    print(f"特征工程: {'禁用' if args.no_feature_engineering else '启用'}")
    print(f"SHAP分析: {'禁用' if args.no_shap else '启用'}")
    print(f"高级图表: {'禁用' if args.no_advanced_plots else '启用'}")
    print(f"快速模式: {'是' if args.fast_mode else '否'}")
    print("=" * 80)

    try:
        # 创建模型实例
        ml = EnhancedMLModels(
            use_feature_engineering=not args.no_feature_engineering,
            use_shap=not args.no_shap,
            use_advanced_plots=not args.no_advanced_plots,
            n_jobs=args.n_jobs
        )

        # 加载数据
        data = ml.load_data(sample_fraction=args.sample)

        # 准备数据
        X_train, X_test, y_train, y_test = ml.prepare_data(data, test_size=args.test_size)

        # 确定要训练的模型
        if args.models == 'all':
            if args.fast_mode:
                models_to_train = ['XGBoost', 'LightGBM', 'Ridge', 'Lasso']
                print(f"快速模式：只训练 {len(models_to_train)} 个最快模型")
            else:
                models_to_train = list(ml.model_configs.keys())
        elif args.models == 'fast':
            models_to_train = ['XGBoost', 'LightGBM', 'Ridge', 'Lasso']
        else:
            models_to_train = [m.strip() for m in args.models.split(',')]

        # 训练模型
        training_results = ml.train_models(
            X_train, y_train,
            models_to_train=models_to_train,
            cv_folds=args.cv_folds
        )

        # 评估模型
        results_df = ml.evaluate_models(X_test, y_test)

        # 生成可视化
        ml.generate_visualizations(X_test, y_test, results_df)

        # 保存总结报告
        ml.save_summary_report(results_df, training_results)

        # 打印最终总结
        print("\n" + "=" * 80)
        print("🎉 训练完成!")
        print("=" * 80)
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"输出目录: {ml.output_dir}")

        if not results_df.empty:
            print(f"\n📊 最终模型排名:")
            print("-" * 80)
            print(f"{'排名':<4} {'模型':<15} {'RMSE':<12} {'R²':<10} {'MAE':<12}")
            print("-" * 80)
            for i, (_, row) in enumerate(results_df.iterrows()):
                print(f"{i + 1:<4} {row['Model']:<15} {row['RMSE']:<12.6f} "
                      f"{row['R2']:<10.4f} {row['MAE']:<12.6f}")

        print("\n📁 生成的文件:")
        print(f"  模型文件: {ml.models_dir}/")
        print(f"  图表文件: {ml.plots_dir}/")
        print(f"  表格文件: {ml.tables_dir}/")
        print(f"  总结报告: {ml.output_dir}/training_summary.json")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())