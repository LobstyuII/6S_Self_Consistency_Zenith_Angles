# ==================== MLs.py ====================
"""
机器学习模型训练与评估模块
用于训练和评估多种机器学习模型预测6S模型误差
"""
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import pickle
import json
import time
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

# 高级模型
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("警告: XGBoost未安装，将跳过XGBoost模型")
    print("Try: conda install -c conda-forge xgboost")

try:
    import lightgbm as lgb

    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("警告: LightGBM未安装，将跳过LightGBM模型")
    print("Try: conda install -c conda-forge lightgbm")

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from config import ExperimentConfig
from utils import setup_logger, calculate_statistics


# ==================== 自定义进度条类 ====================
class ProgressTracker:
    """跟踪和显示训练进度的类"""

    def __init__(self, total_steps: int = 100, description: str = "进度"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.step_times = []
        self.phases = {}

    def start_phase(self, phase_name: str, total_substeps: int = 1):
        """开始一个新阶段"""
        self.phases[phase_name] = {
            'start': time.time(),
            'total': total_substeps,
            'current': 0,
            'completed': False
        }
        print(f"\n[开始阶段] {phase_name}")

    def update_phase(self, phase_name: str, progress: float = None, message: str = None):
        """更新阶段进度"""
        if phase_name in self.phases:
            phase = self.phases[phase_name]
            if progress is not None:
                phase['current'] = int(phase['total'] * progress)

            elapsed = time.time() - phase['start']
            eta = None
            if phase['current'] > 0:
                avg_time = elapsed / phase['current']
                eta = avg_time * (phase['total'] - phase['current'])

            status_line = f"  [{phase_name}] "
            if phase['total'] > 1:
                status_line += f"{phase['current']}/{phase['total']} "

            if message:
                status_line += f"- {message}"

            if eta:
                status_line += f" (预计剩余: {eta:.1f}秒)"

            print(status_line)

    def complete_phase(self, phase_name: str, message: str = None):
        """完成一个阶段"""
        if phase_name in self.phases:
            self.phases[phase_name]['completed'] = True
            elapsed = time.time() - self.phases[phase_name]['start']
            status = f"[完成阶段] {phase_name} - 耗时: {elapsed:.2f}秒"
            if message:
                status += f" ({message})"
            print(status)

    def step(self, message: str = None):
        """更新总体进度"""
        self.current_step += 1
        progress = (self.current_step / self.total_steps) * 100
        elapsed = time.time() - self.start_time

        if self.current_step > 1:
            avg_time = elapsed / self.current_step
            eta = avg_time * (self.total_steps - self.current_step)
        else:
            eta = None

        progress_bar = self._create_progress_bar(progress)
        status = f"\r{progress_bar} {progress:.1f}%"

        if eta:
            status += f" | 预计剩余: {eta:.1f}秒"

        if message:
            status += f" | {message}"

        print(status, end='', flush=True)

        if self.current_step == self.total_steps:
            print()  # 换行

    def _create_progress_bar(self, progress: float, length: int = 30):
        """创建文本进度条"""
        filled = int(length * progress / 100)
        bar = '█' * filled + '░' * (length - filled)
        return f"[{bar}]"

    def print_summary(self):
        """打印进度摘要"""
        print("\n" + "=" * 60)
        print("训练进度摘要:")
        print("=" * 60)

        total_elapsed = time.time() - self.start_time
        print(f"总耗时: {total_elapsed:.2f}秒")

        for phase_name, phase_info in self.phases.items():
            if phase_info['completed']:
                elapsed = phase_info.get('elapsed', 0)
                print(f"  {phase_name}: 完成 (耗时: {elapsed:.2f}秒)")
            else:
                print(f"  {phase_name}: 进行中")

        print("=" * 60)


# ==================== 自定义GridSearchCV回调 ====================
class GridSearchCallback:
    """GridSearchCV进度回调"""

    def __init__(self, n_folds: int, n_params: int):
        self.n_folds = n_folds
        self.n_params = n_params
        self.total_fits = n_folds * n_params
        self.current_fit = 0
        self.start_time = time.time()
        self.fold_start_time = None
        self.param_start_time = None

    def on_fold_start(self, fold_idx: int):
        """开始新折叠"""
        self.fold_start_time = time.time()
        print(f"\n  折叠 {fold_idx + 1}/{self.n_folds}:")

    def on_fold_end(self, fold_idx: int, score: float = None):
        """折叠结束"""
        elapsed = time.time() - self.fold_start_time
        status = f"    完成 (耗时: {elapsed:.1f}秒)"
        if score is not None:
            status += f", 得分: {score:.4f}"
        print(status)

    def on_param_start(self, param_idx: int, params: dict):
        """开始新参数组合"""
        self.param_start_time = time.time()
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()][:3])  # 只显示前3个参数
        if len(params) > 3:
            param_str += "..."
        print(f"    参数 {param_idx + 1}/{self.n_params}: {param_str}")

    def on_param_end(self, param_idx: int, score: float = None):
        """参数组合结束"""
        elapsed = time.time() - self.param_start_time
        self.current_fit += self.n_folds

        progress = (self.current_fit / self.total_fits) * 100
        elapsed_total = time.time() - self.start_time

        status = f"      完成 (耗时: {elapsed:.1f}秒)"
        if score is not None:
            status += f", 平均得分: {score:.4f}"

        # 估计剩余时间
        if self.current_fit > 0:
            avg_time = elapsed_total / self.current_fit
            remaining = avg_time * (self.total_fits - self.current_fit)
            status += f" | 预计剩余: {remaining:.1f}秒"

        print(status)


# ==================== 主要机器学习类 ====================
class MachineLearningModels:
    """机器学习模型训练与评估类"""

    def __init__(self, config: ExperimentConfig = None, logger=None, show_progress: bool = True):
        self.config = config or ExperimentConfig
        self.logger = logger or setup_logger('MLModels')
        self.show_progress = show_progress
        self.progress_tracker = None

        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.results = {}

        # 特征列名
        self.feature_columns = [
            'sza', 'vza', 'rho_true', 'aod550', 'h2o', 'o3',
            'wavelength', 'airmass_sza', 'airmass_vza', 'total_airmass'
        ]

        # 目标列名
        self.target_column = 'error_absolute'

        # 模型配置
        self.model_configs = self._initialize_model_configs()

        # 输出目录
        self.output_dir = self.config.MODELS_DIR / "ml_models"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"机器学习模型初始化完成，输出目录: {self.output_dir}")

        if self.show_progress:
            self.progress_tracker = ProgressTracker(total_steps=12, description="机器学习训练流程")

    def _initialize_model_configs(self) -> Dict[str, Dict]:
        """初始化模型配置"""
        configs = {
            'RandomForest': {
                'model_class': RandomForestRegressor,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2'],
                    'random_state': [self.config.RANDOM_SEED]
                },
                'description': '随机森林回归',
                'show_progress': True
            },
            'GradientBoosting': {
                'model_class': GradientBoostingRegressor,
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'random_state': [self.config.RANDOM_SEED]
                },
                'description': '梯度提升回归树',
                'show_progress': True
            },
            'ExtraTrees': {
                'model_class': ExtraTreesRegressor,
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'random_state': [self.config.RANDOM_SEED]
                },
                'description': '极端随机树',
                'show_progress': True
            },
            'SVR_RBF': {
                'model_class': SVR,
                'params': {
                    'kernel': ['rbf'],
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.01, 0.1],
                    'epsilon': [0.01, 0.1, 0.2]
                },
                'description': '支持向量回归(RBF核)',
                'show_progress': True
            },
            'MLP': {
                'model_class': MLPRegressor,
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive'],
                    'max_iter': [1000],
                    'random_state': [self.config.RANDOM_SEED]
                },
                'description': '多层感知器',
                'show_progress': True
            },
            'Ridge': {
                'model_class': Ridge,
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'random_state': [self.config.RANDOM_SEED]
                },
                'description': '岭回归',
                'show_progress': True
            },
            'Lasso': {
                'model_class': Lasso,
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1.0],
                    'random_state': [self.config.RANDOM_SEED]
                },
                'description': 'Lasso回归',
                'show_progress': True
            }
        }

        # 添加XGBoost（如果可用）
        if XGB_AVAILABLE:
            configs['XGBoost'] = {
                'model_class': xgb.XGBRegressor,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'random_state': [self.config.RANDOM_SEED]
                },
                'description': '极端梯度提升',
                'show_progress': True
            }

        # 添加LightGBM（如果可用）
        if LGB_AVAILABLE:
            configs['LightGBM'] = {
                'model_class': lgb.LGBMRegressor,
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 7, 10, -1],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'num_leaves': [31, 50, 100],
                    'random_state': [self.config.RANDOM_SEED]
                },
                'description': 'LightGBM',
                'show_progress': True
            }

        return configs

    def _print_model_header(self, model_name: str, config: Dict):
        """打印模型训练头部信息"""
        print("\n" + "=" * 70)
        print(f"训练模型: {model_name}")
        print(f"描述: {config['description']}")
        print(f"参数空间大小: {self._count_param_combinations(config['params'])} 种组合")
        print("=" * 70)

    def _count_param_combinations(self, param_grid: Dict) -> int:
        """计算参数组合总数"""
        total = 1
        for values in param_grid.values():
            total *= len(values)
        return total

    def load_all_data(self) -> pd.DataFrame:
        """加载所有波段的数据并合并"""
        if self.show_progress:
            self.progress_tracker.start_phase("加载数据", 1)

        data_files = list(self.config.DATA_DIR.glob("simulation_results_*.nc"))

        if not data_files:
            raise FileNotFoundError(f"在目录 {self.config.DATA_DIR} 中找不到数据文件")

        self.logger.info(f"找到 {len(data_files)} 个数据文件")

        if self.show_progress:
            self.progress_tracker.update_phase("加载数据", 0.1, f"找到 {len(data_files)} 个文件")

        all_data = []
        for i, file in enumerate(data_files):
            try:
                if self.show_progress:
                    self.progress_tracker.update_phase("加载数据",
                                                       (i + 1) / len(data_files),
                                                       f"加载文件 {i + 1}/{len(data_files)}: {file.name}")

                # 使用xarray加载NetCDF文件
                ds = xr.open_dataset(file)

                # 获取波段名称
                band_name = file.stem.replace("simulation_results_", "").replace("_parallel", "")

                # 转换为DataFrame
                df = ds.to_dataframe().reset_index(drop=True)

                # 添加波段信息
                if 'band' not in df.columns:
                    df['band'] = band_name

                # 获取波长
                if band_name in self.config.BANDS:
                    df['wavelength'] = self.config.BANDS[band_name]['wavelength']

                # 只保留成功的样本
                if 'success' in df.columns:
                    if df['success'].dtype == bool:
                        df = df[df['success']]
                    else:
                        df = df[df['success'] == 1]

                # 只保留闭合循环成功的样本
                if 'closed_loop_success' in df.columns:
                    if df['closed_loop_success'].dtype == bool:
                        df = df[df['closed_loop_success']]
                    else:
                        df = df[df['closed_loop_success'] == 1]

                all_data.append(df)
                ds.close()

                self.logger.info(f"加载文件 {file.name}: {len(df)} 个样本")

            except Exception as e:
                self.logger.error(f"加载文件 {file} 时出错: {e}")

        if not all_data:
            raise ValueError("没有成功加载任何数据")

        # 合并所有数据
        combined_data = pd.concat(all_data, ignore_index=True)

        self.logger.info(f"合并后的数据集: {len(combined_data)} 个样本, {combined_data.shape[1]} 个特征")

        if self.show_progress:
            self.progress_tracker.complete_phase("加载数据",
                                                 f"加载 {len(combined_data)} 个样本")
            self.progress_tracker.step("数据加载完成")

        # 检查必需的特征列
        missing_features = [col for col in self.feature_columns if col not in combined_data.columns]
        if missing_features:
            self.logger.warning(f"缺少以下特征: {missing_features}")

        # 检查目标列
        if self.target_column not in combined_data.columns:
            raise ValueError(f"目标列 '{self.target_column}' 不存在于数据中")

        return combined_data

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """数据预处理"""
        if self.show_progress:
            self.progress_tracker.start_phase("数据预处理", 3)

        # 创建特征数据
        X = df.copy()

        if self.show_progress:
            self.progress_tracker.update_phase("数据预处理", 0.33, "准备特征数据")

        # 确保所有特征列都存在
        for col in self.feature_columns:
            if col not in X.columns:
                if col == 'airmass_sza':
                    X['airmass_sza'] = 1.0 / np.cos(np.radians(X['sza']))
                elif col == 'airmass_vza':
                    X['airmass_vza'] = 1.0 / np.cos(np.radians(X['vza']))
                elif col == 'total_airmass':
                    X['total_airmass'] = X.get('airmass_sza', 1.0 / np.cos(np.radians(X['sza']))) + \
                                         X.get('airmass_vza', 1.0 / np.cos(np.radians(X['vza'])))
                else:
                    self.logger.warning(f"特征 {col} 不存在，使用默认值")

        # 选择特征
        X_features = X[self.feature_columns].copy()

        if self.show_progress:
            self.progress_tracker.update_phase("数据预处理", 0.66, "处理缺失值")

        # 处理缺失值
        X_features = X_features.fillna(X_features.median())

        # 目标变量
        y = X[self.target_column].copy()

        if self.show_progress:
            self.progress_tracker.update_phase("数据预处理", 0.8, "处理异常值")

        # 移除异常值（基于IQR）
        if len(y) > 0:
            Q1 = y.quantile(0.25)
            Q3 = y.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            mask = (y >= lower_bound) & (y <= upper_bound)
            X_features = X_features[mask]
            y = y[mask]

            removed_count = len(X) - len(y)
            self.logger.info(f"移除异常值后: {len(y)} 个样本 (移除 {removed_count} 个)")

        if self.show_progress:
            self.progress_tracker.complete_phase("数据预处理",
                                                 f"处理 {len(y)} 个样本")
            self.progress_tracker.step("数据预处理完成")

        return X_features, y

    def train_test_split_data(self, X: pd.DataFrame, y: pd.Series,
                              test_size: float = 0.2, random_state: int = 42):
        """划分训练集和测试集"""
        if self.show_progress:
            self.progress_tracker.start_phase("数据划分", 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.logger.info(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

        if self.show_progress:
            self.progress_tracker.update_phase("数据划分", 0.5, "划分数据集")

        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers['standard'] = scaler

        # 转换为DataFrame以保持列名
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        if self.show_progress:
            self.progress_tracker.complete_phase("数据划分",
                                                 f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")
            self.progress_tracker.step("数据划分完成")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_model_with_progress(self, model_name: str, X_train: pd.DataFrame,
                                  y_train: pd.Series, cv_folds: int = 5,
                                  n_jobs: int = -1) -> Dict:
        """训练单个模型（带进度显示）"""
        if model_name not in self.model_configs:
            raise ValueError(f"模型 {model_name} 不存在于配置中")

        config = self.model_configs[model_name]

        # 显示模型头部信息
        if self.show_progress:
            self._print_model_header(model_name, config)

        start_time = time.time()

        # 创建模型
        model_class = config['model_class']
        param_grid = config['params']

        # 计算参数组合数
        n_param_combinations = self._count_param_combinations(param_grid)

        if self.show_progress:
            print(f"\n开始超参数搜索:")
            print(f"  交叉验证折叠数: {cv_folds}")
            print(f"  参数组合数: {n_param_combinations}")
            print(f"  总训练次数: {cv_folds * n_param_combinations}")
            print(f"  使用进程数: {n_jobs if n_jobs != -1 else '自动'}")
            print()

        # 创建进度回调
        if self.show_progress and config.get('show_progress', False):
            callback = GridSearchCallback(cv_folds, n_param_combinations)
        else:
            callback = None

        # 自定义评分函数，用于显示进度
        def custom_scorer(estimator, X, y):
            score = -mean_squared_error(y, estimator.predict(X))  # 负MSE

            # 如果有回调，更新进度
            if callback:
                # 这里我们无法直接知道当前是哪个参数和折叠，所以只能简单更新
                pass

            return score

        # 使用网格搜索进行超参数调优
        grid_search = GridSearchCV(
            estimator=model_class(),
            param_grid=param_grid,
            cv=KFold(n_splits=cv_folds, shuffle=True, random_state=self.config.RANDOM_SEED),
            scoring='neg_mean_squared_error',
            n_jobs=n_jobs,
            verbose=0  # 我们自己控制输出
        )

        # 训练模型（带进度显示）
        if self.show_progress:
            print("开始训练...")

        # 手动模拟进度
        if self.show_progress and callback:
            # 简单模拟进度
            total_iterations = cv_folds * n_param_combinations

            # 由于GridSearchCV不提供进度回调，我们使用简单的时间估计
            print("  进度: [░░░░░░░░░░░░░░░░░░░░] 0.0%", end='', flush=True)

            # 保存原始fit方法
            original_fit = grid_search.fit

            def fit_with_progress(X, y):
                fit_start = time.time()
                result = original_fit(X, y)
                fit_time = time.time() - fit_start

                # 模拟完成
                print(f"\r  进度: [████████████████████] 100.0% - 耗时: {fit_time:.1f}秒")

                return result

            grid_search.fit = fit_with_progress

        # 训练模型
        grid_search.fit(X_train, y_train)

        # 获取最佳模型
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_  # 转换为正数

        if self.show_progress:
            print(f"\n超参数搜索完成!")
            print(f"  最佳参数: {best_params}")
            print(f"  最佳MSE: {best_score:.6f}")
            print(f"  最佳RMSE: {np.sqrt(best_score):.6f}")

        # 计算交叉验证得分
        if self.show_progress:
            print(f"\n计算交叉验证得分...")

        cv_scores = cross_val_score(
            best_model, X_train, y_train,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=n_jobs
        )
        cv_rmse = np.sqrt(-cv_scores)

        if self.show_progress:
            print(f"  交叉验证RMSE: {cv_rmse.mean():.6f} ± {cv_rmse.std():.6f}")

        # 保存模型
        self.models[model_name] = best_model

        # 计算特征重要性（如果模型支持）
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            self.feature_importance[model_name] = feature_importance

            if self.show_progress:
                print(f"\n特征重要性 (Top 5):")
                for _, row in feature_importance.head(5).iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")

        elapsed_time = time.time() - start_time

        result = {
            'model': best_model,
            'best_params': best_params,
            'best_rmse': np.sqrt(best_score),
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'feature_importance': feature_importance,
            'training_time': elapsed_time,
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0]
        }

        if self.show_progress:
            print(f"\n模型 {model_name} 训练完成!")
            print(f"  总耗时: {elapsed_time:.2f}秒")
            print("=" * 70)

        return result

    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                    cv_folds: int = 5, n_jobs: int = -1) -> Dict:
        """训练单个模型（兼容接口）"""
        if self.show_progress:
            return self.train_model_with_progress(model_name, X_train, y_train, cv_folds, n_jobs)
        else:
            # 原始训练逻辑（不带进度显示）
            return self._train_model_basic(model_name, X_train, y_train, cv_folds, n_jobs)

    def _train_model_basic(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                           cv_folds: int = 5, n_jobs: int = -1) -> Dict:
        """基础训练函数（不带进度显示）"""
        if model_name not in self.model_configs:
            raise ValueError(f"模型 {model_name} 不存在于配置中")

        config = self.model_configs[model_name]
        self.logger.info(f"训练模型: {model_name} ({config['description']})")

        start_time = time.time()

        # 创建模型
        model_class = config['model_class']
        param_grid = config['params']

        # 使用网格搜索进行超参数调优
        grid_search = GridSearchCV(
            estimator=model_class(),
            param_grid=param_grid,
            cv=KFold(n_splits=cv_folds, shuffle=True, random_state=self.config.RANDOM_SEED),
            scoring='neg_mean_squared_error',
            n_jobs=n_jobs,
            verbose=0
        )

        # 训练模型
        grid_search.fit(X_train, y_train)

        # 获取最佳模型
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_  # 转换为正数

        # 计算交叉验证得分
        cv_scores = cross_val_score(
            best_model, X_train, y_train,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=n_jobs
        )
        cv_rmse = np.sqrt(-cv_scores)

        # 保存模型
        self.models[model_name] = best_model

        # 计算特征重要性（如果模型支持）
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            self.feature_importance[model_name] = feature_importance

        elapsed_time = time.time() - start_time

        result = {
            'model': best_model,
            'best_params': best_params,
            'best_rmse': np.sqrt(best_score),
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'feature_importance': feature_importance,
            'training_time': elapsed_time,
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0]
        }

        self.logger.info(f"模型 {model_name} 训练完成: "
                         f"最佳RMSE={result['best_rmse']:.4f}, "
                         f"CV RMSE={result['cv_rmse_mean']:.4f}±{result['cv_rmse_std']:.4f}, "
                         f"耗时={elapsed_time:.2f}秒")

        return result

    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """评估模型性能"""
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 尚未训练")

        if self.show_progress:
            print(f"\n评估模型: {model_name}")

        model = self.models[model_name]

        # 预测
        if self.show_progress:
            print("  进行预测...")

        y_pred = model.predict(X_test)

        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # 计算相对误差
        abs_errors = np.abs(y_test - y_pred)
        relative_errors = abs_errors / np.abs(y_test)
        relative_errors = relative_errors.replace([np.inf, -np.inf], np.nan)
        relative_errors = relative_errors.dropna()

        stats = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'max_error': np.max(abs_errors),
            'mean_abs_error': np.mean(abs_errors),
            'median_abs_error': np.median(abs_errors),
            'std_abs_error': np.std(abs_errors),
            'mean_relative_error': np.mean(relative_errors) if len(relative_errors) > 0 else np.nan,
            'median_relative_error': np.median(relative_errors) if len(relative_errors) > 0 else np.nan,
            'n_samples': len(y_test),
            'predictions': y_pred,
            'actual': y_test.values
        }

        self.results[model_name] = stats

        if self.show_progress:
            print(f"  评估完成:")
            print(f"    RMSE: {rmse:.6f}")
            print(f"    MAE:  {mae:.6f}")
            print(f"    R²:   {r2:.4f}")

        return stats

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                         models_to_train: List[str] = None, n_jobs: int = -1):
        """训练所有模型"""
        if models_to_train is None:
            models_to_train = list(self.model_configs.keys())

        if self.show_progress:
            self.progress_tracker.start_phase("模型训练", len(models_to_train))
            print(f"\n开始训练 {len(models_to_train)} 个模型")
            print("=" * 70)

        training_results = {}
        for i, model_name in enumerate(models_to_train):
            try:
                if self.show_progress:
                    self.progress_tracker.update_phase("模型训练",
                                                       (i) / len(models_to_train),
                                                       f"训练模型 {i + 1}/{len(models_to_train)}: {model_name}")

                result = self.train_model(model_name, X_train, y_train, n_jobs=n_jobs)
                training_results[model_name] = result

                # 保存模型到文件
                model_path = self.output_dir / f"{model_name}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(self.models[model_name], f)

                # 保存特征重要性
                if model_name in self.feature_importance:
                    importance_path = self.output_dir / f"{model_name}_feature_importance.csv"
                    self.feature_importance[model_name].to_csv(importance_path, index=False)

                if self.show_progress:
                    self.progress_tracker.update_phase("模型训练",
                                                       (i + 1) / len(models_to_train),
                                                       f"完成模型 {model_name}")

            except Exception as e:
                self.logger.error(f"训练模型 {model_name} 时出错: {e}")
                if self.show_progress:
                    print(f"  ✗ 模型 {model_name} 训练失败: {e}")

        if self.show_progress:
            self.progress_tracker.complete_phase("模型训练",
                                                 f"完成 {len(training_results)} 个模型")
            self.progress_tracker.step("模型训练完成")

        self.logger.info("所有模型训练完成")
        return training_results

    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """评估所有已训练的模型"""
        if self.show_progress:
            self.progress_tracker.start_phase("模型评估", len(self.models))
            print(f"\n评估 {len(self.models)} 个已训练模型")

        evaluation_results = []

        for i, model_name in enumerate(self.models.keys()):
            try:
                if self.show_progress:
                    self.progress_tracker.update_phase("模型评估",
                                                       (i) / len(self.models),
                                                       f"评估模型 {i + 1}/{len(self.models)}: {model_name}")

                stats = self.evaluate_model(model_name, X_test, y_test)

                # 添加到结果列表
                result_row = {
                    'Model': model_name,
                    'Description': self.model_configs.get(model_name, {}).get('description', ''),
                    'RMSE': stats['rmse'],
                    'MAE': stats['mae'],
                    'R²': stats['r2'],
                    'Max_Error': stats['max_error'],
                    'Mean_Abs_Error': stats['mean_abs_error'],
                    'Std_Abs_Error': stats['std_abs_error'],
                    'Mean_Relative_Error': stats['mean_relative_error'],
                    'n_Samples': stats['n_samples']
                }

                evaluation_results.append(result_row)

            except Exception as e:
                self.logger.error(f"评估模型 {model_name} 时出错: {e}")
                if self.show_progress:
                    print(f"  ✗ 模型 {model_name} 评估失败: {e}")

        # 创建结果DataFrame
        results_df = pd.DataFrame(evaluation_results)

        # 按RMSE排序
        results_df = results_df.sort_values('RMSE')

        # 保存结果
        results_path = self.output_dir / "model_evaluation_results.csv"
        results_df.to_csv(results_path, index=False)

        if self.show_progress:
            self.progress_tracker.complete_phase("模型评估",
                                                 f"评估 {len(evaluation_results)} 个模型")
            self.progress_tracker.step("模型评估完成")
            print(f"\n评估结果已保存到: {results_path}")

        return results_df

    def compare_models(self, results_df: pd.DataFrame):
        """比较模型性能"""
        if self.show_progress:
            self.progress_tracker.start_phase("模型比较", 1)
            print(f"\n生成模型比较图表...")

        # 创建比较图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. RMSE和MAE比较
        ax = axes[0, 0]
        x = np.arange(len(results_df))
        width = 0.35

        ax.bar(x - width / 2, results_df['RMSE'], width, label='RMSE', alpha=0.8)
        ax.bar(x + width / 2, results_df['MAE'], width, label='MAE', alpha=0.8)
        ax.set_xlabel('模型')
        ax.set_ylabel('误差')
        ax.set_title('模型误差比较 (RMSE vs MAE)')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. R²比较
        ax = axes[0, 1]
        colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
        bars = ax.bar(results_df['Model'], results_df['R²'], color=colors, alpha=0.8)
        ax.set_xlabel('模型')
        ax.set_ylabel('R²')
        ax.set_title('模型决定系数 (R²) 比较')
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        # 添加数值标签
        for bar, value in zip(bars, results_df['R²']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        # 3. 最大误差比较
        ax = axes[1, 0]
        ax.bar(results_df['Model'], results_df['Max_Error'], alpha=0.8)
        ax.set_xlabel('模型')
        ax.set_ylabel('最大绝对误差')
        ax.set_title('模型最大绝对误差比较')
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        # 4. 相对误差比较
        ax = axes[1, 1]
        if results_df['Mean_Relative_Error'].notna().any():
            ax.bar(results_df['Model'], results_df['Mean_Relative_Error'], alpha=0.8)
            ax.set_xlabel('模型')
            ax.set_ylabel('平均相对误差')
            ax.set_title('模型平均相对误差比较')
            ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无相对误差数据', ha='center', va='center', fontsize=12)
            ax.set_title('相对误差比较 (无数据)')

        plt.tight_layout()

        # 保存图表
        comparison_path = self.output_dir / "model_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()

        if self.show_progress:
            self.progress_tracker.complete_phase("模型比较", "生成比较图表")
            self.progress_tracker.step("模型比较完成")
            print(f"模型比较图表已保存到: {comparison_path}")

        # 输出最佳模型
        best_model = results_df.iloc[0]

        if self.show_progress:
            print(f"\n{'=' * 60}")
            print(f"最佳模型: {best_model['Model']} ({best_model['Description']})")
            print(f"RMSE: {best_model['RMSE']:.6f}")
            print(f"MAE: {best_model['MAE']:.6f}")
            print(f"R²: {best_model['R²']:.4f}")
            print(f"{'=' * 60}")

        return best_model

    def plot_predictions_vs_actual(self, model_name: str, y_test: pd.Series, y_pred: np.ndarray):
        """绘制预测值与实际值的对比图"""
        if self.show_progress:
            print(f"\n为模型 {model_name} 生成预测对比图...")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 1. 散点图
        ax = axes[0]
        ax.scatter(y_test, y_pred, alpha=0.5, s=10)

        # 添加对角线
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        ax.set_xlabel('实际误差')
        ax.set_ylabel('预测误差')
        ax.set_title(f'{model_name}: 预测值 vs 实际值')
        ax.grid(True, alpha=0.3)

        # 添加统计信息
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.4f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 2. 残差图
        ax = axes[1]
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)

        ax.set_xlabel('预测误差')
        ax.set_ylabel('残差 (实际 - 预测)')
        ax.set_title(f'{model_name}: 残差图')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        plot_path = self.output_dir / f"{model_name}_predictions_vs_actual.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        if self.show_progress:
            print(f"  图表已保存到: {plot_path}")

    def create_lookup_table(self, model_name: str, resolution: Dict = None):
        """创建误差查找表(LUT)"""
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 尚未训练")

        if self.show_progress:
            self.progress_tracker.start_phase("创建查找表", 1)
            print(f"\n开始创建误差查找表(LUT)，使用模型: {model_name}")

        if resolution is None:
            # 默认分辨率
            resolution = {
                'sza': np.arange(0, 86, 5),  # 0-85度，步长5度
                'vza': np.arange(0, 76, 5),  # 0-75度，步长5度
                'rho_true': np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]),
                'aod550': np.array([0.05, 0.10, 0.20, 0.30, 0.50, 1.00]),
                'h2o': np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0]),
                'o3': np.array([0.2, 0.25, 0.30, 0.35, 0.40]),
                'wavelength': np.array([band['wavelength'] for band in self.config.BANDS.values()])
            }

        # 生成所有参数组合
        from itertools import product
        param_names = list(resolution.keys())
        param_values = list(resolution.values())

        total_combinations = np.prod([len(v) for v in param_values])

        if self.show_progress:
            print(f"  参数组合总数: {total_combinations:,}")
            print(f"  参数维度: {param_names}")
            print(f"  各维度大小: {[len(v) for v in param_values]}")
            print(f"  开始生成LUT数据...")

        # 分批处理以避免内存问题
        batch_size = 10000
        all_predictions = []

        # 计算总批次数
        total_batches = (total_combinations + batch_size - 1) // batch_size

        if self.show_progress:
            print(f"  批次大小: {batch_size}")
            print(f"  总批次数: {total_batches}")
            print()

        # 创建进度显示
        if self.show_progress:
            from tqdm import tqdm
            pbar = tqdm(total=total_combinations, desc="生成LUT数据", unit="组合")

        for i in range(0, total_combinations, batch_size):
            batch_combinations = []
            batch_indices = []

            # 生成当前批次的组合
            for idx in range(i, min(i + batch_size, total_combinations)):
                # 计算多维索引
                indices = []
                temp_idx = idx
                for dim_size in [len(v) for v in param_values]:
                    indices.append(temp_idx % dim_size)
                    temp_idx //= dim_size

                # 获取参数值
                combination = {}
                for name, idx_val, values in zip(param_names, indices, param_values):
                    combination[name] = values[idx_val]

                # 计算衍生特征
                combination['airmass_sza'] = 1.0 / np.cos(np.radians(combination['sza']))
                combination['airmass_vza'] = 1.0 / np.cos(np.radians(combination['vza']))
                combination['total_airmass'] = combination['airmass_sza'] + combination['airmass_vza']

                batch_combinations.append(combination)
                batch_indices.append(indices)

            # 转换为DataFrame
            batch_df = pd.DataFrame(batch_combinations)

            # 确保列顺序一致
            batch_df = batch_df[self.feature_columns]

            # 标准化特征
            if 'standard' in self.scalers:
                batch_scaled = self.scalers['standard'].transform(batch_df)
                batch_df_scaled = pd.DataFrame(batch_scaled, columns=batch_df.columns)
            else:
                batch_df_scaled = batch_df

            # 预测误差
            predictions = self.models[model_name].predict(batch_df_scaled)
            all_predictions.extend(predictions)

            # 更新进度
            if self.show_progress:
                pbar.update(len(batch_combinations))

        if self.show_progress:
            pbar.close()

        # 创建多维数组
        lut_shape = tuple(len(v) for v in param_values)
        lut_array = np.array(all_predictions).reshape(lut_shape)

        # 创建xarray Dataset
        coords = {name: values for name, values in zip(param_names, param_values)}
        ds = xr.Dataset(
            {'error_prediction': (param_names, lut_array)},
            coords=coords,
            attrs={
                'model': model_name,
                'description': f'Error prediction LUT generated by {model_name}',
                'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'features': str(self.feature_columns),
                'target': self.target_column
            }
        )

        # 保存LUT
        lut_path = self.output_dir / f"error_lut_{model_name}.nc"
        ds.to_netcdf(lut_path)

        if self.show_progress:
            self.progress_tracker.complete_phase("创建查找表",
                                                 f"生成 {total_combinations} 个组合")
            self.progress_tracker.step("查找表创建完成")
            print(f"\n查找表已保存到: {lut_path}")
            print(f"LUT形状: {lut_shape}")
            print(f"LUT大小: {lut_array.size} 个元素")
            print(f"LUT数据类型: {lut_array.dtype}")

        return ds, lut_path

    def analyze_feature_importance(self):
        """分析特征重要性"""
        if not self.feature_importance:
            self.logger.warning("没有特征重要性数据")
            return

        if self.show_progress:
            self.progress_tracker.start_phase("特征重要性分析", 1)
            print(f"\n分析特征重要性...")

        # 创建特征重要性图表
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        models_with_importance = list(self.feature_importance.keys())

        if self.show_progress:
            print(f"  有特征重要性的模型数: {len(models_with_importance)}")

        for idx, model_name in enumerate(models_with_importance[:6]):  # 最多显示6个模型
            if idx >= len(axes):
                break

            ax = axes[idx]
            importance_df = self.feature_importance[model_name]

            # 取前10个最重要的特征
            top_features = importance_df.head(10)

            ax.barh(range(len(top_features)), top_features['importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('重要性')
            ax.set_title(f'{model_name} - 特征重要性 (Top 10)')
            ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        # 保存图表
        importance_path = self.output_dir / "feature_importance_comparison.png"
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 保存汇总
        summary_path = self.output_dir / "feature_importance_summary.csv"
        all_importance = []
        for model_name, df in self.feature_importance.items():
            df_copy = df.copy()
            df_copy['model'] = model_name
            all_importance.append(df_copy)

        if all_importance:
            combined_importance = pd.concat(all_importance, ignore_index=True)
            combined_importance.to_csv(summary_path, index=False)

            if self.show_progress:
                # 显示最重要的特征
                print(f"\n特征重要性总结:")
                avg_importance = combined_importance.groupby('feature')['importance'].mean().sort_values(
                    ascending=False)
                for i, (feature, importance) in enumerate(avg_importance.head(5).items()):
                    print(f"  {i + 1}. {feature}: {importance:.4f}")

        if self.show_progress:
            self.progress_tracker.complete_phase("特征重要性分析",
                                                 f"分析 {len(models_with_importance)} 个模型")
            self.progress_tracker.step("特征重要性分析完成")
            print(f"特征重要性图表已保存到: {importance_path}")
            print(f"特征重要性汇总已保存到: {summary_path}")

        return combined_importance if all_importance else None

    def save_all_results(self):
        """保存所有结果"""
        if self.show_progress:
            self.progress_tracker.start_phase("保存结果", 1)
            print(f"\n保存所有结果...")

        # 保存模型
        models_path = self.output_dir / "trained_models.pkl"
        with open(models_path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'feature_importance': self.feature_importance,
                'results': self.results
            }, f)

        # 保存配置
        config_path = self.output_dir / "ml_config.json"
        config_data = {
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_configs': {k: {'description': v['description']}
                              for k, v in self.model_configs.items()}
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        # 保存进度摘要
        if self.progress_tracker:
            progress_path = self.output_dir / "training_progress_summary.txt"
            with open(progress_path, 'w') as f:
                f.write("机器学习训练进度摘要\n")
                f.write("=" * 60 + "\n")
                f.write(f"训练时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"训练模型数: {len(self.models)}\n")
                f.write(f"特征数: {len(self.feature_columns)}\n")
                f.write("\n训练模型列表:\n")
                for model_name in self.models:
                    f.write(f"  - {model_name}\n")

        if self.show_progress:
            self.progress_tracker.complete_phase("保存结果", "保存所有结果文件")
            self.progress_tracker.step("结果保存完成")
            print(f"所有结果已保存到目录: {self.output_dir}")

        # 打印最终摘要
        if self.show_progress and self.progress_tracker:
            self.progress_tracker.print_summary()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='训练和评估机器学习模型')
    parser.add_argument('--models', type=str, default='all',
                        help='要训练的模型，用逗号分隔或all')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--create_lut', action='store_true',
                        help='创建查找表')
    parser.add_argument('--lut_model', type=str, default='RandomForest',
                        help='用于创建LUT的模型')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='并行作业数')
    parser.add_argument('--show_progress', action='store_true', default=True,
                        help='显示训练进度')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')

    args = parser.parse_args()

    # 设置日志
    logger = setup_logger('MLMain', level='DEBUG' if args.debug else 'INFO')

    # 打印启动信息
    print("\n" + "=" * 70)
    print("机器学习模型训练与评估系统")
    print("=" * 70)
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"显示进度: {'是' if args.show_progress else '否'}")
    print(f"并行作业数: {args.n_jobs if args.n_jobs != -1 else '自动'}")
    print("=" * 70 + "\n")

    # 创建机器学习模型实例
    ml_models = MachineLearningModels(logger=logger, show_progress=args.show_progress)

    try:
        # 1. 加载数据
        if args.show_progress:
            print("\n阶段1: 数据准备")
            print("-" * 40)

        logger.info("步骤1: 加载数据")
        data = ml_models.load_all_data()

        # 2. 数据预处理
        logger.info("步骤2: 数据预处理")
        X, y = ml_models.preprocess_data(data)

        # 3. 划分训练集和测试集
        logger.info("步骤3: 划分训练集和测试集")
        X_train, X_test, y_train, y_test = ml_models.train_test_split_data(
            X, y, test_size=args.test_size, random_state=ml_models.config.RANDOM_SEED
        )

        # 4. 确定要训练的模型
        if args.models == 'all':
            models_to_train = list(ml_models.model_configs.keys())
        else:
            models_to_train = [m.strip() for m in args.models.split(',')]

        if args.show_progress:
            print(f"\n阶段2: 模型训练")
            print("-" * 40)
            print(f"将训练 {len(models_to_train)} 个模型:")
            for i, model in enumerate(models_to_train):
                desc = ml_models.model_configs.get(model, {}).get('description', '未知')
                print(f"  {i + 1}. {model} ({desc})")

        # 5. 训练所有模型
        logger.info("步骤4: 训练模型")
        training_results = ml_models.train_all_models(
            X_train, y_train, models_to_train=models_to_train, n_jobs=args.n_jobs
        )

        # 6. 评估所有模型
        if args.show_progress:
            print(f"\n阶段3: 模型评估")
            print("-" * 40)

        logger.info("步骤5: 评估模型")
        results_df = ml_models.evaluate_all_models(X_test, y_test)

        # 7. 比较模型
        logger.info("步骤6: 比较模型")
        best_model = ml_models.compare_models(results_df)

        # 8. 绘制预测对比图
        if args.show_progress:
            print(f"\n阶段4: 结果可视化")
            print("-" * 40)

        logger.info("步骤7: 绘制预测对比图")
        for model_name in ml_models.models.keys():
            if model_name in ml_models.results:
                y_pred = ml_models.results[model_name]['predictions']
                ml_models.plot_predictions_vs_actual(model_name, y_test, y_pred)

        # 9. 分析特征重要性
        logger.info("步骤8: 分析特征重要性")
        ml_models.analyze_feature_importance()

        # 10. 创建查找表（如果指定）
        if args.create_lut:
            if args.show_progress:
                print(f"\n阶段5: 创建查找表")
                print("-" * 40)

            logger.info("步骤9: 创建查找表")
            if args.lut_model in ml_models.models:
                lut_ds, lut_path = ml_models.create_lookup_table(args.lut_model)
                logger.info(f"查找表已创建: {lut_path}")
            else:
                logger.warning(f"模型 {args.lut_model} 未训练，使用最佳模型")
                if best_model is not None:
                    lut_model = best_model['Model']
                    lut_ds, lut_path = ml_models.create_lookup_table(lut_model)
                    logger.info(f"查找表已创建: {lut_path}")

        # 11. 保存所有结果
        if args.show_progress:
            print(f"\n阶段6: 保存结果")
            print("-" * 40)

        logger.info("步骤10: 保存所有结果")
        ml_models.save_all_results()

        logger.info("机器学习实验完成!")

        # 输出最终总结
        print("\n" + "=" * 70)
        print("机器学习实验总结")
        print("=" * 70)
        print(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"数据集大小: {len(data)} 个样本")
        print(f"特征数: {X.shape[1]}")
        print(f"训练集: {X_train.shape[0]} 个样本")
        print(f"测试集: {X_test.shape[0]} 个样本")
        print(f"训练的模型数: {len(ml_models.models)}")
        print(f"\n最佳模型: {best_model['Model']} ({best_model['Description']})")
        print(f"RMSE: {best_model['RMSE']:.6f}")
        print(f"MAE: {best_model['MAE']:.6f}")
        print(f"R²: {best_model['R²']:.4f}")

        # 显示所有模型排名
        print(f"\n模型性能排名 (按RMSE升序):")
        print("-" * 70)
        print(f"{'排名':<5} {'模型':<15} {'RMSE':<10} {'MAE':<10} {'R²':<8}")
        print("-" * 70)
        for i, (_, row) in enumerate(results_df.iterrows()):
            rank = i + 1
            print(f"{rank:<5} {row['Model']:<15} {row['RMSE']:.6f}  {row['MAE']:.6f}  {row['R²']:.4f}")

        print("=" * 70)

    except Exception as e:
        logger.error(f"机器学习实验失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()