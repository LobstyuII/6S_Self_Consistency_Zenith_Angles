# ==================== model_trainer.py ====================
"""
模型训练和保存模块
负责训练机器学习模型并保存为pkl文件
"""
import numpy as np
import pandas as pd
import pickle
import json
import time
import psutil
import warnings
from typing import Dict, List
from datetime import datetime

# 机器学习库
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet

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

# 项目模块
from config import ExperimentConfig
from utils import setup_logger
from data_loader import DataLoader

warnings.filterwarnings('ignore')


class ModelTrainer:
    """模型训练器类"""

    def __init__(self, config: ExperimentConfig = None, logger=None,
                 n_jobs: int = -1):

        self.config = config or ExperimentConfig
        self.logger = logger or setup_logger('ModelTrainer')
        self.n_jobs = n_jobs

        # 初始化组件
        self.models = {}
        self.results = {}
        self.feature_importance = {}

        # 输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.config.MODELS_DIR / f"model_training_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        self.models_dir = self.output_dir / "models"
        self.tables_dir = self.output_dir / "tables"
        for d in [self.models_dir, self.tables_dir]:
            d.mkdir(exist_ok=True)

        # 初始化模型配置
        self.model_configs = self._initialize_model_configs()

        self.logger.info(f"模型训练器初始化完成")
        self.logger.info(f"输出目录: {self.output_dir}")

    def _initialize_model_configs(self) -> Dict[str, Dict]:
        """初始化模型配置"""
        configs = {
            'RandomForest': {
                'model_class': RandomForestRegressor,
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [10, 20, 50],
                    'min_samples_leaf': [5, 10, 20],
                    'max_features': ['sqrt', 0.5],
                    'bootstrap': [True],
                    'random_state': [self.config.RANDOM_SEED],
                    'n_jobs': [1],
                    'max_samples': [0.5, 0.7]
                },
                'search_method': 'randomized',
                'n_iter': 8,
                'description': '随机森林回归(优化版)'
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
                    'tree_method': ['hist']
                } if XGB_AVAILABLE else {},
                'search_method': 'randomized',
                'n_iter': 15,
                'description': 'XGBoost回归'
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
                'description': 'LightGBM回归'
            } if LGB_AVAILABLE else None,
            'GradientBoosting': {
                'model_class': GradientBoostingRegressor,
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 4],
                    'min_samples_split': [20, 50],
                    'min_samples_leaf': [10, 20],
                    'subsample': [0.7, 0.8],
                    'max_features': [0.5, 0.7],
                    'random_state': [self.config.RANDOM_SEED]
                },
                'search_method': 'randomized',
                'n_iter': 8,
                'description': '梯度提升回归树(优化版)'
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
                'description': '支持向量回归(RBF核)'
            },
            'MLP': {
                'model_class': MLPRegressor,
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive'],
                    'learning_rate_init': [0.001, 0.01],
                    'max_iter': [300, 500],
                    'early_stopping': [True],
                    'random_state': [self.config.RANDOM_SEED],
                    'batch_size': [128, 256]
                },
                'search_method': 'randomized',
                'n_iter': 10,
                'description': '多层感知器'
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
                'description': '岭回归'
            },
            'Lasso': {
                'model_class': Lasso,
                'params': {
                    'alpha': np.logspace(-3, 1, 5),
                    'max_iter': [1000],
                    'random_state': [self.config.RANDOM_SEED]
                },
                'search_method': 'randomized',
                'n_iter': 10,
                'description': 'Lasso回归'
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
                'description': '弹性网络回归'
            }
        }

        # 移除不可用的模型
        return {k: v for k, v in configs.items() if v is not None and v['model_class'] is not None}

    def train_single_model(self, model_name: str, X_train: pd.DataFrame,
                           y_train: pd.Series, cv_folds: int = 5) -> Dict:
        """训练单个模型"""
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
                n_jobs=1 if model_name in ['RandomForest'] else self.n_jobs,
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
        print(f"   开始训练 {model_name}...")
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
        memory_diff = memory_after - memory_before

        result = {
            'model': best_model,
            'best_params': best_params,
            'best_rmse': best_rmse,
            'best_score': best_score,
            'cv_scores': search.cv_results_,
            'feature_importance': feature_importance,
            'n_samples_used': len(X_train_sampled),
            'training_time': search.refit_time_ if hasattr(search, 'refit_time_') else 0,
            'memory_used_mb': memory_diff
        }

        return result

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

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                     models_to_train: List[str] = None, cv_folds: int = 5) -> Dict:
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

                result = self.train_single_model(model_name, X_train, y_train, cv_folds)

                training_time = time.time() - start_time
                result['training_time'] = training_time

                training_results[model_name] = result

                print(f"   ✅ {model_name} 训练完成")
                print(f"     最佳RMSE (CV): {result['best_rmse']:.6f}")
                print(f"     训练时间: {training_time:.1f}秒")
                print(f"     内存使用: {result['memory_used_mb']:.1f} MB")

                # 保存模型
                self.save_model(model_name, best_model=result['model'])

            except Exception as e:
                print(f"   ❌ {model_name} 训练失败: {e}")
                import traceback
                traceback.print_exc()

        # 保存训练总结
        self.save_training_summary(training_results, X_train.columns.tolist())

        return training_results

    def save_model(self, model_name: str, best_model=None):
        """保存单个模型"""
        if best_model is None:
            best_model = self.models.get(model_name)

        if best_model is not None:
            model_path = self.models_dir / f"{model_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            print(f"     模型已保存: {model_path}")

            # 如果模型有特征重要性，也保存
            if model_name in self.feature_importance:
                importance_path = self.models_dir / f"{model_name}_feature_importance.pkl"
                with open(importance_path, 'wb') as f:
                    pickle.dump(self.feature_importance[model_name], f)

    def save_training_summary(self, training_results: Dict, feature_names: List[str]):
        """保存训练总结"""
        print("=" * 70)
        print("💾 保存训练总结...")

        summary = {
            'training_summary': {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_models': len(training_results),
                'features_used': feature_names,
                'output_dir': str(self.output_dir)
            },
            'model_results': {}
        }

        for model_name, result in training_results.items():
            summary['model_results'][model_name] = {
                'best_params': result.get('best_params', {}),
                'best_rmse': float(result.get('best_rmse', 0)),
                'best_score': float(result.get('best_score', 0)),
                'training_time': float(result.get('training_time', 0)),
                'n_samples_used': result.get('n_samples_used', 0),
                'memory_used_mb': float(result.get('memory_used_mb', 0))
            }

        # 保存JSON总结
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 保存训练结果表格
        results_df = self._create_results_dataframe(training_results)
        results_path = self.tables_dir / "training_results.csv"
        results_df.to_csv(results_path, index=False)

        print(f"✅ 训练总结已保存:")
        print(f"   JSON总结: {summary_path}")
        print(f"   结果表格: {results_path}")
        print(f"   模型文件: {self.models_dir}/")

    def _create_results_dataframe(self, training_results: Dict) -> pd.DataFrame:
        """创建训练结果DataFrame"""
        results = []

        for model_name, result in training_results.items():
            results.append({
                'Model': model_name,
                'Description': self.model_configs[model_name]['description'],
                'CV_RMSE': result['best_rmse'],
                'CV_Score': result['best_score'],
                'Training_Time_s': result['training_time'],
                'Memory_Used_MB': result['memory_used_mb'],
                'N_Samples_Used': result['n_samples_used']
            })

        return pd.DataFrame(results)


def train_main():
    """训练主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='机器学习模型训练系统')
    parser.add_argument('--sample', type=float, default=0.5,
                        help='数据采样比例 (0.01-1.0)')
    parser.add_argument('--models', type=str, default='RandomForest,XGBoost,LightGBM,GradientBoosting',
                        help='要训练的模型列表，用逗号分隔')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='交叉验证折数')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='并行作业数')
    parser.add_argument('--no_feature_engineering', action='store_true',
                        help='禁用特征工程')
    parser.add_argument('--output_suffix', type=str, default='',
                        help='输出目录后缀')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🚀 机器学习模型训练系统")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据采样: {args.sample * 100:.1f}%")
    print(f"测试集比例: {args.test_size * 100:.0f}%")
    print(f"交叉验证: {args.cv_folds}折")
    print(f"并行作业: {args.n_jobs}")
    print(f"特征工程: {'禁用' if args.no_feature_engineering else '启用'}")
    print(f"训练模型: {args.models}")
    print("=" * 80)

    try:
        # 创建数据加载器
        dl = DataLoader()

        # 加载并准备数据
        data_dict = dl.load_and_prepare(
            sample_fraction=args.sample,
            test_size=args.test_size,
            use_feature_engineering=not args.no_feature_engineering
        )

        # 创建模型训练器
        trainer = ModelTrainer(n_jobs=args.n_jobs)

        # 如果指定了输出后缀，修改输出目录
        if args.output_suffix:
            suffix = f"_{args.output_suffix}" if not args.output_suffix.startswith('_') else args.output_suffix
            trainer.output_dir = trainer.output_dir.with_name(trainer.output_dir.name + suffix)
            trainer.output_dir.mkdir(parents=True, exist_ok=True)

            # 更新子目录
            trainer.models_dir = trainer.output_dir / "models"
            trainer.tables_dir = trainer.output_dir / "tables"
            for d in [trainer.models_dir, trainer.tables_dir]:
                d.mkdir(exist_ok=True)

        # 确定要训练的模型
        if args.models == 'all':
            models_to_train = list(trainer.model_configs.keys())
        else:
            models_to_train = [m.strip() for m in args.models.split(',')]

        # 训练模型
        training_results = trainer.train_models(
            data_dict['X_train'], data_dict['y_train'],
            models_to_train=models_to_train,
            cv_folds=args.cv_folds
        )

        # 保存标准化器和特征名称（用于后续评估）
        artifacts = {
            'scaler': data_dict['scaler'],
            'feature_names': data_dict['feature_names']
        }

        artifacts_path = trainer.output_dir / "training_artifacts.pkl"
        with open(artifacts_path, 'wb') as f:
            pickle.dump(artifacts, f)

        print("\n" + "=" * 80)
        print("🎉 训练完成!")
        print("=" * 80)
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"输出目录: {trainer.output_dir}")
        print(f"模型已保存至: {trainer.models_dir}")
        print(f"后续可以使用 model_evaluator.py 进行评估和可视化")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    train_main()