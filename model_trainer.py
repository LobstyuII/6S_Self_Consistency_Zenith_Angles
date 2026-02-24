# ==================== model_trainer.py ====================
"""
模型训练和保存模块
目标变量：delta_toa (ρ_TOA^SA - ρ_TOA^PPA)
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
from pathlib import Path

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from config import ExperimentConfig
from utils import setup_logger
from data_loader import DataLoader

warnings.filterwarnings('ignore')


class ModelTrainer:
    """模型训练器类（目标变量为 delta_toa）"""

    def __init__(self, config: ExperimentConfig = None, logger=None,
                 n_jobs: int = -1, output_dir: Path = None):
        self.config = config or ExperimentConfig
        self.logger = logger or setup_logger('ModelTrainer')
        self.n_jobs = n_jobs

        self.models = {}
        self.results = {}
        self.feature_importance = {}

        if output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = self.config.MODELS_DIR / f"model_training_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir = self.output_dir / "models"
        self.tables_dir = self.output_dir / "tables"
        for d in [self.models_dir, self.tables_dir]:
            d.mkdir(exist_ok=True)

        self.model_configs = self._initialize_model_configs()

        self.logger.info(f"模型训练器初始化完成")
        self.logger.info(f"输出目录: {self.output_dir}")

    def _initialize_model_configs(self) -> Dict[str, Dict]:
        """初始化模型配置（与旧版相同）"""
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
                'description': '随机森林回归'
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
                    'tree_method': ['hist'],
                    'objective': ['reg:squarederror']
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
            # 其余模型配置保持不变（省略以节省篇幅，实际需完整保留）
        }
        # 移除不可用的模型
        return {k: v for k, v in configs.items() if v is not None and v['model_class'] is not None}

    def train_single_model(self, model_name: str, X_train: pd.DataFrame,
                           y_train: pd.Series, cv_folds: int = 5,
                           sample_weight: np.ndarray = None) -> Dict:
        """训练单个模型（支持样本权重）"""
        config = self.model_configs[model_name]
        model_class = config['model_class']
        params = config['params']

        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024

        # 对于内存密集型模型，采样
        if model_name in ['RandomForest', 'GradientBoosting', 'ExtraTrees']:
            if len(X_train) > 10000:
                sample_idx = np.random.choice(len(X_train), 10000, replace=False)
                X_train_sampled = X_train.iloc[sample_idx]
                y_train_sampled = y_train.iloc[sample_idx]
                if sample_weight is not None:
                    sample_weight_sampled = sample_weight[sample_idx]
                else:
                    sample_weight_sampled = None
            else:
                X_train_sampled = X_train
                y_train_sampled = y_train
                sample_weight_sampled = sample_weight
        else:
            X_train_sampled = X_train
            y_train_sampled = y_train
            sample_weight_sampled = sample_weight

        # 调整并行度
        if model_name in ['RandomForest', 'ExtraTrees']:
            if 'n_jobs' in params:
                params['n_jobs'] = [1]
            current_n_jobs = 1
        else:
            current_n_jobs = self.n_jobs

        if model_name in ['XGBoost', 'LightGBM']:
            base_model = model_class(random_state=self.config.RANDOM_SEED,
                                     n_jobs=current_n_jobs, verbose=0)
        else:
            base_model = model_class()

        search_method = config.get('search_method', 'grid')
        n_iter = config.get('n_iter', 10)

        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.config.RANDOM_SEED)

        fit_params = {}
        if sample_weight_sampled is not None:
            fit_params = {'sample_weight': sample_weight_sampled}

        if search_method == 'randomized':
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=params,
                n_iter=n_iter,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=1 if model_name in ['RandomForest', 'ExtraTrees'] else self.n_jobs,
                random_state=self.config.RANDOM_SEED,
                verbose=0
            )
        else:
            search = GridSearchCV(
                estimator=base_model,
                param_grid=params,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=1 if model_name in ['RandomForest', 'ExtraTrees'] else self.n_jobs,
                verbose=0
            )

        print(f"   开始训练 {model_name}...")
        search.fit(X_train_sampled, y_train_sampled, **fit_params)

        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = -search.best_score_
        best_rmse = np.sqrt(best_score)

        self.models[model_name] = best_model

        feature_importance = self._compute_feature_importance(best_model, X_train_sampled)
        if feature_importance is not None:
            self.feature_importance[model_name] = feature_importance

        memory_after = process.memory_info().rss / 1024 / 1024
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
            importances = np.abs(model.coef_)
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            return feature_importance
        else:
            return None

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                     models_to_train: List[str] = None, cv_folds: int = 5,
                     sample_weight: np.ndarray = None) -> Dict:
        """训练所有指定的模型"""
        print("=" * 70)
        print("🚀 训练模型...")

        if models_to_train is None:
            models_to_train = list(self.model_configs.keys())

        print(f"   将训练 {len(models_to_train)} 个模型:")
        for i, model_name in enumerate(models_to_train):
            config = self.model_configs[model_name]
            print(f"     {i+1}. {model_name} - {config['description']}")

        training_results = {}

        for model_name in models_to_train:
            try:
                print(f"\n   🔄 训练 {model_name}...")
                start_time = time.time()
                result = self.train_single_model(model_name, X_train, y_train,
                                                 cv_folds, sample_weight)
                training_time = time.time() - start_time
                result['training_time'] = training_time
                training_results[model_name] = result

                print(f"   ✅ {model_name} 训练完成")
                print(f"     最佳RMSE (CV): {result['best_rmse']:.6f}")
                print(f"     训练时间: {training_time:.1f}秒")
                print(f"     内存使用: {result['memory_used_mb']:.1f} MB")

                self.save_model(model_name, best_model=result['model'])

            except Exception as e:
                print(f"   ❌ {model_name} 训练失败: {e}")
                import traceback
                traceback.print_exc()

        self.save_training_summary(training_results, X_train.columns.tolist())

        return training_results

    def save_model(self, model_name: str, best_model=None):
        if best_model is None:
            best_model = self.models.get(model_name)
        if best_model is not None:
            model_path = self.models_dir / f"{model_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            print(f"     模型已保存: {model_path}")
            if model_name in self.feature_importance:
                importance_path = self.models_dir / f"{model_name}_feature_importance.pkl"
                with open(importance_path, 'wb') as f:
                    pickle.dump(self.feature_importance[model_name], f)

    def save_training_summary(self, training_results: Dict, feature_names: List[str]):
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

        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        results_df = self._create_results_dataframe(training_results)
        results_path = self.tables_dir / "training_results.csv"
        results_df.to_csv(results_path, index=False)

        print(f"✅ 训练总结已保存:")
        print(f"   JSON总结: {summary_path}")
        print(f"   结果表格: {results_path}")

    def _create_results_dataframe(self, training_results: Dict) -> pd.DataFrame:
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
    """训练主函数（命令行接口）"""
    import argparse
    parser = argparse.ArgumentParser(description='机器学习模型训练系统 (目标: delta_toa)')
    parser.add_argument('--sample', type=float, default=0.5,
                        help='数据采样比例 (0.01-1.0)')
    parser.add_argument('--models', type=str,
                        default='RandomForest,XGBoost,LightGBM,GradientBoosting,ExtraTrees',
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
    print("🚀 机器学习模型训练系统 (目标: ΔTOA)")
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
        dl = DataLoader()
        data_dict = dl.load_and_prepare(
            sample_fraction=args.sample,
            test_size=args.test_size,
            use_feature_engineering=not args.no_feature_engineering
        )

        trainer = ModelTrainer(n_jobs=args.n_jobs)

        if args.output_suffix:
            suffix = f"_{args.output_suffix}" if not args.output_suffix.startswith('_') else args.output_suffix
            trainer.output_dir = trainer.output_dir.with_name(trainer.output_dir.name + suffix)
            trainer.output_dir.mkdir(parents=True, exist_ok=True)
            trainer.models_dir = trainer.output_dir / "models"
            trainer.tables_dir = trainer.output_dir / "tables"
            for d in [trainer.models_dir, trainer.tables_dir]:
                d.mkdir(exist_ok=True)

        if args.models.lower() == 'all':
            models_to_train = list(trainer.model_configs.keys())
        else:
            models_to_train = [m.strip() for m in args.models.split(',')]

        # 计算样本权重（极端角度加权）
        X_raw = data_dict['X_raw']
        extreme_threshold = trainer.config.MONTE_CARLO_CONFIG.get('extreme_threshold', 85.0)
        if 'sza' in X_raw.columns and 'vza' in X_raw.columns:
            is_extreme = (X_raw['sza'] > extreme_threshold) | (X_raw['vza'] > extreme_threshold)
            sample_weight = np.where(is_extreme, 10.0, 1.0)
            train_indices = data_dict['X_train'].index
            sample_weight_train = sample_weight[train_indices]
            print(f"   极端样本加权：极端样本数 {is_extreme.sum()}，常规样本数 {(~is_extreme).sum()}")
        else:
            sample_weight_train = None

        training_results = trainer.train_models(
            data_dict['X_train'], data_dict['y_train'],
            models_to_train=models_to_train,
            cv_folds=args.cv_folds,
            sample_weight=sample_weight_train
        )

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
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    train_main()