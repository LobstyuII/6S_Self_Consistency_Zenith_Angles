# ==================== ml_pipeline.py ====================
"""
整合的机器学习流水线
目标变量：delta_toa
"""
import argparse
from datetime import datetime
import pickle
from pathlib import Path

from config import ExperimentConfig
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from data_loader import DataLoader
import numpy as np


class MLPipeline:
    """机器学习流水线类"""

    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig

    def run_full_pipeline(self, sample_fraction: float = 0.5,
                          models_to_train: str = "RandomForest,XGBoost,LightGBM,GradientBoosting",
                          test_size: float = 0.2,
                          cv_folds: int = 5,
                          n_jobs: int = 1,
                          use_feature_engineering: bool = True,
                          use_shap: bool = True,
                          output_suffix: str = ""):
        """运行完整流水线（训练+评估）"""
        print("\n" + "=" * 80)
        print("🚀 机器学习完整流水线 (目标: ΔTOA)")
        print("=" * 80)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        try:
            # 第一步：训练模型
            print("\n📌 第一步：训练模型")
            print("-" * 80)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if output_suffix:
                suffix = f"_{output_suffix}" if not output_suffix.startswith('_') else output_suffix
                model_output_dir = self.config.MODELS_DIR / f"model_training_{timestamp}{suffix}"
            else:
                model_output_dir = self.config.MODELS_DIR / f"model_training_{timestamp}"

            trainer = ModelTrainer(config=self.config, n_jobs=n_jobs, output_dir=model_output_dir)

            dl = DataLoader(config=self.config)
            data_dict = dl.load_and_prepare(
                sample_fraction=sample_fraction,
                test_size=test_size,
                use_feature_engineering=use_feature_engineering
            )

            # 样本权重计算（与旧版相同）
            X_raw = data_dict['X_raw']
            extreme_threshold = self.config.MONTE_CARLO_CONFIG.get('extreme_threshold', 85.0)
            if 'sza' in X_raw.columns and 'vza' in X_raw.columns:
                is_extreme = (X_raw['sza'] > extreme_threshold) | (X_raw['vza'] > extreme_threshold)
                sample_weight = np.where(is_extreme, 10.0, 1.0)
                train_indices = data_dict['X_train'].index
                sample_weight_train = sample_weight[train_indices]
                print(f"   极端样本加权：极端样本数 {is_extreme.sum()}，常规样本数 {(~is_extreme).sum()}")
            else:
                sample_weight_train = None

            if models_to_train.lower() == 'all':
                model_list = list(trainer.model_configs.keys())
            else:
                model_list = [m.strip() for m in models_to_train.split(',')]

            training_results = trainer.train_models(
                data_dict['X_train'], data_dict['y_train'],
                models_to_train=model_list,
                cv_folds=cv_folds,
                sample_weight=sample_weight_train
            )

            artifacts = {
                'scaler': data_dict['scaler'],
                'feature_names': data_dict['feature_names']
            }
            artifacts_path = trainer.output_dir / "training_artifacts.pkl"
            with open(artifacts_path, 'wb') as f:
                pickle.dump(artifacts, f)

            print(f"✅ 模型已保存至: {trainer.output_dir}")

            # 第二步：评估模型
            print("\n📌 第二步：评估模型")
            print("-" * 80)

            evaluator = ModelEvaluator(
                saved_dir=trainer.output_dir,
                config=self.config,
                use_shap=use_shap
            )

            results_df = evaluator.evaluate_models(data_dict['X_test'], data_dict['y_test'])
            evaluator.generate_all_visualizations(data_dict['X_test'], data_dict['y_test'], results_df)
            evaluator.generate_report(results_df)

            print("\n" + "=" * 80)
            print("🎉 完整流水线执行完成!")
            print("=" * 80)
            print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"模型目录: {trainer.output_dir}")
            print("=" * 80)

            return 0

        except Exception as e:
            print(f"\n❌ 流水线执行失败: {e}")
            import traceback
            traceback.print_exc()
            return 1

    def evaluate_existing_models(self, saved_dir: str,
                                 sample_fraction: float = 0.3,
                                 use_feature_engineering: bool = True,
                                 use_shap: bool = True):
        """评估已存在的模型"""
        print("\n" + "=" * 80)
        print("🔍 评估已存在的模型")
        print("=" * 80)

        try:
            evaluator = ModelEvaluator(
                saved_dir=Path(saved_dir),
                config=self.config,
                use_shap=use_shap
            )

            dl = DataLoader(config=self.config)
            test_data = dl.load_data(sample_fraction=sample_fraction)

            X_test, y_test = evaluator.prepare_test_data(
                test_data,
                use_feature_engineering=use_feature_engineering
            )

            results_df = evaluator.evaluate_models(X_test, y_test)
            evaluator.generate_all_visualizations(X_test, y_test, results_df)
            evaluator.generate_report(results_df)

            print("\n" + "=" * 80)
            print("🎉 模型评估完成!")
            print("=" * 80)
            print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"模型目录: {saved_dir}")
            print("=" * 80)

            return 0

        except Exception as e:
            print(f"\n❌ 评估失败: {e}")
            import traceback
            traceback.print_exc()
            return 1


def pipeline_main():
    """流水线主函数"""
    parser = argparse.ArgumentParser(description='机器学习流水线系统 (目标: ΔTOA)')
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # 完整流水线命令
    pipeline_parser = subparsers.add_parser('full', help='运行完整流水线（训练+评估）')
    pipeline_parser.add_argument('--sample', type=float, default=0.5,
                                 help='数据采样比例 (0.01-1.0)')
    pipeline_parser.add_argument('--models', type=str,
                                 default='RandomForest,XGBoost,LightGBM,GradientBoosting',
                                 help='要训练的模型列表')
    pipeline_parser.add_argument('--test_size', type=float, default=0.2,
                                 help='测试集比例')
    pipeline_parser.add_argument('--cv_folds', type=int, default=5,
                                 help='交叉验证折数')
    pipeline_parser.add_argument('--n_jobs', type=int, default=1,
                                 help='并行作业数')
    pipeline_parser.add_argument('--no_feature_engineering', action='store_true',
                                 help='禁用特征工程')
    pipeline_parser.add_argument('--no_shap', action='store_true',
                                 help='禁用SHAP分析')
    pipeline_parser.add_argument('--output_suffix', type=str, default='',
                                 help='输出目录后缀')

    # 评估已存在模型命令
    eval_parser = subparsers.add_parser('evaluate', help='评估已存在的模型')
    eval_parser.add_argument('--saved_dir', type=str, required=True,
                             help='已保存模型的目录路径')
    eval_parser.add_argument('--sample', type=float, default=0.3,
                             help='测试数据采样比例')
    eval_parser.add_argument('--no_feature_engineering', action='store_true',
                             help='禁用特征工程')
    eval_parser.add_argument('--no_shap', action='store_true',
                             help='禁用SHAP分析')

    # 单独训练命令
    train_parser = subparsers.add_parser('train', help='只训练模型')
    train_parser.add_argument('--sample', type=float, default=0.5,
                              help='数据采样比例')
    train_parser.add_argument('--models', type=str,
                              default='RandomForest,XGBoost,LightGBM,GradientBoosting',
                              help='要训练的模型列表')
    train_parser.add_argument('--test_size', type=float, default=0.2,
                              help='测试集比例')
    train_parser.add_argument('--cv_folds', type=int, default=5,
                              help='交叉验证折数')
    train_parser.add_argument('--n_jobs', type=int, default=1,
                              help='并行作业数')
    train_parser.add_argument('--no_feature_engineering', action='store_true',
                              help='禁用特征工程')
    train_parser.add_argument('--output_suffix', type=str, default='',
                              help='输出目录后缀')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    pipeline = MLPipeline()

    if args.command == 'full':
        return pipeline.run_full_pipeline(
            sample_fraction=args.sample,
            models_to_train=args.models,
            test_size=args.test_size,
            cv_folds=args.cv_folds,
            n_jobs=args.n_jobs,
            use_feature_engineering=not args.no_feature_engineering,
            use_shap=not args.no_shap,
            output_suffix=args.output_suffix
        )

    elif args.command == 'evaluate':
        return pipeline.evaluate_existing_models(
            saved_dir=args.saved_dir,
            sample_fraction=args.sample,
            use_feature_engineering=not args.no_feature_engineering,
            use_shap=not args.no_shap
        )

    elif args.command == 'train':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.output_suffix:
            suffix = f"_{args.output_suffix}" if not args.output_suffix.startswith('_') else args.output_suffix
            output_dir = pipeline.config.MODELS_DIR / f"model_training_{timestamp}{suffix}"
        else:
            output_dir = pipeline.config.MODELS_DIR / f"model_training_{timestamp}"

        trainer = ModelTrainer(config=pipeline.config, n_jobs=args.n_jobs, output_dir=output_dir)

        dl = DataLoader(config=pipeline.config)
        data_dict = dl.load_and_prepare(
            sample_fraction=args.sample,
            test_size=args.test_size,
            use_feature_engineering=not args.no_feature_engineering
        )

        if args.models.lower() == 'all':
            model_list = list(trainer.model_configs.keys())
        else:
            model_list = [m.strip() for m in args.models.split(',')]

        trainer.train_models(
            data_dict['X_train'], data_dict['y_train'],
            models_to_train=model_list,
            cv_folds=args.cv_folds
        )

        artifacts = {
            'scaler': data_dict['scaler'],
            'feature_names': data_dict['feature_names']
        }
        artifacts_path = output_dir / "training_artifacts.pkl"
        with open(artifacts_path, 'wb') as f:
            pickle.dump(artifacts, f)

        print(f"\n✅ 训练完成，模型保存至: {output_dir}")
        return 0

    return 0


if __name__ == "__main__":
    pipeline_main()