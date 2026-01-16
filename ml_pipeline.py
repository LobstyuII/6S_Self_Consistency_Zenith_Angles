# ==================== ml_pipeline.py ====================
"""
整合的机器学习流水线
提供一站式训练和评估功能
"""
import argparse
from pathlib import Path
from datetime import datetime
import sys

from config import ExperimentConfig
from data_loader import DataLoader
from model_trainer import ModelTrainer, train_main
from model_evaluator import ModelEvaluator, evaluate_main


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
        print("🚀 机器学习完整流水线")
        print("=" * 80)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        try:
            # 第一步：训练模型
            print("\n📌 第一步：训练模型")
            print("-" * 80)

            # 设置训练参数
            sys.argv = [
                'model_trainer.py',
                '--sample', str(sample_fraction),
                '--models', models_to_train,
                '--test_size', str(test_size),
                '--cv_folds', str(cv_folds),
                '--n_jobs', str(n_jobs),
                '--output_suffix', output_suffix
            ]

            if not use_feature_engineering:
                sys.argv.append('--no_feature_engineering')

            # 运行训练
            train_result = train_main()

            if train_result != 0:
                print("❌ 训练失败，终止流水线")
                return train_result

            # 获取最新创建的模型目录
            models_dir = self.config.MODELS_DIR
            model_dirs = sorted(models_dir.glob("model_training_*"))
            if not model_dirs:
                print("❌ 未找到模型目录")
                return 1

            latest_model_dir = model_dirs[-1]
            if output_suffix:
                latest_model_dir = models_dir / f"model_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{output_suffix}"

            print(f"✅ 模型已保存至: {latest_model_dir}")

            # 第二步：评估模型
            print("\n📌 第二步：评估模型")
            print("-" * 80)

            # 设置评估参数
            sys.argv = [
                'model_evaluator.py',
                '--saved_dir', str(latest_model_dir),
                '--sample', str(sample_fraction / 2),  # 使用更少的数据进行评估
            ]

            if not use_feature_engineering:
                sys.argv.append('--no_feature_engineering')

            if not use_shap:
                sys.argv.append('--no_shap')

            # 运行评估
            eval_result = evaluate_main()

            print("\n" + "=" * 80)
            print("🎉 完整流水线执行完成!")
            print("=" * 80)
            print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"模型目录: {latest_model_dir}")
            print("=" * 80)

            return eval_result

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

        # 设置评估参数
        sys.argv = [
            'model_evaluator.py',
            '--saved_dir', saved_dir,
            '--sample', str(sample_fraction),
        ]

        if not use_feature_engineering:
            sys.argv.append('--no_feature_engineering')

        if not use_shap:
            sys.argv.append('--no_shap')

        # 运行评估
        return evaluate_main()


def pipeline_main():
    """流水线主函数"""
    parser = argparse.ArgumentParser(description='机器学习流水线系统')

    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # 完整流水线命令
    pipeline_parser = subparsers.add_parser('full', help='运行完整流水线（训练+评估）')
    pipeline_parser.add_argument('--sample', type=float, default=0.5,
                                 help='数据采样比例 (0.01-1.0)')
    pipeline_parser.add_argument('--models', type=str,
                                 default='RandomForest,XGBoost,LightGBM,GradientBoosting',
                                 help='要训练的模型列表，用逗号分隔')
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
                             help='测试数据采样比例 (0.01-1.0)')
    eval_parser.add_argument('--no_feature_engineering', action='store_true',
                             help='禁用特征工程')
    eval_parser.add_argument('--no_shap', action='store_true',
                             help='禁用SHAP分析')

    # 单独训练命令
    train_parser = subparsers.add_parser('train', help='只训练模型')
    train_parser.add_argument('--sample', type=float, default=0.5,
                              help='数据采样比例 (0.01-1.0)')
    train_parser.add_argument('--models', type=str,
                              default='RandomForest,XGBoost,LightGBM,GradientBoosting',
                              help='要训练的模型列表，用逗号分隔')
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
        # 直接调用训练模块
        sys.argv = [
            'model_trainer.py',
            '--sample', str(args.sample),
            '--models', args.models,
            '--test_size', str(args.test_size),
            '--cv_folds', str(args.cv_folds),
            '--n_jobs', str(args.n_jobs),
            '--output_suffix', args.output_suffix
        ]

        if args.no_feature_engineering:
            sys.argv.append('--no_feature_engineering')

        return train_main()

    return 0


if __name__ == "__main__":
    pipeline_main()