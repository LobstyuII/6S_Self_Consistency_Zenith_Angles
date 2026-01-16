# ==================== main_workflow.py ====================
"""
主工作流程 - 集成所有重构模块
"""
import argparse
from pathlib import Path
import time
import pandas as pd
import numpy as np
from datetime import datetime

from config import ExperimentConfig
from utils import setup_logger, save_dataset
from data_generator import MonteCarloDataGenerator
from refactored_parallel_simulator import RefactoredParallelSimulator
from enhanced_feature_engineering import EnhancedFeatureEngineering
from ml_pipeline import MLPipeline
from lut_generator import OperationalLUTGenerator


def run_refactored_workflow():
    """
    运行重构后的完整工作流程
    """
    parser = argparse.ArgumentParser(description='重构的6S几何校正工作流程')

    # 数据生成参数
    parser.add_argument('--training_samples', type=int, default=50000,
                        help='每波段训练样本数')
    parser.add_argument('--validation_grid', action='store_true',
                        help='生成验证网格')
    parser.add_argument('--bands', type=str, default='all',
                        help='波段列表，用逗号分隔或all')
    parser.add_argument('--n_workers', type=int, default=6,
                        help='并行工作进程数')

    # 机器学习参数
    parser.add_argument('--train_ml', action='store_true',
                        help='训练机器学习模型')
    parser.add_argument('--ml_models', type=str,
                        default='RandomForest,XGBoost,LightGBM',
                        help='要训练的ML模型')
    parser.add_argument('--ml_sample_ratio', type=float, default=1.0,
                        help='ML训练数据采样比例')

    # LUT生成参数
    parser.add_argument('--generate_lut', action='store_true',
                        help='生成业务化LUT')
    parser.add_argument('--ml_model_path', type=str,
                        help='ML模型路径，用于生成LUT')

    args = parser.parse_args()

    # 初始化
    config = ExperimentConfig
    logger = setup_logger('RefactoredWorkflow', level='INFO')

    logger.info("=" * 80)
    logger.info("重构的6S几何校正工作流程")
    logger.info("=" * 80)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 确定波段
    if args.bands == 'all':
        bands_to_process = list(config.BANDS.keys())
    else:
        bands_to_process = [b.strip() for b in args.bands.split(',')]

    logger.info(f"处理波段: {bands_to_process}")

    # 1. 数据生成阶段
    logger.info("\n" + "=" * 80)
    logger.info("阶段1: 数据生成")
    logger.info("=" * 80)

    simulator = RefactoredParallelSimulator(config, logger)

    # 生成训练数据集（蒙特卡洛混合采样）
    training_data = simulator.generate_training_dataset(
        total_samples_per_band=args.training_samples,
        bands=bands_to_process
    )

    # 模拟训练数据集
    logger.info("模拟训练数据集...")
    training_results = simulator.simulate_dataset(
        training_data,
        max_workers=args.n_workers
    )

    # 保存训练数据
    output_dir = config.DATA_DIR / "refactored"
    output_dir.mkdir(parents=True, exist_ok=True)

    for band_id, df in training_results.items():
        if len(df) > 0:
            # 过滤成功样本
            if 'success' in df.columns:
                df_success = df[df['success']].copy() if df['success'].dtype == bool else df[df['success'] == 1].copy()
            else:
                df_success = df

            if not df_success.empty:
                output_file = output_dir / f"training_data_{band_id}.nc"

                # 转换为字典并保存
                data_dict = {}
                for col in df_success.columns:
                    col_data = df_success[col].values

                    if col_data.dtype == object:
                        try:
                            col_data = col_data.astype(str)
                        except:
                            col_data = np.array([str(x) if pd.notna(x) else '' for x in col_data])

                    data_dict[col] = col_data

                save_dataset(data_dict, output_file)
                logger.info(f"保存训练数据: {output_file} ({len(df_success)} 样本)")

    # 生成验证网格（如果需要）
    if args.validation_grid:
        logger.info("\n生成验证网格...")
        validation_data = simulator.generate_validation_grid(bands_to_process)

        validation_results = simulator.simulate_dataset(
            validation_data,
            max_workers=args.n_workers
        )

        for band_id, df in validation_results.items():
            if len(df) > 0:
                if 'success' in df.columns:
                    df_success = df[df['success']].copy() if df['success'].dtype == bool else df[
                        df['success'] == 1].copy()
                else:
                    df_success = df

                if not df_success.empty:
                    output_file = output_dir / f"validation_grid_{band_id}.nc"

                    data_dict = {}
                    for col in df_success.columns:
                        col_data = df_success[col].values

                        if col_data.dtype == object:
                            try:
                                col_data = col_data.astype(str)
                            except:
                                col_data = np.array([str(x) if pd.notna(x) else '' for x in col_data])

                        data_dict[col] = col_data

                    save_dataset(data_dict, output_file)
                    logger.info(f"保存验证网格: {output_file} ({len(df_success)} 样本)")

    # 2. 机器学习阶段
    if args.train_ml:
        logger.info("\n" + "=" * 80)
        logger.info("阶段2: 机器学习训练")
        logger.info("=" * 80)

        # 使用增强的特征工程
        ml_pipeline = MLPipeline(config)

        # 运行训练
        train_result = ml_pipeline.run_full_pipeline(
            sample_fraction=args.ml_sample_ratio,
            models_to_train=args.ml_models,
            use_feature_engineering=True,  # 使用增强的特征
            output_suffix="refactored"
        )

        if train_result != 0:
            logger.error("ML训练失败")
        else:
            logger.info("ML训练完成")

    # 3. LUT生成阶段
    if args.generate_lut and args.ml_model_path:
        logger.info("\n" + "=" * 80)
        logger.info("阶段3: 业务化LUT生成")
        logger.info("=" * 80)

        lut_generator = OperationalLUTGenerator(config, logger)

        # 生成LUT
        lut_ds = lut_generator.generate_operational_lut(
            ml_model_path=Path(args.ml_model_path),
            output_path=config.RESULTS_DIR / "operational_lut_refactored.nc"
        )

        logger.info("LUT生成完成")

        # 输出LUT统计信息
        logger.info("\nLUT统计信息:")
        logger.info(f"  维度: {dict(lut_ds.dims)}")
        logger.info(
            f"  校正误差范围: {lut_ds['ml_correction'].min().item():.4f} 到 {lut_ds['ml_correction'].max().item():.4f}")
        logger.info(f"  校正误差均值: {lut_ds['ml_correction'].mean().item():.4f}")
        logger.info(f"  校正误差标准差: {lut_ds['ml_correction'].std().item():.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("工作流程完成!")
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)


if __name__ == "__main__":
    run_refactored_workflow()