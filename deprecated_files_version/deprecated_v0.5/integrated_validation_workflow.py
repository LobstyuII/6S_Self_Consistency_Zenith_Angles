# ==================== integrated_validation_workflow.py ====================
"""
整合的外部验证工作流程
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from datetime import datetime

from config import ExperimentConfig
from utils import setup_logger
from external_data_loader import ExternalDataLoader
from external_feature_alignment import ExternalFeatureAlignment
from updated_external_validation_analysis import UpdatedExternalValidationAnalysis


def run_integrated_validation():
    """
    运行整合的验证工作流程
    """
    parser = argparse.ArgumentParser(description='整合的外部验证工作流程')

    # 数据参数
    parser.add_argument('--n_stations', type=int, default=50,
                        help='验证测站数量')
    parser.add_argument('--start_date', type=str, default='20160101',
                        help='开始日期 (YYYYMMDD)')
    parser.add_argument('--end_date', type=str, default='20160110',
                        help='结束日期 (YYYYMMDD)')
    parser.add_argument('--use_existing_data', action='store_true',
                        help='使用现有数据文件')
    parser.add_argument('--data_path', type=str,
                        help='现有数据文件路径')

    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型文件路径 (.pkl)')
    parser.add_argument('--model_type', type=str, choices=['RandomForest', 'XGBoost', 'LightGBM'],
                        default='RandomForest', help='模型类型')

    # 输出参数
    parser.add_argument('--output_dir', type=str,
                        help='输出目录路径')
    parser.add_argument('--output_suffix', type=str, default='',
                        help='输出后缀')

    args = parser.parse_args()

    # 初始化
    config = ExperimentConfig
    logger = setup_logger('IntegratedValidationWorkflow', level='INFO')

    logger.info("=" * 80)
    logger.info("整合的外部验证工作流程")
    logger.info("=" * 80)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config.RESULTS_DIR / f"integrated_validation_{timestamp}"
        if args.output_suffix:
            output_dir = output_dir.with_name(f"{output_dir.name}_{args.output_suffix}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")

    # 1. 数据准备阶段
    logger.info("\n阶段1: 数据准备")
    logger.info("-" * 80)

    raw_data_path = output_dir / "raw_validation_data.parquet"

    if args.use_existing_data and args.data_path and Path(args.data_path).exists():
        logger.info(f"使用现有数据: {args.data_path}")
        raw_data_path = Path(args.data_path)
    else:
        logger.info("生成新的验证数据集")
        data_loader = ExternalDataLoader(config, logger)

        validation_df = data_loader.generate_validation_dataset(
            n_stations=args.n_stations,
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=raw_data_path
        )

        if validation_df.empty:
            logger.error("数据生成失败")
            return 1

    # 2. 特征对齐阶段
    logger.info("\n阶段2: 特征对齐")
    logger.info("-" * 80)

    feature_aligner = ExternalFeatureAlignment(config, logger)

    # 加载原始数据
    raw_data = pd.read_parquet(raw_data_path)
    logger.info(f"原始数据形状: {raw_data.shape}")

    # 准备闭合实验特征
    closed_experiment_features = feature_aligner.prepare_closed_experiment_features(raw_data)

    if closed_experiment_features.empty:
        logger.error("闭合实验特征准备失败")
        return 1

    # 保存闭合实验特征
    closed_exp_path = output_dir / "closed_experiment_features.parquet"
    closed_experiment_features.to_parquet(closed_exp_path, index=False)
    logger.info(f"闭合实验特征已保存: {closed_exp_path}")

    # 3. 模型验证阶段
    logger.info("\n阶段3: 模型验证")
    logger.info("-" * 80)

    analyzer = UpdatedExternalValidationAnalysis(config, logger)

    success = analyzer.run_complete_analysis(
        data_path=raw_data_path,
        model_path=Path(args.model_path),
        output_dir=output_dir / "validation_results"
    )

    if not success:
        logger.error("模型验证失败")
        return 1

    # 4. 生成总结报告
    logger.info("\n阶段4: 生成总结报告")
    logger.info("-" * 80)

    summary_path = output_dir / "workflow_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("整合的外部验证工作流程总结\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"工作流程ID: {timestamp}\n\n")

        f.write("执行阶段:\n")
        f.write("  1. 数据准备: ✓ 完成\n")
        f.write("  2. 特征对齐: ✓ 完成\n")
        f.write("  3. 模型验证: ✓ 完成\n")
        f.write("  4. 总结报告: ✓ 完成\n\n")

        f.write("数据统计:\n")
        f.write(f"  原始数据记录数: {len(raw_data)}\n")
        f.write(f"  测站数量: {raw_data['station'].nunique()}\n")
        f.write(f"  时间范围: {raw_data['datetime'].min()} 到 {raw_data['datetime'].max()}\n\n")

        f.write("模型信息:\n")
        f.write(f"  模型路径: {args.model_path}\n")
        f.write(f"  模型类型: {args.model_type}\n\n")

        f.write("输出文件:\n")
        f.write(f"  1. 原始数据: {raw_data_path}\n")
        f.write(f"  2. 闭合实验特征: {closed_exp_path}\n")
        f.write(f"  3. 验证结果目录: {output_dir / 'validation_results'}\n")
        f.write(f"  4. 工作流程总结: {summary_path}\n")

    logger.info(f"工作流程总结已保存: {summary_path}")

    logger.info("\n" + "=" * 80)
    logger.info("整合的外部验证工作流程完成！")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(run_integrated_validation())