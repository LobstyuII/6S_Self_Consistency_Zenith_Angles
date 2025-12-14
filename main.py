# ==================== main.py ====================
"""
主程序模块
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

from config import ExperimentConfig
from data_generator import BatchSimulator
from error_analyzer import ErrorAnalyzer
from correction_model import ModelTrainer
from validation import ModelValidator
from utils import setup_logger, load_dataset
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='6S几何误差校正实验')
    parser.add_argument('--phase', type=str, choices=['all', 'simulate', 'analyze', 'train', 'validate'],
                        default='all', help='运行阶段')
    parser.add_argument('--band', type=str, default='band1',
                        help='目标波段 (band1, band2, band3)')
    parser.add_argument('--model_type', type=str, default='lut',
                        help='校正模型类型 (lut, linear, polynomial, ml)')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='并行工作进程数')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')
    args = parser.parse_args()

    # 设置配置
    config = ExperimentConfig

    # 更新并行配置
    if args.n_workers:
        config.PARALLEL_CONFIG['n_workers'] = args.n_workers

    # 设置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger('Main', config.BASE_DIR / 'experiment.log', level=log_level)

    logger.info("=" * 60)
    logger.info(f"6S几何误差校正实验 - {config.EXP_NAME} {config.EXP_VERSION}")
    logger.info(f"运行阶段: {args.phase}")
    logger.info(f"目标波段: {args.band}")
    logger.info(f"模型类型: {args.model_type}")
    logger.info("=" * 60)

    try:
        # 阶段1: 数据生成
        if args.phase in ['all', 'simulate']:
            logger.info("阶段1: 数据生成")

            # 检查是否已有数据
            data_file = config.DATA_DIR / f"simulation_results_{args.band}.nc"
            if data_file.exists() and not args.debug:
                logger.info(f"数据文件已存在: {data_file}")
                logger.info("跳过数据生成阶段，使用现有数据")
                results = {args.band: pd.DataFrame(load_dataset(data_file))}
            else:
                # 运行模拟
                simulator = BatchSimulator(config, logger)
                results = simulator.run_all_bands() if args.band == 'all' else {
                    args.band: simulator.run_batch_simulation(args.band)
                }

        else:
            # 加载现有数据
            logger.info("加载现有数据")
            if args.band == 'all':
                results = {}
                for band_id in config.BANDS.keys():
                    data_file = config.DATA_DIR / f"simulation_results_{band_id}.nc"
                    if data_file.exists():
                        results[band_id] = pd.DataFrame(load_dataset(data_file))
            else:
                data_file = config.DATA_DIR / f"simulation_results_{args.band}.nc"
                if not data_file.exists():
                    raise FileNotFoundError(f"数据文件不存在: {data_file}")
                results = {args.band: pd.DataFrame(load_dataset(data_file))}

        # 阶段2: 误差分析
        if args.phase in ['all', 'analyze']:
            logger.info("阶段2: 误差分析")

            analyzer = ErrorAnalyzer(results, logger)

            # 生成误差报告
            report_file = config.RESULTS_DIR / f"error_analysis_report_{args.band}.txt"
            report = analyzer.generate_error_report(report_file)
            logger.info(f"误差分析报告已生成: {report_file}")

            # 绘制误差分布图
            dist_file = config.FIGURES_DIR / f"error_distribution_{args.band}.png"
            analyzer.plot_error_distribution(dist_file)

            # 绘制误差等高线图
            contour_file = config.FIGURES_DIR / f"error_contour_{args.band}.png"
            analyzer.plot_error_contours(save_path=contour_file)

            # 按参数分组分析
            param_file = config.FIGURES_DIR / f"error_by_parameter_{args.band}.png"
            analyzer.plot_error_by_parameter(param_file)

        # 阶段3: 模型训练
        if args.phase in ['all', 'train']:
            logger.info("阶段3: 模型训练")

            # 合并所有波段数据用于训练
            all_data = pd.concat([df for df in results.values()], ignore_index=True)

            trainer = ModelTrainer(config, logger)

            if args.band == 'all':
                # 训练所有波段
                trainer.train_all_bands(all_data, args.model_type)
            else:
                # 训练单个波段
                model = trainer.train_model(all_data, args.band, args.model_type)

                # 比较不同模型
                if args.model_type == 'lut':
                    comparison = trainer.compare_models(all_data, args.band)
                    logger.info(f"模型比较完成，最佳模型: {comparison.loc[comparison['rmse'].idxmin(), 'model_type']}")

        # 阶段4: 模型验证
        if args.phase in ['all', 'validate']:
            logger.info("阶段4: 模型验证")

            # 加载数据
            all_data = pd.concat([df for df in results.values()], ignore_index=True)

            validator = ModelValidator(config, logger)

            if args.band == 'all':
                # 验证所有波段
                validation_results = {}
                for band_id in config.BANDS.keys():
                    try:
                        # 交叉验证
                        cv_results = validator.cross_validate(all_data, band_id, model_type=args.model_type)

                        # 极端条件验证
                        extreme_results = validator.validate_on_extreme_conditions(all_data, band_id)

                        # 绘制验证结果
                        val_fig_file = config.FIGURES_DIR / f"validation_{band_id}_{args.model_type}.png"
                        val_results = validator.plot_validation_results(all_data, band_id, save_path=val_fig_file)

                        # 合并结果
                        validation_results[band_id] = {
                            **cv_results,
                            **extreme_results,
                            **val_results
                        }

                    except Exception as e:
                        logger.error(f"波段 {band_id} 验证失败: {e}")

                # 保存验证结果
                import json
                val_file = config.RESULTS_DIR / f"validation_results_all.json"
                with open(val_file, 'w') as f:
                    json.dump(validation_results, f, indent=2)

            else:
                # 验证单个波段
                # 交叉验证
                cv_results = validator.cross_validate(all_data, args.band, model_type=args.model_type)
                logger.info(f"交叉验证结果 - RMSE: {cv_results.get('cv_rmse_mean', np.nan):.6f}")

                # 极端条件验证
                extreme_results = validator.validate_on_extreme_conditions(all_data, args.band)
                logger.info(f"极端条件验证 - RMSE: {extreme_results.get('extreme_rmse', np.nan):.6f}")

                # 绘制验证结果
                val_fig_file = config.FIGURES_DIR / f"validation_{args.band}_{args.model_type}.png"
                val_results = validator.plot_validation_results(all_data, args.band, save_path=val_fig_file)

                # 应用校正
                corrected_data = validator.apply_correction(all_data, args.band)

                # 保存校正后数据
                corrected_file = config.DATA_DIR / f"corrected_results_{args.band}.nc"
                from utils import save_dataset
                save_dataset(corrected_data.to_dict('list'), corrected_file)
                logger.info(f"校正后数据已保存: {corrected_file}")

        logger.info("=" * 60)
        logger.info("实验完成!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"实验失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()