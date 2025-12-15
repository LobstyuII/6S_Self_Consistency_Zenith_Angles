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
from utils import setup_logger, load_dataset, save_dataset
from sensitivity_analyzer import SensitivityAnalyzer
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False


def validate_data_columns(df: pd.DataFrame, band_id: str) -> bool:
    """验证数据列是否完整"""
    required_columns = ['sza', 'vza', 'error_absolute', 'rho_true', 'rho_retrieved']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"警告: 波段 {band_id} 数据缺少列: {missing_columns}")
        print(f"可用列: {df.columns.tolist()}")
        return False
    return True


def repair_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """修复缺失的列"""
    df_repair = df.copy()

    # 如果 error_absolute 不存在，尝试从其他列计算
    if 'error_absolute' not in df_repair.columns:
        if 'rho_retrieved' in df_repair.columns and 'rho_true' in df_repair.columns:
            print("计算缺失的 error_absolute 列...")
            df_repair['error_absolute'] = df_repair['rho_retrieved'] - df_repair['rho_true']
        else:
            print("无法计算 error_absolute，创建空列")
            df_repair['error_absolute'] = np.nan

    # 添加必要的列
    if 'airmass_sza' not in df_repair.columns and 'sza' in df_repair.columns:
        df_repair['airmass_sza'] = 1.0 / np.cos(np.radians(df_repair['sza']))

    if 'airmass_vza' not in df_repair.columns and 'vza' in df_repair.columns:
        df_repair['airmass_vza'] = 1.0 / np.cos(np.radians(df_repair['vza']))

    if 'airmass_sza' in df_repair.columns and 'airmass_vza' in df_repair.columns:
        df_repair['total_airmass'] = df_repair['airmass_sza'] + df_repair['airmass_vza']

    return df_repair


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='6S几何误差校正实验')
    parser.add_argument('--phase', type=str,
                        choices=['all', 'simulate', 'analyze', 'sensitivity', 'train', 'validate'],
                        default='all', help='运行阶段')
    parser.add_argument('--mode', type=str,
                        choices=['full', 'single_factor', 'multi_factor', 'airmass_only'],
                        default='full', help='实验模式')
    parser.add_argument('--band', type=str, default='band3',
                        help='目标波段 (band1, band2, band3)')
    parser.add_argument('--model_type', type=str, default='lut',
                        help='校正模型类型 (lut, linear, polynomial, ml)')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='并行工作进程数')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')
    parser.add_argument('--force_repair', action='store_true',
                        help='强制修复数据列')
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
            data_file = config.DATA_DIR / f"simulation_results_{args.band}_{args.mode}.nc"
            if data_file.exists() and not args.debug and not args.force_repair:
                logger.info(f"数据文件已存在: {data_file}")
                logger.info("跳过数据生成阶段，使用现有数据")

                # 加载并验证数据
                try:
                    data_dict = load_dataset(data_file)
                    df_result = pd.DataFrame(data_dict)

                    # 验证数据列
                    if not validate_data_columns(df_result, args.band):
                        logger.warning(f"数据列不完整，尝试修复...")
                        df_result = repair_missing_columns(df_result)

                        # 保存修复后的数据
                        save_dataset(df_result.to_dict('list'), data_file)
                        logger.info(f"数据已修复并重新保存: {data_file}")

                    results = {args.band: df_result}

                except Exception as e:
                    logger.error(f"加载数据失败: {e}")
                    logger.info("重新生成数据...")
                    simulator = BatchSimulator(config, logger)
                    results = {args.band: simulator.run_batch_simulation(args.band, args.mode)}
            else:
                # 运行模拟（使用指定模式）
                simulator = BatchSimulator(config, logger)
                if args.band == 'all':
                    results = {}
                    for band_id in config.BANDS.keys():
                        results[band_id] = simulator.run_batch_simulation(band_id, args.mode)
                else:
                    results = {args.band: simulator.run_batch_simulation(args.band, args.mode)}

        else:
            # 加载现有数据
            logger.info("加载现有数据")
            data_file = config.DATA_DIR / f"simulation_results_{args.band}_{args.mode}.nc"

            if not data_file.exists():
                # 尝试加载其他模式的数据
                alt_files = list(config.DATA_DIR.glob(f"simulation_results_{args.band}_*.nc"))
                if alt_files:
                    data_file = alt_files[0]
                    logger.info(f"使用替代数据文件: {data_file}")
                else:
                    raise FileNotFoundError(f"找不到 {args.band} 的数据文件")

            try:
                # 加载数据
                data_dict = load_dataset(data_file)
                df_result = pd.DataFrame(data_dict)

                # 验证和修复数据
                if not validate_data_columns(df_result, args.band) or args.force_repair:
                    logger.info("修复数据列...")
                    df_result = repair_missing_columns(df_result)

                    # 保存修复后的数据
                    save_dataset(df_result.to_dict('list'), data_file)
                    logger.info(f"数据已修复并重新保存: {data_file}")

                results = {args.band: df_result}
                logger.info(f"数据加载成功，形状: {df_result.shape}")
                logger.info(f"数据列: {df_result.columns.tolist()}")

            except Exception as e:
                logger.error(f"加载数据失败: {e}")
                raise

        # 阶段2: 误差分析
        if args.phase in ['all', 'analyze'] and 'error_absolute' in results[args.band].columns:
            logger.info("阶段2: 误差分析")

            # 确保有误差数据
            error_count = results[args.band]['error_absolute'].notna().sum()
            if error_count == 0:
                logger.warning("没有有效的误差数据，跳过误差分析")
            else:
                analyzer = ErrorAnalyzer(results, logger)

                # 生成误差报告
                report_file = config.RESULTS_DIR / f"error_analysis_report_{args.band}.txt"

                # 检查 analyzer 是否有 generate_error_report 方法
                if hasattr(analyzer, 'generate_error_report'):
                    report = analyzer.generate_error_report(report_file)
                else:
                    # 如果没有该方法，只打印基本统计信息
                    stats = analyzer.calculate_overall_statistics()
                    with open(report_file, 'w', encoding='utf-8') as f:
                        f.write("误差分析报告\n")
                        f.write(f"波段: {args.band}\n")
                        f.write(f"样本数: {len(results[args.band])}\n")
                        f.write(f"有效误差数据: {error_count}\n")
                        for band, stat in stats.items():
                            f.write(f"\n{band}:\n")
                            for key, value in stat.items():
                                f.write(f"  {key}: {value:.6f}\n")
                    logger.info(f"误差分析报告已保存: {report_file}")

                # =============== 新增图表类型 ===============
                # 只选择部分图表生成，避免全部失败
                try:
                    # 1. 误差分布图
                    dist_file = config.FIGURES_DIR / f"error_distribution_{args.band}.png"
                    analyzer.plot_error_distribution(dist_file)
                except Exception as e:
                    logger.warning(f"绘制误差分布图失败: {e}")

                try:
                    # 2. 综合摘要图
                    summary_file = config.FIGURES_DIR / f"error_summary_{args.band}.png"
                    analyzer.create_summary_figure(save_path=summary_file)
                except Exception as e:
                    logger.warning(f"绘制综合摘要图失败: {e}")

                try:
                    # 3. 3D误差曲面
                    surface_file = config.FIGURES_DIR / f"3d_surface_{args.band}.png"
                    analyzer.plot_3d_error_surface(save_path=surface_file)
                except Exception as e:
                    logger.warning(f"绘制3D误差曲面失败: {e}")

                logger.info("误差分析完成")

        # 阶段2.5: 敏感性分析
        if args.phase in ['all', 'sensitivity'] and 'error_absolute' in results[args.band].columns:
            logger.info("阶段2.5: 敏感性分析")

            # 合并数据
            all_data = pd.concat([df for df in results.values()], ignore_index=True)

            # 检查是否有足够的误差数据
            if all_data['error_absolute'].notna().sum() < 10:
                logger.warning("误差数据不足，跳过敏感性分析")
            else:
                # 创建敏感性分析器
                sensitivity_analyzer = SensitivityAnalyzer(all_data, logger)

                # 生成敏感性报告
                sens_report_file = config.RESULTS_DIR / f"sensitivity_report_{args.band}_{args.mode}.txt"

                try:
                    sens_report = sensitivity_analyzer.generate_sensitivity_report(sens_report_file)
                    logger.info(f"敏感性分析报告已生成: {sens_report_file}")
                except Exception as e:
                    logger.error(f"生成敏感性报告失败: {e}")
                    # 创建简单的报告
                    with open(sens_report_file, 'w', encoding='utf-8') as f:
                        f.write(f"敏感性分析报告\n")
                        f.write(f"波段: {args.band}\n")
                        f.write(f"模式: {args.mode}\n")
                        f.write(f"样本数: {len(all_data)}\n")
                        f.write(f"错误: {str(e)}\n")

                # 尝试绘制敏感性分析图
                try:
                    single_factor_file = config.FIGURES_DIR / f"sensitivity_single_factor_{args.band}_{args.mode}.png"
                    sensitivity_analyzer.plot_single_factor_sensitivity(single_factor_file)
                except Exception as e:
                    logger.warning(f"绘制单因素敏感性图失败: {e}")

                try:
                    airmass_file = config.FIGURES_DIR / f"sensitivity_airmass_{args.band}_{args.mode}.png"
                    sensitivity_analyzer.plot_airmass_breakdown(airmass_file)
                except Exception as e:
                    logger.warning(f"绘制大气质量分解图失败: {e}")

                logger.info("敏感性分析完成")

        # 阶段3: 模型训练
        if args.phase in ['all', 'train'] and 'error_absolute' in results[args.band].columns:
            logger.info("阶段3: 模型训练")

            # 合并所有波段数据用于训练
            all_data = pd.concat([df for df in results.values()], ignore_index=True)

            # 检查是否有足够的训练数据
            error_data = all_data['error_absolute'].dropna()
            if len(error_data) < 50:
                logger.warning(f"有效误差数据不足 ({len(error_data)} < 50)，跳过模型训练")
            else:
                trainer = ModelTrainer(config, logger)

                if args.band == 'all':
                    # 训练所有波段
                    trainer.train_all_bands(all_data, args.model_type)
                else:
                    # 训练单个波段
                    try:
                        model = trainer.train_model(all_data, args.band, args.model_type)

                        # 比较不同模型
                        if args.model_type == 'lut':
                            comparison = trainer.compare_models(all_data, args.band)
                            logger.info(
                                f"模型比较完成，最佳模型: {comparison.loc[comparison['rmse'].idxmin(), 'model_type']}")
                    except Exception as e:
                        logger.error(f"模型训练失败: {e}")

        # 阶段4: 模型验证
        if args.phase in ['all', 'validate'] and 'error_absolute' in results[args.band].columns:
            logger.info("阶段4: 模型验证")

            # 合并数据
            all_data = pd.concat([df for df in results.values()], ignore_index=True)

            # 检查是否有验证数据
            if all_data['error_absolute'].notna().sum() < 20:
                logger.warning("验证数据不足，跳过模型验证")
            else:
                validator = ModelValidator(config, logger)

                if args.band == 'all':
                    # 验证所有波段
                    validation_results = {}
                    for band_id in config.BANDS.keys():
                        try:
                            # 检查该波段是否有足够数据
                            band_data = all_data[all_data['band'] == band_id]
                            if len(band_data) < 10:
                                logger.warning(f"波段 {band_id} 数据不足，跳过验证")
                                continue

                            # 交叉验证
                            cv_results = validator.cross_validate(all_data, band_id, model_type=args.model_type)

                            # 绘制验证结果
                            val_fig_file = config.FIGURES_DIR / f"validation_{band_id}_{args.model_type}.png"
                            val_results = validator.plot_validation_results(all_data, band_id, save_path=val_fig_file)

                            # 合并结果
                            validation_results[band_id] = {
                                **cv_results,
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
                    try:
                        # 交叉验证
                        cv_results = validator.cross_validate(all_data, args.band, model_type=args.model_type)
                        logger.info(f"交叉验证结果 - RMSE: {cv_results.get('cv_rmse_mean', np.nan):.6f}")

                        # 绘制验证结果
                        val_fig_file = config.FIGURES_DIR / f"validation_{args.band}_{args.model_type}.png"
                        val_results = validator.plot_validation_results(all_data, args.band, save_path=val_fig_file)

                        # 应用校正
                        corrected_data = validator.apply_correction(all_data, args.band)

                        # 保存校正后数据
                        corrected_file = config.DATA_DIR / f"corrected_results_{args.band}.nc"
                        save_dataset(corrected_data.to_dict('list'), corrected_file)
                        logger.info(f"校正后数据已保存: {corrected_file}")

                    except Exception as e:
                        logger.error(f"验证失败: {e}")

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