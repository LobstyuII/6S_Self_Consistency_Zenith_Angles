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


# 修改 main.py 中的错误信息显示
def validate_data_columns(df: pd.DataFrame, band_id: str) -> bool:
    """验证数据列是否完整"""
    required_columns = ['sza', 'vza', 'error_absolute', 'rho_true', 'rho_retrieved']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Warning: Band {band_id} data missing columns: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        return False
    return True


def repair_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """修复缺失的列"""
    df_repair = df.copy()

    # 如果 error_absolute 不存在，尝试从其他列计算
    if 'error_absolute' not in df_repair.columns:
        if 'rho_retrieved' in df_repair.columns and 'rho_true' in df_repair.columns:
            print("Calculating missing error_absolute column...")
            df_repair['error_absolute'] = df_repair['rho_retrieved'] - df_repair['rho_true']
        else:
            print("Cannot calculate error_absolute, creating empty column")
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
                        help='目标波段 (band1, band2, band3, all)')
    parser.add_argument('--model_type', type=str, default='lut',
                        help='校正模型类型 (lut, linear, polynomial, ml)')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='并行工作进程数')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')
    parser.add_argument('--force_repair', action='store_true',
                        help='强制修复数据列')
    parser.add_argument('--force_regenerate', action='store_true',
                        help='强制重新生成数据')
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

            results = {}

            if args.band == 'all':
                # 运行所有波段
                simulator = BatchSimulator(config, logger)

                for band_id in config.BANDS.keys():
                    logger.info(f"开始处理波段: {band_id}")

                    # 检查是否已有数据
                    data_file = config.DATA_DIR / f"simulation_results_{band_id}_{args.mode}.nc"

                    # 检查是否需要重新生成
                    should_regenerate = False
                    if args.force_regenerate:
                        should_regenerate = True
                        logger.info(f"强制重新生成波段 {band_id} 的数据")
                    elif not data_file.exists():
                        should_regenerate = True
                        logger.info(f"数据文件不存在: {data_file}")
                    elif data_file.stat().st_size < 100:  # 文件太小，可能损坏
                        should_regenerate = True
                        logger.warning(f"数据文件可能损坏（大小: {data_file.stat().st_size} 字节）")
                    elif not args.debug and not args.force_repair:
                        # 尝试加载并验证数据
                        try:
                            data_dict = load_dataset(data_file)
                            if not data_dict or len(data_dict) == 0:
                                logger.warning(f"数据文件为空或损坏: {data_file}")
                                should_regenerate = True
                            else:
                                df_result = pd.DataFrame(data_dict)
                                if len(df_result) == 0:
                                    logger.warning(f"DataFrame为空: {data_file}")
                                    should_regenerate = True
                                else:
                                    # 检查必要的列
                                    required_columns = ['sza', 'vza', 'error_absolute', 'rho_true', 'rho_retrieved']
                                    missing_columns = [col for col in required_columns if col not in df_result.columns]

                                    if missing_columns:
                                        logger.warning(f"波段 {band_id} 数据缺少列: {missing_columns}")
                                        should_regenerate = True
                                    else:
                                        results[band_id] = df_result
                                        logger.info(f"波段 {band_id} 数据加载成功，形状: {df_result.shape}")
                                        logger.info(f"数据示例 - sza: {df_result['sza'].iloc[0]:.1f}, "
                                                    f"vza: {df_result['vza'].iloc[0]:.1f}, "
                                                    f"error: {df_result['error_absolute'].iloc[0]:.6f}")
                        except Exception as e:
                            logger.error(f"加载数据失败: {e}")
                            should_regenerate = True

                    # 如果需要重新生成数据
                    if should_regenerate:
                        logger.info(f"生成波段 {band_id} 的数据...")
                        results[band_id] = simulator.run_batch_simulation(band_id, args.mode)
                    else:
                        logger.info(f"使用现有数据: {data_file}")

            else:
                # 单个波段
                # 检查是否已有数据
                data_file = config.DATA_DIR / f"simulation_results_{args.band}_{args.mode}.nc"

                should_regenerate = False
                if args.force_regenerate:
                    should_regenerate = True
                    logger.info(f"强制重新生成波段 {args.band} 的数据")
                elif not data_file.exists():
                    should_regenerate = True
                    logger.info(f"数据文件不存在: {data_file}")
                elif data_file.stat().st_size < 100:
                    should_regenerate = True
                    logger.warning(f"数据文件可能损坏（大小: {data_file.stat().st_size} 字节）")
                elif not args.debug and not args.force_repair:
                    # 尝试加载并验证数据
                    try:
                        data_dict = load_dataset(data_file)
                        if not data_dict or len(data_dict) == 0:
                            logger.warning(f"数据文件为空或损坏: {data_file}")
                            should_regenerate = True
                        else:
                            df_result = pd.DataFrame(data_dict)
                            if len(df_result) == 0:
                                logger.warning(f"DataFrame为空: {data_file}")
                                should_regenerate = True
                            else:
                                # 检查必要的列
                                required_columns = ['sza', 'vza', 'error_absolute', 'rho_true', 'rho_retrieved']
                                missing_columns = [col for col in required_columns if col not in df_result.columns]

                                if missing_columns:
                                    logger.warning(f"数据缺少列: {missing_columns}")
                                    should_regenerate = True
                                else:
                                    results = {args.band: df_result}
                                    logger.info(f"数据加载成功，形状: {df_result.shape}")
                                    logger.info(f"数据示例 - sza: {df_result['sza'].iloc[0]:.1f}, "
                                                f"vza: {df_result['vza'].iloc[0]:.1f}, "
                                                f"error: {df_result['error_absolute'].iloc[0]:.6f}")
                    except Exception as e:
                        logger.error(f"加载数据失败: {e}")
                        should_regenerate = True

                # 如果需要重新生成数据
                if should_regenerate:
                    logger.info(f"生成波段 {args.band} 的数据...")
                    simulator = BatchSimulator(config, logger)
                    results = {args.band: simulator.run_batch_simulation(args.band, args.mode)}
                else:
                    logger.info(f"使用现有数据: {data_file}")

        else:
            # 加载现有数据
            logger.info("加载现有数据")

            if args.band == 'all':
                results = {}
                for band_id in config.BANDS.keys():
                    data_file = config.DATA_DIR / f"simulation_results_{band_id}_{args.mode}.nc"

                    if not data_file.exists():
                        # 尝试加载其他模式的数据
                        alt_files = list(config.DATA_DIR.glob(f"simulation_results_{band_id}_*.nc"))
                        if alt_files:
                            data_file = alt_files[0]
                            logger.info(f"波段 {band_id} 使用替代数据文件: {data_file}")
                        else:
                            logger.warning(f"找不到波段 {band_id} 的数据文件，跳过")
                            continue

                    try:
                        # 加载数据
                        data_dict = load_dataset(data_file)
                        if not data_dict or len(data_dict) == 0:
                            logger.warning(f"波段 {band_id} 数据文件为空")
                            continue

                        df_result = pd.DataFrame(data_dict)

                        if len(df_result) == 0:
                            logger.warning(f"波段 {band_id} DataFrame为空")
                            continue

                        # 验证和修复数据
                        if not validate_data_columns(df_result, band_id) or args.force_repair:
                            logger.info(f"修复波段 {band_id} 数据列...")
                            df_result = repair_missing_columns(df_result)

                            # 保存修复后的数据
                            save_dataset({col: df_result[col].values for col in df_result.columns}, data_file)
                            logger.info(f"波段 {band_id} 数据已修复并重新保存: {data_file}")

                        results[band_id] = df_result
                        logger.info(f"波段 {band_id} 数据加载成功，形状: {df_result.shape}")

                    except Exception as e:
                        logger.error(f"波段 {band_id} 加载数据失败: {e}")

            else:
                # 单个波段
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
                    if not data_dict or len(data_dict) == 0:
                        raise ValueError(f"数据文件为空: {data_file}")

                    df_result = pd.DataFrame(data_dict)

                    if len(df_result) == 0:
                        raise ValueError(f"DataFrame为空: {data_file}")

                    # 验证和修复数据
                    if not validate_data_columns(df_result, args.band) or args.force_repair:
                        logger.info("修复数据列...")
                        df_result = repair_missing_columns(df_result)

                        # 保存修复后的数据
                        save_dataset({col: df_result[col].values for col in df_result.columns}, data_file)
                        logger.info(f"数据已修复并重新保存: {data_file}")

                    results = {args.band: df_result}
                    logger.info(f"数据加载成功，形状: {df_result.shape}")
                    logger.info(f"数据列: {df_result.columns.tolist()}")

                    # 显示数据统计
                    if 'error_absolute' in df_result.columns:
                        error_data = df_result['error_absolute'].dropna()
                        logger.info(f"误差统计 - 均值: {error_data.mean():.6f}, "
                                    f"标准差: {error_data.std():.6f}, "
                                    f"有效样本: {len(error_data)}")

                except Exception as e:
                    logger.error(f"加载数据失败: {e}")
                    raise

        # 检查是否有有效数据
        if not results:
            logger.error("没有有效数据，程序退出")
            return

        valid_bands = [band_id for band_id, df in results.items()
                       if isinstance(df, pd.DataFrame) and len(df) > 0]

        if not valid_bands:
            logger.error("所有波段都没有有效数据，程序退出")
            return

        logger.info(f"有效波段: {valid_bands}")

        # 阶段2: 误差分析
        if args.phase in ['all', 'analyze'] and results:
            logger.info("阶段2: 误差分析")

            # 检查results中是否有实际数据
            valid_results = {}
            for band_id, df in results.items():
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    # 检查是否有必要的列
                    required_columns = ['sza', 'vza', 'error_absolute']
                    if all(col in df.columns for col in required_columns):
                        # 检查是否有误差数据
                        if df['error_absolute'].notna().sum() > 0:
                            valid_results[band_id] = df
                            logger.info(f"波段 {band_id} 有 {df['error_absolute'].notna().sum()} 个有效误差样本")
                        else:
                            logger.warning(f"波段 {band_id} 没有有效的误差数据")
                    else:
                        missing = [col for col in required_columns if col not in df.columns]
                        logger.warning(f"波段 {band_id} 缺少列: {missing}")

            if not valid_results:
                logger.warning("没有有效数据用于误差分析")
            else:
                # 如果分析所有波段且多于一个波段有效，先合并数据
                if args.band == 'all' and len(valid_results) > 1:
                    try:
                        logger.info("分析所有波段合并数据...")

                        # 为每个波段数据添加波段标识
                        band_data_list = []
                        for band_id, df in valid_results.items():
                            df_copy = df.copy()
                            df_copy['band'] = band_id
                            df_copy['wavelength'] = config.BANDS.get(band_id, {}).get('wavelength', np.nan)
                            band_data_list.append(df_copy)

                        all_data = pd.concat(band_data_list, ignore_index=True)
                        all_results = {'all': all_data}

                        # 检查合并后的数据
                        logger.info(f"合并数据形状: {all_data.shape}")
                        logger.info(f"合并数据有效误差样本: {all_data['error_absolute'].notna().sum()}")

                        analyzer_all = ErrorAnalyzer(all_results, logger)

                        # 生成综合摘要图（所有波段）
                        summary_file = config.FIGURES_DIR / f"error_summary_all_bands.png"
                        analyzer_all.create_summary_figure(band_id='all', save_path=summary_file)
                        logger.info(f"所有波段综合摘要图已保存: {summary_file}")

                        # 生成误差分布图（所有波段）
                        dist_file = config.FIGURES_DIR / f"error_distribution_all_bands.png"
                        analyzer_all.plot_error_distribution(dist_file)
                        logger.info(f"所有波段误差分布图已保存: {dist_file}")

                    except Exception as e:
                        logger.error(f"所有波段合并分析失败: {e}")
                        import traceback
                        logger.error(traceback.format_exc())

                # 分析每个波段单独的数据
                for band_id in list(valid_results.keys()):
                    try:
                        logger.info(f"分析波段 {band_id}...")

                        df_band = valid_results[band_id]

                        # 检查数据量
                        if len(df_band) < 10:
                            logger.warning(f"波段 {band_id} 数据量不足 ({len(df_band)})，跳过详细分析")
                            continue

                        # 检查误差数据量
                        error_count = df_band['error_absolute'].notna().sum()
                        if error_count < 10:
                            logger.warning(f"波段 {band_id} 有效误差数据不足 ({error_count})，跳过详细分析")
                            continue

                        # 创建分析器
                        band_results = {band_id: df_band}
                        analyzer = ErrorAnalyzer(band_results, logger)

                        # 生成误差报告
                        report_file = config.RESULTS_DIR / f"error_analysis_report_{band_id}.txt"

                        # 计算并保存基本统计信息
                        try:
                            stats = analyzer.calculate_overall_statistics()
                            with open(report_file, 'w', encoding='utf-8') as f:
                                f.write("误差分析报告\n")
                                f.write(f"波段: {band_id}\n")
                                f.write(f"样本数: {len(df_band)}\n")
                                f.write(f"有效误差数据: {error_count}\n")
                                if band_id in stats:
                                    for key, value in stats[band_id].items():
                                        if key != 'error':
                                            f.write(f"{key}: {value:.6f}\n")
                                else:
                                    f.write("无法计算统计量\n")
                            logger.info(f"波段 {band_id} 误差分析报告已保存: {report_file}")
                        except Exception as e:
                            logger.warning(f"波段 {band_id} 生成报告失败: {e}")

                        # 尝试生成关键图表
                        try:
                            summary_file = config.FIGURES_DIR / f"error_summary_{band_id}.png"
                            analyzer.create_summary_figure(band_id=band_id, save_path=summary_file)
                            logger.info(f"波段 {band_id} 综合摘要图已保存: {summary_file}")
                        except Exception as e:
                            logger.warning(f"波段 {band_id} 绘制综合摘要图失败: {e}")

                    except Exception as e:
                        logger.error(f"波段 {band_id} 误差分析失败: {e}")

                else:
                    # 单个波段
                    if args.band in results:
                        df_band = results[args.band]

                        if 'error_absolute' in df_band.columns:
                            # 确保有误差数据
                            error_count = df_band['error_absolute'].notna().sum()
                            if error_count == 0:
                                logger.warning("没有有效的误差数据，跳过误差分析")
                            else:
                                analyzer = ErrorAnalyzer(results, logger)

                                # 生成误差报告
                                report_file = config.RESULTS_DIR / f"error_analysis_report_{args.band}.txt"

                                try:
                                    stats = analyzer.calculate_overall_statistics()
                                    with open(report_file, 'w', encoding='utf-8') as f:
                                        f.write("误差分析报告\n")
                                        f.write(f"波段: {args.band}\n")
                                        f.write(f"样本数: {len(df_band)}\n")
                                        f.write(f"有效误差数据: {error_count}\n")
                                        if args.band in stats:
                                            for key, value in stats[args.band].items():
                                                if key != 'error':
                                                    f.write(f"{key}: {value:.6f}\n")
                                        else:
                                            f.write("无法计算统计量\n")
                                    logger.info(f"误差分析报告已保存: {report_file}")
                                except Exception as e:
                                    logger.warning(f"生成误差报告失败: {e}")

                                # 尝试绘制图表
                                try:
                                    summary_file = config.FIGURES_DIR / f"error_summary_{args.band}.png"
                                    analyzer.create_summary_figure(band_id=args.band, save_path=summary_file)
                                    logger.info(f"综合摘要图已保存: {summary_file}")
                                except Exception as e:
                                    logger.warning(f"绘制综合摘要图失败: {e}")

                                try:
                                    dist_file = config.FIGURES_DIR / f"error_distribution_{args.band}.png"
                                    analyzer.plot_error_distribution(dist_file)
                                    logger.info(f"误差分布图已保存: {dist_file}")
                                except Exception as e:
                                    logger.warning(f"绘制误差分布图失败: {e}")

                                try:
                                    surface_file = config.FIGURES_DIR / f"3d_surface_{args.band}.png"
                                    analyzer.plot_3d_error_surface(save_path=surface_file)
                                    logger.info(f"3D误差曲面图已保存: {surface_file}")
                                except Exception as e:
                                    logger.warning(f"绘制3D误差曲面失败: {e}")

                # 阶段2.5: 敏感性分析
                if args.phase in ['all', 'sensitivity'] and results:
                    logger.info("阶段2.5: 敏感性分析")

                    if args.band == 'all':
                        # 合并所有波段数据
                        band_data_list = []
                        for band_id, df in results.items():
                            if isinstance(df, pd.DataFrame) and len(df) > 0:
                                df_copy = df.copy()
                                df_copy['band'] = band_id
                                band_data_list.append(df_copy)

                        if band_data_list:
                            all_data = pd.concat(band_data_list, ignore_index=True)
                        else:
                            all_data = pd.DataFrame()

                        # 检查是否有足够的误差数据
                        if 'error_absolute' in all_data.columns:
                            error_count = all_data['error_absolute'].notna().sum()
                        else:
                            error_count = 0

                        if error_count < 10:
                            logger.warning(f"误差数据不足 ({error_count} < 10)，跳过敏感性分析")
                        else:
                            # 创建敏感性分析器
                            sensitivity_analyzer = SensitivityAnalyzer(all_data, logger)

                            # 生成敏感性报告
                            sens_report_file = config.RESULTS_DIR / f"sensitivity_report_all_{args.mode}.txt"

                            try:
                                sens_report = sensitivity_analyzer.generate_sensitivity_report(sens_report_file)
                                logger.info(f"敏感性分析报告已生成: {sens_report_file}")
                            except Exception as e:
                                logger.error(f"生成敏感性报告失败: {e}")
                                # 创建简单的报告
                                with open(sens_report_file, 'w', encoding='utf-8') as f:
                                    f.write(f"敏感性分析报告\n")
                                    f.write(f"波段: all\n")
                                    f.write(f"模式: {args.mode}\n")
                                    f.write(f"样本数: {len(all_data)}\n")
                                    f.write(f"错误: {str(e)}\n")

                            # 尝试绘制敏感性分析图
                            try:
                                # 单因素敏感性图
                                single_factor_file = config.FIGURES_DIR / f"sensitivity_single_factor_all_{args.mode}.png"
                                sensitivity_analyzer.plot_single_factor_sensitivity(single_factor_file)
                            except Exception as e:
                                logger.warning(f"绘制单因素敏感性图失败: {e}")

                            try:
                                # 净效应敏感性分析图
                                net_sensitivity_file = config.FIGURES_DIR / f"sensitivity_net_effect_all_{args.mode}.png"
                                sensitivity_analyzer.plot_net_sensitivity_analysis(net_sensitivity_file)
                            except Exception as e:
                                logger.warning(f"绘制净效应敏感性图失败: {e}")

                            try:
                                # 部分依赖图（绝对误差）
                                pdp_abs_file = config.FIGURES_DIR / f"sensitivity_pdp_absolute_all_{args.mode}.png"
                                sensitivity_analyzer.plot_partial_dependence_analysis('error_absolute', pdp_abs_file)
                            except Exception as e:
                                logger.warning(f"绘制绝对误差部分依赖图失败: {e}")

                            try:
                                # 部分依赖图（相对误差，如果存在）
                                if 'error_relative' in all_data.columns:
                                    pdp_rel_file = config.FIGURES_DIR / f"sensitivity_pdp_relative_all_{args.mode}.png"
                                    sensitivity_analyzer.plot_partial_dependence_analysis('error_relative',
                                                                                          pdp_rel_file)
                            except Exception as e:
                                logger.warning(f"绘制相对误差部分依赖图失败: {e}")

                            try:
                                # 交互效应图
                                interaction_file = config.FIGURES_DIR / f"sensitivity_interaction_all_{args.mode}.png"
                                sensitivity_analyzer.plot_interaction_effects(interaction_file)
                            except Exception as e:
                                logger.warning(f"绘制交互效应图失败: {e}")

                            try:
                                # 大气质量分解图
                                airmass_file = config.FIGURES_DIR / f"sensitivity_airmass_all_{args.mode}.png"
                                sensitivity_analyzer.plot_airmass_breakdown(airmass_file)
                            except Exception as e:
                                logger.warning(f"绘制大气质量分解图失败: {e}")

                            # 生成全面敏感性报告
                            try:
                                comprehensive_dir = config.RESULTS_DIR / "comprehensive_sensitivity_all"
                                comprehensive_report = sensitivity_analyzer.generate_comprehensive_sensitivity_report(
                                    comprehensive_dir)
                                logger.info(f"全面敏感性分析报告已生成到: {comprehensive_dir}")
                            except Exception as e:
                                logger.warning(f"生成全面敏感性报告失败: {e}")

                            logger.info("敏感性分析完成")

                    else:
                        # 单个波段
                        if args.band in results:
                            band_data = results[args.band]

                            # 检查是否有足够的误差数据
                            if 'error_absolute' in band_data.columns:
                                error_count = band_data['error_absolute'].notna().sum()
                            else:
                                error_count = 0

                            if error_count < 10:
                                logger.warning(f"误差数据不足 ({error_count} < 10)，跳过敏感性分析")
                            else:
                                # 创建敏感性分析器
                                sensitivity_analyzer = SensitivityAnalyzer(band_data, logger)

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
                                        f.write(f"样本数: {len(band_data)}\n")
                                        f.write(f"错误: {str(e)}\n")

                                # 尝试绘制敏感性分析图
                                try:
                                    single_factor_file = config.FIGURES_DIR / f"sensitivity_single_factor_{args.band}_{args.mode}.png"
                                    sensitivity_analyzer.plot_single_factor_sensitivity(single_factor_file)
                                except Exception as e:
                                    logger.warning(f"绘制单因素敏感性图失败: {e}")

                                try:
                                    # 净效应敏感性分析图
                                    net_sensitivity_file = config.FIGURES_DIR / f"sensitivity_net_effect_{args.band}_{args.mode}.png"
                                    sensitivity_analyzer.plot_net_sensitivity_analysis(net_sensitivity_file)
                                except Exception as e:
                                    logger.warning(f"绘制净效应敏感性图失败: {e}")

                                try:
                                    # 部分依赖图（绝对误差）
                                    pdp_abs_file = config.FIGURES_DIR / f"sensitivity_pdp_absolute_{args.band}_{args.mode}.png"
                                    sensitivity_analyzer.plot_partial_dependence_analysis('error_absolute',
                                                                                          pdp_abs_file)
                                except Exception as e:
                                    logger.warning(f"绘制绝对误差部分依赖图失败: {e}")

                                try:
                                    # 部分依赖图（相对误差，如果存在）
                                    if 'error_relative' in band_data.columns:
                                        pdp_rel_file = config.FIGURES_DIR / f"sensitivity_pdp_relative_{args.band}_{args.mode}.png"
                                        sensitivity_analyzer.plot_partial_dependence_analysis('error_relative',
                                                                                              pdp_rel_file)
                                except Exception as e:
                                    logger.warning(f"绘制相对误差部分依赖图失败: {e}")

                                try:
                                    # 交互效应图
                                    interaction_file = config.FIGURES_DIR / f"sensitivity_interaction_{args.band}_{args.mode}.png"
                                    sensitivity_analyzer.plot_interaction_effects(interaction_file)
                                except Exception as e:
                                    logger.warning(f"绘制交互效应图失败: {e}")

                                try:
                                    airmass_file = config.FIGURES_DIR / f"sensitivity_airmass_{args.band}_{args.mode}.png"
                                    sensitivity_analyzer.plot_airmass_breakdown(airmass_file)
                                except Exception as e:
                                    logger.warning(f"绘制大气质量分解图失败: {e}")

                                # 生成全面敏感性报告
                                try:
                                    comprehensive_dir = config.RESULTS_DIR / f"comprehensive_sensitivity_{args.band}"
                                    comprehensive_report = sensitivity_analyzer.generate_comprehensive_sensitivity_report(
                                        comprehensive_dir)
                                    logger.info(f"全面敏感性分析报告已生成到: {comprehensive_dir}")
                                except Exception as e:
                                    logger.warning(f"生成全面敏感性报告失败: {e}")

                                logger.info("敏感性分析完成")

        # 阶段3: 模型训练
        if args.phase in ['all', 'train'] and results:
            logger.info("阶段3: 模型训练")

            if args.band == 'all':
                # 合并所有波段数据
                band_data_list = []
                for band_id, df in results.items():
                    if isinstance(df, pd.DataFrame) and len(df) > 0:
                        df_copy = df.copy()
                        df_copy['band'] = band_id
                        band_data_list.append(df_copy)

                if band_data_list:
                    all_data = pd.concat(band_data_list, ignore_index=True)
                else:
                    all_data = pd.DataFrame()

                # 检查是否有足够的训练数据
                if 'error_absolute' in all_data.columns:
                    error_data = all_data['error_absolute'].dropna()
                else:
                    error_data = pd.Series([], dtype=float)

                if len(error_data) < 50:
                    logger.warning(f"有效误差数据不足 ({len(error_data)} < 50)，跳过模型训练")
                else:
                    trainer = ModelTrainer(config, logger)

                    for band_id in config.BANDS.keys():
                        try:
                            # 检查该波段是否有足够数据
                            if band_id in results and isinstance(results[band_id], pd.DataFrame):
                                band_data = results[band_id]
                                if len(band_data) < 50:
                                    logger.warning(f"波段 {band_id} 数据不足 ({len(band_data)} < 50)，跳过训练")
                                    continue

                                # 训练模型
                                model_name = f"{args.model_type}_{band_id}"
                                model = trainer.train_model(all_data, band_id, args.model_type, model_name)
                                logger.info(f"波段 {band_id} 模型训练完成")

                        except Exception as e:
                            logger.error(f"波段 {band_id} 模型训练失败: {e}")

            else:
                # 训练单个波段
                if args.band in results:
                    all_data = pd.concat([df for df in results.values() if isinstance(df, pd.DataFrame)],
                                         ignore_index=True)

                    # 检查是否有足够的训练数据
                    if 'error_absolute' in all_data.columns:
                        error_data = all_data['error_absolute'].dropna()
                    else:
                        error_data = pd.Series([], dtype=float)

                    if len(error_data) < 50:
                        logger.warning(f"有效误差数据不足 ({len(error_data)} < 50)，跳过模型训练")
                    else:
                        trainer = ModelTrainer(config, logger)

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
        if args.phase in ['all', 'validate'] and results:
            logger.info("阶段4: 模型验证")

            if args.band == 'all':
                # 合并所有波段数据
                band_data_list = []
                for band_id, df in results.items():
                    if isinstance(df, pd.DataFrame) and len(df) > 0:
                        df_copy = df.copy()
                        df_copy['band'] = band_id
                        band_data_list.append(df_copy)

                if band_data_list:
                    all_data = pd.concat(band_data_list, ignore_index=True)
                else:
                    all_data = pd.DataFrame()

                # 检查是否有验证数据
                if 'error_absolute' in all_data.columns:
                    error_count = all_data['error_absolute'].notna().sum()
                else:
                    error_count = 0

                if error_count < 20:
                    logger.warning(f"验证数据不足 ({error_count} < 20)，跳过模型验证")
                else:
                    validator = ModelValidator(config, logger)
                    validation_results = {}

                    for band_id in config.BANDS.keys():
                        try:
                            # 检查该波段是否有足够数据
                            if band_id in results and isinstance(results[band_id], pd.DataFrame):
                                band_data = results[band_id]
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
                if args.band in results:
                    all_data = pd.concat([df for df in results.values() if isinstance(df, pd.DataFrame)],
                                         ignore_index=True)

                    # 检查是否有验证数据
                    if 'error_absolute' in all_data.columns:
                        error_count = all_data['error_absolute'].notna().sum()
                    else:
                        error_count = 0

                    if error_count < 20:
                        logger.warning(f"验证数据不足 ({error_count} < 20)，跳过模型验证")
                    else:
                        validator = ModelValidator(config, logger)

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
                            save_dataset({col: corrected_data[col].values for col in corrected_data.columns},
                                         corrected_file)
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