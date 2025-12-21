# ==================== run_sensitivity_experiments.py ====================
"""
运行敏感性实验的专用脚本
"""
import argparse
import sys
import pandas as pd
import numpy as np
import logging

from config import ExperimentConfig
from data_generator import BatchSimulator
from sensitivity_analyzer_backup12191943 import SensitivityAnalyzer
from utils import setup_logger, load_dataset


def run_single_factor_experiment(band_id='band3'):
    """运行单因素敏感性实验"""
    logger = setup_logger('SingleFactor', level=logging.INFO)

    config = ExperimentConfig
    logger.info(f"开始单因素敏感性实验 - 波段: {band_id}")

    # 运行模拟
    simulator = BatchSimulator(config, logger)
    results = simulator.run_batch_simulation(band_id, mode='single_factor')

    # 分析结果
    analyzer = SensitivityAnalyzer(results, logger)

    # 保存结果
    output_dir = config.RESULTS_DIR / 'sensitivity' / 'single_factor'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成报告
    report_file = output_dir / f'single_factor_report_{band_id}.txt'
    analyzer.generate_sensitivity_report(report_file)

    # 生成图表
    fig_file = output_dir / f'single_factor_analysis_{band_id}.png'
    analyzer.plot_single_factor_sensitivity(fig_file)

    logger.info(f"单因素敏感性实验完成，结果保存在: {output_dir}")
    return results


def run_multi_factor_experiment(band_id='band3'):
    """运行多因素协同敏感性实验"""
    logger = setup_logger('MultiFactor', level=logging.INFO)

    config = ExperimentConfig
    logger.info(f"开始多因素协同敏感性实验 - 波段: {band_id}")

    # 运行模拟
    simulator = BatchSimulator(config, logger)
    results = simulator.run_batch_simulation(band_id, mode='multi_factor')

    # 分析结果
    analyzer = SensitivityAnalyzer(results, logger)

    # 保存结果
    output_dir = config.RESULTS_DIR / 'sensitivity' / 'multi_factor'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成报告
    report_file = output_dir / f'multi_factor_report_{band_id}.txt'
    analyzer.generate_sensitivity_report(report_file)

    # 生成交互效应图
    fig_file = output_dir / f'multi_factor_interaction_{band_id}.png'
    analyzer.plot_interaction_effects(fig_file)

    logger.info(f"多因素协同敏感性实验完成，结果保存在: {output_dir}")
    return results


def run_comprehensive_comparison():
    """运行全面比较实验"""
    logger = setup_logger('Comprehensive', level=logging.INFO)

    config = ExperimentConfig
    logger.info("开始全面比较实验")

    all_results = {}

    for band_id in ['band1', 'band3', 'band6']:  # 选择代表性波段
        logger.info(f"处理波段: {band_id}")

        # 运行不同模式的实验
        simulator = BatchSimulator(config, logger)

        # 1. 全参数模式
        results_full = simulator.run_batch_simulation(band_id, mode='full')

        # 2. 单因素模式
        results_single = simulator.run_batch_simulation(band_id, mode='single_factor')

        # 3. 多因素模式
        results_multi = simulator.run_batch_simulation(band_id, mode='multi_factor')

        # 存储结果
        all_results[band_id] = {
            'full': results_full,
            'single_factor': results_single,
            'multi_factor': results_multi
        }

    # 比较不同模式的结果
    comparison_results = compare_experiment_modes(all_results)

    # 保存比较结果
    output_dir = config.RESULTS_DIR / 'sensitivity' / 'comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存为CSV
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(output_dir / 'mode_comparison.csv')

    logger.info(f"全面比较实验完成，结果保存在: {output_dir}")
    return all_results


def compare_experiment_modes(all_results):
    """比较不同实验模式的结果"""
    comparison = []

    for band_id, modes_results in all_results.items():
        for mode_name, results_df in modes_results.items():
            # 计算统计指标
            errors = results_df['error_absolute'].dropna()

            if len(errors) > 0:
                stats = {
                    'band': band_id,
                    'mode': mode_name,
                    'mean_error': errors.mean(),
                    'std_error': errors.std(),
                    'rmse': np.sqrt(np.mean(errors ** 2)),
                    'max_error': errors.max(),
                    'min_error': errors.min(),
                    'n_samples': len(errors),
                    'success_rate': results_df['closed_loop_success'].mean() * 100
                }

                # 计算大气参数的敏感性
                if 'h2o' in results_df.columns and 'o3' in results_df.columns:
                    # 水汽敏感性
                    valid_h2o = results_df[['h2o', 'error_absolute']].dropna()
                    if len(valid_h2o) > 10:
                        h2o_corr = np.corrcoef(valid_h2o['h2o'], valid_h2o['error_absolute'])[0, 1]
                        stats['h2o_sensitivity'] = h2o_corr

                    # 臭氧敏感性
                    valid_o3 = results_df[['o3', 'error_absolute']].dropna()
                    if len(valid_o3) > 10:
                        o3_corr = np.corrcoef(valid_o3['o3'], valid_o3['error_absolute'])[0, 1]
                        stats['o3_sensitivity'] = o3_corr

                comparison.append(stats)

    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行敏感性实验')
    parser.add_argument('--experiment', type=str,
                        choices=['single', 'multi', 'comprehensive', 'all'],
                        default='all', help='实验类型')
    parser.add_argument('--band', type=str, default='band3',
                        help='目标波段')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    try:
        if args.experiment in ['single', 'all']:
            run_single_factor_experiment(args.band)

        if args.experiment in ['multi', 'all']:
            run_multi_factor_experiment(args.band)

        if args.experiment in ['comprehensive', 'all']:
            run_comprehensive_comparison()

        print("敏感性实验完成!")

    except Exception as e:
        print(f"实验失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)