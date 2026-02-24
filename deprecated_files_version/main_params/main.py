# ==================== main.py (完整修改版) ====================
"""
主程序模块 - 支持分块化LUT模拟
"""
import argparse
import sys
import pandas as pd
import logging
from pathlib import Path

from config import ExperimentConfig
from deprecated.error_analyzer import ErrorAnalyzer
from deprecated.paper_figures import PaperFiguresGenerator
from utils import setup_logger, load_dataset
from deprecated.sensitivity_analyzer import SensitivityAnalyzer

import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='6S几何误差校正实验')
    parser.add_argument('--phase', type=str,
                        choices=['analyze', 'sensitivity', 'paper_figures'],
                        default='analyze', help='运行阶段')
    parser.add_argument('--band', type=str, default='band1',
                        choices=['band1', 'band2', 'band3', 'band4', 'band5', 'band6', 'all'],
                        help='目标波段')
    parser.add_argument('--data_file', type=str, default=None,
                        help='数据文件路径（可选，默认自动查找）')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')

    args = parser.parse_args()

    config = ExperimentConfig
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger('MainSimple', config.BASE_DIR / 'experiment_simple.log', level=log_level)

    logger.info("=" * 60)
    logger.info(f"6S几何误差校正实验 - 精简版")
    logger.info(f"运行阶段: {args.phase}")
    logger.info(f"目标波段: {args.band}")
    logger.info("=" * 60)

    # 确保目录存在
    config.MANU_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # 加载数据
        results = {}

        if args.data_file:
            data_files = [Path(args.data_file)]
        else:
            # 自动查找数据文件
            if args.band == 'all':
                data_files = list(config.DATA_DIR.glob("simulation_results_band*_parallel.nc"))
            else:
                data_files = list(config.DATA_DIR.glob(f"simulation_results_{args.band}_parallel.nc"))

        if not data_files:
            logger.error("找不到数据文件")
            return

        for data_file in data_files:
            try:
                logger.info(f"加载数据文件: {data_file}")
                data_dict = load_dataset(data_file)
                df = pd.DataFrame(data_dict)

                # 提取波段信息
                if 'band' in df.columns:
                    band_ids = df['band'].unique()
                    for band_id in band_ids:
                        band_data = df[df['band'] == band_id].copy()
                        if len(band_data) > 0:
                            results[str(band_id)] = band_data
                            logger.info(f"波段 {band_id}: {len(band_data)} 个样本")
                else:
                    # 从文件名提取波段
                    filename = data_file.stem
                    band_id = filename.split('_')[2]  # simulation_results_band1_parallel
                    results[band_id] = df

            except Exception as e:
                logger.error(f"加载文件 {data_file} 失败: {e}")

        if not results:
            logger.error("没有加载到有效数据")
            return

        # 阶段1: 误差分析
        if args.phase in ['analyze', 'all']:
            logger.info("阶段1: 误差分析")

            for band_id, df in results.items():
                try:
                    logger.info(f"分析波段 {band_id}...")
                    band_results = {band_id: df}
                    analyzer = ErrorAnalyzer(band_results, logger)

                    # 生成误差报告
                    report_file = config.RESULTS_DIR / f"error_analysis_report_{band_id}.txt"
                    stats = analyzer.calculate_overall_statistics()

                    with open(report_file, 'w', encoding='utf-8') as f:
                        f.write("误差分析报告\n")
                        f.write(f"波段: {band_id}\n")
                        f.write(f"样本数: {len(df)}\n")
                        f.write(f"有效误差数据: {df['error_absolute'].notna().sum()}\n")
                        if band_id in stats:
                            for key, value in stats[band_id].items():
                                f.write(f"{key}: {value:.6f}\n")

                    logger.info(f"误差分析报告已保存: {report_file}")

                except Exception as e:
                    logger.error(f"波段 {band_id} 误差分析失败: {e}")

        # 阶段2: 敏感性分析
        if args.phase in ['sensitivity', 'all']:
            logger.info("阶段2: 敏感性分析")

            # 合并所有波段数据
            all_data_list = []
            for band_id, df in results.items():
                df_copy = df.copy()
                df_copy['band'] = band_id
                all_data_list.append(df_copy)

            if all_data_list:
                all_data = pd.concat(all_data_list, ignore_index=True)

                if 'error_absolute' in all_data.columns and all_data['error_absolute'].notna().sum() > 10:
                    sensitivity_analyzer = SensitivityAnalyzer(all_data, logger)
                    sens_report_file = config.RESULTS_DIR / f"sensitivity_report_all.txt"

                    try:
                        sens_report = sensitivity_analyzer.generate_sensitivity_report(sens_report_file)
                        logger.info(f"敏感性分析报告已生成: {sens_report_file}")
                    except Exception as e:
                        logger.error(f"生成敏感性报告失败: {e}")

        # 阶段3: 生成论文图表
        if args.phase == 'paper_figures':
            logger.info("阶段3: 生成论文图表")

            try:
                generator = PaperFiguresGenerator(results, logger)
                generator.generate_all_figures()
                logger.info(f"论文图表已生成并保存到: {config.MANU_FIGURES_DIR}")
            except Exception as e:
                logger.error(f"生成论文图表失败: {e}")

        logger.info("=" * 60)
        logger.info("分析完成!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"程序失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()