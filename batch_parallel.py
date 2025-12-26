# ==================== batch_parallel.py ====================
"""
批量并行处理脚本 - 高效处理所有波段的模拟
"""
import argparse
import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import ExperimentConfig
from parallel_simulator import ParallelBlockSimulatorImproved
from utils import setup_logger, save_dataset


def run_parallel_simulation_directly():
    """直接运行并行模拟，避免复杂的任务管理器"""
    import argparse
    from pathlib import Path
    import time
    import pandas as pd
    from tqdm import tqdm

    from config import ExperimentConfig
    from parallel_simulator import ParallelBlockSimulatorImproved
    from utils import setup_logger, save_dataset

    parser = argparse.ArgumentParser(description='批量并行6S模拟')
    parser.add_argument('--band', type=str, default='band6',
                        help='要处理的波段')
    parser.add_argument('--n_workers', type=int, default=6,
                        help='工作进程数')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='每个任务块的大小')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'paper_figures', 'sensitivity'],
                        help='模拟模式')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')

    args = parser.parse_args()

    config = ExperimentConfig
    logger = setup_logger('DirectParallel', level='DEBUG' if args.debug else 'INFO')

    # 创建改进的并行模拟器
    parallel_simulator = ParallelBlockSimulatorImproved(config, logger)

    # 生成参数组合（与单线程保持一致）
    logger.info("生成参数组合...")
    all_combinations = parallel_simulator.generate_all_param_combinations_consistent(args.mode)

    # 准备任务块
    task_blocks = []
    total_combinations = 0

    if args.band == 'all':
        bands_to_process = list(config.BANDS.keys())
    else:
        bands_to_process = [args.band]

    for band_id in bands_to_process:
        if band_id in all_combinations:
            param_list = all_combinations[band_id]
            total_combinations += len(param_list)
            logger.info(f"波段 {band_id}: {len(param_list)} 个参数组合")

            # 分批
            chunk_size = args.chunk_size if args.chunk_size else config.PARALLEL_CONFIG.get('chunk_size', 1000)
            for i in range(0, len(param_list), chunk_size):
                chunk = param_list[i:i + chunk_size]
                task_blocks.append((band_id, chunk))

    if not task_blocks:
        logger.error(f"没有为波段 {args.band} 找到参数组合")
        return

    logger.info(f"任务块数: {len(task_blocks)}")
    logger.info(f"总参数组合数: {total_combinations}")

    # 运行并行模拟
    start_time = time.time()

    logger.info("开始并行模拟...")
    results = parallel_simulator.simulate_blocks_parallel_improved(
        task_blocks,
        max_workers=args.n_workers
    )

    elapsed_time = time.time() - start_time

    # 保存结果
    output_dir = config.DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    success_samples = 0

    for band_id, df in results.items():
        if len(df) > 0:
            # 统计成功和失败的样本
            if 'success' in df.columns:
                success_count = df['success'].sum() if df['success'].dtype == bool else (df['success'] == 1).sum()
                success_samples += success_count

            # 过滤成功的结果
            if 'success' in df.columns and not df.empty:
                df_success = df[df['success']].copy() if df['success'].dtype == bool else df[df['success'] == 1].copy()
            else:
                df_success = df

            total_samples += len(df_success)

            if not df_success.empty:
                output_file = output_dir / f"simulation_results_{band_id}_parallel.nc"

                # 确保所有列都是数值类型或可序列化
                data_dict = {}
                for col in df_success.columns:
                    col_data = df_success[col].values

                    # 处理不同类型的数据
                    if col_data.dtype == object:
                        try:
                            # 尝试转换为字符串
                            col_data = col_data.astype(str)
                        except:
                            # 如果无法转换，尝试其他方法
                            try:
                                col_data = np.array([str(x) if pd.notna(x) else '' for x in col_data])
                            except:
                                logger.warning(f"列 {col} 无法转换为字符串，跳过")
                                continue

                    data_dict[col] = col_data

                if data_dict:
                    save_dataset(data_dict, output_file)
                    logger.info(f"波段 {band_id}: 保存 {len(df_success)} 个样本到 {output_file}")
                else:
                    logger.warning(f"波段 {band_id}: 没有有效数据保存")

    # 汇总统计
    logger.info("=" * 60)
    logger.info(f"批量并行模拟完成!")
    logger.info(f"总耗时: {elapsed_time:.2f} 秒")
    logger.info(f"总样本数: {total_samples}")
    if total_samples > 0:
        logger.info(f"平均速度: {total_samples / elapsed_time:.2f} 样本/秒")
    logger.info(f"成功样本: {success_samples}")
    if total_combinations > 0:
        logger.info(f"成功率: {(success_samples / total_combinations) * 100:.2f}%")
    logger.info("=" * 60)


def main():
    """主函数 - 批量并行处理所有波段"""
    parser = argparse.ArgumentParser(description='批量并行6S模拟')
    parser.add_argument('--bands', type=str, default='all',
                        help='要处理的波段，用逗号分隔或all')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='工作进程数')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='每个任务块的大小')
    parser.add_argument('--max_combinations', type=int, default=None,
                        help='最大参数组合数（用于测试）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'paper_figures', 'sensitivity'],
                        help='模拟模式')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')

    args = parser.parse_args()

    config = ExperimentConfig
    logger = setup_logger('BatchParallel', level='DEBUG' if args.debug else 'INFO')

    # 确定要处理的波段
    if args.bands == 'all':
        bands_to_process = list(config.BANDS.keys())
    else:
        bands_to_process = [b.strip() for b in args.bands.split(',')]

    logger.info(f"处理波段: {bands_to_process}")

    # 创建改进的并行模拟器
    parallel_simulator = ParallelBlockSimulatorImproved(config, logger)

    # 生成参数组合
    logger.info("生成参数组合...")
    all_combinations = parallel_simulator.generate_all_param_combinations_consistent(args.mode)

    # 限制参数组合数（用于测试）
    if args.max_combinations:
        for band_id in all_combinations:
            if len(all_combinations[band_id]) > args.max_combinations:
                all_combinations[band_id] = all_combinations[band_id][:args.max_combinations]

    # 准备任务块
    task_blocks = []
    total_combinations = 0

    for band_id in bands_to_process:
        if band_id in all_combinations:
            param_list = all_combinations[band_id]
            total_combinations += len(param_list)

            # 分批
            chunk_size = args.chunk_size if args.chunk_size else config.PARALLEL_CONFIG.get('chunk_size', 1000)
            for i in range(0, len(param_list), chunk_size):
                chunk = param_list[i:i + chunk_size]
                task_blocks.append((band_id, chunk))

    logger.info(f"总参数组合数: {total_combinations}")
    logger.info(f"任务块数: {len(task_blocks)}")

    # 运行并行模拟
    start_time = time.time()

    # 设置工作进程数
    if args.n_workers:
        n_workers = args.n_workers
    else:
        import multiprocessing as mp
        physical_cores = mp.cpu_count() // 2
        max_concurrent = config.PARALLEL_CONFIG.get('max_concurrent_6s', 4)
        n_workers = min(physical_cores, max_concurrent)

    # 运行改进的并行模拟
    results = parallel_simulator.simulate_blocks_parallel_improved(
        task_blocks,
        max_workers=n_workers
    )

    elapsed_time = time.time() - start_time

    # 保存结果
    output_dir = Path(args.output_dir) if args.output_dir else config.DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    success_samples = 0

    for band_id, df in results.items():
        if len(df) > 0:
            # 统计成功和失败的样本
            if 'success' in df.columns:
                success_count = df['success'].sum() if df['success'].dtype == bool else (df['success'] == 1).sum()
                success_samples += success_count

            # 过滤成功的结果
            if 'success' in df.columns and not df.empty:
                df_success = df[df['success']].copy() if df['success'].dtype == bool else df[df['success'] == 1].copy()
            else:
                df_success = df

            total_samples += len(df_success)

            if not df_success.empty:
                output_file = output_dir / f"simulation_results_{band_id}_parallel.nc"

                # 确保所有列都是数值类型或可序列化
                data_dict = {}
                for col in df_success.columns:
                    col_data = df_success[col].values

                    # 处理不同类型的数据
                    if col_data.dtype == object:
                        try:
                            # 尝试转换为字符串
                            col_data = col_data.astype(str)
                        except:
                            # 如果无法转换，尝试其他方法
                            try:
                                col_data = np.array([str(x) if pd.notna(x) else '' for x in col_data])
                            except:
                                logger.warning(f"列 {col} 无法转换为字符串，跳过")
                                continue

                    data_dict[col] = col_data

                if data_dict:
                    save_dataset(data_dict, output_file)
                    logger.info(f"波段 {band_id}: 保存 {len(df_success)} 个样本到 {output_file}")
                else:
                    logger.warning(f"波段 {band_id}: 没有有效数据保存")

    # 汇总统计
    logger.info("=" * 60)
    logger.info(f"批量并行模拟完成!")
    logger.info(f"总耗时: {elapsed_time:.2f} 秒")
    logger.info(f"总样本数: {total_samples}")
    if total_samples > 0:
        logger.info(f"平均速度: {total_samples / elapsed_time:.2f} 样本/秒")
    logger.info(f"成功样本: {success_samples}")
    if total_combinations > 0:
        logger.info(f"成功率: {(success_samples / total_combinations) * 100:.2f}%")
    logger.info(f"工作进程数: {n_workers}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_parallel_simulation_directly()