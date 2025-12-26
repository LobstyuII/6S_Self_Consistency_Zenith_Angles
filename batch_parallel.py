# ==================== batch_parallel.py ====================
"""
批量并行处理脚本 - 高效处理所有波段的模拟
"""
import argparse
import sys
from pathlib import Path
import time
import pandas as pd

from config import ExperimentConfig
from parallel_simulator import ParallelBlockSimulator
from utils import setup_logger, save_dataset


def main():
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

    args = parser.parse_args()

    config = ExperimentConfig
    logger = setup_logger('BatchParallel')

    # 确定要处理的波段
    if args.bands == 'all':
        bands_to_process = list(config.BANDS.keys())
    else:
        bands_to_process = [b.strip() for b in args.bands.split(',')]

    logger.info(f"处理波段: {bands_to_process}")

    # 创建并行模拟器
    parallel_simulator = ParallelBlockSimulator(config, logger)

    # 生成参数组合
    logger.info("生成参数组合...")
    all_combinations = parallel_simulator.generate_all_param_combinations()

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
            for i in range(0, len(param_list), args.chunk_size):
                chunk = param_list[i:i + args.chunk_size]
                task_blocks.append((band_id, chunk))

    logger.info(f"总参数组合数: {total_combinations}")
    logger.info(f"任务块数: {len(task_blocks)}")

    # 运行并行模拟
    start_time = time.time()
    results = parallel_simulator.simulate_blocks_parallel(
        task_blocks,
        max_workers=args.n_workers
    )
    elapsed_time = time.time() - start_time

    # 保存结果
    output_dir = Path(args.output_dir) if args.output_dir else config.DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    for band_id, df in results.items():
        if len(df) > 0:
            output_file = output_dir / f"simulation_results_{band_id}_parallel.nc"

            data_dict = {}
            for col in df.columns:
                col_data = df[col].values
                if col_data.dtype == object:
                    try:
                        col_data = col_data.astype(str)
                    except:
                        pass
                data_dict[col] = col_data

            save_dataset(data_dict, output_file)
            logger.info(f"波段 {band_id}: 保存 {len(df)} 个样本到 {output_file}")
            total_samples += len(df)

    # 汇总统计
    logger.info("=" * 60)
    logger.info(f"批量并行模拟完成!")
    logger.info(f"总耗时: {elapsed_time:.2f} 秒")
    logger.info(f"总样本数: {total_samples}")
    logger.info(f"平均速度: {total_samples / elapsed_time:.2f} 样本/秒")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()