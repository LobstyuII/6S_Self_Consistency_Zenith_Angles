#!/usr/bin/env python
"""
清理并重新生成数据
"""
import shutil
from pathlib import Path
import sys

from config import ExperimentConfig
from data_generator import BatchSimulator
from utils import setup_logger


def recreate_data():
    """重新生成所有数据"""
    logger = setup_logger('RecreateData')

    # 清理数据目录
    data_dir = ExperimentConfig.DATA_DIR
    if data_dir.exists():
        logger.info(f"清理数据目录: {data_dir}")
        for file in data_dir.glob("*.nc"):
            file.unlink()
            logger.info(f"删除: {file}")

    # 重新生成数据
    logger.info("开始重新生成数据...")
    simulator = BatchSimulator(ExperimentConfig, logger)

    # 生成所有波段数据
    results = {}
    for band_id in ExperimentConfig.BANDS.keys():
        logger.info(f"生成波段 {band_id} 数据...")
        try:
            results[band_id] = simulator.run_batch_simulation(band_id, mode='full')
            logger.info(f"波段 {band_id} 数据生成完成: {len(results[band_id])} 行")
        except Exception as e:
            logger.error(f"波段 {band_id} 生成失败: {e}")

    logger.info("数据重新生成完成!")
    return results


if __name__ == "__main__":
    recreate_data()