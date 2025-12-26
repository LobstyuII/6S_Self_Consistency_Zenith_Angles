# ==================== data_merger.py ====================
"""
数据合并器 - 合并分块数据
"""
import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings

from config import ExperimentConfig, LUTTaskConfig
from utils import setup_logger, load_dataset
from task_manager import TaskManager


class DataMerger:
    """数据合并器"""

    def __init__(self, config: ExperimentConfig, task_manager: TaskManager, logger=None):
        self.config = config
        self.task_manager = task_manager
        self.logger = logger or setup_logger('DataMerger')

    def merge_blocks(self, band_id: str, output_file: Optional[Path] = None) -> pd.DataFrame:
        """合并所有完成的块"""
        self.logger.info(f"开始合并波段 {band_id} 的数据块")

        # 获取所有已完成的任务
        completed_tasks = self._get_completed_tasks(band_id)

        if not completed_tasks:
            self.logger.warning(f"波段 {band_id} 没有已完成的数据块")
            return pd.DataFrame()

        self.logger.info(f"找到 {len(completed_tasks)} 个已完成的数据块")

        # 合并数据
        all_data = []
        for task in completed_tasks:
            if task.data_file and Path(task.data_file).exists():
                try:
                    data_dict = load_dataset(Path(task.data_file))
                    df = pd.DataFrame(data_dict)
                    all_data.append(df)
                    self.logger.debug(f"加载块 {task.block_id}: {len(df)} 行")
                except Exception as e:
                    self.logger.error(f"加载块 {task.block_id} 失败: {e}")

        if not all_data:
            self.logger.error("没有成功加载任何数据块")
            return pd.DataFrame()

        # 合并所有数据
        merged_df = pd.concat(all_data, ignore_index=True)

        self.logger.info(f"合并完成: 总共 {len(merged_df)} 行数据")

        # 保存合并后的数据
        if output_file is None:
            output_file = Path(self.config.DATA_DIR) / f"simulation_results_{band_id}_merged.nc"

        data_dict = {}
        for col in merged_df.columns:
            col_data = merged_df[col].values
            if col_data.dtype == object:
                try:
                    col_data = col_data.astype(str)
                except:
                    pass
            data_dict[col] = col_data

        from utils import save_dataset
        save_dataset(data_dict, output_file)

        self.logger.info(f"合并数据已保存: {output_file}")

        return merged_df

    def merge_selective(self, band_id: str, param_constraints: Dict[str, Tuple[float, float]],
                        output_file: Optional[Path] = None) -> pd.DataFrame:
        """选择性合并满足条件的数据块"""
        self.logger.info(f"选择性合并波段 {band_id} 的数据")

        # 获取所有已完成的任务
        completed_tasks = self._get_completed_tasks(band_id)

        if not completed_tasks:
            self.logger.warning(f"波段 {band_id} 没有已完成的数据块")
            return pd.DataFrame()

        # 筛选满足条件的数据块
        selected_tasks = []
        for task in completed_tasks:
            if self._check_param_constraints(task.param_ranges, param_constraints):
                selected_tasks.append(task)

        self.logger.info(f"找到 {len(selected_tasks)} 个满足条件的数据块")

        # 合并选中的数据块
        all_data = []
        for task in selected_tasks:
            if task.data_file and Path(task.data_file).exists():
                try:
                    data_dict = load_dataset(Path(task.data_file))
                    df = pd.DataFrame(data_dict)

                    # 进一步筛选数据行
                    for param, (min_val, max_val) in param_constraints.items():
                        if param in df.columns:
                            df = df[(df[param] >= min_val) & (df[param] <= max_val)]

                    all_data.append(df)
                except Exception as e:
                    self.logger.error(f"加载块 {task.block_id} 失败: {e}")

        if not all_data:
            self.logger.error("没有成功加载任何数据块")
            return pd.DataFrame()

        # 合并数据
        merged_df = pd.concat(all_data, ignore_index=True)

        self.logger.info(f"选择性合并完成: 总共 {len(merged_df)} 行数据")

        # 保存数据
        if output_file is None:
            constraints_str = "_".join([f"{k}_{v[0]}-{v[1]}" for k, v in param_constraints.items()])
            output_file = Path(self.config.DATA_DIR) / f"simulation_results_{band_id}_{constraints_str}.nc"

        data_dict = {}
        for col in merged_df.columns:
            col_data = merged_df[col].values
            if col_data.dtype == object:
                try:
                    col_data = col_data.astype(str)
                except:
                    pass
            data_dict[col] = col_data

        from utils import save_dataset
        save_dataset(data_dict, output_file)

        return merged_df

    def _get_completed_tasks(self, band_id: str) -> List:
        """获取已完成的任务"""
        # 这里需要从任务管理器获取已完成任务
        # 简化实现
        import sqlite3
        from task_manager import TaskStatus

        db_path = Path(self.config.BASE_DIR) / LUTTaskConfig.TASK_MANAGEMENT['task_db_file']
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM tasks 
            WHERE band_id = ? AND status = ?
        ''', (band_id, TaskStatus.COMPLETED.value))

        tasks = []
        for row in cursor.fetchall():
            task_dict = {
                'block_id': row[0],
                'band_id': row[1],
                'param_ranges': eval(row[2]),
                'data_file': row[9]
            }
            tasks.append(type('Task', (), task_dict)())

        conn.close()
        return tasks

    def _check_param_constraints(self, param_ranges: Dict,
                                 constraints: Dict[str, Tuple[float, float]]) -> bool:
        """检查参数范围是否满足约束"""
        for param, (min_val, max_val) in constraints.items():
            if param in param_ranges:
                block_min, block_max = param_ranges[param]
                # 检查是否有重叠
                if block_max < min_val or block_min > max_val:
                    return False
        return True