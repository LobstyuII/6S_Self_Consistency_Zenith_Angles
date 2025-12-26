# ==================== block_simulator.py ====================
"""
分块模拟器 - 执行分块LUT模拟
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from tqdm import tqdm

from config import ExperimentConfig, LUTTaskConfig
from utils import setup_logger, save_dataset
from data_generator import SixSSimulator
from task_manager import TaskManager, TaskStatus, TaskBlock


class BlockSimulator:
    """分块模拟器"""

    def __init__(self, config: ExperimentConfig, task_manager: TaskManager, logger=None):
        self.config = config
        self.task_manager = task_manager
        self.logger = logger or setup_logger('BlockSimulator')

        # 创建波段模拟器
        self.simulators = {
            band_id: SixSSimulator(band_info['wavelength'], logger)
            for band_id, band_info in config.BANDS.items()
        }

    def simulate_block(self, task_block: TaskBlock) -> Optional[Path]:
        """模拟单个任务块"""
        try:
            # 更新状态为运行中
            self.task_manager.update_task_status(task_block.block_id, TaskStatus.RUNNING)

            simulator = self.simulators[task_block.band_id]

            # 生成参数组合
            param_combinations = self._generate_param_combinations(task_block.param_ranges)

            results = []
            for params in tqdm(param_combinations, desc=f"模拟块 {task_block.block_id}"):
                result = simulator.run_closed_loop(params)
                result['band'] = task_block.band_id
                result['wavelength'] = self.config.BANDS[task_block.band_id]['wavelength']
                results.append(result)

            # 保存结果
            df_results = pd.DataFrame(results)
            data_file = self._save_block_data(df_results, task_block)

            # 更新状态为完成
            self.task_manager.update_task_status(
                task_block.block_id,
                TaskStatus.COMPLETED,
                data_file=str(data_file)
            )

            return data_file

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"任务块 {task_block.block_id} 失败: {error_msg}")

            # 更新状态为失败
            self.task_manager.update_task_status(
                task_block.block_id,
                TaskStatus.FAILED,
                error_message=error_msg
            )

            return None

    def _generate_param_combinations(self, param_ranges: Dict[str, Tuple[float, float]]) -> List[Dict]:
        """根据参数范围生成参数组合"""
        # 从配置获取参数值
        from config import ExperimentConfig
        param_configs = ExperimentConfig.PARAM_RANGES

        param_values = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            if param_name in param_configs:
                config = param_configs[param_name]
                if isinstance(config, dict):
                    # 动态生成值
                    step = config.get('step', 1.0)
                    values = np.arange(
                        max(min_val, config.get('min', min_val)),
                        min(max_val, config.get('max', max_val)) + step / 2,
                        step
                    )
                else:
                    # 使用预设值但过滤范围
                    values = [v for v in config if min_val <= v <= max_val]

                param_values[param_name] = values

        # 生成所有组合
        from itertools import product
        param_names = list(param_values.keys())
        value_lists = [param_values[name] for name in param_names]

        combinations = []
        for values in product(*value_lists):
            params = dict(zip(param_names, values))
            combinations.append(params)

        return combinations

    def _save_block_data(self, df: pd.DataFrame, task_block: TaskBlock) -> Path:
        """保存块数据"""
        block_dir = Path(self.config.DATA_DIR) / "blocks" / task_block.band_id
        block_dir.mkdir(parents=True, exist_ok=True)

        data_file = block_dir / f"{task_block.block_id}.nc"

        data_dict = {}
        for col in df.columns:
            col_data = df[col].values
            if col_data.dtype == object:
                try:
                    col_data = col_data.astype(str)
                except:
                    pass
            data_dict[col] = col_data

        # 添加块元数据
        metadata = {
            'block_id': task_block.block_id,
            'band_id': task_block.band_id,
            'param_ranges': task_block.param_ranges,
            'simulation_time': pd.Timestamp.now().isoformat(),
            'n_samples': len(df)
        }

        save_dataset(data_dict, data_file)

        # 保存元数据文件
        metadata_file = block_dir / f"{task_block.block_id}_meta.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        return data_file

    def run_batch(self, band_id: str, max_blocks: int = None,
                  resume: bool = True) -> Dict[str, Any]:
        """运行一批任务块"""
        self.logger.info(f"开始处理波段 {band_id} 的模拟任务")

        # 获取待处理任务
        if resume:
            tasks = self.task_manager.get_pending_tasks(band_id, max_blocks)
        else:
            # 重新开始 - 获取所有任务
            tasks = self._get_all_tasks_for_band(band_id, max_blocks)

        if not tasks:
            self.logger.info(f"波段 {band_id} 没有待处理任务")
            return {'completed': 0, 'failed': 0, 'total': 0}

        self.logger.info(f"找到 {len(tasks)} 个待处理任务块")

        completed = 0
        failed = 0

        for task in tqdm(tasks, desc=f"处理波段 {band_id}"):
            result_file = self.simulate_block(task)

            if result_file:
                completed += 1
            else:
                failed += 1

            # 定期输出进度
            if (completed + failed) % 10 == 0:
                progress = self.task_manager.get_progress(band_id)
                self.logger.info(f"进度: {progress['completion_rate']:.1f}%")

        return {
            'completed': completed,
            'failed': failed,
            'total': len(tasks)
        }

    def _get_all_tasks_for_band(self, band_id: str, limit: int = None) -> List[TaskBlock]:
        """获取波段的所有任务"""
        # 这里需要实现获取所有任务的方法
        # 简化为获取参数配置生成新任务
        param_grids = self._generate_param_grids(band_id)
        tasks = self.task_manager.generate_task_blocks(band_id, param_grids)
        self.task_manager.save_tasks(tasks)

        if limit:
            tasks = tasks[:limit]

        return tasks

    def _generate_param_grids(self, band_id: str) -> Dict[str, np.ndarray]:
        """生成参数网格"""
        from config import ExperimentConfig
        param_configs = ExperimentConfig.PARAM_RANGES

        param_grids = {}
        for param_name, config in param_configs.items():
            if isinstance(config, dict):
                values = np.arange(
                    config['min'],
                    config['max'] + config['step'] / 2,
                    config['step']
                )
            else:
                values = np.array(config)

            param_grids[param_name] = values

        return param_grids