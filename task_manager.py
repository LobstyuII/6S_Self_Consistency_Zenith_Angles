# ==================== task_manager.py ====================
"""
任务管理器 - 管理分块LUT模拟任务
"""
import sqlite3
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum

from config import ExperimentConfig, LUTTaskConfig
from utils import setup_logger


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskBlock:
    """任务块定义"""
    block_id: str
    band_id: str
    param_ranges: Dict[str, Tuple[float, float]]
    status: TaskStatus
    created_time: str
    started_time: Optional[str] = None
    completed_time: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    data_file: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return {
            **asdict(self),
            'status': self.status.value
        }


class TaskManager:
    """任务管理器"""

    def __init__(self, config: ExperimentConfig, logger=None):
        self.config = config
        self.logger = logger or setup_logger('TaskManager')
        self.task_db = self._init_database()

    def _init_database(self) -> sqlite3.Connection:
        """初始化任务数据库"""
        db_path = Path(self.config.BASE_DIR) / LUTTaskConfig.TASK_MANAGEMENT['task_db_file']
        db_path.parent.mkdir(exist_ok=True)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # 创建任务表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                block_id TEXT PRIMARY KEY,
                band_id TEXT NOT NULL,
                param_ranges TEXT NOT NULL,
                status TEXT NOT NULL,
                created_time TEXT NOT NULL,
                started_time TEXT,
                completed_time TEXT,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                data_file TEXT,
                metadata TEXT,
                FOREIGN KEY (band_id) REFERENCES bands (band_id)
            )
        ''')

        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON tasks(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_band ON tasks(band_id)')

        conn.commit()
        return conn

    def generate_task_blocks(self, band_id: str,
                             param_grids: Dict[str, np.ndarray]) -> List[TaskBlock]:
        """生成任务块"""
        task_blocks = []

        # 获取参数分块配置
        block_configs = LUTTaskConfig.PARAM_BLOCKS

        # 为每个参数维度生成分块
        param_blocks = {}
        for param_name, values in param_grids.items():
            if param_name in block_configs:
                config = block_configs[param_name]
                block_size = config['size']
                overlap = config.get('overlap', 0)

                blocks = self._split_into_blocks(values, block_size, overlap)
                param_blocks[param_name] = blocks

        # 生成所有参数组合的任务块
        from itertools import product
        param_names = list(param_blocks.keys())
        block_indices = [range(len(param_blocks[name])) for name in param_names]

        for indices in product(*block_indices):
            param_ranges = {}
            for i, param_name in enumerate(param_names):
                block_idx = indices[i]
                block_range = param_blocks[param_name][block_idx]
                param_ranges[param_name] = (float(block_range[0]), float(block_range[-1]))

            # 生成块ID
            block_id = self._generate_block_id(band_id, param_ranges)

            task_block = TaskBlock(
                block_id=block_id,
                band_id=band_id,
                param_ranges=param_ranges,
                status=TaskStatus.PENDING,
                created_time=datetime.now().isoformat()
            )

            task_blocks.append(task_block)

        self.logger.info(f"为波段 {band_id} 生成了 {len(task_blocks)} 个任务块")
        return task_blocks

    def _split_into_blocks(self, values: np.ndarray, block_size: float,
                           overlap: float = 0) -> List[np.ndarray]:
        """将值数组分块"""
        blocks = []
        n = len(values)

        if n == 0:
            return blocks

        # 计算步长（考虑重叠）
        step = block_size - overlap

        start_idx = 0
        while start_idx < n:
            # 找到块结束位置
            start_val = values[start_idx]
            end_val = start_val + block_size

            # 找到所有在范围内的值
            mask = (values >= start_val) & (values <= end_val)
            block_values = values[mask]

            if len(block_values) > 0:
                blocks.append(block_values)

            # 移动到下一个块
            next_start = values[values > start_val + step]
            if len(next_start) > 0:
                start_idx = np.where(values == next_start[0])[0][0]
            else:
                break

        return blocks

    def _generate_block_id(self, band_id: str, param_ranges: Dict) -> str:
        """生成唯一的块ID"""
        param_str = json.dumps(param_ranges, sort_keys=True)
        hash_str = hashlib.md5(f"{band_id}_{param_str}".encode()).hexdigest()[:16]
        return f"{band_id}_{hash_str}"

    def save_tasks(self, task_blocks: List[TaskBlock]):
        """保存任务到数据库"""
        cursor = self.task_db.cursor()

        for task in task_blocks:
            cursor.execute('''
                INSERT OR REPLACE INTO tasks 
                (block_id, band_id, param_ranges, status, created_time, 
                 started_time, completed_time, error_message, retry_count, 
                 data_file, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task.block_id,
                task.band_id,
                json.dumps(task.param_ranges),
                task.status.value,
                task.created_time,
                task.started_time,
                task.completed_time,
                task.error_message,
                task.retry_count,
                task.data_file,
                json.dumps(task.metadata) if task.metadata else None
            ))

        self.task_db.commit()

    def get_pending_tasks(self, band_id: Optional[str] = None,
                          limit: int = None) -> List[TaskBlock]:
        """获取待处理任务"""
        query = "SELECT * FROM tasks WHERE status = ?"
        params = [TaskStatus.PENDING.value]

        if band_id:
            query += " AND band_id = ?"
            params.append(band_id)

        query += " ORDER BY created_time"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = self.task_db.cursor()
        cursor.execute(query, params)

        tasks = []
        for row in cursor.fetchall():
            task = self._row_to_task(row)
            tasks.append(task)

        return tasks

    def update_task_status(self, block_id: str, status: TaskStatus,
                           error_message: str = None,
                           data_file: str = None):
        """更新任务状态"""
        cursor = self.task_db.cursor()

        update_time = datetime.now().isoformat()

        if status == TaskStatus.RUNNING:
            cursor.execute('''
                UPDATE tasks 
                SET status = ?, started_time = ?, retry_count = retry_count + 1
                WHERE block_id = ?
            ''', (status.value, update_time, block_id))
        elif status == TaskStatus.COMPLETED:
            cursor.execute('''
                UPDATE tasks 
                SET status = ?, completed_time = ?, data_file = ?
                WHERE block_id = ?
            ''', (status.value, update_time, data_file, block_id))
        elif status == TaskStatus.FAILED:
            cursor.execute('''
                UPDATE tasks 
                SET status = ?, error_message = ?
                WHERE block_id = ?
            ''', (status.value, error_message, block_id))

        self.task_db.commit()

    def _row_to_task(self, row) -> TaskBlock:
        """数据库行转换为TaskBlock"""
        return TaskBlock(
            block_id=row[0],
            band_id=row[1],
            param_ranges=json.loads(row[2]),
            status=TaskStatus(row[3]),
            created_time=row[4],
            started_time=row[5],
            completed_time=row[6],
            error_message=row[7],
            retry_count=row[8],
            data_file=row[9],
            metadata=json.loads(row[10]) if row[10] else None
        )

    def get_progress(self, band_id: str) -> Dict[str, Any]:
        """获取进度统计"""
        cursor = self.task_db.cursor()

        cursor.execute('''
            SELECT status, COUNT(*) 
            FROM tasks 
            WHERE band_id = ?
            GROUP BY status
        ''', (band_id,))

        stats = {status.value: 0 for status in TaskStatus}
        for row in cursor.fetchall():
            stats[row[0]] = row[1]

        total = sum(stats.values())
        if total > 0:
            completion_rate = stats[TaskStatus.COMPLETED.value] / total * 100
        else:
            completion_rate = 0

        return {
            'total_tasks': total,
            'completion_rate': completion_rate,
            'status_counts': stats
        }