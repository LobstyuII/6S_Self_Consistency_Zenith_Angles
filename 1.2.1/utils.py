# ==================== utils.py ====================
"""
工具函数模块
"""
import logging
import json
import pickle
import hashlib
from pathlib import Path
from functools import wraps
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import xarray as xr
import pandas as pd
from scipy import interpolate
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from datetime import datetime
from config import ExperimentConfig


# 设置日志
def setup_logger(name: str, log_file: Optional[Path] = None, level=logging.INFO):
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除已有处理器
    if logger.hasHandlers():
        logger.handlers.clear()

    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器（如果提供了文件路径）
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# 缓存装饰器
def cache_result(func):
    """缓存函数结果的装饰器"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # 生成缓存键
        key = hashlib.md5(
            f"{func.__name__}{args}{kwargs}".encode()
        ).hexdigest()

        cache_dir = ExperimentConfig.PARALLEL_CONFIG['cache_dir']
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"{key}.pkl"

        # 检查缓存
        if ExperimentConfig.PARALLEL_CONFIG['use_cache'] and cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        # 执行函数
        result = func(*args, **kwargs)

        # 保存缓存
        if ExperimentConfig.PARALLEL_CONFIG['use_cache']:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)

        return result

    return wrapper


# 数据保存/加载
def save_dataset(data: Dict[str, np.ndarray], filename: Path):
    """保存数据集"""
    if ExperimentConfig.DATA_STORAGE['format'] == 'netcdf':
        # 转换为xarray Dataset
        ds = xr.Dataset()
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                ds[key] = xr.DataArray(value)

        # 添加属性
        ds.attrs['experiment'] = ExperimentConfig.EXP_NAME
        ds.attrs['version'] = ExperimentConfig.EXP_VERSION
        ds.attrs['created'] = datetime.now().isoformat()

        # 保存
        encoding = {}
        if ExperimentConfig.DATA_STORAGE['compression']:
            encoding = {var: {
                'zlib': True,
                'complevel': ExperimentConfig.DATA_STORAGE['compression_level']
            } for var in ds.data_vars}

        ds.to_netcdf(filename, encoding=encoding)

    elif ExperimentConfig.DATA_STORAGE['format'] == 'hdf5':
        # 保存为HDF5
        with pd.HDFStore(filename, 'w') as store:
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    store.put(key, pd.DataFrame(value))

    else:
        raise ValueError(f"不支持的数据格式: {ExperimentConfig.DATA_STORAGE['format']}")


def load_dataset(filename: Path) -> Dict[str, np.ndarray]:
    """加载数据集"""
    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    if ExperimentConfig.DATA_STORAGE['format'] == 'netcdf':
        ds = xr.open_dataset(filename)
        data = {key: ds[key].values for key in ds.data_vars}
        ds.close()
        return data

    elif ExperimentConfig.DATA_STORAGE['format'] == 'hdf5':
        data = {}
        with pd.HDFStore(filename, 'r') as store:
            for key in store.keys():
                df = store[key]
                data[key.lstrip('/')] = df.values
        return data

    else:
        raise ValueError(f"Unsupported data format: {ExperimentConfig.DATA_STORAGE['format']}")


# 几何计算工具
def calculate_airmass(zenith_angle: float) -> float:
    """计算大气质量（使用Kasten和Young公式）"""
    cos_z = np.cos(np.radians(zenith_angle))
    # 避免除零
    cos_z = np.clip(cos_z, 0.001, 1.0)
    airmass = 1.0 / (cos_z + 0.50572 * (96.07995 - zenith_angle) ** -1.6364)
    return airmass


def calculate_secz(zenith_angle: float) -> float:
    """计算sec(θ)"""
    cos_z = np.cos(np.radians(zenith_angle))
    cos_z = np.clip(cos_z, 0.001, 1.0)
    return 1.0 / cos_z


def spherical_to_cartesian(zenith: np.ndarray, azimuth: np.ndarray, radius: float = 1.0):
    """球坐标转笛卡尔坐标"""
    zenith_rad = np.radians(zenith)
    azimuth_rad = np.radians(azimuth)

    x = radius * np.sin(zenith_rad) * np.cos(azimuth_rad)
    y = radius * np.sin(zenith_rad) * np.sin(azimuth_rad)
    z = radius * np.cos(zenith_rad)

    return x, y, z


# 统计工具
def calculate_statistics(errors: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """计算误差统计量"""
    errors = errors[~np.isnan(errors)]
    if len(errors) == 0:
        return {}

    stats = {
        f"{prefix}mean": np.mean(errors),
        f"{prefix}std": np.std(errors),
        f"{prefix}rmse": np.sqrt(np.mean(errors ** 2)),
        f"{prefix}mae": np.mean(np.abs(errors)),
        f"{prefix}max": np.max(errors),
        f"{prefix}min": np.min(errors),
        f"{prefix}median": np.median(errors),
        f"{prefix}q25": np.percentile(errors, 25),
        f"{prefix}q75": np.percentile(errors, 75),
        f"{prefix}n": len(errors)
    }
    return stats


def fit_error_model(x: np.ndarray, y: np.ndarray, model_type: str = 'linear'):
    """拟合误差模型"""
    valid_idx = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid_idx]
    y_valid = y[valid_idx]

    if len(x_valid) < 2:
        return None, None

    if model_type == 'linear':
        slope, intercept, r_value, p_value, std_err = linregress(x_valid, y_valid)
        model = lambda x_pred: slope * x_pred + intercept
        params = {'slope': slope, 'intercept': intercept, 'r2': r_value ** 2}
        return model, params

    elif model_type == 'poly2':
        coeffs = np.polyfit(x_valid, y_valid, 2)
        model = np.poly1d(coeffs)
        params = {'coeffs': coeffs.tolist()}
        return model, params

    else:
        raise ValueError(f"Unsupported model type: {model_type}")