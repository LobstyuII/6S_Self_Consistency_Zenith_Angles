# ==================== utils.py (修改版) ====================
"""
工具函数模块 - 重构版
添加蒙特卡洛采样和物理特征计算函数
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
from tqdm import tqdm
from datetime import datetime

from config import ExperimentConfig


def setup_logger(name: str, log_file: Optional[Path] = None, level=logging.INFO):
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def cache_result(func):
    """缓存函数结果的装饰器"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = hashlib.md5(f"{func.__name__}{args}{kwargs}".encode()).hexdigest()
        cache_dir = ExperimentConfig.PARALLEL_CONFIG['cache_dir']
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"{key}.pkl"

        if ExperimentConfig.PARALLEL_CONFIG['use_cache'] and cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        result = func(*args, **kwargs)

        if ExperimentConfig.PARALLEL_CONFIG['use_cache']:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)

        return result

    return wrapper


def save_dataset(data: Dict[str, np.ndarray], filename: Path):
    """保存数据集"""
    if ExperimentConfig.DATA_STORAGE['format'] == 'netcdf':
        ds = xr.Dataset()
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                ds[key] = xr.DataArray(value)

        ds.attrs['experiment'] = ExperimentConfig.EXP_NAME
        ds.attrs['version'] = ExperimentConfig.EXP_VERSION
        ds.attrs['created'] = datetime.now().isoformat()

        encoding = {}
        if ExperimentConfig.DATA_STORAGE['compression']:
            encoding = {var: {
                'zlib': True,
                'complevel': ExperimentConfig.DATA_STORAGE['compression_level']
            } for var in ds.data_vars}

        ds.to_netcdf(filename, encoding=encoding)

    elif ExperimentConfig.DATA_STORAGE['format'] == 'hdf5':
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


# ========== 重构：添加物理特征计算函数 ==========

def calculate_airmass(zenith_angle: float) -> float:
    """计算大气质量因子"""
    cos_z = np.cos(np.radians(zenith_angle))
    cos_z = np.clip(cos_z, 0.001, 1.0)
    airmass = 1.0 / cos_z
    return airmass


def calculate_scattering_angle(sza: float, vza: float, raa: float) -> float:
    """
    计算散射角 (degrees)
    公式: cos(θ_scat) = -cos(θ_s) * cos(θ_v) + sin(θ_s) * sin(θ_v) * cos(φ)
    """
    sza_rad = np.radians(sza)
    vza_rad = np.radians(vza)
    raa_rad = np.radians(raa)

    cos_scat = -np.cos(sza_rad) * np.cos(vza_rad) + \
               np.sin(sza_rad) * np.sin(vza_rad) * np.cos(raa_rad)

    cos_scat = np.clip(cos_scat, -1.0, 1.0)
    scattering_angle = np.degrees(np.arccos(cos_scat))

    return scattering_angle


def calculate_total_airmass(sza: float, vza: float) -> float:
    """计算总大气质量"""
    airmass_sza = calculate_airmass(sza)
    airmass_vza = calculate_airmass(vza)
    return airmass_sza + airmass_vza


def is_extreme_geometry(sza: float, vza: float, threshold: float = 60.0) -> bool:
    """判断是否为极端几何条件"""
    return (sza > threshold) or (vza > threshold)


def calculate_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算物理特征

    Args:
        df: 包含sza, vza, raa列的DataFrame

    Returns:
        包含物理特征的DataFrame
    """
    features = pd.DataFrame(index=df.index)

    # 三角函数
    sza_rad = np.radians(df['sza'])
    vza_rad = np.radians(df['vza'])
    raa_rad = np.radians(df['raa'])

    features['cos_sza'] = np.cos(sza_rad)
    features['sin_sza'] = np.sin(sza_rad)
    features['cos_vza'] = np.cos(vza_rad)
    features['sin_vza'] = np.sin(vza_rad)
    features['cos_raa'] = np.cos(raa_rad)
    features['sin_raa'] = np.sin(raa_rad)

    # 大气质量
    features['airmass_sza'] = 1.0 / np.clip(features['cos_sza'], 0.001, 1.0)
    features['airmass_vza'] = 1.0 / np.clip(features['cos_vza'], 0.001, 1.0)
    features['total_airmass'] = features['airmass_sza'] + features['airmass_vza']

    # 散射角
    cos_scat = -features['cos_sza'] * features['cos_vza'] + \
               features['sin_sza'] * features['sin_vza'] * features['cos_raa']
    cos_scat = np.clip(cos_scat, -1.0, 1.0)
    features['scattering_angle'] = np.degrees(np.arccos(cos_scat))

    # 归一化相对方位角
    features['raa_norm'] = df['raa'] / 180.0

    # 极端角度标识
    threshold = ExperimentConfig.MONTE_CARLO_CONFIG.get('extreme_threshold', 60.0)
    features['is_extreme_sza'] = (df['sza'] > threshold).astype(float)
    features['is_extreme_vza'] = (df['vza'] > threshold).astype(float)
    features['is_extreme_geometry'] = ((df['sza'] > threshold) | (df['vza'] > threshold)).astype(float)

    return features


def calculate_interaction_features(df: pd.DataFrame, physical_features: pd.DataFrame) -> pd.DataFrame:
    """
    计算交互特征

    Args:
        df: 原始DataFrame
        physical_features: 物理特征DataFrame

    Returns:
        包含交互特征的DataFrame
    """
    features = pd.DataFrame(index=df.index)

    # 大气-几何交互
    features['aod_airmass'] = df['aod550'] * physical_features['total_airmass']
    features['h2o_airmass'] = df['h2o'] * physical_features['total_airmass']
    features['o3_airmass'] = df['o3'] * physical_features['total_airmass']

    # 波长相关交互
    features['aod_wavelength'] = df['aod550'] * df['wavelength']
    features['wavelength_airmass'] = df['wavelength'] * physical_features['total_airmass']

    # 反射率相关交互
    if 'rho_toa' in df.columns and 'rho_retrieved' in df.columns:
        features['rho_ratio'] = df['rho_retrieved'] / (df['rho_toa'] + 1e-6)
        features['rho_diff'] = df['rho_toa'] - df['rho_retrieved']
        features['rho_product'] = df['rho_toa'] * df['rho_retrieved']

    # 角度-波长交互
    features['wavelength_cos_sza'] = df['wavelength'] * physical_features['cos_sza']
    features['wavelength_cos_vza'] = df['wavelength'] * physical_features['cos_vza']
    features['wavelength_scattering_angle'] = df['wavelength'] * physical_features['scattering_angle']

    # 气溶胶-散射角交互
    features['aod_scattering_angle'] = df['aod550'] * physical_features['scattering_angle']

    return features


# ========== 重构：添加蒙特卡洛采样函数 ==========

def monte_carlo_sample(param_ranges: dict, n_samples: int,
                       strategy: str = 'mixed', random_seed: int = 42) -> pd.DataFrame:
    """
    蒙特卡洛采样函数

    Args:
        param_ranges: 参数范围配置
        n_samples: 样本数
        strategy: 采样策略 ('mixed', 'uniform', 'extreme_only')
        random_seed: 随机种子

    Returns:
        采样参数的DataFrame
    """
    np.random.seed(random_seed)

    if strategy == 'mixed':
        # 混合采样：部分常规，部分极端
        regular_ratio = ExperimentConfig.MONTE_CARLO_CONFIG['regular_ratio']
        n_regular = int(n_samples * regular_ratio)
        n_extreme = n_samples - n_regular

        # 采样常规几何条件
        regular_params = _sample_uniform_geometry(param_ranges, n_regular,
                                                  include_extreme=False)

        # 采样极端几何条件（过采样）
        extreme_params = _sample_extreme_geometry(param_ranges, n_extreme)

        # 合并
        params = _merge_sampled_params(regular_params, extreme_params)

    elif strategy == 'uniform':
        # 均匀采样
        params = _sample_uniform_geometry(param_ranges, n_samples,
                                          include_extreme=True)

    elif strategy == 'extreme_only':
        # 仅极端角度
        params = _sample_extreme_geometry(param_ranges, n_samples)

    else:
        raise ValueError(f"不支持的采样策略: {strategy}")

    # 采样大气参数
    params.update(_sample_atmospheric_params(param_ranges, n_samples))

    return pd.DataFrame(params)


def _sample_uniform_geometry(param_ranges: dict, n_samples: int,
                             include_extreme: bool = True) -> dict:
    """均匀采样几何参数"""
    geo_ranges = param_ranges['geometry']

    params = {
        'sza': np.random.uniform(geo_ranges['sza']['min'],
                                 geo_ranges['sza']['max'],
                                 n_samples),
        'vza': np.random.uniform(geo_ranges['vza']['min'],
                                 geo_ranges['vza']['max'],
                                 n_samples),
        'raa': np.random.uniform(geo_ranges['raa']['min'],
                                 geo_ranges['raa']['max'],
                                 n_samples),
    }

    if not include_extreme:
        # 过滤掉极端角度
        threshold = ExperimentConfig.MONTE_CARLO_CONFIG['extreme_threshold']
        mask = ~((params['sza'] > threshold) | (params['vza'] > threshold))

        for key in params:
            params[key] = params[key][mask]

        # 重新采样以补足数量
        n_needed = n_samples - len(params['sza'])
        if n_needed > 0:
            additional = _sample_uniform_geometry(param_ranges, n_needed,
                                                  include_extreme=False)
            for key in params:
                params[key] = np.concatenate([params[key], additional[key]])

    return params


def _sample_extreme_geometry(param_ranges: dict, n_samples: int) -> dict:
    """采样极端几何参数"""
    geo_ranges = param_ranges['geometry']
    threshold = ExperimentConfig.MONTE_CARLO_CONFIG['extreme_threshold']

    params = {'sza': [], 'vza': [], 'raa': []}
    attempts = 0
    max_attempts = n_samples * 10

    while len(params['sza']) < n_samples and attempts < max_attempts:
        attempts += 1

        sza = np.random.uniform(geo_ranges['sza']['min'],
                                geo_ranges['sza']['max'])
        vza = np.random.uniform(geo_ranges['vza']['min'],
                                geo_ranges['vza']['max'])
        raa = np.random.uniform(geo_ranges['raa']['min'],
                                geo_ranges['raa']['max'])

        if is_extreme_geometry(sza, vza, threshold):
            params['sza'].append(sza)
            params['vza'].append(vza)
            params['raa'].append(raa)

    # 转换为数组
    for key in params:
        params[key] = np.array(params[key][:n_samples])

    return params


def _sample_atmospheric_params(param_ranges: dict, n_samples: int) -> dict:
    """采样大气参数"""
    params = {}

    # 地表反射率：均匀分布
    surf_ranges = param_ranges['surface']
    params['rho_true'] = np.random.uniform(
        surf_ranges['rho_true']['min'],
        surf_ranges['rho_true']['max'],
        n_samples
    )

    # 气溶胶：对数均匀分布
    atmos_ranges = param_ranges['atmosphere']
    if atmos_ranges['aod550']['distribution'] == 'log_uniform':
        aod_log_min = np.log10(atmos_ranges['aod550']['min'])
        aod_log_max = np.log10(atmos_ranges['aod550']['max'])
        params['aod550'] = 10 ** np.random.uniform(aod_log_min, aod_log_max, n_samples)
    else:
        params['aod550'] = np.random.uniform(
            atmos_ranges['aod550']['min'],
            atmos_ranges['aod550']['max'],
            n_samples
        )

    # 水汽和臭氧：均匀分布
    params['h2o'] = np.random.uniform(
        atmos_ranges['h2o']['min'],
        atmos_ranges['h2o']['max'],
        n_samples
    )
    params['o3'] = np.random.uniform(
        atmos_ranges['o3']['min'],
        atmos_ranges['o3']['max'],
        n_samples
    )

    return params


def _merge_sampled_params(regular_params: dict, extreme_params: dict) -> dict:
    """合并采样参数"""
    params = {}

    for key in regular_params:
        if key in extreme_params:
            params[key] = np.concatenate([regular_params[key], extreme_params[key]])
        else:
            params[key] = regular_params[key]

    return params


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
        raise ValueError(f"不支持的模型类型: {model_type}")