# ==================== config.py ====================
"""
实验配置参数模块
"""
import numpy as np
from pathlib import Path
from datetime import datetime


class ExperimentConfig:
    """实验配置参数"""

    # 实验基本信息
    EXP_NAME = "6S_Geometry_Correction"
    EXP_VERSION = "v1.0"
    EXP_DATE = datetime.now().strftime("%Y%m%d")

    # 路径配置
    BASE_DIR = Path("D:/PythonProject2_data_band1")  # 修改为实际路径
    DATA_DIR = BASE_DIR / "data"
    RESULTS_DIR = BASE_DIR / "results"
    FIGURES_DIR = BASE_DIR / "figures"
    MODELS_DIR = BASE_DIR / "models"

    # 创建目录
    for dir_path in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Himawari波段配置
    BANDS = {
        'band1': {'wavelength': 0.46, 'name': 'Himawari Band 1 (0.46um)'},
        'band2': {'wavelength': 0.51, 'name': 'Himawari Band 2 (0.51um)'},
        'band3': {'wavelength': 0.64, 'name': 'Himawari Band 3 (0.64um)'},
        'band4': {'wavelength': 0.86, 'name': 'Band 4 (0.86um)'},
        'band5': {'wavelength': 1.6, 'name': 'Band 5 (1.6um)'},
        'band6': {'wavelength': 2.3, 'name': 'Band 6 (2.3um)'}
    }

    # 实验参数空间
    PARAM_SPACE = {
        'sza': np.arange(0, 86, 5),  # 太阳天顶角 (0-85°, 5°步长)
        'vza': np.arange(0, 76, 5),  # 观测天顶角 (0-75°, 5°步长)
        # 'raa': [0, 90, 180],  # 相对方位角
        'raa': [0],  # 相对方位角
        'rho_true': [0.05, 0.1, 0.2, 0.4],  # 地表真实反射率
        'aod550': [0.1, 0.3, 0.5],  # 550nm气溶胶光学厚度
        'atmos_profile': ['MidlatitudeSummer'],  # 大气廓线
        'aero_profile': ['Continental']  # 气溶胶类型
    }

    # 6S模型配置
    SIXS_CONFIG = {
        'altitudes': 'satellite_level',
        'target_altitude': 0.0,  # km, 海平面
        'ground_type': 'HomogeneousLambertian',
        'gases': ['H2O', 'O3', 'O2']  # 考虑的气体
    }

    # 并行计算配置
    PARALLEL_CONFIG = {
        'n_workers': 8,  # 并行进程数
        'chunk_size': 100,  # 每个进程处理的任务数
        'use_cache': True,  # 是否使用缓存
        'cache_dir': DATA_DIR / "cache"
    }

    # 数据存储配置
    DATA_STORAGE = {
        'format': 'netcdf',  # 或 'hdf5'
        'compression': True,
        'compression_level': 4
    }

    # 随机种子
    RANDOM_SEED = 42

    # 实验阶段控制
    PHASES = {
        'run_forward': True,  # 运行正向模拟
        'run_inversion': True,  # 运行反演
        'analyze_errors': True,  # 分析误差
        'build_model': True,  # 构建校正模型
        'validate': True  # 验证
    }

    @classmethod
    def get_param_combinations(cls):
        """获取所有参数组合"""
        from itertools import product

        # 基础参数组合
        base_params = product(
            cls.PARAM_SPACE['sza'],
            cls.PARAM_SPACE['vza'],
            cls.PARAM_SPACE['raa'],
            cls.PARAM_SPACE['rho_true'],
            cls.PARAM_SPACE['aod550']
        )

        # 转换为参数字典列表
        param_list = []
        for sza, vza, raa, rho_true, aod550 in base_params:
            param_dict = {
                'sza': float(sza),
                'vza': float(vza),
                'raa': float(raa),
                'rho_true': float(rho_true),
                'aod550': float(aod550),
                'atmos_profile': cls.PARAM_SPACE['atmos_profile'][0],
                'aero_profile': cls.PARAM_SPACE['aero_profile'][0],
                'target_altitude': cls.SIXS_CONFIG['target_altitude']
            }
            param_list.append(param_dict)

        return param_list

