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
        'band1': {'wavelength': 0.46, 'name': 'Himawari-AHI Band 1 (0.46um)'},
        'band2': {'wavelength': 0.51, 'name': 'Himawari-AHI Band 2 (0.51um)'},
        'band3': {'wavelength': 0.64, 'name': 'Himawari-AHI Band 3 (0.64um)'},
        'band4': {'wavelength': 0.86, 'name': 'Himawari-AHI Band 4 (0.86um)'},
        'band5': {'wavelength': 1.6, 'name': 'Himawari-AHI Band 5 (1.6um)'},
        'band6': {'wavelength': 2.3, 'name': 'Himawari-AHI Band 6 (2.3um)'}
    }

    # 实验参数空间
    PARAM_SPACE = {
        'sza': np.arange(0, 86, 5),  # 太阳天顶角 (0-85°, 5°步长)
        'vza': np.arange(0, 76, 5),  # 观测天顶角 (0-75°, 5°步长)
        'raa': [0],  # 相对方位角
        'rho_true': [0.05, 0.1, 0.2, 0.4],  # 地表真实反射率
        'aod550': [0.1, 0.3, 0.5],  # 550nm气溶胶光学厚度
        # 'h2o': [1.0, 2.0, 3.0],  # 水汽含量 (g/cm²)
        'h2o': [2.0],  # 水汽含量 (g/cm²)
        # 'o3': [0.2, 0.3, 0.4],  # 臭氧含量 (cm-atm)
        'o3': [0.3],  # 臭氧含量 (cm-atm)
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

    EXPERIMENT_MODES = {
        'full': '全参数模拟',
        'single_factor': '单因素敏感性测试',
        'multi_factor': '多因素协同敏感性测试',
        'airmass_only': '仅大气质量测试'
    }

    @classmethod
    def get_param_combinations(cls, mode='full'):
        """获取参数组合"""
        from itertools import product

        if mode == 'full':
            # 全参数组合
            param_combinations = product(
                cls.PARAM_SPACE['sza'],
                cls.PARAM_SPACE['vza'],
                cls.PARAM_SPACE['raa'],
                cls.PARAM_SPACE['rho_true'],
                cls.PARAM_SPACE['aod550'],
                cls.PARAM_SPACE['h2o'],
                cls.PARAM_SPACE['o3']
            )

            param_list = []
            for sza, vza, raa, rho_true, aod550, h2o, o3 in param_combinations:
                param_dict = {
                    'sza': float(sza),
                    'vza': float(vza),
                    'raa': float(raa),
                    'rho_true': float(rho_true),
                    'aod550': float(aod550),
                    'h2o': float(h2o),
                    'o3': float(o3),
                    'atmos_profile': cls.PARAM_SPACE['atmos_profile'][0],
                    'aero_profile': cls.PARAM_SPACE['aero_profile'][0],
                    'target_altitude': cls.SIXS_CONFIG['target_altitude']
                }
                param_list.append(param_dict)

        elif mode == 'single_factor':
            # 单因素敏感性测试
            param_list = []

            # 基础参数
            base_params = {
                'sza': 30.0,
                'vza': 0.0,
                'raa': 0.0,
                'rho_true': 0.2,
                'aod550': 0.3,
                'h2o': 2.0,
                'o3': 0.3,
                'atmos_profile': cls.PARAM_SPACE['atmos_profile'][0],
                'aero_profile': cls.PARAM_SPACE['aero_profile'][0],
                'target_altitude': cls.SIXS_CONFIG['target_altitude']
            }

            # 1. SZA敏感性
            for sza in cls.PARAM_SPACE['sza']:
                params = base_params.copy()
                params['sza'] = float(sza)
                param_list.append(params)

            # 2. VZA敏感性
            for vza in cls.PARAM_SPACE['vza']:
                params = base_params.copy()
                params['vza'] = float(vza)
                param_list.append(params)

            # 3. AOD敏感性
            for aod550 in cls.PARAM_SPACE['aod550']:
                params = base_params.copy()
                params['aod550'] = float(aod550)
                param_list.append(params)

            # 4. 水汽敏感性
            for h2o in cls.PARAM_SPACE['h2o']:
                params = base_params.copy()
                params['h2o'] = float(h2o)
                param_list.append(params)

            # 5. 臭氧敏感性
            for o3 in cls.PARAM_SPACE['o3']:
                params = base_params.copy()
                params['o3'] = float(o3)
                param_list.append(params)

        elif mode == 'multi_factor':
            # 多因素协同敏感性测试（部分因子设计）
            param_list = []

            # 设计协同测试场景
            scenarios = [
                # 高AOD + 高水汽
                {'aod550': 0.5, 'h2o': 3.0, 'o3': 0.3, 'sza': 60.0, 'vza': 30.0},
                # 高AOD + 高臭氧
                {'aod550': 0.5, 'h2o': 2.0, 'o3': 0.4, 'sza': 60.0, 'vza': 30.0},
                # 高水汽 + 高臭氧
                {'aod550': 0.3, 'h2o': 3.0, 'o3': 0.4, 'sza': 60.0, 'vza': 30.0},
                # 所有因素都高 + 高角度
                {'aod550': 0.5, 'h2o': 3.0, 'o3': 0.4, 'sza': 75.0, 'vza': 60.0},
                # 所有因素都低
                {'aod550': 0.1, 'h2o': 1.0, 'o3': 0.2, 'sza': 15.0, 'vza': 0.0},
            ]

            base_params = {
                'rho_true': 0.2,
                'raa': 0.0,
                'atmos_profile': cls.PARAM_SPACE['atmos_profile'][0],
                'aero_profile': cls.PARAM_SPACE['aero_profile'][0],
                'target_altitude': cls.SIXS_CONFIG['target_altitude']
            }

            for scenario in scenarios:
                params = base_params.copy()
                params.update(scenario)
                param_list.append(params)

        elif mode == 'airmass_only':
            # 仅大气质量测试（固定其他大气参数）
            param_list = []

            base_params = {
                'rho_true': 0.2,
                'aod550': 0.3,
                'h2o': 2.0,
                'o3': 0.3,
                'raa': 0.0,
                'atmos_profile': cls.PARAM_SPACE['atmos_profile'][0],
                'aero_profile': cls.PARAM_SPACE['aero_profile'][0],
                'target_altitude': cls.SIXS_CONFIG['target_altitude']
            }

            # 不同SZA和VZA组合
            for sza in cls.PARAM_SPACE['sza']:
                for vza in cls.PARAM_SPACE['vza']:
                    params = base_params.copy()
                    params['sza'] = float(sza)
                    params['vza'] = float(vza)
                    param_list.append(params)

        else:
            raise ValueError(f"不支持的实验模式: {mode}")

        return param_list

    @classmethod
    def get_sensitivity_analysis_params(cls):
        """获取敏感性分析参数配置"""
        return {
            'sza': {'values': cls.PARAM_SPACE['sza'], 'label': '太阳天顶角 (°)'},
            'vza': {'values': cls.PARAM_SPACE['vza'], 'label': '观测天顶角 (°)'},
            'aod550': {'values': cls.PARAM_SPACE['aod550'], 'label': 'AOD550'},
            'h2o': {'values': cls.PARAM_SPACE['h2o'], 'label': '水汽含量 (g/cm²)'},
            'o3': {'values': cls.PARAM_SPACE['o3'], 'label': '臭氧含量 (cm-atm)'},
            'rho_true': {'values': cls.PARAM_SPACE['rho_true'], 'label': '地表反射率'}
        }

