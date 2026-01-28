# ==================== config.py (瘦身优化版) ====================
"""
实验配置参数模块 - 重构版
支持蒙特卡洛采样和连续变量
已优化验证网格规模
"""
import numpy as np
from pathlib import Path
from datetime import datetime


class ExperimentConfig:
    """实验配置参数 - 重构版"""

    EXP_NAME = "6S_Geometry_Correction_Refactored"
    EXP_VERSION = "v2.0"
    EXP_DATE = datetime.now().strftime("%Y%m%d")

    # 根据您的环境调整路径
    BASE_DIR = Path("D:/6S_Self_Consistency_Zenith_Angles_v0.4")
    DATA_DIR = BASE_DIR / "data" / "refactored"
    RESULTS_DIR = BASE_DIR / "results" / "refactored"
    FIGURES_DIR = BASE_DIR / "figures" / "refactored"
    MODELS_DIR = BASE_DIR / "models" / "refactored"
    MANU_FIGURES_DIR = FIGURES_DIR / "Manu_figures"

    for dir_path in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR, MANU_FIGURES_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 波段配置
    BANDS = {
        'band1': {'wavelength': 0.46, 'name': 'Himawari-AHI Band 1 (0.46um)'},
        'band2': {'wavelength': 0.51, 'name': 'Himawari-AHI Band 2 (0.51um)'},
        'band3': {'wavelength': 0.64, 'name': 'Himawari-AHI Band 3 (0.64um)'},
        'band4': {'wavelength': 0.86, 'name': 'Himawari-AHI Band 4 (0.86um)'},
        'band5': {'wavelength': 1.6, 'name': 'Himawari-AHI Band 5 (1.6um)'},
        'band6': {'wavelength': 2.3, 'name': 'Himawari-AHI Band 6 (2.3um)'}
    }

    # ========== 蒙特卡洛采样配置 (训练用) ==========
    MONTE_CARLO_CONFIG = {
        'sampling_strategy': 'mixed',
        'regular_ratio': 0.7,
        'extreme_ratio': 0.3,
        'extreme_threshold': 60.0,
        'total_samples_per_band': 50000,
    }

    # ========== 连续参数范围配置 (训练用) ==========
    PARAM_RANGES_CONTINUOUS = {
        'geometry': {
            'sza': {'min': 0.0, 'max': 85.0, 'distribution': 'uniform'},
            'vza': {'min': 0.0, 'max': 75.0, 'distribution': 'uniform'},
            'raa': {'min': 0.0, 'max': 180.0, 'distribution': 'uniform'},
        },
        'surface': {
            'rho_true': {'min': 0.01, 'max': 0.6, 'distribution': 'uniform'},
        },
        'atmosphere': {
            'aod550': {'min': 0.05, 'max': 1.0, 'distribution': 'log_uniform'},
            'h2o': {'min': 0.5, 'max': 5.0, 'distribution': 'uniform'},
            'o3': {'min': 0.2, 'max': 0.4, 'distribution': 'uniform'},
        },
        'profiles': {
            'atmos_profile': ['MidlatitudeSummer'],  # 简化为单一轮廓
            'aero_profile': ['Continental']  # 简化为单一轮廓
        }
    }

    # ========== LUT配置 (生成业务化LUT用) ==========
    LUT_CONFIG = {
        'sza': {
            'regular': {'range': (0, 70), 'step': 5},
            'extreme': {'range': (70, 85), 'step': 1}
        },
        'vza': {
            'regular': {'range': (0, 70), 'step': 5},
            'extreme': {'range': (70, 75), 'step': 1}
        },
        'raa': {'range': (0, 180), 'step': 15},
        'rho_apparent': {'range': (0.01, 0.6), 'step': 0.01},
        'aod550': [0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
        'wavelength': [band['wavelength'] for band in BANDS.values()]
    }

    # ========== 验证网格配置 (瘦身版) ==========
    # 之前是全排列组合导致数据爆炸 (200万+)，现在优化核心变量
    VALIDATION_GRID = {
        # 几何角度保持一定的覆盖度，特别是大角度
        'sza': [0, 20, 40, 60, 70, 75, 80, 85],  # 8个点
        'vza': [0, 20, 40, 60, 70, 75],  # 6个点
        'raa': [0, 90, 180],  # 3个点 (减少中间角度)

        # 物理参数保留关键变化
        'rho_true': [0.1, 0.3, 0.5],  # 3个点 (低中高反射率)
        'aod550': [0.05, 0.2, 0.5, 1.0],  # 4个点 (清洁到浑浊)

        # 次要大气参数固定为典型值，减少维度爆炸
        'h2o': [2.0],  # 1个点 (固定典型值)
        'o3': [0.3],  # 1个点 (固定典型值)
    }
    # 预计每波段总数: 8*6*3*3*4*1*1 = 1728 个样本
    # 6个波段合计约 10,368 个样本 (极快)

    # ========== 特征工程配置 ==========
    FEATURE_ENGINEERING = {
        'use_physical_features': True,
        'physical_features': [
            'cos_sza', 'cos_vza',
            'airmass_sza', 'airmass_vza', 'total_airmass',
            'scattering_angle',
            'raa_norm'
        ],
        'use_interaction_features': True,
        'interaction_features': [
            'aod_airmass',
            'wavelength_scattering_angle',
            'aod_scattering_angle'
        ],
        'use_extreme_flags': True,
        'extreme_flags': [
            'is_extreme_sza',
            'is_extreme_vza',
            'is_extreme_geometry'
        ]
    }

    # 6S 基础配置
    SIXS_CONFIG = {
        'altitudes': 'satellite_level',
        'target_altitude': 0.0,
        'ground_type': 'HomogeneousLambertian',
        'gases': ['H2O', 'O3', 'O2']
    }

    PARALLEL_CONFIG = {
        'n_workers': 8,
        'max_concurrent_6s': 4,
        'use_shared_memory': True,
        'shared_memory_size': 1024 * 1024 * 100,
        'chunk_size': 1000,
        'use_cache': True,
        'cache_dir': DATA_DIR / "cache"
    }

    DATA_STORAGE = {
        'format': 'netcdf',
        'compression': True,
        'compression_level': 4
    }

    RANDOM_SEED = 42# ==================== config.py (角度范围调整版) ====================
"""
实验配置参数模块 - 重构版
支持蒙特卡洛采样和连续变量
已优化验证网格规模
"""
import numpy as np
from pathlib import Path
from datetime import datetime


class ExperimentConfig:
    """实验配置参数 - 重构版"""

    EXP_NAME = "6S_Geometry_Correction_Refactored"
    EXP_VERSION = "v2.0"
    EXP_DATE = datetime.now().strftime("%Y%m%d")

    # 根据您的环境调整路径
    BASE_DIR = Path("D:/6S_Self_Consistency_Zenith_Angles_v0.4")
    DATA_DIR = BASE_DIR / "data" / "refactored"
    RESULTS_DIR = BASE_DIR / "results" / "refactored"
    FIGURES_DIR = BASE_DIR / "figures" / "refactored"
    MODELS_DIR = BASE_DIR / "models" / "refactored"
    MANU_FIGURES_DIR = FIGURES_DIR / "Manu_figures"

    for dir_path in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR, MANU_FIGURES_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 波段配置
    BANDS = {
        'band1': {'wavelength': 0.46, 'name': 'Himawari-AHI Band 1 (0.46um)'},
        'band2': {'wavelength': 0.51, 'name': 'Himawari-AHI Band 2 (0.51um)'},
        'band3': {'wavelength': 0.64, 'name': 'Himawari-AHI Band 3 (0.64um)'},
        'band4': {'wavelength': 0.86, 'name': 'Himawari-AHI Band 4 (0.86um)'},
        'band5': {'wavelength': 1.6, 'name': 'Himawari-AHI Band 5 (1.6um)'},
        'band6': {'wavelength': 2.3, 'name': 'Himawari-AHI Band 6 (2.3um)'}
    }

    # ========== 蒙特卡洛采样配置 (训练用) ==========
    MONTE_CARLO_CONFIG = {
        'sampling_strategy': 'mixed',
        'regular_ratio': 0.6,  # 调整为60%常规，40%极端
        'extreme_ratio': 0.4,  # 增加极端采样比例
        'extreme_threshold': 60.0,  # 极端角度阈值
        'total_samples_per_band': 100000,  # 增加到10万/波段
    }

    # ========== 连续参数范围配置 (训练用) ==========
    PARAM_RANGES_CONTINUOUS = {
        'geometry': {
            'sza': {'min': 0.0, 'max': 85.0, 'distribution': 'uniform'},
            'vza': {'min': 0.0, 'max': 85.0, 'distribution': 'uniform'},  # 改为85°
            'raa': {'min': 0.0, 'max': 180.0, 'distribution': 'uniform'},
        },
        'surface': {
            'rho_true': {'min': 0.01, 'max': 0.6, 'distribution': 'uniform'},
        },
        'atmosphere': {
            'aod550': {'min': 0.05, 'max': 1.0, 'distribution': 'log_uniform'},
            'h2o': {'min': 0.5, 'max': 5.0, 'distribution': 'uniform'},
            'o3': {'min': 0.2, 'max': 0.4, 'distribution': 'uniform'},
        },
        'profiles': {
            'atmos_profile': ['MidlatitudeSummer'],  # 简化为单一轮廓
            'aero_profile': ['Continental']  # 简化为单一轮廓
        }
    }

    # ========== LUT配置 (生成业务化LUT用) ==========
    LUT_CONFIG = {
        'sza': {
            'regular': {'range': (0, 60), 'step': 5},
            'extreme': {'range': (60, 85), 'step': 2}  # 极端部分更细致
        },
        'vza': {
            'regular': {'range': (0, 60), 'step': 5},
            'extreme': {'range': (60, 85), 'step': 2}  # 极端部分更细致
        },
        'raa': {'range': (0, 180), 'step': 15},
        'rho_apparent': {'range': (0.01, 0.6), 'step': 0.01},
        'aod550': [0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
        'wavelength': [band['wavelength'] for band in BANDS.values()]
    }

    # ========== 验证网格配置 (调整版) ==========
    VALIDATION_GRID = {
        # 几何角度 - 覆盖全范围，极端部分更细致
        'sza': [0, 15, 30, 45, 60, 65, 70, 75, 80, 85],  # 10个点，极端部分更密
        'vza': [0, 15, 30, 45, 60, 65, 70, 75, 80, 85],  # 10个点，与SZA对称
        'raa': [0, 45, 90, 135, 180],  # 5个点 (更细致的方位角覆盖)

        # 物理参数保留关键变化
        'rho_true': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],  # 6个点
        'aod550': [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],  # 7个点

        # 次要大气参数固定为典型值
        'h2o': [2.0],
        'o3': [0.3],
    }
    # 预计每波段总数: 10*10*5*6*7*1*1 = 21,000 个样本
    # 6个波段合计约 126,000 个样本 (可接受)

    # ========== 特征工程配置 ==========
    FEATURE_ENGINEERING = {
        'use_physical_features': True,
        'physical_features': [
            'cos_sza', 'cos_vza',
            'airmass_sza', 'airmass_vza', 'total_airmass',
            'scattering_angle',
            'raa_norm'
        ],
        'use_interaction_features': True,
        'interaction_features': [
            'aod_airmass',
            'wavelength_scattering_angle',
            'aod_scattering_angle'
        ],
        'use_extreme_flags': True,
        'extreme_flags': [
            'is_extreme_sza',
            'is_extreme_vza',
            'is_extreme_geometry'
        ]
    }

    # 6S 基础配置
    SIXS_CONFIG = {
        'altitudes': 'satellite_level',
        'target_altitude': 0.0,
        'ground_type': 'HomogeneousLambertian',
        'gases': ['H2O', 'O3', 'O2']
    }

    PARALLEL_CONFIG = {
        'n_workers': 8,
        'max_concurrent_6s': 4,
        'use_shared_memory': True,
        'shared_memory_size': 1024 * 1024 * 100,
        'chunk_size': 1000,
        'use_cache': True,
        'cache_dir': DATA_DIR / "cache"
    }

    DATA_STORAGE = {
        'format': 'netcdf',
        'compression': True,
        'compression_level': 4
    }

    RANDOM_SEED = 42

    @classmethod
    def get_monte_carlo_params(cls, n_samples: int, param_type: str = 'training') -> dict:
        if param_type == 'training':
            return {
                'n_samples': n_samples,
                'strategy': cls.MONTE_CARLO_CONFIG['sampling_strategy'],
                'regular_ratio': cls.MONTE_CARLO_CONFIG['regular_ratio'],
                'extreme_ratio': cls.MONTE_CARLO_CONFIG['extreme_ratio'],
                'extreme_threshold': cls.MONTE_CARLO_CONFIG['extreme_threshold'],
                'param_ranges': cls.PARAM_RANGES_CONTINUOUS
            }
        elif param_type == 'validation':
            return {
                'grid_config': cls.VALIDATION_GRID,
                'description': '增强验证网格'
            }
        else:
            raise ValueError(f"不支持的参数类型: {param_type}")

    @classmethod
    def get_lut_axes(cls) -> dict:
        axes = {}
        sza_regular = np.arange(
            cls.LUT_CONFIG['sza']['regular']['range'][0],
            cls.LUT_CONFIG['sza']['regular']['range'][1] + 0.1,
            cls.LUT_CONFIG['sza']['regular']['step']
        )
        sza_extreme = np.arange(
            cls.LUT_CONFIG['sza']['extreme']['range'][0],
            cls.LUT_CONFIG['sza']['extreme']['range'][1] + 0.1,
            cls.LUT_CONFIG['sza']['extreme']['step']
        )
        axes['sza'] = np.unique(np.concatenate([sza_regular, sza_extreme]))

        vza_regular = np.arange(
            cls.LUT_CONFIG['vza']['regular']['range'][0],
            cls.LUT_CONFIG['vza']['regular']['range'][1] + 0.1,
            cls.LUT_CONFIG['vza']['regular']['step']
        )
        vza_extreme = np.arange(
            cls.LUT_CONFIG['vza']['extreme']['range'][0],
            cls.LUT_CONFIG['vza']['extreme']['range'][1] + 0.1,
            cls.LUT_CONFIG['vza']['extreme']['step']
        )
        axes['vza'] = np.unique(np.concatenate([vza_regular, vza_extreme]))

        axes['raa'] = np.arange(
            cls.LUT_CONFIG['raa']['range'][0],
            cls.LUT_CONFIG['raa']['range'][1] + 0.1,
            cls.LUT_CONFIG['raa']['step']
        )
        axes['rho_apparent'] = np.arange(
            cls.LUT_CONFIG['rho_apparent']['range'][0],
            cls.LUT_CONFIG['rho_apparent']['range'][1] + 0.0001,
            cls.LUT_CONFIG['rho_apparent']['step']
        )
        axes['aod550'] = np.array(cls.LUT_CONFIG['aod550'])
        axes['wavelength'] = np.array(cls.LUT_CONFIG['wavelength'])

        return axes