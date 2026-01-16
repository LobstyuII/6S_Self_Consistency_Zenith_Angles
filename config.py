# ==================== config.py (修改版) ====================
"""
实验配置参数模块 - 重构版
支持蒙特卡洛采样和连续变量
"""
import numpy as np
from pathlib import Path
from datetime import datetime


class ExperimentConfig:
    """实验配置参数 - 重构版"""

    EXP_NAME = "6S_Geometry_Correction_Refactored"
    EXP_VERSION = "v2.0"
    EXP_DATE = datetime.now().strftime("%Y%m%d")

    BASE_DIR = Path("D:/6S_Self_Consistency_Zenith_Angles_v0.4")
    DATA_DIR = BASE_DIR / "data" / "refactored"  # 修改：新的数据目录
    RESULTS_DIR = BASE_DIR / "results" / "refactored"
    FIGURES_DIR = BASE_DIR / "figures" / "refactored"
    MODELS_DIR = BASE_DIR / "models" / "refactored"
    MANU_FIGURES_DIR = FIGURES_DIR / "Manu_figures"

    for dir_path in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR, MANU_FIGURES_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 波段配置保持不变
    BANDS = {
        'band1': {'wavelength': 0.46, 'name': 'Himawari-AHI Band 1 (0.46um)'},
        'band2': {'wavelength': 0.51, 'name': 'Himawari-AHI Band 2 (0.51um)'},
        'band3': {'wavelength': 0.64, 'name': 'Himawari-AHI Band 3 (0.64um)'},
        'band4': {'wavelength': 0.86, 'name': 'Himawari-AHI Band 4 (0.86um)'},
        'band5': {'wavelength': 1.6, 'name': 'Himawari-AHI Band 5 (1.6um)'},
        'band6': {'wavelength': 2.3, 'name': 'Himawari-AHI Band 6 (2.3um)'}
    }

    # ========== 重构：蒙特卡洛采样配置 ==========
    MONTE_CARLO_CONFIG = {
        'sampling_strategy': 'mixed',  # 'mixed' = 混合采样, 'uniform' = 均匀采样
        'regular_ratio': 0.7,  # 70% 常规采样
        'extreme_ratio': 0.3,  # 30% 极端角度过采样
        'extreme_threshold': 60.0,  # 极端角度阈值（度）
        'total_samples_per_band': 50000,  # 每波段总样本数
    }

    # ========== 重构：连续参数范围配置 ==========
    # 注意：这里定义的是采样范围，不是离散值
    PARAM_RANGES_CONTINUOUS = {
        'geometry': {
            'sza': {'min': 0.0, 'max': 85.0, 'distribution': 'uniform'},
            'vza': {'min': 0.0, 'max': 75.0, 'distribution': 'uniform'},
            'raa': {'min': 0.0, 'max': 180.0, 'distribution': 'uniform'},
        },
        'surface': {
            'rho_true': {'min': 0.01, 'max': 0.6, 'distribution': 'uniform'},  # 连续范围
        },
        'atmosphere': {
            'aod550': {'min': 0.05, 'max': 1.0, 'distribution': 'log_uniform'},  # 对数均匀
            'h2o': {'min': 0.5, 'max': 5.0, 'distribution': 'uniform'},  # 连续
            'o3': {'min': 0.2, 'max': 0.4, 'distribution': 'uniform'},  # 连续
        },
        'profiles': {
            'atmos_profile': ['MidlatitudeSummer', 'MidlatitudeWinter', 'Tropical'],
            'aero_profile': ['Continental', 'Maritime', 'Urban', 'Desert']
        }
    }

    # ========== 重构：LUT配置 ==========
    LUT_CONFIG = {
        # SZA轴：非均匀采样
        'sza': {
            'regular': {'range': (0, 70), 'step': 5},  # 0-70°, 步长5°
            'extreme': {'range': (70, 85), 'step': 1}  # 70-85°, 步长1° (加密)
        },
        # VZA轴：非均匀采样
        'vza': {
            'regular': {'range': (0, 70), 'step': 5},  # 0-70°, 步长5°
            'extreme': {'range': (70, 75), 'step': 1}  # 70-75°, 步长1° (加密)
        },
        # RAA轴
        'raa': {'range': (0, 180), 'step': 15},
        # 表观反射率轴（查找索引）
        'rho_apparent': {'range': (0.01, 0.6), 'step': 0.01},
        # AOD典型值
        'aod550': [0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
        # 波段波长
        'wavelength': [band['wavelength'] for band in BANDS.values()]
    }

    # 验证网格配置（稀疏规则网格，仅用于可视化）
    VALIDATION_GRID = {
        'sza': [0, 20, 40, 60, 70, 75, 80, 85],
        'vza': [0, 20, 40, 60, 70, 75],
        'raa': [0, 30, 60, 90, 120, 150, 180],
        'rho_true': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        'aod550': [0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
        'h2o': [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
        'o3': [0.2, 0.25, 0.3, 0.35, 0.4],
    }

    # ========== 重构：特征工程配置 ==========
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

    # 保留原有配置（向后兼容）
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

    # 新方法：生成蒙特卡洛采样参数
    @classmethod
    def get_monte_carlo_params(cls, n_samples: int, param_type: str = 'training') -> dict:
        """
        生成蒙特卡洛采样参数配置

        Args:
            n_samples: 样本数
            param_type: 'training' 或 'validation'

        Returns:
            参数配置字典
        """
        if param_type == 'training':
            # 训练数据使用连续采样
            return {
                'n_samples': n_samples,
                'strategy': cls.MONTE_CARLO_CONFIG['sampling_strategy'],
                'regular_ratio': cls.MONTE_CARLO_CONFIG['regular_ratio'],
                'extreme_ratio': cls.MONTE_CARLO_CONFIG['extreme_ratio'],
                'extreme_threshold': cls.MONTE_CARLO_CONFIG['extreme_threshold'],
                'param_ranges': cls.PARAM_RANGES_CONTINUOUS
            }
        elif param_type == 'validation':
            # 验证数据使用稀疏网格
            return {
                'grid_config': cls.VALIDATION_GRID,
                'description': '稀疏验证网格'
            }
        else:
            raise ValueError(f"不支持的参数类型: {param_type}")

    # 新方法：获取LUT轴
    @classmethod
    def get_lut_axes(cls) -> dict:
        """获取LUT坐标轴配置"""
        axes = {}

        # SZA轴（非均匀）
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

        # VZA轴（非均匀）
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

        # 其他轴
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