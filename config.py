# ==================== config.py ====================
"""
实验配置参数模块
"""
import numpy as np
from pathlib import Path
from datetime import datetime


class LUTTaskConfig:
    """LUT任务配置"""

    # 参数分块配置
    PARAM_BLOCKS = {
        'sza': {'size': 5, 'overlap': 1},  # 每5度一个块，重叠1度
        'vza': {'size': 5, 'overlap': 1},
        'aod550': {'size': 0.1, 'overlap': 0.02},
        'rho_true': {'size': 0.05, 'overlap': 0.01},
        'h2o': {'size': 0.5, 'overlap': 0.1},
        'o3': {'size': 0.01, 'overlap': 0.002}
    }

    # 任务管理
    TASK_MANAGEMENT = {
        'task_db_file': 'tasks.db',  # 任务状态数据库
        'max_retries': 3,  # 最大重试次数
        'chunk_size': 100,  # 每个任务块的大小
        'checkpoint_interval': 100,  # 检查点间隔
        'backup_interval': 1000,  # 备份间隔
    }

    # 数据存储
    DATA_STORAGE = {
        'block_format': 'netcdf',  # 块数据格式
        'merge_format': 'netcdf',  # 合并数据格式
        'compress_blocks': True,  # 压缩块数据
        'keep_blocks': True,  # 保留块数据
    }


class ExperimentConfig:
    """实验配置参数"""

    EXP_NAME = "6S_Geometry_Correction"
    EXP_VERSION = "v1.0"
    EXP_DATE = datetime.now().strftime("%Y%m%d")

    BASE_DIR = Path("D:/6S_Self_Consistency_Zenith_Angles")
    DATA_DIR = BASE_DIR / "data"
    RESULTS_DIR = BASE_DIR / "results"
    FIGURES_DIR = BASE_DIR / "figures"
    MODELS_DIR = BASE_DIR / "models"
    MANU_FIGURES_DIR = FIGURES_DIR / "Manu_figures"  # 新增：论文图表目录

    for dir_path in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR, MANU_FIGURES_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    BANDS = {
        'band1': {'wavelength': 0.46, 'name': 'Himawari-AHI Band 1 (0.46um)'},
        'band2': {'wavelength': 0.51, 'name': 'Himawari-AHI Band 2 (0.51um)'},
        'band3': {'wavelength': 0.64, 'name': 'Himawari-AHI Band 3 (0.64um)'},
        'band4': {'wavelength': 0.86, 'name': 'Himawari-AHI Band 4 (0.86um)'},
        'band5': {'wavelength': 1.6, 'name': 'Himawari-AHI Band 5 (1.6um)'},
        'band6': {'wavelength': 2.3, 'name': 'Himawari-AHI Band 6 (2.3um)'}
    }

    # 参数范围配置 - 支持灵活定义
    PARAM_RANGES = {
        'sza': {'min': 0.0, 'max': 85.0, 'step': 5.0},  # 动态生成
        'vza': {'min': 0.0, 'max': 75.0, 'step': 5.0},
        'aod550': [0.05, 0.1, 0.2, 0.3, 0.5, 1.0],  # 预设值
        'rho_true': {'min': 0.05, 'max': 0.5, 'step': 0.05},
        'h2o': [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
        'o3': [0.2, 0.25, 0.3, 0.35, 0.4],
    }

    # 新增：模拟模式配置
    SIMULATION_MODES = {
        'full': {'use_all_params': True},
        'paper_figures': {
            'sza': [0, 30, 60],
            'vza': [0, 30, 60],
            'aod550': [0.05, 0.2, 0.5],
            'rho_true': [0.05, 0.2, 0.4],
            'h2o': [1.0, 2.0],
            'o3': [0.25, 0.35]
        },
        'sensitivity': {
            'sza': [0, 30, 60],
            'vza': [0, 30, 60],
            'aod550': [0.1, 0.3],
            'rho_true': [0.1, 0.3],
            'h2o': [1.0, 3.0],
            'o3': [0.25, 0.35]
        }
    }

    SIXS_CONFIG = {
        'altitudes': 'satellite_level',
        'target_altitude': 0.0,
        'ground_type': 'HomogeneousLambertian',
        'gases': ['H2O', 'O3', 'O2']
    }

    PARALLEL_CONFIG = {
        'n_workers': 8,
        'chunk_size': 100,
        'use_cache': True,
        'cache_dir': DATA_DIR / "cache"
    }

    DATA_STORAGE = {
        'format': 'netcdf',
        'compression': True,
        'compression_level': 4
    }

    RANDOM_SEED = 42

    PHASES = {
        'run_forward': True,
        'run_inversion': True,
        'analyze_errors': True,
        'build_model': True,
        'validate': True
    }

    # 新增：论文图表配置
    PAPER_FIGURES = {
        # 基础固定参数配置（所有横截面共用的固定值）
        'fixed_params': {
            'aod550': 0.3,
            'rho_true': 0.2,
            'h2o': 2.0,  # 固定h2o
            'o3': 0.3,  # 固定o3
            'atmos_profile': 'MidlatitudeSummer',
            'aero_profile': 'Continental'
        },
        'contour_fixed_all': {
            'layout': (2, 3),  # 2行3列，6个波段
            'figsize': (18, 12)
        },
        'contour_varying_aod': {
            'rho_true': 0.2,
            'h2o': 2.0,  # 固定h2o
            'o3': 0.3,  # 固定o3
            'aod550_values': [0.1, 0.3, 0.5],
            'band': 'band3',
            'layout': (1, 3),  # 1行3列
            'figsize': (18, 6)
        },
        'contour_varying_rho': {
            'aod550': 0.3,
            'h2o': 2.0,  # 固定h2o
            'o3': 0.3,  # 固定o3
            'rho_true_values': [0.1, 0.2, 0.4],
            'band': 'band3',
            'layout': (1, 3),  # 1行3列
            'figsize': (18, 6)
        },
        'contour_varying_h2o': {
            'aod550': 0.3,
            'rho_true': 0.2,
            'o3': 0.3,  # 固定o3
            'h2o_values': [1.0, 2.0],  # h2o只有2个值
            'band': 'band3',
            'layout': (1, 2),  # 1行2列
            'figsize': (12, 6)
        },
        'contour_varying_o3': {
            'aod550': 0.3,
            'rho_true': 0.2,
            'h2o': 2.0,  # 固定h2o
            'o3_values': [0.2, 0.3],  # o3只有2个值
            'band': 'band3',
            'layout': (1, 2),  # 1行2列
            'figsize': (12, 6)
        },
        'contour_varying_band': {
            'aod550': 0.3,
            'rho_true': 0.2,
            'h2o': 2.0,  # 固定h2o
            'o3': 0.3,  # 固定o3
            'bands': ['band1', 'band2', 'band3', 'band4', 'band5', 'band6'],
            'layout': (2, 3),  # 2行3列
            'figsize': (18, 12)
        },
        'single_factor_sensitivity': {
            'layout': (3, 4),  # 3行4列（共12个参数）
            'figsize': (20, 15)
        },
        'error_distribution': {
            'layout': (2, 3),  # 2行3列（共6个波段）
            'figsize': (18, 12)
        }
    }

    @classmethod
    def get_param_combinations(cls, mode='full'):
        """获取参数组合"""
        from itertools import product

        if mode == 'full':
            param_combinations = product(
                cls.PARAM_SPACE['sza'],
                cls.PARAM_SPACE['vza'],
                cls.PARAM_SPACE['rho_true'],
                cls.PARAM_SPACE['aod550'],
                cls.PARAM_SPACE['h2o'],
                cls.PARAM_SPACE['o3']
            )

            param_list = []
            for sza, vza, rho_true, aod550, h2o, o3 in param_combinations:
                param_dict = {
                    'sza': float(sza),
                    'vza': float(vza),
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
            param_list = []
            base_params = {
                'sza': 30.0,
                'vza': 0.0,
                'rho_true': 0.2,
                'aod550': 0.2,
                'h2o': 2.0,
                'o3': 0.3,
                'atmos_profile': cls.PARAM_SPACE['atmos_profile'][0],
                'aero_profile': cls.PARAM_SPACE['aero_profile'][0],
                'target_altitude': cls.SIXS_CONFIG['target_altitude']
            }

            for sza in cls.PARAM_SPACE['sza']:
                params = base_params.copy()
                params['sza'] = float(sza)
                param_list.append(params)

            for vza in cls.PARAM_SPACE['vza']:
                params = base_params.copy()
                params['vza'] = float(vza)
                param_list.append(params)

            for aod550 in cls.PARAM_SPACE['aod550']:
                params = base_params.copy()
                params['aod550'] = float(aod550)
                param_list.append(params)

            for h2o in cls.PARAM_SPACE['h2o']:
                params = base_params.copy()
                params['h2o'] = float(h2o)
                param_list.append(params)

            for o3 in cls.PARAM_SPACE['o3']:
                params = base_params.copy()
                params['o3'] = float(o3)
                param_list.append(params)

        elif mode == 'airmass_only':
            param_list = []
            base_params = {
                'rho_true': 0.2,
                'aod550': 0.3,
                'h2o': 2.0,
                'o3': 0.3,
                'atmos_profile': cls.PARAM_SPACE['atmos_profile'][0],
                'aero_profile': cls.PARAM_SPACE['aero_profile'][0],
                'target_altitude': cls.SIXS_CONFIG['target_altitude']
            }

            for sza in cls.PARAM_SPACE['sza']:
                for vza in cls.PARAM_SPACE['vza']:
                    params = base_params.copy()
                    params['sza'] = float(sza)
                    params['vza'] = float(vza)
                    param_list.append(params)

        elif mode == 'paper_figures':  # 新增：论文图表模式
            # 为论文图表生成完整数据
            param_list = []
            param_combinations = product(
                cls.PARAM_SPACE['sza'],
                cls.PARAM_SPACE['vza'],
                cls.PARAM_SPACE['rho_true'],
                cls.PARAM_SPACE['aod550'],
                cls.PARAM_SPACE['h2o'],
                cls.PARAM_SPACE['o3']
            )

            for sza, vza, rho_true, aod550, h2o, o3 in param_combinations:
                param_dict = {
                    'sza': float(sza),
                    'vza': float(vza),
                    'rho_true': float(rho_true),
                    'aod550': float(aod550),
                    'h2o': float(h2o),
                    'o3': float(o3),
                    'atmos_profile': cls.PARAM_SPACE['atmos_profile'][0],
                    'aero_profile': cls.PARAM_SPACE['aero_profile'][0],
                    'target_altitude': cls.SIXS_CONFIG['target_altitude']
                }
                param_list.append(param_dict)

        else:
            raise ValueError(f"不支持的实验模式: {mode}")

        return param_list

    @classmethod
    def get_sensitivity_analysis_params(cls):
        """获取敏感性分析参数配置"""
        return {
            'sza': {'values': cls.PARAM_SPACE['sza'], 'label': 'Solar zenith angle (°)'},
            'vza': {'values': cls.PARAM_SPACE['vza'], 'label': 'View zenith angle (°)'},
            'aod550': {'values': cls.PARAM_SPACE['aod550'], 'label': 'AOD550'},
            'h2o': {'values': cls.PARAM_SPACE['h2o'], 'label': 'Water vapor (g/cm²)'},
            'o3': {'values': cls.PARAM_SPACE['o3'], 'label': 'Ozone (cm-atm)'},
            'rho_true': {'values': cls.PARAM_SPACE['rho_true'], 'label': 'Surface reflectance'}
        }