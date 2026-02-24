# ==================== data_generator.py (角度范围修复版) ====================
"""
重构的数据生成模块 - 采用蒙特卡洛采样和极端角度过采样
修复：验证网格现在正确读取配置文件中的设置
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
from itertools import product

from config import ExperimentConfig
from utils import setup_logger

warnings.filterwarnings('ignore')


class MonteCarloDataGenerator:
    """
    蒙特卡洛数据生成器
    采用混合采样策略：60%常规采样 + 40%极端角度过采样
    """

    def __init__(self, config: ExperimentConfig = None, logger=None):
        self.config = config or ExperimentConfig
        self.logger = logger or setup_logger('MonteCarloDataGenerator')

        # 采样参数从配置中读取
        mc_config = getattr(self.config, 'MONTE_CARLO_CONFIG', {})
        self.extreme_threshold = mc_config.get('extreme_threshold', 60.0)
        self.regular_ratio = mc_config.get('regular_ratio', 0.6)  # 调整为0.6
        self.extreme_ratio = mc_config.get('extreme_ratio', 0.4)  # 调整为0.4

        # 参数范围
        if hasattr(self.config, 'PARAM_RANGES_CONTINUOUS'):
            # 扁平化配置以便于内部使用
            self.param_ranges = {}
            ranges = self.config.PARAM_RANGES_CONTINUOUS
            if 'geometry' in ranges: self.param_ranges.update(ranges['geometry'])
            if 'surface' in ranges: self.param_ranges.update(ranges['surface'])
            if 'atmosphere' in ranges: self.param_ranges.update(ranges['atmosphere'])
        else:
            # 回退默认值
            self.param_ranges = {
                'sza': {'min': 0.0, 'max': 85.0},
                'vza': {'min': 0.0, 'max': 85.0},  # 改为85°
                'raa': {'min': 0.0, 'max': 180.0},
                'rho_true': {'min': 0.01, 'max': 0.6},
                'aod550': {'min': 0.05, 'max': 1.0},
                'h2o': {'min': 0.5, 'max': 5.0},
                'o3': {'min': 0.2, 'max': 0.4},
            }

        # 波段信息
        self.bands = self.config.BANDS

    def calculate_scattering_angle(self, sza: float, vza: float, raa: float) -> float:
        """
        计算散射角 (degrees)
        公式: cos(θ_scat) = -cos(θ_s) * cos(θ_v) + sin(θ_s) * sin(θ_v) * cos(φ)
        """
        sza_rad = np.radians(sza)
        vza_rad = np.radians(vza)
        raa_rad = np.radians(raa)

        cos_scat = -np.cos(sza_rad) * np.cos(vza_rad) + \
                   np.sin(sza_rad) * np.sin(vza_rad) * np.cos(raa_rad)

        # 防止数值误差
        cos_scat = np.clip(cos_scat, -1.0, 1.0)
        scattering_angle = np.degrees(np.arccos(cos_scat))

        return scattering_angle

    def is_extreme_geometry(self, sza: float, vza: float) -> bool:
        """判断是否为极端几何条件"""
        return (sza > self.extreme_threshold) or (vza > self.extreme_threshold)

    def sample_regular_geometry(self, n_samples: int) -> Dict[str, np.ndarray]:
        """
        常规几何条件采样 - 均匀分布
        """
        params = {}
        sza_range = self.param_ranges.get('sza', {'min': 0, 'max': 85})
        vza_range = self.param_ranges.get('vza', {'min': 0, 'max': 85})  # 改为85
        raa_range = self.param_ranges.get('raa', {'min': 0, 'max': 180})

        params['sza'] = np.random.uniform(sza_range['min'], sza_range['max'], n_samples)
        params['vza'] = np.random.uniform(vza_range['min'], vza_range['max'], n_samples)
        params['raa'] = np.random.uniform(raa_range['min'], raa_range['max'], n_samples)

        return params

    def sample_extreme_geometry(self, n_samples: int) -> Dict[str, np.ndarray]:
        """
        极端几何条件过采样
        专门采样 SZA > 60° 或 VZA > 60° 的情况
        增强采样：极端角度内部分布更偏向大角度
        """
        params = {'sza': [], 'vza': [], 'raa': []}

        sza_range = self.param_ranges.get('sza', {'min': 0, 'max': 85})
        vza_range = self.param_ranges.get('vza', {'min': 0, 'max': 85})
        raa_range = self.param_ranges.get('raa', {'min': 0, 'max': 180})

        # 定义极端角度子范围
        extreme_sza_min = 60
        extreme_vza_min = 60

        samples_generated = 0
        max_attempts = n_samples * 20  # 增加尝试次数

        # 使用更偏向大角度的分布
        for _ in range(max_attempts):
            if samples_generated >= n_samples:
                break

            # 在极端范围内使用偏向大角度的分布（平方分布）
            sza_random = np.random.uniform(0, 1)
            # 平方分布使得更多样本靠近85°
            sza = extreme_sza_min + (sza_range['max'] - extreme_sza_min) * (sza_random ** 0.5)

            vza_random = np.random.uniform(0, 1)
            vza = extreme_vza_min + (vza_range['max'] - extreme_vza_min) * (vza_random ** 0.5)

            raa = np.random.uniform(raa_range['min'], raa_range['max'])

            # 确保至少一个是极端角度
            if self.is_extreme_geometry(sza, vza):
                params['sza'].append(sza)
                params['vza'].append(vza)
                params['raa'].append(raa)
                samples_generated += 1

        # 如果没采够，用常规方式补充
        if samples_generated < n_samples:
            remaining = n_samples - samples_generated
            for _ in range(remaining * 5):
                if samples_generated >= n_samples:
                    break

                sza = np.random.uniform(extreme_sza_min, sza_range['max'])
                vza = np.random.uniform(extreme_vza_min, vza_range['max'])
                raa = np.random.uniform(raa_range['min'], raa_range['max'])

                if self.is_extreme_geometry(sza, vza):
                    params['sza'].append(sza)
                    params['vza'].append(vza)
                    params['raa'].append(raa)
                    samples_generated += 1

        for key in params:
            params[key] = np.array(params[key][:n_samples])

        return params

    def sample_atmospheric_params(self, n_samples: int) -> Dict[str, np.ndarray]:
        """
        采样大气参数 - 连续随机变量
        """
        params = {}

        # 获取范围
        rho_range = self.param_ranges.get('rho_true', {'min': 0.01, 'max': 0.6})
        aod_range = self.param_ranges.get('aod550', {'min': 0.05, 'max': 1.0})
        h2o_range = self.param_ranges.get('h2o', {'min': 0.5, 'max': 5.0})
        o3_range = self.param_ranges.get('o3', {'min': 0.2, 'max': 0.4})

        # 地表反射率: 连续均匀分布
        params['rho_true'] = np.random.uniform(rho_range['min'], rho_range['max'], n_samples)

        # 气溶胶光学厚度: 对数均匀分布
        aod_log_min = np.log10(aod_range['min'])
        aod_log_max = np.log10(aod_range['max'])
        params['aod550'] = 10 ** np.random.uniform(aod_log_min, aod_log_max, n_samples)

        # 水汽: 均匀分布
        params['h2o'] = np.random.uniform(h2o_range['min'], h2o_range['max'], n_samples)

        # 臭氧: 均匀分布
        params['o3'] = np.random.uniform(o3_range['min'], o3_range['max'], n_samples)

        return params

    def generate_mixed_sampling_dataset(self, total_samples: int,
                                        bands: List[str] = None) -> Dict[str, List[Dict]]:
        """
        生成混合采样数据集
        Args:
            total_samples: 总样本数
            bands: 波段列表，None表示所有波段
        Returns:
            按波段分组的参数组合字典
        """
        if bands is None:
            bands = list(self.bands.keys())

        # 计算各类样本数
        n_regular = int(total_samples * self.regular_ratio)
        n_extreme = total_samples - n_regular

        self.logger.info(f"混合采样配置:")
        self.logger.info(f"  总样本数: {total_samples}")
        self.logger.info(f"  常规采样: {n_regular} ({self.regular_ratio * 100:.0f}%)")
        self.logger.info(f"  极端过采样: {n_extreme} ({self.extreme_ratio * 100:.0f}%)")

        # 采样几何参数
        self.logger.info("采样几何参数...")
        regular_geo = self.sample_regular_geometry(n_regular)
        extreme_geo = self.sample_extreme_geometry(n_extreme)

        # 合并几何参数
        geo_params = {}
        for key in ['sza', 'vza', 'raa']:
            geo_params[key] = np.concatenate([regular_geo[key], extreme_geo[key]])

        # 采样大气参数
        self.logger.info("采样大气参数...")
        atmos_params = self.sample_atmospheric_params(total_samples)

        # 为每个波段生成参数组合
        all_combinations = {}

        for band_id in bands:
            self.logger.info(f"为波段 {band_id} 生成参数组合...")

            band_combinations = []
            wavelength = self.bands[band_id]['wavelength']

            for i in range(total_samples):
                params = {
                    'sza': float(geo_params['sza'][i]),
                    'vza': float(geo_params['vza'][i]),
                    'raa': float(geo_params['raa'][i]),
                    'rho_true': float(atmos_params['rho_true'][i]),
                    'aod550': float(atmos_params['aod550'][i]),
                    'h2o': float(atmos_params['h2o'][i]),
                    'o3': float(atmos_params['o3'][i]),
                    'wavelength': wavelength,
                    'band': band_id,
                    'atmos_profile': 'MidlatitudeSummer',
                    'aero_profile': 'Continental',
                    'target_altitude': self.config.SIXS_CONFIG['target_altitude'],
                    'is_extreme': self.is_extreme_geometry(
                        geo_params['sza'][i],
                        geo_params['vza'][i]
                    )
                }

                band_combinations.append(params)

            all_combinations[band_id] = band_combinations

        return all_combinations

    def generate_validation_grid(self, bands: List[str] = None) -> Dict[str, List[Dict]]:
        """
        生成稀疏规则网格用于验证（不参与训练）
        读取 config.py 中的 VALIDATION_GRID 配置
        """
        if bands is None:
            bands = list(self.bands.keys())

        # 直接读取 Config 中的配置
        if hasattr(self.config, 'VALIDATION_GRID'):
            grid_config = self.config.VALIDATION_GRID
            self.logger.info("成功读取 Config.VALIDATION_GRID 配置")
        else:
            self.logger.warning("未找到 VALIDATION_GRID 配置，使用内置默认值")
            grid_config = {
                'sza': [0, 20, 40, 60, 70, 75, 80],
                'vza': [0, 20, 40, 60, 70],
                'raa': [0, 90, 180],
                'rho_true': [0.1, 0.3, 0.5],
                'aod550': [0.1, 0.3, 0.5, 0.8],
                'h2o': [2.0],
                'o3': [0.3],
            }

        all_combinations = {}
        total_count_all_bands = 0

        for band_id in bands:
            wavelength = self.bands[band_id]['wavelength']
            combinations = []

            # 获取各个维度的列表
            sza_list = grid_config.get('sza', [0])
            vza_list = grid_config.get('vza', [0])
            raa_list = grid_config.get('raa', [0])
            rho_list = grid_config.get('rho_true', [0.2])
            aod_list = grid_config.get('aod550', [0.1])
            h2o_list = grid_config.get('h2o', [2.0])
            o3_list = grid_config.get('o3', [0.3])

            # 使用itertools生成所有组合
            for values in product(
                    sza_list, vza_list, raa_list,
                    rho_list, aod_list, h2o_list, o3_list
            ):
                sza, vza, raa, rho_true, aod550, h2o, o3 = values

                params = {
                    'sza': float(sza),
                    'vza': float(vza),
                    'raa': float(raa),
                    'rho_true': float(rho_true),
                    'aod550': float(aod550),
                    'h2o': float(h2o),
                    'o3': float(o3),
                    'wavelength': wavelength,
                    'band': band_id,
                    'atmos_profile': 'MidlatitudeSummer',
                    'aero_profile': 'Continental',
                    'target_altitude': self.config.SIXS_CONFIG['target_altitude'],
                    'is_extreme': self.is_extreme_geometry(sza, vza)
                }

                combinations.append(params)

            all_combinations[band_id] = combinations
            total_count_all_bands += len(combinations)
            self.logger.info(f"验证网格 - 波段 {band_id}: {len(combinations)} 个组合")

        self.logger.info(f"验证网格生成完毕，总计: {total_count_all_bands} 个样本")
        return all_combinations