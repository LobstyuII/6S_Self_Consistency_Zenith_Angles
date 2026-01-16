# ==================== data_generator.py ====================
"""
重构的数据生成模块 - 采用蒙特卡洛采样和极端角度过采样
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
    采用混合采样策略：70%常规采样 + 30%极端角度过采样
    """

    def __init__(self, config: ExperimentConfig = None, logger=None):
        self.config = config or ExperimentConfig
        self.logger = logger or setup_logger('MonteCarloDataGenerator')

        # 极端角度阈值
        self.extreme_threshold = 60.0  # 角度 > 60° 视为极端
        self.regular_ratio = 0.7  # 70% 常规采样
        self.extreme_ratio = 0.3  # 30% 极端角度过采样

        # 参数范围
        self.param_ranges = {
            'sza': {'min': 0.0, 'max': 85.0},
            'vza': {'min': 0.0, 'max': 75.0},
            'raa': {'min': 0.0, 'max': 180.0},
            'rho_true': {'min': 0.01, 'max': 0.6},  # 连续范围
            'aod550': {'min': 0.05, 'max': 1.0},  # 连续范围
            'h2o': {'min': 0.5, 'max': 5.0},  # 连续范围
            'o3': {'min': 0.2, 'max': 0.4},  # 连续范围
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

    def calculate_airmass(self, zenith_angle: float) -> float:
        """
        计算大气质量因子
        公式: airmass = 1/cos(θ) (简单近似)
        """
        cos_z = np.cos(np.radians(zenith_angle))
        # 防止除零
        cos_z = np.clip(cos_z, 0.001, 1.0)
        return 1.0 / cos_z

    def is_extreme_geometry(self, sza: float, vza: float) -> bool:
        """判断是否为极端几何条件"""
        return (sza > self.extreme_threshold) or (vza > self.extreme_threshold)

    def sample_regular_geometry(self, n_samples: int) -> Dict[str, np.ndarray]:
        """
        常规几何条件采样 - 均匀分布
        """
        params = {}

        # SZA: 整个范围内均匀采样
        params['sza'] = np.random.uniform(
            self.param_ranges['sza']['min'],
            self.param_ranges['sza']['max'],
            n_samples
        )

        # VZA: 整个范围内均匀采样
        params['vza'] = np.random.uniform(
            self.param_ranges['vza']['min'],
            self.param_ranges['vza']['max'],
            n_samples
        )

        # RAA: 0-180°均匀采样
        params['raa'] = np.random.uniform(
            self.param_ranges['raa']['min'],
            self.param_ranges['raa']['max'],
            n_samples
        )

        return params

    def sample_extreme_geometry(self, n_samples: int) -> Dict[str, np.ndarray]:
        """
        极端几何条件过采样
        专门采样 SZA > 60° 或 VZA > 60° 的情况
        """
        params = {'sza': [], 'vza': [], 'raa': []}

        samples_generated = 0
        max_attempts = n_samples * 10  # 防止无限循环

        for _ in range(max_attempts):
            if samples_generated >= n_samples:
                break

            # 生成随机角度
            sza = np.random.uniform(
                self.param_ranges['sza']['min'],
                self.param_ranges['sza']['max']
            )
            vza = np.random.uniform(
                self.param_ranges['vza']['min'],
                self.param_ranges['vza']['max']
            )
            raa = np.random.uniform(
                self.param_ranges['raa']['min'],
                self.param_ranges['raa']['max']
            )

            # 检查是否为极端角度
            if self.is_extreme_geometry(sza, vza):
                params['sza'].append(sza)
                params['vza'].append(vza)
                params['raa'].append(raa)
                samples_generated += 1

        # 转换为numpy数组
        for key in params:
            params[key] = np.array(params[key][:n_samples])

        return params

    def sample_atmospheric_params(self, n_samples: int) -> Dict[str, np.ndarray]:
        """
        采样大气参数 - 连续随机变量
        """
        params = {}

        # 地表反射率: 连续均匀分布
        params['rho_true'] = np.random.uniform(
            self.param_ranges['rho_true']['min'],
            self.param_ranges['rho_true']['max'],
            n_samples
        )

        # 气溶胶光学厚度: 对数均匀分布（更符合实际情况）
        aod_log_min = np.log10(self.param_ranges['aod550']['min'])
        aod_log_max = np.log10(self.param_ranges['aod550']['max'])
        params['aod550'] = 10 ** np.random.uniform(aod_log_min, aod_log_max, n_samples)

        # 水汽: 均匀分布
        params['h2o'] = np.random.uniform(
            self.param_ranges['h2o']['min'],
            self.param_ranges['h2o']['max'],
            n_samples
        )

        # 臭氧: 均匀分布
        params['o3'] = np.random.uniform(
            self.param_ranges['o3']['min'],
            self.param_ranges['o3']['max'],
            n_samples
        )

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
            self.logger.info(f"  波段 {band_id}: 生成 {len(band_combinations)} 个参数组合")

            # 统计极端角度比例
            extreme_count = sum(1 for p in band_combinations if p['is_extreme'])
            self.logger.info(f"  极端角度比例: {extreme_count / len(band_combinations) * 100:.1f}%")

        return all_combinations

    def generate_validation_grid(self, bands: List[str] = None) -> Dict[str, List[Dict]]:
        """
        生成稀疏规则网格用于验证（不参与训练）
        保持原有网格特性以便可视化对比
        """
        if bands is None:
            bands = list(self.bands.keys())

        # 稀疏网格配置
        grid_config = {
            'sza': [0, 20, 40, 60, 70, 75, 80, 85],
            'vza': [0, 20, 40, 60, 70, 75],
            'raa': [0, 30, 60, 90, 120, 150, 180],
            'rho_true': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
            'aod550': [0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
            'h2o': [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
            'o3': [0.2, 0.25, 0.3, 0.35, 0.4],
        }

        all_combinations = {}

        for band_id in bands:
            wavelength = self.bands[band_id]['wavelength']
            combinations = []

            # 使用itertools生成所有组合
            for values in product(
                    grid_config['sza'],
                    grid_config['vza'],
                    grid_config['raa'],
                    grid_config['rho_true'],
                    grid_config['aod550'],
                    grid_config['h2o'],
                    grid_config['o3']
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
            self.logger.info(f"验证网格 - 波段 {band_id}: {len(combinations)} 个组合")

        return all_combinations
