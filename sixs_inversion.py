# ==================== sixs_inversion.py ====================
"""
6S反演核心模块（带缓存）
提供从TOA反射率反演LSR的功能，并缓存计算结果避免重复运行。
"""
import os
import hashlib
import pickle
import numpy as np
import gc
from pathlib import Path
from datetime import datetime

from Py6S import *

# 默认缓存目录
DEFAULT_CACHE_DIR = Path("./sixs_cache")
DEFAULT_CACHE_DIR.mkdir(exist_ok=True)


class SixSInversion:
    """单次6S反演（TOA -> LSR）"""

    def __init__(self, wavelength: float):
        self.wavelength = wavelength
        self._init_profiles()

    def _init_profiles(self):
        self.atmos_profiles = {
            'MidlatitudeSummer': AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeSummer),
            'MidlatitudeWinter': AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeWinter),
            'Tropical': AtmosProfile.PredefinedType(AtmosProfile.Tropical),
            'SubarcticSummer': AtmosProfile.PredefinedType(AtmosProfile.SubarcticSummer),
            'SubarcticWinter': AtmosProfile.PredefinedType(AtmosProfile.SubarcticWinter),
        }
        self.aero_profiles = {
            'Continental': AeroProfile.PredefinedType(AeroProfile.Continental),
            'Maritime': AeroProfile.PredefinedType(AeroProfile.Maritime),
            'Urban': AeroProfile.PredefinedType(AeroProfile.Urban),
            'Desert': AeroProfile.PredefinedType(AeroProfile.Desert),
            'BiomassBurning': AeroProfile.PredefinedType(AeroProfile.BiomassBurning),
        }

    def run(self, params: dict) -> dict:
        """
        执行单次反演
        参数:
            params: 字典，必须包含以下键：
                - sza, vza, raa, rho_toa (待反演的TOA反射率)
                - aod550, h2o, o3
                - wavelength (可选，若未提供则使用初始化时波长)
                - atmos_profile, aero_profile (可选，默认 MidlatitudeSummer, Continental)
                - target_altitude (默认 0.0)
                - date, lat (用于自动大气廓线，暂未使用)
        返回:
            dict: 包含 'rho_lsr' 及其他诊断信息
        """
        try:
            s = SixS()
            wl = params.get('wavelength', self.wavelength)
            s.wavelength = Wavelength(wl)

            # 大气廓线
            atmos_key = params.get('atmos_profile', 'MidlatitudeSummer')
            if atmos_key == 'UserWaterAndOzone':
                water = params.get('h2o', 2.0)
                ozone = params.get('o3', 0.3)
                if not np.isnan(water) and not np.isnan(ozone) and water > 0 and ozone > 0:
                    s.atmos_profile = AtmosProfile.UserWaterAndOzone(water, ozone)
                else:
                    s.atmos_profile = self.atmos_profiles.get('MidlatitudeSummer')
            else:
                s.atmos_profile = self.atmos_profiles.get(atmos_key, self.atmos_profiles['MidlatitudeSummer'])

            # 气溶胶
            aero_key = params.get('aero_profile', 'Continental')
            s.aero_profile = self.aero_profiles.get(aero_key, self.aero_profiles['Continental'])

            # AOD
            aod = params.get('aod550', 0.1)
            if not np.isnan(aod) and aod >= 0:
                s.aot550 = aod
            else:
                s.aot550 = 0.1

            # 几何
            s.geometry = Geometry.User()
            s.geometry.solar_z = params['sza']
            s.geometry.solar_a = params.get('phi', 0.0)
            s.geometry.view_z = params['vza']
            s.geometry.view_a = params.get('raa', 0.0)  # 注意：6S中view_a是相对方位角

            # 高度
            s.altitudes = Altitudes()
            s.altitudes.set_target_custom_altitude(params.get('target_altitude', 0.0))
            s.altitudes.set_sensor_satellite_level()

            # 执行大气校正
            rho_toa = params['rho_toa']
            s.atmos_corr = AtmosCorr.AtmosCorrLambertianFromReflectance(rho_toa)
            s.run()

            rho_lsr = s.outputs.values['pixel_reflectance']

            # 清理
            del s
            gc.collect()

            return {
                'success': True,
                'rho_lsr': rho_lsr,
                'rho_toa_input': rho_toa,
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'rho_lsr': np.nan,
            }


def _hash_params(params: dict, precision: dict = None) -> str:
    """
    将参数字典转换为哈希键，用于缓存。
    参数按一定精度四舍五入以避免微小差异导致缓存失效。
    """
    if precision is None:
        precision = {
            'sza': 2,
            'vza': 2,
            'raa': 2,
            'aod550': 4,
            'h2o': 4,
            'o3': 4,
            'rho_toa': 6,
            'wavelength': 4,
        }

    # 提取关键参数并量化
    key_parts = []
    for key in ['sza', 'vza', 'raa', 'aod550', 'h2o', 'o3', 'rho_toa', 'wavelength']:
        val = params.get(key)
        if val is None:
            val = 0.0
        prec = precision.get(key, 4)
        # 四舍五入到指定小数位
        rounded = round(float(val), prec)
        key_parts.append(f"{rounded:.{prec}f}")

    # 添加配置字符串
    key_parts.append(params.get('atmos_profile', 'MidlatitudeSummer'))
    key_parts.append(params.get('aero_profile', 'Continental'))
    key_parts.append(str(params.get('target_altitude', 0.0)))

    # 合并为字符串
    key_str = '_'.join(key_parts)
    # 生成哈希
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()


def run_inversion_cached(params: dict, cache_dir: Path = DEFAULT_CACHE_DIR) -> dict:
    """
    带缓存的6S反演。先检查缓存，若无则运行并保存。
    返回的字典包含 'rho_lsr' 和 'success' 等字段。
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)

    # 生成缓存键
    cache_key = _hash_params(params)
    cache_file = cache_dir / f"{cache_key}.pkl"

    # 检查缓存
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
            # 添加缓存命中标记
            result['from_cache'] = True
            return result
        except Exception:
            # 缓存损坏，重新计算
            pass

    # 创建反演器并运行
    wavelength = params.get('wavelength')
    if wavelength is None:
        raise ValueError("Missing 'wavelength' in params")
    inverter = SixSInversion(wavelength)
    result = inverter.run(params)
    result['from_cache'] = False

    # 保存到缓存（仅成功的结果可缓存，失败也可缓存避免反复尝试？根据需求决定）
    if result['success']:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"Warning: failed to save cache {cache_file}: {e}")

    return result