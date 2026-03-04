# ==================== sixs_inversion.py (调试版) ====================
"""
6S反演核心模块（带缓存）
提供从TOA反射率反演LSR的功能，并缓存计算结果避免重复运行。
修正：几何参数传递方式，明确使用太阳方位角和传感器方位角。
增加调试打印，输出每次反演的输入输出。
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
                - sza, vza, rho_toa
                - aod550, h2o, o3
                - wavelength (可选，若未提供则使用初始化时波长)
                - atmos_profile, aero_profile (可选，默认 MidlatitudeSummer, Continental)
                - target_altitude (默认 0.0)
                - solar_a: 太阳方位角（可选，默认0）
                - view_a: 传感器方位角（可选，默认与solar_a相对，若只提供raa则退化为solar_a=0, view_a=raa）
                - raa: 相对方位角（可选，用于特征但不直接用于几何设置）
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

            # 几何设置
            s.geometry = Geometry.User()
            s.geometry.solar_z = params['sza']

            # 优先使用绝对方位角（solar_a, view_a）
            if 'solar_a' in params and 'view_a' in params:
                s.geometry.solar_a = params['solar_a']
                s.geometry.view_a = params['view_a']
            else:
                # 如果没有绝对方位角，则使用简化模式：solar_a=0, view_a=raa
                raa = params.get('raa', 0.0)
                s.geometry.solar_a = 0.0
                s.geometry.view_a = raa
                # 发出警告，提醒用户
                if 'solar_a' not in params or 'view_a' not in params:
                    print("Warning: solar_a or view_a missing, using solar_a=0, view_a=raa (relative azimuth)")

            s.geometry.view_z = params['vza']

            # 高度
            s.altitudes = Altitudes()
            s.altitudes.set_target_custom_altitude(params.get('target_altitude', 0.0))
            s.altitudes.set_sensor_satellite_level()

            # 执行大气校正
            rho_toa = params['rho_toa']
            s.atmos_corr = AtmosCorr.AtmosCorrLambertianFromReflectance(rho_toa)
            s.run()

            rho_lsr = s.outputs.values['pixel_reflectance']

            # ========== 调试输出 ==========
            print(f"[SixSInversion] rho_toa_in={rho_toa:.8f} -> rho_lsr_out={rho_lsr:.8f}")
            # 可选：打印一些大气参数，检查它们是否变化
            try:
                path_ref = s.outputs.values.get('path_reflectance', np.nan)
                trans = s.outputs.values.get('total_gas_transmittance', np.nan)
                print(f"                 path_reflectance={path_ref:.6f}, transmittance={trans:.6f}")
            except:
                pass
            # ============================

            # 清理
            del s
            gc.collect()

            return {
                'success': True,
                'rho_lsr': rho_lsr,
                'rho_toa_input': rho_toa,
            }

        except Exception as e:
            print(f"[SixSInversion] ERROR: {e}")
            return {
                'success': False,
                'error': str(e),
                'rho_lsr': np.nan,
            }


def _hash_params(params: dict, precision: dict = None) -> str:
    """将参数字典转换为哈希键，用于缓存。"""
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
    key_parts = []
    for key in ['sza', 'vza', 'raa', 'aod550', 'h2o', 'o3', 'rho_toa', 'wavelength']:
        val = params.get(key)
        if val is None:
            val = 0.0
        prec = precision.get(key, 4)
        rounded = round(float(val), prec)
        key_parts.append(f"{rounded:.{prec}f}")
    key_parts.append(params.get('atmos_profile', 'MidlatitudeSummer'))
    key_parts.append(params.get('aero_profile', 'Continental'))
    key_parts.append(str(params.get('target_altitude', 0.0)))
    key_str = '_'.join(key_parts)
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()


def run_inversion_cached(params: dict, cache_dir: Path = DEFAULT_CACHE_DIR) -> dict:
    """带缓存的6S反演（可选用）"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    cache_key = _hash_params(params)
    cache_file = cache_dir / f"{cache_key}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
            result['from_cache'] = True
            return result
        except Exception:
            pass
    wavelength = params.get('wavelength')
    if wavelength is None:
        raise ValueError("Missing 'wavelength' in params")
    inverter = SixSInversion(wavelength)
    result = inverter.run(params)
    result['from_cache'] = False
    if result['success']:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"Warning: failed to save cache {cache_file}: {e}")
    return result