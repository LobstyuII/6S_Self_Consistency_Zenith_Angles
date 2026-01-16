# ==================== data_generator.py ====================
"""
数据生成模块
"""
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple, Any
import numpy as np
from Py6S import *
import pandas as pd
from tqdm import tqdm

from config import ExperimentConfig
from utils import setup_logger, cache_result, save_dataset, calculate_airmass


class SixSSimulator:
    """6S模拟器类"""

    def __init__(self, band_wavelength: float, logger=None):
        """
        初始化6S模拟器
        """
        self.band_wavelength = band_wavelength
        self.logger = logger or setup_logger('SixSSimulator')
        self._precompute_atmos_profiles()

    def _precompute_atmos_profiles(self):
        """预计算大气廓线映射"""
        self.atmos_profile_map = {
            'MidlatitudeSummer': AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeSummer),
            'MidlatitudeWinter': AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeWinter),
            'Tropical': AtmosProfile.PredefinedType(AtmosProfile.Tropical),
            'SubarcticSummer': AtmosProfile.PredefinedType(AtmosProfile.SubarcticSummer),
            'SubarcticWinter': AtmosProfile.PredefinedType(AtmosProfile.SubarcticWinter),
            'UserWaterAndOzone': None
        }

        self.aero_profile_map = {
            'Continental': AeroProfile.PredefinedType(AeroProfile.Continental),
            'Maritime': AeroProfile.PredefinedType(AeroProfile.Maritime),
            'Urban': AeroProfile.PredefinedType(AeroProfile.Urban),
            'Desert': AeroProfile.PredefinedType(AeroProfile.Desert),
            'BiomassBurning': AeroProfile.PredefinedType(AeroProfile.BiomassBurning),
        }

    def create_sixs_instance(self, params: Dict[str, float],
                             mode: str = 'forward') -> SixS:
        """
        创建并配置6S实例
        """
        s = SixS()
        s.wavelength = Wavelength(self.band_wavelength)

        atmos_key = params.get('atmos_profile', 'MidlatitudeSummer')

        if 'h2o' in params and 'o3' in params:
            water = params['h2o']
            ozone = params['o3']
            s.atmos_profile = AtmosProfile.UserWaterAndOzone(water, ozone)
            self.logger.debug(f"使用自定义水汽({water}g/cm²)和臭氧({ozone}cm-atm)")
        else:
            s.atmos_profile = self.atmos_profile_map.get(atmos_key)

        aero_key = params.get('aero_profile', 'Continental')
        s.aero_profile = self.aero_profile_map.get(aero_key)
        s.aot550 = params.get('aod550', 0.2)

        s.geometry = Geometry.User()
        s.geometry.solar_z = params['sza']
        s.geometry.solar_a = 0.0
        s.geometry.view_z = params['vza']
        # 移除raa参数
        s.geometry.view_a = 0.0

        s.altitudes = Altitudes()
        s.altitudes.set_target_custom_altitude(params.get('target_altitude', 0.0))
        s.altitudes.set_sensor_satellite_level()

        if mode == 'forward':
            s.ground_reflectance = GroundReflectance.HomogeneousLambertian(
                params.get('rho_true', 0.2)
            )
            s.atmos_corr = AtmosCorr.NoAtmosCorr()

        return s

    @cache_result
    def run_forward_simulation(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        运行正向模拟
        """
        try:
            s = self.create_sixs_instance(params, mode='forward')
            s.run()

            rho_toa = s.outputs.values['apparent_reflectance']
            airmass_sza = calculate_airmass(params['sza'])
            airmass_vza = calculate_airmass(params['vza'])

            result = {
                'rho_true': params.get('rho_true', 0.2),
                'rho_toa': rho_toa,
                'airmass_sza': airmass_sza,
                'airmass_vza': airmass_vza,
                'total_airmass': airmass_sza + airmass_vza,
                'sza': params['sza'],
                'vza': params['vza'],
                'aod550': params.get('aod550', 0.2),
                'h2o': params.get('h2o', np.nan),
                'o3': params.get('o3', np.nan),
                'success': True
            }

            return result

        except Exception as e:
            self.logger.error(f"正向模拟失败: {e}")
            return {
                'rho_true': params.get('rho_true', 0.2),
                'rho_toa': np.nan,
                'airmass_sza': np.nan,
                'airmass_vza': np.nan,
                'total_airmass': np.nan,
                'sza': params['sza'],
                'vza': params['vza'],
                'aod550': params.get('aod550', 0.2),
                'h2o': params.get('h2o', np.nan),
                'o3': params.get('o3', np.nan),
                'success': False,
                'error': str(e)
            }

    def run_inversion(self, rho_toa: float, params: Dict[str, float]) -> Dict[str, float]:
        """
        运行反演
        """
        try:
            s = self.create_sixs_instance(params, mode='inversion')
            s.atmos_corr = AtmosCorr.AtmosCorrLambertianFromReflectance(rho_toa)
            s.run()

            rho_retrieved = s.outputs.values['pixel_reflectance']

            return {
                'rho_toa_input': rho_toa,
                'rho_retrieved': rho_retrieved,
                'success': True
            }

        except Exception as e:
            self.logger.error(f"反演失败: {e}")
            return {
                'rho_toa_input': rho_toa,
                'rho_retrieved': np.nan,
                'success': False,
                'error': str(e)
            }

    def run_closed_loop(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        运行闭合循环
        """
        forward_result = self.run_forward_simulation(params)

        if not forward_result['success'] or np.isnan(forward_result['rho_toa']):
            return {
                **forward_result,
                'rho_retrieved': np.nan,
                'error_absolute': np.nan,
                'error_relative': np.nan,
                'closed_loop_success': False
            }

        inv_params = {k: v for k, v in params.items() if k != 'rho_true'}
        inv_result = self.run_inversion(forward_result['rho_toa'], inv_params)

        if not inv_result['success'] or np.isnan(inv_result['rho_retrieved']):
            return {
                **forward_result,
                **inv_result,
                'error_absolute': np.nan,
                'error_relative': np.nan,
                'closed_loop_success': False
            }

        error_abs = inv_result['rho_retrieved'] - params.get('rho_true', 0.2)
        rho_true = params.get('rho_true', 0.2)
        error_rel = error_abs / rho_true if rho_true > 0 else np.nan

        return {
            **forward_result,
            **inv_result,
            'error_absolute': error_abs,
            'error_relative': error_rel,
            'closed_loop_success': True
        }


class BatchSimulator:
    """批量模拟器"""

    def __init__(self, config: ExperimentConfig, logger=None):
        """
        初始化批量模拟器
        """
        self.config = config
        self.logger = logger or setup_logger('BatchSimulator')

        self.simulators = {
            band_id: SixSSimulator(band_info['wavelength'], logger)
            for band_id, band_info in config.BANDS.items()
        }

    def simulate_single_task(self, task: Tuple[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        模拟单个任务
        """
        band_id, params = task

        try:
            simulator = self.simulators[band_id]
            result = simulator.run_closed_loop(params)
            result['band'] = band_id
            result['wavelength'] = self.config.BANDS[band_id]['wavelength']
            return result

        except Exception as e:
            self.logger.error(f"任务失败: {e}")
            return {
                'band': band_id,
                'success': False,
                'error': str(e),
                **params
            }

    def run_batch_simulation(self, band_id: str = 'band3', mode: str = 'full') -> pd.DataFrame:
        """
        运行批量模拟
        """
        self.logger.info(f"开始批量模拟 - 波段: {band_id}, 模式: {mode}")

        param_list = self.config.get_param_combinations(mode)
        tasks = [(band_id, params) for params in param_list]
        results = []

        n_workers = min(self.config.PARALLEL_CONFIG['n_workers'], cpu_count())
        chunk_size = self.config.PARALLEL_CONFIG['chunk_size']

        with Pool(processes=n_workers) as pool:
            with tqdm(total=len(tasks), desc=f"模拟 {band_id} ({mode})") as pbar:
                for result in pool.imap_unordered(
                        self.simulate_single_task, tasks, chunksize=chunk_size
                ):
                    results.append(result)
                    pbar.update(1)

        df_results = pd.DataFrame(results)

        success_count = df_results['closed_loop_success'].sum() if 'closed_loop_success' in df_results.columns else 0
        success_rate = (success_count / len(df_results)) * 100 if len(df_results) > 0 else 0

        error_count = df_results['error_absolute'].notna().sum() if 'error_absolute' in df_results.columns else 0
        valid_rate = (error_count / len(df_results)) * 100 if len(df_results) > 0 else 0

        self.logger.info(f"模拟完成 - 成功率: {success_rate:.1f}%, 有效数据率: {valid_rate:.1f}%")

        output_file = self.config.DATA_DIR / f"simulation_results_{band_id}_{mode}.nc"
        data_dict = {}
        for col in df_results.columns:
            col_data = df_results[col].values
            if col_data.dtype == object:
                try:
                    col_data = col_data.astype(str)
                except:
                    pass
            data_dict[col] = col_data

        save_dataset(data_dict, output_file)
        self.logger.info(f"结果已保存: {output_file}")

        return df_results

    def run_all_bands(self) -> Dict[str, pd.DataFrame]:
        """运行所有波段"""
        results = {}
        for band_id in self.config.BANDS.keys():
            self.logger.info(f"开始处理波段: {band_id}")
            results[band_id] = self.run_batch_simulation(band_id)
        return results