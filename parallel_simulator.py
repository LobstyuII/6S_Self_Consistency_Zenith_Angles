# ==================== parallel_simulator.py (双正向模拟版) ====================
"""
重构的并行模拟器 - 采用双正向模拟策略
目标：生成 ΔTOA = ρ_TOA^SA - ρ_TOA^PPA
用于训练机器学习模型，在TOA层进行前置校正
"""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
import gc
import traceback
from Py6S import *
from tqdm import tqdm

from config import ExperimentConfig
from utils import setup_logger
from data_generator import MonteCarloDataGenerator

mp.set_start_method('spawn', force=True)


class SixSProcessWorker:
    """6S进程工作器 - 仅执行正向模拟（等效角度 vs 真实角度）"""

    def __init__(self, band_wavelength: float):
        self.band_wavelength = band_wavelength

    @staticmethod
    def kasten_young_airmass(z_deg: float) -> float:
        """Kasten & Young (1989) 球形大气质量"""
        z_clipped = np.clip(z_deg, 0, 89.9)
        cos_z = np.cos(np.radians(z_clipped))
        m_sphere = 1.0 / (cos_z + 0.50572 * (96.07995 - z_clipped) ** (-1.6364))
        return m_sphere

    @staticmethod
    def get_effective_angle(true_angle_deg: float) -> float:
        """根据真实角度计算等效角度，使PPA光程等于球形光程"""
        m_sphere = SixSProcessWorker.kasten_young_airmass(true_angle_deg)
        cos_eff = 1.0 / m_sphere
        cos_eff = np.clip(cos_eff, 0.001, 1.0)
        return np.degrees(np.arccos(cos_eff))

    def _get_atmos_profile(self, params: Dict[str, float]):
        """创建大气廓线对象"""
        atmos_key = params.get('atmos_profile', 'MidlatitudeSummer')
        water = params.get('h2o', np.nan)
        ozone = params.get('o3', np.nan)
        if not np.isnan(water) and not np.isnan(ozone) and water > 0 and ozone > 0:
            return AtmosProfile.UserWaterAndOzone(water, ozone)
        mapping = {
            'MidlatitudeSummer': AtmosProfile.MidlatitudeSummer,
            'MidlatitudeWinter': AtmosProfile.MidlatitudeWinter,
            'Tropical': AtmosProfile.Tropical,
            'SubarcticSummer': AtmosProfile.SubarcticSummer,
            'SubarcticWinter': AtmosProfile.SubarcticWinter,
        }
        profile_const = mapping.get(atmos_key, AtmosProfile.MidlatitudeSummer)
        return AtmosProfile.PredefinedType(profile_const)

    def _get_aero_profile(self, params: Dict[str, float]):
        """创建气溶胶对象"""
        aero_key = params.get('aero_profile', 'Continental')
        mapping = {
            'Continental': AeroProfile.Continental,
            'Maritime': AeroProfile.Maritime,
            'Urban': AeroProfile.Urban,
            'Desert': AeroProfile.Desert,
            'BiomassBurning': AeroProfile.BiomassBurning,
        }
        profile_const = mapping.get(aero_key, AeroProfile.Continental)
        return AeroProfile.PredefinedType(profile_const)

    def create_sixs_instance(self, params: Dict[str, float], use_effective_angles: bool) -> SixS:
        """
        创建并配置6S实例
        Args:
            params: 参数字典
            use_effective_angles: True -> 使用等效角度（模拟球形大气）
                                  False -> 使用真实角度（平面平行假设）
        """
        try:
            s = SixS()
            s.wavelength = Wavelength(self.band_wavelength)

            s.atmos_profile = self._get_atmos_profile(params)
            s.aero_profile = self._get_aero_profile(params)

            aod_val = params.get('aod550', 0.2)
            if np.isnan(aod_val) or aod_val < 0:
                aod_val = 0.2
            s.aot550 = aod_val

            s.geometry = Geometry.User()

            true_sza = params['sza']
            true_vza = params['vza']

            if use_effective_angles:
                eff_sza = self.get_effective_angle(true_sza)
                eff_vza = self.get_effective_angle(true_vza)
                s.geometry.solar_z = eff_sza
                s.geometry.view_z = eff_vza
            else:
                s.geometry.solar_z = true_sza
                s.geometry.view_z = true_vza

            raa = float(params.get('raa', 0.0))
            s.geometry.solar_a = 0.0
            s.geometry.view_a = raa

            s.altitudes = Altitudes()
            s.altitudes.set_target_custom_altitude(params.get('target_altitude', 0.0))
            s.altitudes.set_sensor_satellite_level()

            s.ground_reflectance = GroundReflectance.HomogeneousLambertian(params.get('rho_true', 0.2))
            s.atmos_corr = AtmosCorr.NoAtmosCorr()

            return s
        except Exception as e:
            print(f"创建6S实例失败: {e}")
            return None

    def run_forward(self, params: Dict[str, float], use_effective_angles: bool) -> Dict[str, Any]:
        """
        运行正向模拟，返回表观反射率
        """
        try:
            s = self.create_sixs_instance(params, use_effective_angles)
            if s is None:
                return {'success': False, 'error': '创建6S实例失败'}

            s.run()
            rho_toa = s.outputs.values['apparent_reflectance']

            del s
            gc.collect()

            return {
                'success': True,
                'rho_toa': rho_toa,
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def run_dual_forward(self, params: Dict[str, float]) -> Dict[str, Any]:
        """
        双正向模拟：先运行等效角度（模拟真实球形大气），再运行真实角度（理想PPA）
        返回两个rho_toa及差值delta_toa = rho_sa - rho_ppa
        """
        # 1. 等效角度正向（模拟真实观测）
        res_sa = self.run_forward(params, use_effective_angles=True)
        if not res_sa.get('success', False):
            return {**params, 'success': False, 'error': res_sa.get('error', 'SA forward failed')}

        # 2. 真实角度正向（理想PPA）
        res_ppa = self.run_forward(params, use_effective_angles=False)
        if not res_ppa.get('success', False):
            return {**params, 'success': False, 'error': res_ppa.get('error', 'PPA forward failed')}

        rho_sa = res_sa['rho_toa']
        rho_ppa = res_ppa['rho_toa']

        # 计算差值
        delta_toa = rho_sa - rho_ppa

        # 返回所有参数及关键模拟值
        return {
            'success': True,
            'rho_true': params['rho_true'],
            'rho_toa_sa': rho_sa,          # 模拟的真实观测
            'rho_toa_ppa': rho_ppa,         # 理想PPA下的观测
            'delta_toa': delta_toa,          # 机器学习目标
            'sza': params['sza'],
            'vza': params['vza'],
            'raa': params.get('raa', 0.0),
            'aod550': params.get('aod550', 0.2),
            'h2o': params.get('h2o', np.nan),
            'o3': params.get('o3', np.nan),
            'is_extreme': params.get('is_extreme', False),
            **params
        }


class ParallelSimulator:
    """重构的并行模拟器 - 使用双正向模拟生成数据集"""

    def __init__(self, config: ExperimentConfig = None, logger=None):
        self.config = config or ExperimentConfig
        self.logger = logger or setup_logger('ParallelSimulator')
        self.data_generator = MonteCarloDataGenerator(config, logger)

        self.logger.info("初始化双正向模拟器 (等效角度 vs 真实角度)")
        self.logger.info("目标：生成 ΔTOA = ρ_SA - ρ_PPA，用于TOA层前置校正")

    def generate_training_dataset(self, total_samples_per_band: int = 100000,
                                  bands: List[str] = None) -> Dict[str, List[Dict]]:
        """生成训练数据集 - 蒙特卡洛混合采样"""
        self.logger.info(f"生成训练数据集，每波段 {total_samples_per_band:,} 个样本")
        training_data = self.data_generator.generate_mixed_sampling_dataset(
            total_samples=total_samples_per_band,
            bands=bands
        )
        return training_data

    def generate_validation_grid(self, bands: List[str] = None) -> Dict[str, List[Dict]]:
        """生成验证网格 - 稀疏规则网格"""
        validation_data = self.data_generator.generate_validation_grid(bands)
        return validation_data

    def simulate_dataset(self, dataset: Dict[str, List[Dict]],
                         max_workers: int = None) -> Dict[str, pd.DataFrame]:
        """
        并行模拟数据集，调用每个样本的 run_dual_forward
        """
        if max_workers is None:
            import multiprocessing as mp
            physical_cores = mp.cpu_count() // 2
            max_workers = min(physical_cores,
                              self.config.PARALLEL_CONFIG.get('max_concurrent_6s', 4))

        self.logger.info(f"开始双正向并行模拟，使用 {max_workers} 个工作进程")
        self.logger.info("模式：等效角度（球形大气）vs 真实角度（PPA）")

        task_blocks = []
        for band_id, param_list in dataset.items():
            chunk_size = self.config.PARALLEL_CONFIG.get('chunk_size', 1000)
            for i in range(0, len(param_list), chunk_size):
                chunk = param_list[i:i + chunk_size]
                task_blocks.append((band_id, chunk))

        self.logger.info(f"任务块数: {len(task_blocks)}")
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {}
            for batch in task_blocks:
                future = executor.submit(self._worker_process_wrapper, batch)
                future_to_batch[future] = batch

            results_by_band = {}
            total_batches = len(task_blocks)

            with tqdm(total=total_batches, desc="并行模拟", unit="批次") as pbar:
                for future in as_completed(future_to_batch):
                    try:
                        band_id, batch_results = future.result(timeout=3600)
                        if band_id not in results_by_band:
                            results_by_band[band_id] = []
                        results_by_band[band_id].extend(batch_results)

                        success_count = sum(1 for r in batch_results if r.get('success', False))
                        total_count = len(batch_results)
                        pbar.update(1)
                        pbar.set_postfix_str(f"{band_id}: {success_count}/{total_count}")
                    except Exception as e:
                        self.logger.error(f"任务批次处理失败: {e}")
                        pbar.update(1)

        elapsed_time = time.time() - start_time
        final_results = {}
        total_samples = 0
        total_success = 0

        for band_id, result_list in results_by_band.items():
            if result_list:
                df = pd.DataFrame(result_list)
                final_results[band_id] = df
                total_samples += len(df)
                if 'success' in df.columns:
                    success_count = df['success'].sum() if df['success'].dtype == bool else (df['success'] == 1).sum()
                    total_success += success_count

                # 统计delta_toa特性
                if 'delta_toa' in df.columns:
                    delta_mean = df['delta_toa'].mean()
                    delta_std = df['delta_toa'].std()
                    delta_min = df['delta_toa'].min()
                    delta_max = df['delta_toa'].max()
                    self.logger.info(f"波段 {band_id}: ΔTOA 统计: 均值={delta_mean:.6f}, 标准差={delta_std:.6f}, "
                                     f"范围=[{delta_min:.6f}, {delta_max:.6f}]")

        success_rate = (total_success / total_samples) * 100 if total_samples > 0 else 0
        self.logger.info(f"并行模拟完成，耗时: {elapsed_time:.2f}秒")
        self.logger.info(f"总样本数: {total_samples} (成功率: {success_rate:.1f}%)")
        if elapsed_time > 0:
            self.logger.info(f"平均速度: {total_samples / elapsed_time:.2f} 样本/秒")

        return final_results

    def _worker_process_wrapper(self, task_batch: Tuple[str, List[Dict]]) -> Tuple[str, List[Dict]]:
        """工作进程包装器，调用 run_dual_forward"""
        import gc
        band_id, param_list = task_batch
        band_config = self.config.BANDS[band_id]
        wavelength = band_config['wavelength']

        worker = SixSProcessWorker(wavelength)
        results = []

        for i, params in enumerate(param_list):
            try:
                result = worker.run_dual_forward(params)
                result['band'] = band_id
                result['wavelength'] = wavelength
                results.append(result)

                if i % 100 == 0:
                    gc.collect()
            except Exception as e:
                error_result = {
                    'band': band_id,
                    'success': False,
                    'error': str(e),
                    **params
                }
                results.append(error_result)

        del worker
        gc.collect()
        return band_id, results