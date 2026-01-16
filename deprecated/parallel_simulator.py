# ==================== parallel_simulator.py (修改版) ====================
"""
并行模拟器 - 重构版
集成蒙特卡洛采样策略
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
from utils import setup_logger, monte_carlo_sample, calculate_physical_features

# 设置多进程启动方法
mp.set_start_method('spawn', force=True)


class RefactoredSixSProcessWorker:
    """重构的6S进程工作器"""

    def __init__(self, band_wavelength: float):
        self.band_wavelength = band_wavelength
        self._precompute_atmos_profiles()

    def _precompute_atmos_profiles(self):
        """预计算大气廓线映射"""
        self.atmos_profile_map = {
            'MidlatitudeSummer': AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeSummer),
            'MidlatitudeWinter': AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeWinter),
            'Tropical': AtmosProfile.PredefinedType(AtmosProfile.Tropical),
            'SubarcticSummer': AtmosProfile.PredefinedType(AtmosProfile.SubarcticSummer),
            'SubarcticWinter': AtmosProfile.PredefinedType(AtmosProfile.SubarcticWinter),
        }

        self.aero_profile_map = {
            'Continental': AeroProfile.PredefinedType(AeroProfile.Continental),
            'Maritime': AeroProfile.PredefinedType(AeroProfile.Maritime),
            'Urban': AeroProfile.PredefinedType(AeroProfile.Urban),
            'Desert': AeroProfile.PredefinedType(AeroProfile.Desert),
            'BiomassBurning': AeroProfile.PredefinedType(AeroProfile.BiomassBurning),
        }

    def create_sixs_instance(self, params: Dict[str, float]) -> SixS:
        """
        创建并配置6S实例 - 支持连续变量
        """
        try:
            s = SixS()
            s.wavelength = Wavelength(self.band_wavelength)

            # 配置大气廓线
            atmos_key = params.get('atmos_profile', 'MidlatitudeSummer')

            # 使用连续的水汽和臭氧值
            if 'h2o' in params and 'o3' in params:
                water = params['h2o']
                ozone = params['o3']
                if not np.isnan(water) and not np.isnan(ozone) and water > 0 and ozone > 0:
                    s.atmos_profile = AtmosProfile.UserWaterAndOzone(water, ozone)
                else:
                    s.atmos_profile = self.atmos_profile_map.get(atmos_key)
            else:
                s.atmos_profile = self.atmos_profile_map.get(atmos_key)

            # 配置气溶胶
            aero_key = params.get('aero_profile', 'Continental')
            s.aero_profile = self.aero_profile_map.get(aero_key)

            # 设置连续的气溶胶光学厚度
            if 'aod550' in params:
                aod_val = params['aod550']
                if not np.isnan(aod_val) and aod_val >= 0:
                    s.aot550 = aod_val
                else:
                    s.aot550 = 0.2
            else:
                s.aot550 = 0.2

            # 配置几何参数
            s.geometry = Geometry.User()
            s.geometry.solar_z = params['sza']
            s.geometry.solar_a = params.get('phi', 0.0)  # 使用实际的相对方位角
            s.geometry.view_z = params['vza']
            s.geometry.view_a = 0.0  # 固定观测方位角

            # 配置高度
            s.altitudes = Altitudes()
            s.altitudes.set_target_custom_altitude(params.get('target_altitude', 0.0))
            s.altitudes.set_sensor_satellite_level()

            return s

        except Exception as e:
            print(f"创建6S实例失败: {e}")
            return None

    def run_closed_loop(self, params: Dict[str, float]) -> Dict[str, Any]:
        """
        运行闭合循环模拟 - 包含物理特征
        """
        try:
            # 1. 正向模拟
            s = self.create_sixs_instance(params)
            if s is None:
                return {
                    'success': False,
                    'error': '创建6S实例失败',
                    'closed_loop_success': False,
                    **params
                }

            # 配置连续的地表反射率
            rho_true = params.get('rho_true', 0.2)
            s.ground_reflectance = GroundReflectance.HomogeneousLambertian(rho_true)
            s.atmos_corr = AtmosCorr.NoAtmosCorr()

            # 运行正向模拟
            s.run()
            rho_toa = s.outputs.values['apparent_reflectance']

            # 2. 反演模拟
            s_inv = self.create_sixs_instance({k: v for k, v in params.items() if k != 'rho_true'})
            if s_inv is None:
                del s
                gc.collect()
                return {
                    'success': False,
                    'error': '创建反演6S实例失败',
                    'closed_loop_success': False,
                    **params
                }

            s_inv.atmos_corr = AtmosCorr.AtmosCorrLambertianFromReflectance(rho_toa)
            s_inv.run()
            rho_retrieved = s_inv.outputs.values['pixel_reflectance']

            # 3. 计算物理特征
            sza = params['sza']
            vza = params['vza']
            raa = params.get('raa', 0.0)

            # 大气质量因子
            cos_sza = np.cos(np.radians(sza)) if sza < 88 else 0.0349
            cos_vza = np.cos(np.radians(vza)) if vza < 88 else 0.0349
            airmass_sza = 1.0 / cos_sza if cos_sza > 0.01 else 1.0 / 0.01
            airmass_vza = 1.0 / cos_vza if cos_vza > 0.01 else 1.0 / 0.01

            # 散射角
            cos_scat = -np.cos(np.radians(sza)) * np.cos(np.radians(vza)) + \
                       np.sin(np.radians(sza)) * np.sin(np.radians(vza)) * np.cos(np.radians(raa))
            cos_scat = np.clip(cos_scat, -1.0, 1.0)
            scattering_angle = np.degrees(np.arccos(cos_scat))

            # 4. 计算误差
            error_abs = rho_retrieved - rho_true
            error_rel = error_abs / rho_true if rho_true > 0 else np.nan

            # 5. 清理实例
            del s, s_inv
            gc.collect()

            # 6. 返回结果
            return {
                'success': True,
                'closed_loop_success': True,
                'rho_true': rho_true,
                'rho_toa': rho_toa,
                'rho_retrieved': rho_retrieved,
                'error_absolute': error_abs,
                'error_relative': error_rel,
                'airmass_sza': airmass_sza,
                'airmass_vza': airmass_vza,
                'total_airmass': airmass_sza + airmass_vza,
                'scattering_angle': scattering_angle,
                'cos_sza': cos_sza,
                'cos_vza': cos_vza,
                'sza': sza,
                'vza': vza,
                'raa': raa,
                'aod550': params.get('aod550', 0.2),
                'h2o': params.get('h2o', 2.0),
                'o3': params.get('o3', 0.3),
                'wavelength': self.band_wavelength,
                'band': params.get('band', 'unknown'),
                'is_extreme': params.get('is_extreme', False),
                **params
            }

        except Exception as e:
            error_msg = f"闭合循环模拟失败: {str(e)}"
            return {
                'success': False,
                'error': error_msg,
                'closed_loop_success': False,
                **params
            }


class RefactoredParallelSimulator:
    """重构的并行模拟器"""

    def __init__(self, config: ExperimentConfig = None, logger=None):
        self.config = config or ExperimentConfig
        self.logger = logger or setup_logger('RefactoredParallelSimulator')

    def generate_monte_carlo_dataset(self, bands: List[str] = None,
                                     n_samples_per_band: int = None,
                                     strategy: str = None) -> Dict[str, List[Dict]]:
        """
        生成蒙特卡洛数据集

        Args:
            bands: 波段列表，None表示所有波段
            n_samples_per_band: 每波段样本数
            strategy: 采样策略

        Returns:
            按波段分组的参数组合字典
        """
        if bands is None:
            bands = list(self.config.BANDS.keys())

        if n_samples_per_band is None:
            n_samples_per_band = self.config.MONTE_CARLO_CONFIG['total_samples_per_band']

        if strategy is None:
            strategy = self.config.MONTE_CARLO_CONFIG['sampling_strategy']

        self.logger.info(f"生成蒙特卡洛数据集:")
        self.logger.info(f"  波段: {bands}")
        self.logger.info(f"  每波段样本数: {n_samples_per_band}")
        self.logger.info(f"  采样策略: {strategy}")

        all_combinations = {}

        for band_id in bands:
            self.logger.info(f"为波段 {band_id} 生成参数...")

            # 采样参数
            param_df = monte_carlo_sample(
                param_ranges=self.config.PARAM_RANGES_CONTINUOUS,
                n_samples=n_samples_per_band,
                strategy=strategy,
                random_seed=self.config.RANDOM_SEED
            )

            # 添加波段信息
            wavelength = self.config.BANDS[band_id]['wavelength']
            param_df['wavelength'] = wavelength
            param_df['band'] = band_id

            # 添加固定参数
            param_df['atmos_profile'] = 'MidlatitudeSummer'
            param_df['aero_profile'] = 'Continental'
            param_df['target_altitude'] = self.config.SIXS_CONFIG['target_altitude']

            # 标记极端角度
            threshold = self.config.MONTE_CARLO_CONFIG['extreme_threshold']
            param_df['is_extreme'] = (
                    (param_df['sza'] > threshold) |
                    (param_df['vza'] > threshold)
            )

            # 转换为参数列表
            param_list = param_df.to_dict('records')
            all_combinations[band_id] = param_list

            # 统计信息
            extreme_count = param_df['is_extreme'].sum()
            extreme_ratio = extreme_count / len(param_df) * 100

            self.logger.info(f"  波段 {band_id}: 生成 {len(param_list)} 个参数组合")
            self.logger.info(f"  极端角度比例: {extreme_ratio:.1f}% ({extreme_count} 个)")

        total_combinations = sum(len(v) for v in all_combinations.values())
        self.logger.info(f"蒙特卡洛数据集总计: {total_combinations:,} 个参数组合")

        return all_combinations

    def generate_validation_grid(self, bands: List[str] = None) -> Dict[str, List[Dict]]:
        """
        生成验证网格（稀疏规则网格）
        """
        from itertools import product

        if bands is None:
            bands = list(self.config.BANDS.keys())

        grid_config = self.config.VALIDATION_GRID
        all_combinations = {}

        for band_id in bands:
            wavelength = self.config.BANDS[band_id]['wavelength']
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
                    'is_extreme': self.config.MONTE_CARLO_CONFIG['extreme_threshold']
                }

                combinations.append(params)

            all_combinations[band_id] = combinations
            self.logger.info(f"验证网格 - 波段 {band_id}: {len(combinations)} 个组合")

        return all_combinations

    def simulate_dataset(self, dataset: Dict[str, List[Dict]],
                         max_workers: int = None) -> Dict[str, pd.DataFrame]:
        """
        模拟数据集

        Args:
            dataset: 参数数据集
            max_workers: 最大工作进程数

        Returns:
            模拟结果字典（按波段）
        """
        if max_workers is None:
            import multiprocessing as mp
            physical_cores = mp.cpu_count() // 2
            max_workers = min(physical_cores,
                              self.config.PARALLEL_CONFIG.get('max_concurrent_6s', 4))

        self.logger.info(f"开始并行模拟，使用 {max_workers} 个工作进程")

        # 准备任务块
        task_blocks = []
        for band_id, param_list in dataset.items():
            chunk_size = self.config.PARALLEL_CONFIG.get('chunk_size', 1000)
            for i in range(0, len(param_list), chunk_size):
                chunk = param_list[i:i + chunk_size]
                task_blocks.append((band_id, chunk))

        self.logger.info(f"任务块数: {len(task_blocks)}")

        # 运行并行模拟
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_batch = {}
            for batch in task_blocks:
                future = executor.submit(self._worker_process_wrapper, batch)
                future_to_batch[future] = batch

            # 收集结果
            results_by_band = {}
            total_batches = len(task_blocks)

            with tqdm(total=total_batches, desc="并行模拟", unit="批次") as pbar:
                for future in as_completed(future_to_batch):
                    try:
                        band_id, batch_results = future.result(timeout=3600)

                        if band_id not in results_by_band:
                            results_by_band[band_id] = []

                        results_by_band[band_id].extend(batch_results)

                        # 统计成功样本
                        success_count = sum(1 for r in batch_results if r.get('success', False))
                        total_count = len(batch_results)

                        pbar.update(1)
                        pbar.set_postfix_str(f"{band_id}: {success_count}/{total_count}")

                    except Exception as e:
                        self.logger.error(f"任务批次处理失败: {e}")
                        traceback.print_exc()
                        pbar.update(1)

        elapsed_time = time.time() - start_time

        # 转换为DataFrame
        final_results = {}
        total_samples = 0
        total_success = 0

        for band_id, result_list in results_by_band.items():
            if result_list:
                df = pd.DataFrame(result_list)
                final_results[band_id] = df
                total_samples += len(df)

                # 统计成功样本
                if 'success' in df.columns:
                    success_count = df['success'].sum() if df['success'].dtype == bool else (df['success'] == 1).sum()
                    total_success += success_count

                self.logger.info(f"波段 {band_id}: 处理了 {len(df)} 个样本")

        # 计算统计信息
        success_rate = (total_success / total_samples) * 100 if total_samples > 0 else 0

        self.logger.info(f"并行模拟完成，耗时: {elapsed_time:.2f}秒")
        self.logger.info(f"总样本数: {total_samples} (成功率: {success_rate:.1f}%)")
        if elapsed_time > 0:
            self.logger.info(f"平均速度: {total_samples / elapsed_time:.2f} 样本/秒")

        return final_results

    def _worker_process_wrapper(self, task_batch: Tuple[str, List[Dict]]) -> Tuple[str, List[Dict]]:
        """
        工作进程包装器
        """
        import numpy as np
        import gc

        band_id, param_list = task_batch

        # 获取波段波长
        wavelength = self.config.BANDS[band_id]['wavelength']

        # 创建工作器
        worker = RefactoredSixSProcessWorker(wavelength)

        results = []

        for i, params in enumerate(param_list):
            try:
                # 运行模拟
                result = worker.run_closed_loop(params)
                results.append(result)

                # 定期清理内存
                if i % 100 == 0:
                    gc.collect()

            except Exception as e:
                error_result = {
                    'band': band_id,
                    'success': False,
                    'error': str(e),
                    'closed_loop_success': False,
                    **params
                }
                results.append(error_result)

        # 最终清理
        del worker
        gc.collect()

        return band_id, results