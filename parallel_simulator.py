# ==================== parallel_simulator.py ====================
"""
并行模拟器 - 使用多进程共享内存高效运行6S模拟
参考6S+BRDF.py成功模式进行重构
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
import multiprocessing

# 设置多进程启动方法
multiprocessing.set_start_method('spawn', force=True)


class SixSProcessWorker:
    """6S进程工作器 - 参考6S+BRDF.py设计，每个任务创建新实例"""

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
        创建并配置6S实例 - 参考6S+BRDF.py中每个任务创建新实例
        """
        try:
            s = SixS()
            s.wavelength = Wavelength(self.band_wavelength)

            # 配置大气廓线
            atmos_key = params.get('atmos_profile', 'MidlatitudeSummer')

            # 检查是否有自定义水汽和臭氧
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

            # 设置气溶胶光学厚度
            if 'aod550' in params:
                aod_val = params['aod550']
                if not np.isnan(aod_val) and aod_val >= 0:
                    s.aot550 = aod_val
                else:
                    s.aot550 = 0.2  # 默认值
            else:
                s.aot550 = 0.2  # 默认值

            # 配置几何参数
            s.geometry = Geometry.User()
            s.geometry.solar_z = params['sza']
            s.geometry.solar_a = 0.0  # 固定太阳方位角
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

    def run_forward_simulation(self, params: Dict[str, float]) -> Dict[str, Any]:
        """运行正向模拟"""
        try:
            # 创建新的6S实例
            s = self.create_sixs_instance(params)
            if s is None:
                return {
                    'success': False,
                    'error': '创建6S实例失败',
                    **params
                }

            # 配置地表反射率
            rho_true = params.get('rho_true', 0.2)
            s.ground_reflectance = GroundReflectance.HomogeneousLambertian(rho_true)
            s.atmos_corr = AtmosCorr.NoAtmosCorr()

            # 运行正向模拟
            s.run()
            rho_toa = s.outputs.values['apparent_reflectance']

            # 计算空气质量
            sza = params['sza']
            vza = params['vza']
            cos_sza = np.cos(np.radians(sza)) if sza < 88 else 0.0349
            cos_vza = np.cos(np.radians(vza)) if vza < 88 else 0.0349
            airmass_sza = 1.0 / cos_sza if cos_sza > 0.01 else 1.0 / 0.01
            airmass_vza = 1.0 / cos_vza if cos_vza > 0.01 else 1.0 / 0.01

            # 清理实例
            del s
            gc.collect()

            return {
                'success': True,
                'rho_true': rho_true,
                'rho_toa': rho_toa,
                'airmass_sza': airmass_sza,
                'airmass_vza': airmass_vza,
                'total_airmass': airmass_sza + airmass_vza
            }

        except Exception as e:
            error_msg = f"正向模拟失败: {str(e)}"
            return {
                'success': False,
                'error': error_msg,
                **params
            }

    def run_inversion(self, rho_toa: float, params: Dict[str, float]) -> Dict[str, Any]:
        """运行反演模拟"""
        try:
            # 创建新的6S实例
            s = self.create_sixs_instance(params)
            if s is None:
                return {
                    'success': False,
                    'error': '创建6S实例失败',
                    **params
                }

            # 配置反演
            s.atmos_corr = AtmosCorr.AtmosCorrLambertianFromReflectance(rho_toa)

            # 运行反演模拟
            s.run()
            rho_retrieved = s.outputs.values['pixel_reflectance']

            # 清理实例
            del s
            gc.collect()

            return {
                'success': True,
                'rho_toa_input': rho_toa,
                'rho_retrieved': rho_retrieved
            }

        except Exception as e:
            error_msg = f"反演模拟失败: {str(e)}"
            return {
                'success': False,
                'error': error_msg,
                **params
            }

    def run_closed_loop(self, params: Dict[str, float]) -> Dict[str, Any]:
        """
        运行闭合循环模拟 - 参考6S+BRDF.py模式
        """
        try:
            # 1. 正向模拟
            forward_result = self.run_forward_simulation(params)

            if not forward_result.get('success', False):
                return {
                    **forward_result,
                    'closed_loop_success': False,
                    **params
                }

            # 2. 反演模拟（移除rho_true参数）
            inv_params = {k: v for k, v in params.items() if k != 'rho_true'}
            inversion_result = self.run_inversion(forward_result['rho_toa'], inv_params)

            if not inversion_result.get('success', False):
                return {
                    **forward_result,
                    **inversion_result,
                    'closed_loop_success': False,
                    **params
                }

            # 3. 计算误差
            error_abs = inversion_result['rho_retrieved'] - params.get('rho_true', 0.2)
            rho_true = params.get('rho_true', 0.2)
            error_rel = error_abs / rho_true if rho_true > 0 else np.nan

            return {
                **forward_result,
                **inversion_result,
                'error_absolute': error_abs,
                'error_relative': error_rel,
                'closed_loop_success': True,
                'success': True,
                'sza': params['sza'],
                'vza': params['vza'],
                'aod550': params.get('aod550', 0.2),
                'h2o': params.get('h2o', np.nan),
                'o3': params.get('o3', np.nan),
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


def worker_process_improved(task_batch: Tuple[str, List[Dict]], config_dict: Dict) -> Tuple[str, List[Dict]]:
    """
    改进的工作进程函数 - 参考6S+BRDF.py设计
    """
    import numpy as np
    import gc

    band_id, param_list = task_batch
    band_config = config_dict['bands'][band_id]
    wavelength = band_config['wavelength']

    # 创建工作器
    worker = SixSProcessWorker(wavelength)

    results = []

    for i, params in enumerate(param_list):
        try:
            # 运行模拟
            result = worker.run_closed_loop(params)

            # 添加波段信息
            result['band'] = band_id
            result['wavelength'] = wavelength

            results.append(result)

            # 定期清理内存
            if i % 100 == 0:
                gc.collect()

        except Exception as e:
            # 记录失败的任务
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


class ParallelBlockSimulatorImproved:
    """改进的并行分块模拟器"""

    def __init__(self, config: ExperimentConfig, logger=None):
        self.config = config
        self.logger = logger or setup_logger('ParallelBlockSimulatorImproved')

    def simulate_blocks_parallel_improved(self,
                                          task_blocks: List[Tuple[str, List[Dict]]],
                                          max_workers: int = None) -> Dict[str, pd.DataFrame]:
        """
        改进的并行模拟方法
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        if max_workers is None:
            import multiprocessing as mp
            physical_cores = mp.cpu_count() // 2
            max_workers = min(physical_cores,
                              self.config.PARALLEL_CONFIG.get('max_concurrent_6s', 4))

        self.logger.info(f"开始并行模拟，使用 {max_workers} 个工作进程")

        # 准备配置字典（可序列化）
        config_dict = {
            'bands': self.config.BANDS,
            'param_ranges': self.config.PARAM_RANGES
        }

        # 使用进程池执行
        results_by_band = {}
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_batch = {
                executor.submit(worker_process_improved, batch, config_dict): batch
                for batch in task_blocks
            }

            # 收集结果
            total_batches = len(task_blocks)
            with tqdm(total=total_batches, desc="并行模拟", unit="批次") as pbar:
                for future in as_completed(future_to_batch):
                    try:
                        band_id, batch_results = future.result(timeout=3600)

                        # 合并结果
                        if band_id not in results_by_band:
                            results_by_band[band_id] = []

                        results_by_band[band_id].extend(batch_results)

                        # 统计成功和失败的样本
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
        total_params = sum(len(params) for _, params in task_blocks)
        success_rate = (total_success / total_params) * 100 if total_params > 0 else 0

        self.logger.info(f"并行模拟完成，耗时: {elapsed_time:.2f}秒")
        self.logger.info(f"总样本数: {total_samples} (成功率: {success_rate:.1f}%)")
        if elapsed_time > 0:
            self.logger.info(f"平均速度: {total_samples / elapsed_time:.2f} 样本/秒")

        return final_results

    def generate_all_param_combinations_consistent(self, mode: str = 'full') -> Dict[str, List[Dict]]:
        """
        生成与单线程一致的所有参数组合
        解决单线程432000 vs 多线程518400的差异
        """
        from itertools import product

        param_grids = {}
        param_configs = self.config.PARAM_RANGES

        # 生成参数网格 - 与单线程的get_param_combinations保持一致
        if mode == 'full':
            # 使用与单线程相同的参数范围
            for param_name, config in param_configs.items():
                if isinstance(config, dict):
                    # 动态生成值 - 确保范围一致
                    min_val = config['min']
                    max_val = config['max']
                    step = config['step']

                    # 生成值，确保不超出范围
                    values = np.arange(min_val, max_val + step / 2, step)
                    # 确保最大值不超过配置的max
                    values = values[values <= max_val]

                    param_grids[param_name] = values
                    self.logger.debug(f"参数 {param_name}: {len(values)} 个值, 范围: {min(values)}-{max(values)}")
                else:
                    # 使用预设值
                    values = np.array(config)
                    param_grids[param_name] = values
                    self.logger.debug(f"参数 {param_name}: {len(values)} 个预设值")
        elif mode == 'paper_figures':
            # 使用论文图表模式的参数
            mode_config = self.config.SIMULATION_MODES.get('paper_figures', {})
            for param_name in ['sza', 'vza', 'aod550', 'rho_true', 'h2o', 'o3']:
                if param_name in mode_config:
                    param_grids[param_name] = np.array(mode_config[param_name])
                elif param_name in param_configs:
                    config = param_configs[param_name]
                    if isinstance(config, dict):
                        values = np.arange(config['min'], config['max'] + config['step'] / 2, config['step'])
                        param_grids[param_name] = values
                    else:
                        param_grids[param_name] = np.array(config)

        # 检查关键参数的组合数
        if 'sza' in param_grids and 'vza' in param_grids:
            sza_count = len(param_grids['sza'])
            vza_count = len(param_grids['vza'])
            self.logger.info(f"SZA: {sza_count}个值, VZA: {vza_count}个值")

        # 生成所有波段的所有参数组合
        all_combinations = {}

        for band_id in self.config.BANDS.keys():
            # 选择要组合的参数
            param_names = ['sza', 'vza', 'rho_true', 'aod550', 'h2o', 'o3']
            value_lists = [param_grids.get(name, []) for name in param_names]

            # 检查是否有空列表
            if any(len(lst) == 0 for lst in value_lists):
                self.logger.warning(f"波段 {band_id}: 某些参数值为空")
                continue

            combinations = []

            # 使用product生成所有组合
            for values in product(*value_lists):
                params = dict(zip(param_names, values))
                # 添加固定参数
                params.update({
                    'atmos_profile': 'MidlatitudeSummer',
                    'aero_profile': 'Continental',
                    'target_altitude': self.config.SIXS_CONFIG['target_altitude']
                })
                combinations.append(params)

            all_combinations[band_id] = combinations

            # 计算理论组合数
            theoretical_count = 1
            for values in value_lists:
                theoretical_count *= len(values)

            self.logger.info(f"波段 {band_id}: 生成 {len(combinations)} 个参数组合 (理论值: {theoretical_count})")

            # 调试输出前几个组合
            if len(combinations) > 0 and self.logger.level == 'DEBUG':
                for i in range(min(3, len(combinations))):
                    self.logger.debug(f"  组合 {i + 1}: {combinations[i]}")

        return all_combinations


# 保留原始类用于向后兼容
class ParallelBlockSimulator:
    """原始并行分块模拟器（向后兼容）"""

    def __init__(self, config: ExperimentConfig, logger=None):
        self.config = config
        self.logger = logger or setup_logger('ParallelBlockSimulator')
        # 创建改进版本的实例
        self.improved_simulator = ParallelBlockSimulatorImproved(config, logger)

    def simulate_blocks_parallel(self, task_blocks, max_workers=None):
        """向后兼容的模拟方法"""
        return self.improved_simulator.simulate_blocks_parallel_improved(task_blocks, max_workers)

    def generate_all_param_combinations(self):
        """向后兼容的参数生成方法"""
        return self.improved_simulator.generate_all_param_combinations_consistent('full')