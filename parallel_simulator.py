# ==================== parallel_simulator.py ====================
"""
并行模拟器 - 使用多进程共享内存高效运行6S模拟
"""
import multiprocessing as mp
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
import gc
import traceback
from Py6S import *

from config import ExperimentConfig
from utils import setup_logger, calculate_airmass
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)


class SixSProcessWorker:
    """6S进程工作器 - 在独立进程中运行6S"""

    def __init__(self, band_wavelength: float):
        self.band_wavelength = band_wavelength
        self.sixs_instance = None
        self._init_sixs()

    def _init_sixs(self):
        """初始化6S实例"""
        try:
            self.sixs_instance = SixS()
            self.sixs_instance.wavelength = Wavelength(self.band_wavelength)
            self.sixs_instance.altitudes.set_target_custom_altitude(0.0)
            self.sixs_instance.altitudes.set_sensor_satellite_level()
            self.sixs_instance.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.2)
            self.sixs_instance.atmos_corr = AtmosCorr.NoAtmosCorr()
        except Exception as e:
            print(f"初始化6S失败: {e}")
            self.sixs_instance = None

    def run_simulation(self, params: Dict[str, float]) -> Dict[str, float]:
        """运行单个模拟"""
        if self.sixs_instance is None:
            return {'success': False, 'error': '6S实例未初始化'}

        try:
            # 配置几何参数
            self.sixs_instance.geometry = Geometry.User()
            self.sixs_instance.geometry.solar_z = params['sza']
            self.sixs_instance.geometry.solar_a = 0.0
            self.sixs_instance.geometry.view_z = params['vza']
            self.sixs_instance.geometry.view_a = 0.0

            # 配置大气参数
            if 'h2o' in params and 'o3' in params:
                water = params['h2o']
                ozone = params['o3']
                self.sixs_instance.atmos_profile = AtmosProfile.UserWaterAndOzone(water, ozone)
            else:
                self.sixs_instance.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeSummer)

            # 配置气溶胶
            self.sixs_instance.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
            if 'aod550' in params:
                self.sixs_instance.aot550 = params['aod550']

            # 配置地表反射率
            if 'rho_true' in params:
                self.sixs_instance.ground_reflectance = GroundReflectance.HomogeneousLambertian(params['rho_true'])

            # 运行正向模拟
            self.sixs_instance.run()
            rho_toa = self.sixs_instance.outputs.values['apparent_reflectance']

            # 运行反演
            self.sixs_instance.atmos_corr = AtmosCorr.AtmosCorrLambertianFromReflectance(rho_toa)
            inv_params = {k: v for k, v in params.items() if k != 'rho_true'}

            # 重新配置反演的参数
            self.sixs_instance.geometry = Geometry.User()
            self.sixs_instance.geometry.solar_z = inv_params.get('sza', params['sza'])
            self.sixs_instance.geometry.solar_a = 0.0
            self.sixs_instance.geometry.view_z = inv_params.get('vza', params['vza'])
            self.sixs_instance.geometry.view_a = 0.0

            if 'h2o' in inv_params and 'o3' in inv_params:
                self.sixs_instance.atmos_profile = AtmosProfile.UserWaterAndOzone(
                    inv_params['h2o'], inv_params['o3']
                )

            self.sixs_instance.run()
            rho_retrieved = self.sixs_instance.outputs.values['pixel_reflectance']

            error_abs = rho_retrieved - params.get('rho_true', 0.2)
            rho_true = params.get('rho_true', 0.2)
            error_rel = error_abs / rho_true if rho_true > 0 else np.nan

            return {
                'success': True,
                'rho_true': params.get('rho_true', 0.2),
                'rho_toa': rho_toa,
                'rho_retrieved': rho_retrieved,
                'error_absolute': error_abs,
                'error_relative': error_rel,
                'sza': params['sza'],
                'vza': params['vza'],
                'aod550': params.get('aod550', 0.2),
                'h2o': params.get('h2o', np.nan),
                'o3': params.get('o3', np.nan),
                'airmass_sza': calculate_airmass(params['sza']),
                'airmass_vza': calculate_airmass(params['vza'])
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                **params
            }

    def cleanup(self):
        """清理资源"""
        del self.sixs_instance
        gc.collect()


class SharedMemoryManager:
    """共享内存管理器"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.shared_buffers = {}

    def create_shared_array(self, name: str, shape: tuple, dtype: np.dtype):
        """创建共享数组"""
        size = int(np.prod(shape) * np.dtype(dtype).itemsize)

        try:
            # 尝试连接现有的共享内存
            shm = shared_memory.SharedMemory(name=name, create=False)
        except FileNotFoundError:
            # 创建新的共享内存
            shm = shared_memory.SharedMemory(name=name, create=True, size=size)

        array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        self.shared_buffers[name] = (shm, array)
        return array

    def get_shared_array(self, name: str):
        """获取共享数组"""
        if name in self.shared_buffers:
            return self.shared_buffers[name][1]
        return None

    def cleanup(self):
        """清理所有共享内存"""
        for name, (shm, _) in self.shared_buffers.items():
            shm.close()
            try:
                shm.unlink()
            except:
                pass


def worker_process(task_batch: Tuple[str, List[Dict]],
                   result_shm_name: str,
                   progress_shm_name: str,
                   config_dict: Dict) -> Dict:
    """
    工作进程函数 - 在独立进程中运行
    """
    import numpy as np

    band_id, param_list = task_batch
    band_config = config_dict['bands'][band_id]
    wavelength = band_config['wavelength']

    # 连接共享内存
    result_shm = shared_memory.SharedMemory(name=result_shm_name)
    progress_shm = shared_memory.SharedMemory(name=progress_shm_name)

    # 获取结果数组和进度数组
    n_tasks = len(param_list)
    result_shape = (n_tasks, 15)  # 15个输出字段
    result_array = np.ndarray(result_shape, dtype=np.float32, buffer=result_shm.buf)
    progress_array = np.ndarray((2,), dtype=np.int32, buffer=progress_shm.buf)  # [已完成, 失败数]

    # 初始化6S工作器
    worker = SixSProcessWorker(wavelength)

    completed = 0
    failed = 0

    for i, params in enumerate(param_list):
        try:
            result = worker.run_simulation(params)

            if result['success']:
                # 填充结果到共享数组
                result_array[i, 0] = result['sza']
                result_array[i, 1] = result['vza']
                result_array[i, 2] = result.get('aod550', 0.2)
                result_array[i, 3] = result.get('rho_true', 0.2)
                result_array[i, 4] = result.get('h2o', np.nan)
                result_array[i, 5] = result.get('o3', np.nan)
                result_array[i, 6] = result['rho_toa']
                result_array[i, 7] = result['rho_retrieved']
                result_array[i, 8] = result['error_absolute']
                result_array[i, 9] = result.get('error_relative', np.nan)
                result_array[i, 10] = result['airmass_sza']
                result_array[i, 11] = result['airmass_vza']
                result_array[i, 12] = 1.0  # success flag
                result_array[i, 13] = float(i)  # 原始索引
                result_array[i, 14] = float(ord(band_id[-1]))  # 波段ID编码
                completed += 1
            else:
                # 标记为失败
                result_array[i, 12] = 0.0
                result_array[i, 13] = float(i)
                result_array[i, 14] = float(ord(band_id[-1]))
                failed += 1

        except Exception as e:
            # 标记为失败
            result_array[i, 12] = 0.0
            result_array[i, 13] = float(i)
            result_array[i, 14] = float(ord(band_id[-1]))
            failed += 1

        # 更新进度
        progress_array[0] = completed
        progress_array[1] = failed

    # 清理
    worker.cleanup()
    result_shm.close()
    progress_shm.close()

    return {'band_id': band_id, 'completed': completed, 'failed': failed, 'total': n_tasks}


class ParallelBlockSimulator:
    """并行分块模拟器"""

    def __init__(self, config: ExperimentConfig, logger=None):
        self.config = config
        self.logger = logger or setup_logger('ParallelBlockSimulator')
        self.shm_manager = SharedMemoryManager(config)

        # 预计算每个波段的6S工作器
        self.workers = {}

    def simulate_blocks_parallel(self,
                                 task_blocks: List[Tuple[str, List[Dict]]],
                                 max_workers: int = None) -> Dict[str, pd.DataFrame]:
        """并行模拟多个任务块"""

        if max_workers is None:
            # 使用物理核心数的一半，但不超过配置的最大值
            physical_cores = mp.cpu_count() // 2
            max_workers = min(physical_cores,
                              self.config.PARALLEL_CONFIG.get('max_concurrent_6s', 4))

        self.logger.info(f"开始并行模拟，使用 {max_workers} 个工作进程")

        # 准备任务批次
        task_batches = self._prepare_task_batches(task_blocks, max_workers)

        # 创建共享内存用于结果和进度
        shm_names = {}
        for i, (band_id, param_list) in enumerate(task_batches):
            n_tasks = len(param_list)
            result_shm_name = f"results_{band_id}_{i}"
            progress_shm_name = f"progress_{band_id}_{i}"

            # 创建结果共享内存
            result_shape = (n_tasks, 15)
            self.shm_manager.create_shared_array(result_shm_name, result_shape, np.float32)

            # 创建进度共享内存
            self.shm_manager.create_shared_array(progress_shm_name, (2,), np.int32)

            shm_names[(band_id, i)] = (result_shm_name, progress_shm_name)

        # 准备配置字典（可序列化）
        config_dict = {
            'bands': self.config.BANDS,
            'param_ranges': self.config.PARAM_RANGES
        }

        # 使用进程池执行
        results = {}
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for i, (band_id, param_list) in enumerate(task_batches):
                result_shm_name, progress_shm_name = shm_names[(band_id, i)]

                future = executor.submit(
                    worker_process,
                    (band_id, param_list),
                    result_shm_name,
                    progress_shm_name,
                    config_dict
                )
                futures.append((band_id, future))

            # 收集结果
            for band_id, future in futures:
                try:
                    result = future.result(timeout=3600)  # 1小时超时
                    self.logger.info(f"波段 {band_id} 完成: {result}")

                    # 从共享内存获取数据
                    result_shm_name, progress_shm_name = shm_names.get((band_id, 0), (None, None))
                    if result_shm_name:
                        result_array = self.shm_manager.get_shared_array(result_shm_name)

                        # 转换为DataFrame
                        df = self._array_to_dataframe(result_array, band_id)
                        results[band_id] = df

                except Exception as e:
                    self.logger.error(f"波段 {band_id} 处理失败: {e}")

        elapsed_time = time.time() - start_time
        self.logger.info(f"并行模拟完成，耗时: {elapsed_time:.2f}秒")

        # 清理共享内存
        self.shm_manager.cleanup()

        return results

    def _prepare_task_batches(self,
                              task_blocks: List[Tuple[str, List[Dict]]],
                              max_workers: int) -> List[Tuple[str, List[Dict]]]:
        """准备任务批次，确保每个批次大小合适"""
        task_batches = []
        chunk_size = self.config.PARALLEL_CONFIG.get('chunk_size', 1000)

        for band_id, param_list in task_blocks:
            # 按chunk_size分割参数列表
            for i in range(0, len(param_list), chunk_size):
                chunk = param_list[i:i + chunk_size]
                task_batches.append((band_id, chunk))

        # 重新平衡批次，使每个工作进程负载均衡
        if len(task_batches) > max_workers * 2:
            # 合并小批次
            merged_batches = []
            current_batch = []
            current_size = 0

            for band_id, param_list in task_batches:
                if current_size + len(param_list) <= chunk_size * 2:
                    current_batch.extend([(band_id, params) for params in param_list])
                    current_size += len(param_list)
                else:
                    if current_batch:
                        # 按波段分组合并
                        band_groups = {}
                        for b_id, params in current_batch:
                            if b_id not in band_groups:
                                band_groups[b_id] = []
                            band_groups[b_id].append(params)

                        for b_id, params_list in band_groups.items():
                            merged_batches.append((b_id, params_list))

                    current_batch = [(band_id, params) for params in param_list]
                    current_size = len(param_list)

            if current_batch:
                band_groups = {}
                for b_id, params in current_batch:
                    if b_id not in band_groups:
                        band_groups[b_id] = []
                    band_groups[b_id].append(params)

                for b_id, params_list in band_groups.items():
                    merged_batches.append((b_id, params_list))

            task_batches = merged_batches

        return task_batches

    def _array_to_dataframe(self, array: np.ndarray, band_id: str) -> pd.DataFrame:
        """将numpy数组转换为DataFrame"""
        df = pd.DataFrame({
            'sza': array[:, 0],
            'vza': array[:, 1],
            'aod550': array[:, 2],
            'rho_true': array[:, 3],
            'h2o': array[:, 4],
            'o3': array[:, 5],
            'rho_toa': array[:, 6],
            'rho_retrieved': array[:, 7],
            'error_absolute': array[:, 8],
            'error_relative': array[:, 9],
            'airmass_sza': array[:, 10],
            'airmass_vza': array[:, 11],
            'success': array[:, 12] > 0.5,
            'original_idx': array[:, 13].astype(int),
            'band': band_id,
            'wavelength': self.config.BANDS[band_id]['wavelength']
        })

        # 过滤成功的结果
        df = df[df['success']].drop(columns=['success', 'original_idx'])

        return df

    def generate_all_param_combinations(self) -> Dict[str, List[Dict]]:
        """生成所有参数组合"""
        from itertools import product

        param_grids = {}
        param_configs = self.config.PARAM_RANGES

        # 生成参数网格
        for param_name, config in param_configs.items():
            if isinstance(config, dict):
                values = np.arange(
                    config['min'],
                    config['max'] + config['step'] / 2,
                    config['step']
                )
            else:
                values = np.array(config)
            param_grids[param_name] = values

        # 生成所有波段的所有参数组合
        all_combinations = {}

        for band_id in self.config.BANDS.keys():
            param_names = list(param_grids.keys())
            value_lists = [param_grids[name] for name in param_names]

            combinations = []
            for values in product(*value_lists):
                params = dict(zip(param_names, values))
                combinations.append(params)

            all_combinations[band_id] = combinations
            self.logger.info(f"波段 {band_id}: 生成 {len(combinations)} 个参数组合")

        total_combinations = sum(len(c) for c in all_combinations.values())
        self.logger.info(f"总共生成 {total_combinations} 个参数组合")

        return all_combinations