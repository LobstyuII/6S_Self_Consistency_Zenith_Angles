# ==================== ahi_6s_inversion_10min.py ====================
"""
10分钟分辨率的AHI TOA到LSR 6S反演
使用10分钟分辨率的AHI数据和MERRA2辅助数据
"""

import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import pickle
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import warnings
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import gc
import json
import hashlib

# Py6S for 6S simulations
from Py6S import *
import sixs_inversion

warnings.filterwarnings('ignore')


# ==================== Configuration ====================
class InversionConfig:
    """6S反演配置类 - 10分钟版本"""

    # Data paths
    DATA_PATHS = {
        "h8l1_10min": "D:/H8_data/H8L1_10min/",
        "h8l2_10min": "D:/H8_data/H8L2ARP_10min/",
        "merra2_10min": "D:/H8_data/MERRA2_10min/combined/",
        "lucc": "D:/H8_data/LC_2015_2024.nc",
        "luts": "D:/H8_data/LUTs.nc",
        "mod_red": "D:/H8_Data/MODIS_Red_nadir/",
        "mod_nir": "D:/H8_Data/MODIS_NIR_nadir/",
        "output": "D:/H8_data/LSR_10min/"
    }

    # Band configuration - 改为分别处理
    BAND_CONFIG = {
        '01': {'wavelength': 0.47, 'name': 'band1'},
        '02': {'wavelength': 0.51, 'name': 'band2'},
        '03': {'wavelength': 0.64, 'name': 'band3', 'modis_band': 'Red'},
        '04': {'wavelength': 0.86, 'name': 'band4', 'modis_band': 'NIR'},
        '05': {'wavelength': 1.60, 'name': 'band5'},
        '06': {'wavelength': 2.30, 'name': 'band6'}
    }

    # 6S Configuration
    SIXS_CONFIG = {
        'atmos_profile': 'FromLatitudeAndDate',  # 改为自动选择
        'aero_profile': 'Continental',
        'target_altitude': 0.0,
        'default_aod': 0.1,
        'default_h2o': 2.0,
        'default_o3': 0.3
    }

    # Parallel processing
    PARALLEL_CONFIG = {
        'max_workers': 8,
        'chunk_size': 1000,
        'max_tasks_per_worker': 10000
    }

    # ========== 新增：站点选择配置 ==========
    STATION_SELECTION = {
        'by_vza_group': True,           # 是否启用VZA分组选择
        'top_per_vza_group': 5,          # 每组选择的站点数
        'vza_bin_size': 10               # VZA分组间隔（度）
    }


# ==================== MERRA2数据加载和单位转换 ====================
class MERRA2DataLoader:
    """MERRA2数据加载器，处理单位转换"""

    @staticmethod
    def load_merra2_data(file_path, station_indices):
        """加载MERRA2数据并应用单位转换"""
        data = {
            'AOD550': np.full(len(station_indices), np.nan),
            'water': np.full(len(station_indices), np.nan),
            'ozone': np.full(len(station_indices), np.nan),
            'is_default': np.zeros(len(station_indices), dtype=bool)  # 标记是否使用默认值
        }

        if not os.path.exists(file_path):
            print(f"Warning: MERRA2 file not found: {file_path}")
            # 使用默认值并标记
            data['AOD550'][:] = InversionConfig.SIXS_CONFIG['default_aod']
            data['water'][:] = InversionConfig.SIXS_CONFIG['default_h2o']
            data['ozone'][:] = InversionConfig.SIXS_CONFIG['default_o3']
            data['is_default'][:] = True
            return data

        try:
            with nc.Dataset(file_path) as ds:
                # 加载AOD550
                if 'AOD550' in ds.variables:
                    aod_data = ds.variables['AOD550'][:][station_indices]
                    if isinstance(aod_data, np.ma.MaskedArray):
                        aod_data = aod_data.filled(np.nan)
                    # 替换无效值
                    aod_data[aod_data == -9999.0] = np.nan
                    data['AOD550'] = aod_data

                # 加载水汽和臭氧，应用单位转换（与6S+BRDF代码一致）
                if 'H2O' in ds.variables:
                    h2o_data = ds.variables['H2O'][:][station_indices]
                    if isinstance(h2o_data, np.ma.MaskedArray):
                        h2o_data = h2o_data.filled(np.nan)
                    h2o_data[h2o_data == -9999.0] = np.nan
                    # 单位转换: kg/m² -> g/cm² (乘以0.1)
                    data['water'] = h2o_data * 0.1

                if 'O3' in ds.variables:
                    o3_data = ds.variables['O3'][:][station_indices]
                    if isinstance(o3_data, np.ma.MaskedArray):
                        o3_data = o3_data.filled(np.nan)
                    o3_data[o3_data == -9999.0] = np.nan
                    # 单位转换: Dobson -> cm-atm (乘以0.001)
                    data['ozone'] = o3_data * 0.001

                # 标记缺失值
                nan_mask = np.isnan(data['AOD550']) | np.isnan(data['water']) | np.isnan(data['ozone'])
                if np.any(nan_mask):
                    # 使用默认值填充缺失值
                    data['AOD550'][nan_mask] = InversionConfig.SIXS_CONFIG['default_aod']
                    data['water'][nan_mask] = InversionConfig.SIXS_CONFIG['default_h2o']
                    data['ozone'][nan_mask] = InversionConfig.SIXS_CONFIG['default_o3']
                    data['is_default'][nan_mask] = True

        except Exception as e:
            print(f"Error reading MERRA2 file {file_path}: {e}")
            # 使用默认值
            data['AOD550'][:] = InversionConfig.SIXS_CONFIG['default_aod']
            data['water'][:] = InversionConfig.SIXS_CONFIG['default_h2o']
            data['ozone'][:] = InversionConfig.SIXS_CONFIG['default_o3']
            data['is_default'][:] = True

        return data


# ==================== 6S Worker ====================
class SixSInversionWorker:
    """6S反演工作器，现在调用缓存函数"""

    def __init__(self, band_wavelength: float):
        self.band_wavelength = band_wavelength

    def run_inversion(self, params: dict):
        """
        运行反演：TOA -> LSR，使用缓存函数
        """
        try:
            # 准备传入缓存模块的参数
            inv_params = {
                'sza': params['sza'],
                'vza': params['vza'],
                'raa': params.get('raa', 0.0),
                'phi': params.get('phi', 0.0),
                'aod550': params.get('aod550', 0.1),
                'h2o': params.get('h2o', 2.0),
                'o3': params.get('o3', 0.3),
                'rho_toa': params['rho_toa'],
                'wavelength': self.band_wavelength,
                'atmos_profile': params.get('atmos_profile', 'MidlatitudeSummer'),
                'aero_profile': params.get('aero_profile', 'Continental'),
                'target_altitude': params.get('target_altitude', 0.0),
            }

            # 调用缓存函数
            result = sixs_inversion.run_inversion_cached(inv_params)

            if result['success']:
                return {
                    'success': True,
                    'rho_toa': params['rho_toa'],
                    'rho_lsr': result['rho_lsr'],
                    'rho_retrieved': result['rho_lsr'],
                    'sza': params['sza'],
                    'vza': params['vza'],
                    'raa': params.get('raa', 0.0),
                    'aod550': params.get('aod550', 0.1),
                    'h2o': params.get('h2o', 2.0),
                    'o3': params.get('o3', 0.3),
                    'wavelength': self.band_wavelength,
                    'band': params.get('band', '01'),
                    'station': params.get('station', 'unknown'),
                    'datetime_utc': params.get('datetime_utc'),
                    'datetime_bj': params.get('datetime_bj'),
                    'original_index': params.get('original_index', -1),
                    'atmos_profile': params.get('atmos_profile', 'Unknown'),
                    'is_default_merra2': params.get('is_default', False),
                    'merra2_warning': params.get('merra2_warning', ''),
                    'from_cache': result.get('from_cache', False),
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', '6S inversion failed'),
                    **params
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                **params
            }


# ==================== Data Loader ====================
class AHIInversionDataLoader10min:
    """AHI数据加载器 - 10分钟版本"""

    def __init__(self, config):
        self.config = config
        self.station_coords = None
        self.station_lc_info = None

    def load_stations(self):
        """加载所有站点坐标"""
        print("Loading all stations...")
        with nc.Dataset(self.config.DATA_PATHS['luts']) as ds:
            stations_raw = ds.variables['Station'][:]
            stations = []
            for s in stations_raw:
                if isinstance(s, bytes):
                    s = s.decode('utf-8')
                stations.append(str(s).strip())

            lats = ds.variables['Lat'][:]
            lons = ds.variables['Lon'][:]

            if isinstance(lats, np.ma.MaskedArray):
                lats = lats.filled(np.nan)
            if isinstance(lons, np.ma.MaskedArray):
                lons = lons.filled(np.nan)

            df = pd.DataFrame({
                'station': stations,
                'lat': lats,
                'lon': lons
            }).dropna(subset=['lat', 'lon']).drop_duplicates('station')

        self.station_coords = df
        print(f"Loaded {len(df)} stations")
        return df

    def load_lc_info(self):
        """加载站点LC信息"""
        if self.station_lc_info is not None:
            return self.station_lc_info

        station_lc_info = {}
        try:
            with nc.Dataset(self.config.DATA_PATHS['lucc']) as ds:
                stations = ds.variables['Station'][:]
                station_names = [station.strip() for station in stations]
                lc_data = ds.variables['LC_type1'][:]
                recent_lc = lc_data[1, :]  # 使用第二年的数据

                for i, station in enumerate(station_names):
                    if recent_lc[i] != -9999:
                        station_lc_info[station] = int(recent_lc[i])
        except Exception as e:
            print(f"Error reading LC data: {e}")

        self.station_lc_info = station_lc_info
        return station_lc_info

    def filter_stations_by_lc(self, stations, lc_types):
        """按LC类型过滤站点"""
        if not lc_types:
            return stations

        lc_info = self.load_lc_info()
        filtered_stations = []

        for station in stations:
            if station in lc_info and lc_info[station] in lc_types:
                filtered_stations.append(station)

        print(f"Filtered stations: {len(filtered_stations)}/{len(stations)} with LC types {lc_types}")
        return filtered_stations

    def load_10min_ahi_data(self, date_str, hour, minute, stations, band_id=None):
        """加载10分钟分辨率的AHI数据 - 修改为支持按波段加载"""
        import os

        time_str = f"{hour:02d}{minute:02d}"

        # L1 TOA数据文件路径
        l1_file_path = os.path.join(
            self.config.DATA_PATHS['h8l1_10min'],
            date_str[:4],
            date_str[4:6],
            f"H8_{date_str}_{time_str}.nc"
        )

        # L2 ARP数据文件路径
        l2_file_path = os.path.join(
            self.config.DATA_PATHS['h8l2_10min'],
            date_str[:4],
            date_str[4:6],
            f"H8L2ARP_{date_str}_{time_str}.nc"
        )

        if not os.path.exists(l1_file_path) or not os.path.exists(l2_file_path):
            return pd.DataFrame()

        try:
            # 加载L1数据
            with nc.Dataset(l1_file_path) as l1_ds:
                file_stations_raw = l1_ds.variables['Station'][:]
                file_stations = []
                for s in file_stations_raw:
                    if isinstance(s, bytes):
                        s = s.decode('utf-8')
                    file_stations.append(str(s).strip())

                station_indices = []
                valid_stations = []
                for station in stations:
                    if station in file_stations:
                        idx = file_stations.index(station)
                        station_indices.append(idx)
                        valid_stations.append(station)

                if not station_indices:
                    return pd.DataFrame()

                data_dict = {'station': valid_stations}

                # 角度数据
                angle_cols = ['SOZ', 'SAZ', 'SOA', 'SAA']
                for col in angle_cols:
                    if col in l1_ds.variables:
                        var_data = l1_ds.variables[col][:][station_indices]
                        if isinstance(var_data, np.ma.MaskedArray):
                            var_data = var_data.filled(np.nan)
                        data_dict[col] = var_data

                # 如果指定了波段，只加载该波段数据
                if band_id is not None:
                    col = f'Albedo_{band_id}'
                    if col in l1_ds.variables:
                        var_data = l1_ds.variables[col][:][station_indices]
                        if isinstance(var_data, np.ma.MaskedArray):
                            var_data = var_data.filled(np.nan)
                        # 注意：此处直接使用 var_data，不再除以100
                        data_dict[f'TOA_Albedo_{band_id}'] = var_data
                else:
                    # 加载所有波段数据
                    for band in ['01', '02', '03', '04', '05', '06']:
                        col = f'Albedo_{band}'
                        if col in l1_ds.variables:
                            var_data = l1_ds.variables[col][:][station_indices]
                            if isinstance(var_data, np.ma.MaskedArray):
                                var_data = var_data.filled(np.nan)
                            # 注意：此处直接使用 var_data，不再除以100
                            data_dict[f'TOA_Albedo_{band}'] = var_data

            # 加载L2数据 (可用性)
            with nc.Dataset(l2_file_path) as l2_ds:
                l2_stations_raw = l2_ds.variables['Station'][:]
                l2_stations = []
                for s in l2_stations_raw:
                    if isinstance(s, bytes):
                        s = s.decode('utf-8')
                    l2_stations.append(str(s).strip())

                # 对齐站点索引
                l2_indices = []
                for station in valid_stations:
                    if station in l2_stations:
                        l2_indices.append(l2_stations.index(station))

                if len(l2_indices) != len(valid_stations):
                    print(f"Warning: L1 and L2 station mismatch")
                    return pd.DataFrame()

                # 提取可用性数据
                if 'Data_Availability' in l2_ds.variables:
                    avail_data = l2_ds.variables['Data_Availability'][:][l2_indices]
                    if isinstance(avail_data, np.ma.MaskedArray):
                        avail_data = avail_data.filled(np.nan)
                    data_dict['Data_Availability'] = avail_data

                if 'Cloud_Flag' in l2_ds.variables:
                    cloud_data = l2_ds.variables['Cloud_Flag'][:][l2_indices]
                    if isinstance(cloud_data, np.ma.MaskedArray):
                        cloud_data = cloud_data.filled(np.nan)
                    data_dict['Cloud_Flag'] = cloud_data

            df = pd.DataFrame(data_dict)

            if df.empty:
                return df

            # 添加时间
            dt_utc = datetime.strptime(date_str, "%Y%m%d") + timedelta(hours=hour, minutes=minute)
            df['datetime_utc'] = dt_utc
            df['datetime_bj'] = dt_utc + timedelta(hours=8)  # UTC+8

            # 加载MERRA2 10分钟辅助数据（使用新的MERRA2加载器）
            merra2_file = os.path.join(
                self.config.DATA_PATHS['merra2_10min'],
                date_str[:4],
                date_str[4:6],
                f"MERRA2_combined_{date_str}_{time_str}.nc"
            )

            # 使用新的MERRA2数据加载器
            merra2_loader = MERRA2DataLoader()
            merra2_data = merra2_loader.load_merra2_data(merra2_file, station_indices)

            # 添加MERRA2数据到DataFrame
            for key in ['AOD550', 'water', 'ozone', 'is_default']:
                if key in merra2_data:
                    series = pd.Series(merra2_data[key], index=valid_stations)
                    df[key] = df['station'].map(series)

            # 记录使用默认值的比例
            if 'is_default' in df.columns:
                default_ratio = df['is_default'].mean()
                if default_ratio > 0:
                    print(
                        f"Warning: {default_ratio * 100:.1f}% of stations using default MERRA2 values for {date_str} {time_str}")

            return df

        except Exception as e:
            print(f"Error loading 10min AHI data for {date_str} {time_str}: {e}")
            return pd.DataFrame()

    def filter_valid_data(self, df):
        """过滤有效数据：SOZ < 90度，SAZ < 90度，可用性为0"""
        if df.empty:
            return df

        # 复制以避免SettingWithCopyWarning
        df_filtered = df.copy()

        # 应用过滤条件
        mask = (
                (df_filtered['SOZ'].notna()) & (df_filtered['SOZ'] < 90) &
                (df_filtered['SAZ'].notna()) & (df_filtered['SAZ'] < 90)
        )

        # 如果有可用性数据，添加到过滤条件
        if 'Data_Availability' in df_filtered.columns:
            mask = mask & (df_filtered['Data_Availability'] == 0)

        if 'Cloud_Flag' in df_filtered.columns:
            mask = mask & (df_filtered['Cloud_Flag'] == 0)

        return df_filtered[mask].reset_index(drop=True)

    def load_ahi_data_for_band(self, stations, start_date, end_date, band_id):
        """加载指定波段的数据（10分钟分辨率）- 按波段分别处理"""
        print(f"Loading AHI data for band {band_id} from {start_date} to {end_date} for {len(stations)} stations")

        # 生成日期范围
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")

        all_data = []

        # 生成所有时间点（10分钟间隔）
        time_points = []
        current_dt = start_dt
        while current_dt <= end_dt:
            for hour in range(24):
                for minute in [0, 10, 20, 30, 40, 50]:
                    time_points.append((current_dt, hour, minute))
            current_dt += timedelta(days=1)

        for date_obj, hour, minute in tqdm(time_points, desc=f"Loading 10min data for band {band_id}"):
            date_str = date_obj.strftime("%Y%m%d")

            df_10min = self.load_10min_ahi_data(date_str, hour, minute, stations, band_id)
            if not df_10min.empty:
                # 过滤有效数据
                df_filtered = self.filter_valid_data(df_10min)
                if not df_filtered.empty:
                    all_data.append(df_filtered)

        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            print(f"AHI data for band {band_id} loaded: {len(final_df)} records from {len(stations)} stations")

            # 统计信息
            if not final_df.empty:
                print(f"  Time range: {final_df['datetime_bj'].min()} to {final_df['datetime_bj'].max()}")
                print(f"  Average records per station: {len(final_df) / len(stations):.1f}")
                # 统计使用默认MERRA2数据的比例
                if 'is_default' in final_df.columns:
                    default_ratio = final_df['is_default'].mean()
                    print(f"  Stations using default MERRA2 data: {default_ratio * 100:.1f}%")

            return final_df

        return pd.DataFrame()


# ==================== Cached Inversion Processor ====================
class CachedInversionProcessor10min:
    """带有缓存功能的并行反演处理器 - 10分钟版本"""

    def __init__(self, config):
        self.config = config
        self.data_loader = AHIInversionDataLoader10min(config)

    def generate_cache_key(self, start_date, end_date, stations, band_id=None, params_hash=None):
        """生成缓存键 - 添加波段信息"""
        # 使用日期范围和站点数量生成缓存键
        station_hash = hashlib.md5('_'.join(sorted(stations)).encode()).hexdigest()[:8]
        date_range = f"{start_date}_{end_date}"

        if band_id:
            band_prefix = f"band{band_id}_"
        else:
            band_prefix = ""

        if params_hash:
            cache_key = f"lsr_cache_10min_{band_prefix}{date_range}_{station_hash}_{params_hash}"
        else:
            cache_key = f"lsr_cache_10min_{band_prefix}{date_range}_{station_hash}"

        return cache_key

    def check_cache_exists(self, cache_key):
        """检查缓存是否存在"""
        cache_dir = Path("./lsr_cache_10min")
        cache_dir.mkdir(exist_ok=True)

        cache_files = [
            cache_dir / f"{cache_key}.parquet",
            cache_dir / f"{cache_key}_stats.json"
        ]

        # 检查所有缓存文件是否存在
        all_exist = all(f.exists() for f in cache_files)

        if all_exist:
            print(f"Cache found: {cache_key}")
            return True, cache_files
        else:
            print(f"Cache not found: {cache_key}")
            return False, cache_files

    def save_to_cache(self, results_df, station_stats, cache_key, band_id=None):
        """保存到缓存"""
        cache_dir = Path("./lsr_cache_10min")
        cache_dir.mkdir(exist_ok=True)

        # 保存LSR数据
        lsr_path = cache_dir / f"{cache_key}.parquet"
        results_df.to_parquet(lsr_path)
        print(f"LSR data saved to cache: {lsr_path}")

        # 保存站点统计信息
        stats_path = cache_dir / f"{cache_key}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(station_stats, f, indent=2, default=str)
        print(f"Station stats saved to cache: {stats_path}")

        # 保存缓存元数据
        meta_path = cache_dir / f"{cache_key}_meta.json"
        meta_data = {
            'cache_key': cache_key,
            'band_id': band_id,
            'created_at': datetime.now().isoformat(),
            'total_records': len(results_df),
            'successful_inversions': int(results_df['success'].sum() if results_df['success'].dtype == bool else (
                    results_df['success'] == 1).sum()),
            'default_merra2_ratio': float(results_df.get('is_default_merra2', pd.Series(
                [False])).mean()) if 'is_default_merra2' in results_df.columns else 0.0
        }
        with open(meta_path, 'w') as f:
            json.dump(meta_data, f, indent=2)

        return lsr_path, stats_path

    def load_from_cache(self, cache_key):
        """从缓存加载"""
        cache_dir = Path("./lsr_cache_10min")

        lsr_path = cache_dir / f"{cache_key}.parquet"
        stats_path = cache_dir / f"{cache_key}_stats.json"

        if not lsr_path.exists() or not stats_path.exists():
            print(f"Cache files not found for key: {cache_key}")
            return None, None

        try:
            # 加载LSR数据
            results_df = pd.read_parquet(lsr_path)
            print(f"LSR data loaded from cache: {len(results_df)} records")

            # 加载站点统计信息
            with open(stats_path, 'r') as f:
                station_stats = json.load(f)
            print(f"Station stats loaded from cache: {len(station_stats)} stations")

            # 加载缓存元数据
            meta_path = cache_dir / f"{cache_key}_meta.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta_data = json.load(f)
                print(f"Cache created at: {meta_data.get('created_at', 'unknown')}")
                print(f"Band ID: {meta_data.get('band_id', 'unknown')}")
                if 'default_merra2_ratio' in meta_data:
                    print(f"Default MERRA2 ratio: {meta_data['default_merra2_ratio'] * 100:.1f}%")

            return results_df, station_stats

        except Exception as e:
            print(f"Error loading from cache: {e}")
            return None, None

    # ========== 新增：按VZA分组选择代表性站点 ==========
    def select_stations_by_vza_group(self, ahi_data, station_stats, top_per_group=5, vza_bin_size=10):
        """
        按VZA每vza_bin_size度分组，每组选择有效数据最多的top_per_group个站点
        ahi_data: DataFrame，必须包含'station'和'SAZ'列
        station_stats: 字典，包含每个站点的'records'等信息
        """
        # 计算每个站点的平均VZA（也可用中位数）
        station_vza = ahi_data.groupby('station')['SAZ'].mean().to_dict()

        # 构建DataFrame
        df_stations = []
        for station, stats in station_stats.items():
            if station in station_vza:
                vza = station_vza[station]
                records = stats.get('records', 0)
                df_stations.append({'station': station, 'vza': vza, 'records': records})
        df = pd.DataFrame(df_stations)
        if df.empty:
            return []

        # 定义VZA区间
        bins = np.arange(0, 91, vza_bin_size)
        labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
        df['vza_bin'] = pd.cut(df['vza'], bins=bins, labels=labels, right=False)

        selected_stations = []
        for bin_label in labels:
            df_bin = df[df['vza_bin'] == bin_label]
            if df_bin.empty:
                continue
            # 按记录数降序排序，取前top_per_group
            df_bin_sorted = df_bin.sort_values('records', ascending=False)
            selected = df_bin_sorted.head(top_per_group)['station'].tolist()
            selected_stations.extend(selected)
            print(f"  VZA bin {bin_label}: {len(df_bin)} stations, selected {len(selected)}")

        return selected_stations

    def prepare_inversion_params_for_band(self, ahi_data, band_id):
        """准备反演参数 - 针对单个波段"""
        print(f"Preparing inversion parameters for band {band_id}...")
        all_params = []

        band_wavelength = self.config.BAND_CONFIG[band_id]['wavelength']
        band_data = ahi_data.copy()

        # 筛选有TOA数据的记录
        toa_col = f'TOA_Albedo_{band_id}'
        if toa_col not in band_data.columns:
            print(f"Warning: No TOA data for band {band_id}")
            return []

        band_data = band_data[band_data[toa_col].notna()].copy()

        if band_data.empty:
            print(f"Warning: No valid data for band {band_id}")
            return []

        print(f"  Band {band_id}: {len(band_data)} records")

        # 计算RAA
        band_data['raa'] = np.abs(band_data['SOA'] - band_data['SAA'])

        # 获取站点坐标信息
        station_coords = self.data_loader.station_coords

        # 为每条记录创建参数
        for idx, row in band_data.iterrows():
            # 获取站点坐标
            station = row['station']
            lat, lon = 0.0, 0.0
            if station_coords is not None and not station_coords.empty:
                station_info = station_coords[station_coords['station'] == station]
                if not station_info.empty:
                    lat = station_info['lat'].values[0]
                    lon = station_info['lon'].values[0]

            params = {
                'sza': float(row['SOZ']),
                'vza': float(row['SAZ']),
                'raa': float(row['raa']),
                'phi': float(row['SOA']),  # 太阳方位角
                'rho_toa': float(row[toa_col]),
                'aod550': float(row['AOD550']),
                'h2o': float(row['water']),
                'o3': float(row['ozone']),
                'atmos_profile': self.config.SIXS_CONFIG['atmos_profile'],
                'aero_profile': self.config.SIXS_CONFIG['aero_profile'],
                'target_altitude': self.config.SIXS_CONFIG['target_altitude'],
                'wavelength': band_wavelength,
                'band': band_id,
                'station': row['station'],
                'datetime_utc': row['datetime_utc'],
                'datetime_bj': row['datetime_bj'],
                'lat': lat,
                'lon': lon,
                'date': row['datetime_bj'],  # 用于大气廓线选择
                'is_default': bool(row.get('is_default', False)) if 'is_default' in row else False,
                'original_index': idx
            }
            all_params.append(params)

        print(f"Total inversion tasks for band {band_id}: {len(all_params)}")
        return all_params

    def calculate_station_availability(self, ahi_data, band_id=None):
        """计算站点数据可用性（10分钟版本）- 支持按波段"""
        print(
            f"\nCalculating station data availability (10min resolution){f' for band {band_id}' if band_id else ''}...")

        station_stats = {}
        all_stations = ahi_data['station'].unique()

        for station in tqdm(all_stations, desc="Analyzing stations"):
            station_data = ahi_data[ahi_data['station'] == station]

            if not station_data.empty:
                # 计算数据可用性
                if band_id:
                    # 按波段计算
                    toa_col = f'TOA_Albedo_{band_id}'
                    band_available = 0
                    if toa_col in station_data.columns:
                        band_available = station_data[toa_col].notna().sum()
                    total_available = band_available
                else:
                    # 计算所有波段
                    band_available = {}
                    total_available = 0
                    for bid in self.config.BAND_CONFIG.keys():
                        toa_col = f'TOA_Albedo_{bid}'
                        if toa_col in station_data.columns:
                            band_count = station_data[toa_col].notna().sum()
                            band_available[bid] = band_count
                            total_available += band_count

                # 10分钟分辨率：每天有144个时间点 (24小时 * 6)
                unique_dates = station_data['datetime_bj'].dt.date.nunique()

                if band_id:
                    # 单个波段：每个时间点1个观测值
                    total_possible = unique_dates * 144
                else:
                    # 所有6个波段：每个时间点6个观测值
                    total_possible = unique_dates * 144 * 6

                availability = total_available / total_possible if total_possible > 0 else 0

                station_stats[station] = {
                    'availability': availability,
                    'records': len(station_data),
                    'unique_dates': unique_dates,
                    'total_possible': total_possible,
                    'total_available': total_available
                }

                if band_id:
                    station_stats[station][f'band{band_id}_available'] = band_available
                elif not band_id and 'band_available' in locals():
                    station_stats[station]['bands_available'] = band_available

        return station_stats

    def filter_stations_by_availability(self, station_stats, min_availability=0.10, max_stations=None):
        """按可用性过滤站点"""
        filtered_stations = []

        for station, stats in station_stats.items():
            if stats['availability'] >= min_availability:
                filtered_stations.append((station, stats['availability']))

        # 按可用性排序
        filtered_stations.sort(key=lambda x: x[1], reverse=True)

        # 如果指定了最大站点数
        if max_stations is not None:
            filtered_stations = filtered_stations[:max_stations]

        filtered_station_names = [station for station, _ in filtered_stations]

        print(f"\nStation filtering (threshold: {min_availability * 100:.0f}%):")
        print(f"  Total stations available: {len(station_stats)}")
        print(f"  Stations meeting criteria: {len(filtered_stations)}")

        if len(filtered_stations) > 0:
            print(f"\nTop 5 filtered stations:")
            for i, (station, avail) in enumerate(filtered_stations[:5]):
                stats = station_stats[station]
                print(f"  {i + 1}. {station:20s}: {avail * 100:5.1f}% "
                      f"({stats['records']} records, {stats['unique_dates']} days)")

        return filtered_station_names

    def run_parallel_inversion_for_band(self, inversion_params, band_id, output_path=None):
        """运行并行反演 - 针对单个波段"""
        max_workers = self.config.PARALLEL_CONFIG['max_workers']
        chunk_size = self.config.PARALLEL_CONFIG['chunk_size']

        print(f"\nStarting parallel inversion for band {band_id} with {max_workers} workers, chunk size: {chunk_size}")

        band_wavelength = self.config.BAND_CONFIG[band_id]['wavelength']

        if not inversion_params:
            print(f"No inversion parameters for band {band_id}")
            return pd.DataFrame()

        # 分块处理
        chunks = [inversion_params[i:i + chunk_size] for i in range(0, len(inversion_params), chunk_size)]
        total_chunks = len(chunks)

        all_results = []
        start_time = time.time()

        # 统计使用默认MERRA2数据的比例
        default_count = sum(1 for p in inversion_params if p.get('is_default', False))
        default_ratio = default_count / len(inversion_params) if inversion_params else 0
        if default_ratio > 0:
            print(
                f"Warning: {default_ratio * 100:.1f}% of inversion tasks using default MERRA2 data for band {band_id}")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for chunk_idx, chunk in enumerate(chunks):
                future = executor.submit(self._process_chunk, chunk, band_wavelength, chunk_idx)
                futures.append(future)

            # 收集结果
            for future in tqdm(as_completed(futures), total=total_chunks, desc=f"Band {band_id}"):
                try:
                    chunk_results = future.result(timeout=3600)
                    all_results.extend(chunk_results)
                except Exception as e:
                    print(f"Error processing chunk: {e}")

        elapsed_time = time.time() - start_time

        # 转换为DataFrame
        results_df = pd.DataFrame(all_results)

        # 统计成功率和使用默认数据的比例
        if len(results_df) > 0:
            success_count = results_df['success'].sum() if results_df['success'].dtype == bool else (
                    results_df['success'] == 1).sum()
            success_rate = (success_count / len(results_df)) * 100 if len(results_df) > 0 else 0

            # 统计使用默认MERRA2数据的比例
            if 'is_default_merra2' in results_df.columns:
                default_merra2_ratio = results_df['is_default_merra2'].mean()
            else:
                default_merra2_ratio = 0.0

            print(f"\nInversion for band {band_id} completed in {elapsed_time:.2f} seconds")
            print(f"Total tasks: {len(results_df)}")
            print(f"Successful inversions: {success_count} ({success_rate:.1f}%)")
            print(f"Default MERRA2 data used: {default_merra2_ratio * 100:.1f}%")
            print(f"Speed: {len(results_df) / elapsed_time:.2f} tasks/second")

            # 输出大气廓线使用统计
            if 'atmos_profile' in results_df.columns:
                profile_counts = results_df['atmos_profile'].value_counts()
                print(f"Atmospheric profiles used:")
                for profile, count in profile_counts.items():
                    print(f"  {profile}: {count} ({count / len(results_df) * 100:.1f}%)")

        # 保存结果
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存为Parquet格式
            results_df.to_parquet(output_path)
            print(f"\nResults saved to: {output_path}")

            # 保存统计信息
            stats_path = output_path.with_suffix('.inversion_stats.json')
            stats_summary = {
                'band_id': band_id,
                'wavelength': self.config.BAND_CONFIG[band_id]['wavelength'],
                'total_records': len(results_df),
                'successful_inversions': int(success_count) if 'success_count' in locals() else 0,
                'success_rate': float(success_rate) if 'success_rate' in locals() else 0,
                'default_merra2_ratio': float(default_merra2_ratio) if 'default_merra2_ratio' in locals() else 0,
                'processing_time_seconds': float(elapsed_time),
                'speed_tasks_per_second': float(len(results_df) / elapsed_time) if elapsed_time > 0 else 0,
                'resolution': '10min'
            }
            with open(stats_path, 'w') as f:
                json.dump(stats_summary, f, indent=2)
            print(f"Inversion statistics saved to: {stats_path}")

        return results_df

    def _process_chunk(self, chunk_params, band_wavelength, chunk_idx):
        """处理一个数据块"""
        worker = SixSInversionWorker(band_wavelength)
        chunk_results = []

        for params in chunk_params:
            result = worker.run_inversion(params)
            chunk_results.append(result)

        # 定期清理内存
        if chunk_idx % 10 == 0:
            gc.collect()

        return chunk_results

    def process_single_band(self, band_id, start_date, end_date, output_path=None,
                            min_availability=0.10, max_stations=None,
                            lc_types=None, use_cache=True):
        """处理单个波段的完整流程 - 10分钟版本"""
        print("=" * 70)
        print(f"AHI TOA TO LSR 6S INVERSION - BAND {band_id} ({self.config.BAND_CONFIG[band_id]['wavelength']}µm)")
        print("=" * 70)

        # 1. 加载所有站点
        print(f"\nStep 1: Loading all stations for band {band_id}...")
        stations_df = self.data_loader.load_stations()
        all_stations = stations_df['station'].tolist()
        print(f"Total stations available: {len(all_stations)}")

        # 2. 按LC类型过滤
        if lc_types:
            print(f"\nFiltering stations by LC types: {lc_types}")
            all_stations = self.data_loader.filter_stations_by_lc(all_stations, lc_types)

        # 3. 生成缓存键
        cache_key = self.generate_cache_key(start_date, end_date, all_stations, band_id)

        # 4. 检查缓存
        if use_cache:
            cache_exists, cache_files = self.check_cache_exists(cache_key)
            if cache_exists:
                print(f"\nCache found for band {band_id}: {cache_key}")
                user_input = input(f"Do you want to use cached data for band {band_id}? (y/n): ").strip().lower()
                if user_input == 'y':
                    # 从缓存加载
                    results_df, station_stats = self.load_from_cache(cache_key)
                    if results_df is not None:
                        print(f"\nUsing cached data for band {band_id}. Skipping 6S inversion.")
                        return results_df, station_stats, cache_key
                else:
                    print(f"Proceeding with new 6S inversion for band {band_id}...")
            else:
                print(f"\nNo cache found for band {band_id}: {cache_key}")

        # 5. 加载AHI TOA数据 (10分钟分辨率) - 仅加载当前波段
        print(f"\nStep 2: Loading AHI TOA data for band {band_id} ({start_date} to {end_date}) - 10min resolution...")
        ahi_data = self.data_loader.load_ahi_data_for_band(
            all_stations, start_date, end_date, band_id
        )

        if ahi_data.empty:
            print(f"Error: No AHI data loaded for band {band_id}")
            return None, None, None

        print(f"AHI data for band {band_id} loaded: {len(ahi_data)} records")

        # 6. 计算站点可用性（10分钟分辨率）
        print(f"\nStep 3: Calculating station data availability for band {band_id} (10min resolution)...")
        station_stats = self.calculate_station_availability(ahi_data, band_id)

        # 7. 按可用性过滤站点（初步过滤）
        print(f"\nStep 4: Filtering stations by availability for band {band_id}...")
        filtered_stations = self.filter_stations_by_availability(
            station_stats, min_availability, max_stations
        )

        # ========== 新增：按VZA分组选择代表性站点 ==========
        station_selection_config = getattr(self.config, 'STATION_SELECTION', {})
        if station_selection_config.get('by_vza_group', False):
            print(f"\nStep 4b: Selecting top stations per VZA group...")
            top_per_group = station_selection_config.get('top_per_vza_group', 5)
            vza_bin_size = station_selection_config.get('vza_bin_size', 10)
            # 获取可用性过滤后的AHI数据（用于计算VZA）
            ahi_data_avail = ahi_data[ahi_data['station'].isin(filtered_stations)].copy()
            if not ahi_data_avail.empty:
                # 仅传递在filtered_stations范围内的station_stats子集
                filtered_stats = {s: station_stats[s] for s in filtered_stations if s in station_stats}
                selected_by_vza = self.select_stations_by_vza_group(
                    ahi_data_avail,
                    filtered_stats,
                    top_per_group=top_per_group,
                    vza_bin_size=vza_bin_size
                )
                if selected_by_vza:
                    filtered_stations = selected_by_vza
                    print(f"    Selected {len(filtered_stations)} stations after VZA grouping.")
                else:
                    print("    VZA grouping returned no stations, using previous filtered list.")
            else:
                print("    No data available for VZA grouping, using previous filtered list.")
        # ====================================================

        if not filtered_stations:
            print(f"Warning: No stations meet the availability criteria for band {band_id}")
            # 使用可用性最高的站点
            sorted_stations = sorted(station_stats.items(), key=lambda x: x[1]['availability'], reverse=True)
            filtered_stations = [s[0] for s in sorted_stations[:min(100, len(sorted_stations))]]
            print(f"Using top {len(filtered_stations)} stations by availability for band {band_id}")

        # 8. 只处理过滤后的站点数据
        print(f"\nStep 5: Processing filtered data for band {band_id} ({len(filtered_stations)} stations)...")
        ahi_data_filtered = ahi_data[ahi_data['station'].isin(filtered_stations)].copy()

        if ahi_data_filtered.empty:
            print(f"Error: No data after filtering for band {band_id}")
            return None, None, None

        print(f"Filtered AHI data for band {band_id}: {len(ahi_data_filtered)} records")

        # 9. 准备反演参数
        print(f"\nStep 6: Preparing inversion parameters for band {band_id}...")
        inversion_params = self.prepare_inversion_params_for_band(ahi_data_filtered, band_id)

        if not inversion_params:
            print(f"Error: No inversion parameters prepared for band {band_id}")
            return None, None, None

        # 10. 运行并行反演
        print(f"\nStep 7: Running parallel 6S inversion for band {band_id}...")

        # 设置输出路径
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"./ahi_lsr_10min_band{band_id}_{start_date}_{end_date}_{timestamp}.parquet"

        results_df = self.run_parallel_inversion_for_band(inversion_params, band_id, output_path)

        if results_df is None or results_df.empty:
            print(f"Error: Inversion failed for band {band_id}")
            return None, None, None

        # 11. 保存到缓存
        print(f"\nStep 8: Saving results to cache for band {band_id}...")
        cache_lsr_path, cache_stats_path = self.save_to_cache(results_df, station_stats, cache_key, band_id)

        print(f"\n" + "=" * 70)
        print(f"PROCESSING FOR BAND {band_id} COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Cache key: {cache_key}")
        print(f"Output file: {output_path}")
        print(f"Cached LSR data: {cache_lsr_path}")
        print(f"Stations processed: {len(filtered_stations)}")
        print(f"Total LSR records: {len(results_df)}")
        print(f"Time resolution: 10 minutes")
        print(f"Band: {band_id} ({self.config.BAND_CONFIG[band_id]['wavelength']}µm)")
        if 'is_default_merra2' in results_df.columns:
            default_ratio = results_df['is_default_merra2'].mean()
            print(f"Default MERRA2 data used: {default_ratio * 100:.1f}%")

        return results_df, station_stats, cache_key

    def process_all_bands(self, start_date, end_date, output_dir=None,
                          min_availability=0.10, max_stations=None,
                          lc_types=None, use_cache=True, bands_to_process=None):
        """处理所有波段的完整流程 - 分别处理每个波段"""
        print("=" * 70)
        print("AHI TOA TO LSR 6S INVERSION - ALL BANDS (SEPARATE PROCESSING)")
        print("=" * 70)

        # 确定要处理的波段
        if bands_to_process is None:
            bands_to_process = list(self.config.BAND_CONFIG.keys())

        all_results = {}

        for band_id in bands_to_process:
            print(f"\n\n{'=' * 60}")
            print(f"PROCESSING BAND {band_id}")
            print(f"{'=' * 60}")

            # 设置波段特定的输出目录
            if output_dir:
                band_output_dir = Path(output_dir) / f"band{band_id}"
                band_output_dir.mkdir(parents=True, exist_ok=True)
                band_output_path = band_output_dir / f"lsr_band{band_id}_{start_date}_{end_date}.parquet"
            else:
                band_output_path = None

            # 处理单个波段
            results_df, station_stats, cache_key = self.process_single_band(
                band_id=band_id,
                start_date=start_date,
                end_date=end_date,
                output_path=band_output_path,
                min_availability=min_availability,
                max_stations=max_stations,
                lc_types=lc_types,
                use_cache=use_cache
            )

            if results_df is not None:
                all_results[band_id] = {
                    'data': results_df,
                    'stats': station_stats,
                    'cache_key': cache_key
                }

                # 保存波段数据
                if band_output_path:
                    results_df.to_parquet(band_output_path)
                    print(f"Band {band_id} data saved to: {band_output_path}")

            # 清理内存
            gc.collect()
            print(f"\nCompleted processing for band {band_id}")

        # 生成整体统计报告
        self._generate_overall_report(all_results, start_date, end_date, output_dir)

        return all_results

    def _generate_overall_report(self, all_results, start_date, end_date, output_dir):
        """生成整体处理报告"""
        if not all_results:
            return

        print("\n" + "=" * 70)
        print("OVERALL PROCESSING SUMMARY")
        print("=" * 70)

        total_records = 0
        total_success = 0
        total_default_merra2 = 0
        band_stats = []

        for band_id, result_info in all_results.items():
            results_df = result_info['data']
            if results_df is None or results_df.empty:
                continue

            band_records = len(results_df)
            band_success = results_df['success'].sum() if results_df['success'].dtype == bool else (
                        results_df['success'] == 1).sum()
            band_success_rate = band_success / band_records * 100 if band_records > 0 else 0

            if 'is_default_merra2' in results_df.columns:
                band_default = results_df['is_default_merra2'].sum()
                band_default_ratio = band_default / band_records * 100 if band_records > 0 else 0
            else:
                band_default = 0
                band_default_ratio = 0

            wavelength = self.config.BAND_CONFIG[band_id]['wavelength']

            band_stats.append({
                'band': band_id,
                'wavelength': wavelength,
                'records': band_records,
                'success': band_success,
                'success_rate': band_success_rate,
                'default_merra2': band_default,
                'default_ratio': band_default_ratio
            })

            total_records += band_records
            total_success += band_success
            total_default_merra2 += band_default

        # 打印波段统计
        print(f"\nBand-by-band statistics:")
        print(f"{'Band':<6} {'Wavelength':<12} {'Records':<10} {'Success':<10} {'Success%':<10} {'Default%':<10}")
        print("-" * 68)

        for stats in band_stats:
            print(f"{stats['band']:<6} {stats['wavelength']:<12.3f} {stats['records']:<10} "
                  f"{stats['success']:<10} {stats['success_rate']:<9.1f} {stats['default_ratio']:<9.1f}")

        # 整体统计
        overall_success_rate = total_success / total_records * 100 if total_records > 0 else 0
        overall_default_ratio = total_default_merra2 / total_records * 100 if total_records > 0 else 0

        print(f"\nOverall statistics:")
        print(f"  Total records: {total_records}")
        print(f"  Successful inversions: {total_success} ({overall_success_rate:.1f}%)")
        print(f"  Default MERRA2 data used: {total_default_merra2} ({overall_default_ratio:.1f}%)")
        print(f"  Bands processed: {len(band_stats)}/{len(self.config.BAND_CONFIG)}")

        # 保存报告
        if output_dir:
            report_path = Path(output_dir) / f"overall_report_{start_date}_{end_date}.txt"
            with open(report_path, 'w') as f:
                f.write("=" * 70 + "\n")
                f.write("AHI TOA TO LSR 6S INVERSION - OVERALL REPORT\n")
                f.write("=" * 70 + "\n")
                f.write(f"Time period: {start_date} to {end_date}\n")
                f.write(f"Time resolution: 10 minutes\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("Band-by-band statistics:\n")
                f.write(
                    f"{'Band':<6} {'Wavelength':<12} {'Records':<10} {'Success':<10} {'Success%':<10} {'Default%':<10}\n")
                f.write("-" * 68 + "\n")

                for stats in band_stats:
                    f.write(f"{stats['band']:<6} {stats['wavelength']:<12.3f} {stats['records']:<10} "
                            f"{stats['success']:<10} {stats['success_rate']:<9.1f} {stats['default_ratio']:<9.1f}\n")

                f.write(f"\nOverall statistics:\n")
                f.write(f"  Total records: {total_records}\n")
                f.write(f"  Successful inversions: {total_success} ({overall_success_rate:.1f}%)\n")
                f.write(f"  Default MERRA2 data used: {total_default_merra2} ({overall_default_ratio:.1f}%)\n")
                f.write(f"  Bands processed: {len(band_stats)}/{len(self.config.BAND_CONFIG)}\n")

            print(f"\nOverall report saved to: {report_path}")


# ==================== Main Function ====================
def main():
    parser = argparse.ArgumentParser(
        description='AHI TOA to LSR 6S Inversion - 10min Resolution (Separate Band Processing)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 生成2020年5月1-7日所有6个波段的LSR数据，分别处理
  python ahi_6s_inversion_10min.py --start_date 20200501 --end_date 20200507 --all_bands

  # 只处理波段3和4
  python ahi_6s_inversion_10min.py --start_date 20200501 --end_date 20200507 --bands 03 04

  # 处理单个波段（波段1）
  python ahi_6s_inversion_10min.py --start_date 20200501 --end_date 20200507 --band 01

  # 不使用缓存，强制重新生成
  python ahi_6s_inversion_10min.py --start_date 20200501 --end_date 20200507 --band 03 --no_cache

  # 只处理可用性>10%的站点（10分钟分辨率）
  python ahi_6s_inversion_10min.py --min_availability 0.10 --start_date 20200501 --end_date 20200507 --band 04

  # 只处理前200个高可用性站点
  python ahi_6s_inversion_10min.py --max_stations 200 --start_date 20200501 --end_date 20200507 --band 03

  # 使用更多工作进程加速处理
  python ahi_6s_inversion_10min.py --max_workers 12 --chunk_size 2000 --start_date 20200501 --end_date 20200507 --band 04

  # 只处理农田和城市站点（LC类型12,13）
  python ahi_6s_inversion_10min.py --lc_types 12 13 --start_date 20200501 --end_date 20200507 --band 03
        """
    )

    parser.add_argument('--start_date', type=str, default='20200501',
                        help='Start date YYYYMMDD (default: 20200501)')
    parser.add_argument('--end_date', type=str, default='20200507',
                        help='End date YYYYMMDD (default: 20200507)')
    parser.add_argument('--output_dir', type=str, default='./lsr_results_10min',
                        help='Output directory (default: ./lsr_results_10min)')
    parser.add_argument('--max_workers', type=int, default=8,
                        help='Maximum number of parallel workers (default: 8)')
    parser.add_argument('--lc_types', type=int, nargs='+', default=None,
                        help='Filter stations by LC types (e.g., 12 13 for Croplands and Urban)')
    parser.add_argument('--min_availability', type=float, default=0.10,
                        help='Minimum data availability ratio to include station (default: 0.10 for 10min)')
    parser.add_argument('--max_stations', type=int, default=None,
                        help='Maximum number of stations to process (default: all)')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Chunk size for parallel processing (default: 1000)')
    parser.add_argument('--no_cache', action='store_true',
                        help='Do not use cache, always run 6S inversion')

    # 波段选择参数
    parser.add_argument('--band', type=str, default=None,
                        help='Process single band (e.g., 01, 02, 03, 04, 05, 06)')
    parser.add_argument('--bands', type=str, nargs='+', default=None,
                        help='Process multiple bands (e.g., 03 04)')
    parser.add_argument('--all_bands', action='store_true',
                        help='Process all 6 bands (separately)')

    args = parser.parse_args()

    print("=" * 70)
    print("AHI TOA TO LSR 6S INVERSION - 10MIN RESOLUTION (SEPARATE BAND PROCESSING)")
    print("=" * 70)
    print(f"Time period: {args.start_date} to {args.end_date}")
    print(f"Time resolution: 10 minutes")
    print(f"Max workers: {args.max_workers}")
    print(f"LC types filter: {args.lc_types if args.lc_types else 'All'}")
    print(f"Minimum availability: {args.min_availability * 100:.0f}% (adjusted for 10min resolution)")
    print(f"Max stations: {args.max_stations if args.max_stations else 'All'}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Use cache: {not args.no_cache}")

    # 确定要处理的波段
    bands_to_process = []
    if args.band:
        bands_to_process = [args.band]
    elif args.bands:
        bands_to_process = args.bands
    elif args.all_bands:
        bands_to_process = ['01', '02', '03', '04', '05', '06']
    else:
        # 默认处理波段3和4
        bands_to_process = ['03', '04']

    print(f"Bands to process: {', '.join(bands_to_process)}")
    print("=" * 70)

    # 配置
    config = InversionConfig()
    config.PARALLEL_CONFIG['max_workers'] = args.max_workers
    config.PARALLEL_CONFIG['chunk_size'] = args.chunk_size

    # 创建处理器
    processor = CachedInversionProcessor10min(config)

    # 运行处理流程
    if len(bands_to_process) == 1:
        # 处理单个波段
        results_df, station_stats, cache_key = processor.process_single_band(
            band_id=bands_to_process[0],
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=None,  # 让函数自动生成路径
            min_availability=args.min_availability,
            max_stations=args.max_stations,
            lc_types=args.lc_types,
            use_cache=not args.no_cache
        )
    else:
        # 处理多个波段
        all_results = processor.process_all_bands(
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            min_availability=args.min_availability,
            max_stations=args.max_stations,
            lc_types=args.lc_types,
            use_cache=not args.no_cache,
            bands_to_process=bands_to_process
        )

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETED!")
    print("=" * 70)
    print(f"Output directory: {args.output_dir}")
    print(f"Bands processed: {', '.join(bands_to_process)}")
    print(f"To validate the results, run:")
    print(
        f"python enhanced_validation_10min.py --lsr_data {args.output_dir}/band*/lsr_band*_{args.start_date}_{args.end_date}.parquet --output_dir ./validation_results_10min")

    return 0


if __name__ == "__main__":
    main()