# ==================== ahi_6s_inversion_cached.py ====================
"""
带有缓存功能的AHI TOA到LSR 6S反演 - 完整版本
可以跳过已存在的LSR数据，直接从缓存读取
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

warnings.filterwarnings('ignore')


# ==================== Configuration ====================
class InversionConfig:
    """6S反演配置类"""

    # Data paths
    DATA_PATHS = {
        "hourly_sozSR": "D:/H8_data/Hourly_sozSR_Angles/",
        "merra2_slv": "D:/H8_data/MERRA2_slv/",
        "merra2_aer": "D:/H8_data/MERRA2_aer/",
        "lucc": "D:/H8_data/LC_2015_2024.nc",
        "luts": "D:/H8_data/LUTs.nc",
        "mod_red": "D:/H8_Data/MODIS_Red_nadir/",
        "mod_nir": "D:/H8_Data/MODIS_NIR_nadir/",
        "h8sr": "D:/H8_data/NBAR_Noatmoscorr/"
    }

    # Band configuration
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
        'atmos_profile': 'MidlatitudeSummer',
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


# ==================== 6S Worker ====================
class SixSInversionWorker:
    """6S反演工作器"""

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

    def create_sixs_instance(self, params: dict):
        """创建6S实例"""
        try:
            s = SixS()
            s.wavelength = Wavelength(self.band_wavelength)

            # 配置大气廓线
            atmos_key = params.get('atmos_profile', 'MidlatitudeSummer')

            # 使用连续的水汽和臭氧值
            water = params.get('h2o', 2.0)
            ozone = params.get('o3', 0.3)
            if not np.isnan(water) and not np.isnan(ozone) and water > 0 and ozone > 0:
                s.atmos_profile = AtmosProfile.UserWaterAndOzone(water, ozone)
            else:
                s.atmos_profile = self.atmos_profile_map.get(atmos_key)

            # 配置气溶胶
            aero_key = params.get('aero_profile', 'Continental')
            s.aero_profile = self.aero_profile_map.get(aero_key)

            # 设置气溶胶光学厚度
            aod_val = params.get('aod550', 0.1)
            if not np.isnan(aod_val) and aod_val >= 0:
                s.aot550 = aod_val
            else:
                s.aot550 = 0.1

            # 配置几何参数
            s.geometry = Geometry.User()
            s.geometry.solar_z = params['sza']
            s.geometry.solar_a = params.get('phi', 0.0)
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

    def run_inversion(self, params: dict):
        """运行6S反演：TOA -> LSR"""
        try:
            # 创建6S实例
            s = self.create_sixs_instance(params)
            if s is None:
                return {
                    'success': False,
                    'error': '创建6S实例失败',
                    **params
                }

            # TOA反射率
            rho_toa = params.get('rho_toa', 0.2)

            # 运行大气校正
            s.atmos_corr = AtmosCorr.AtmosCorrLambertianFromReflectance(rho_toa)
            s.run()

            # 获取反演的地表反射率
            rho_lsr = s.outputs.values['pixel_reflectance']
            rho_retrieved = rho_lsr  # 添加这个别名用于兼容性

            # 清理实例
            del s
            gc.collect()

            return {
                'success': True,
                'rho_toa': rho_toa,
                'rho_lsr': rho_lsr,
                'rho_retrieved': rho_retrieved,  # 添加这个字段
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
                'original_index': params.get('original_index', -1)
            }

        except Exception as e:
            error_msg = f"6S反演失败: {str(e)}"
            return {
                'success': False,
                'error': error_msg,
                **params
            }


# ==================== Data Loader ====================
class AHIInversionDataLoader:
    """AHI数据加载器"""

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

    def load_hourly_data(self, date_str, hour, stations):
        """加载单小时数据"""
        import os

        hour_str = f"{hour * 100:04d}"
        file_path = os.path.join(
            self.config.DATA_PATHS['hourly_sozSR'],
            date_str[:4],
            date_str[4:6],
            f"H8_hourly_sozSR_angles_{date_str}_{hour_str}.nc"
        )

        if not os.path.exists(file_path):
            return pd.DataFrame()

        try:
            with nc.Dataset(file_path) as ds:
                file_stations_raw = ds.variables['Station'][:]
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
                    if col in ds.variables:
                        var_data = ds.variables[col][:][station_indices]
                        if isinstance(var_data, np.ma.MaskedArray):
                            var_data = var_data.filled(np.nan)
                        data_dict[col] = var_data

                # 波段数据 (1-6)
                for band in ['01', '02', '03', '04', '05', '06']:
                    col = f'Albedo_{band}'
                    if col in ds.variables:
                        var_data = ds.variables[col][:][station_indices]
                        if isinstance(var_data, np.ma.MaskedArray):
                            var_data = var_data.filled(np.nan)
                        data_dict[f'TOA_Albedo_{band}'] = var_data / 100.0

                df = pd.DataFrame(data_dict)

                if df.empty:
                    return df

                # 添加时间
                dt_utc = datetime.strptime(date_str, "%Y%m%d") + timedelta(hours=hour)
                df['datetime_utc'] = dt_utc
                df['datetime_bj'] = dt_utc  # 转换为北京时间 (UTC+8)

                # 添加默认大气参数（可以从MERRA2数据加载，这里用默认值）
                df['AOD550'] = self.config.SIXS_CONFIG['default_aod']
                df['water'] = self.config.SIXS_CONFIG['default_h2o']
                df['ozone'] = self.config.SIXS_CONFIG['default_o3']

                return df

        except Exception as e:
            print(f"Error loading hourly data for {date_str} {hour_str}: {e}")
            return pd.DataFrame()

    def load_modis_data(self, date_obj, stations, band_type):
        """加载MODIS数据"""
        date_str = date_obj.strftime("%Y%m%d")
        year = date_str[:4]
        month = date_str[4:6]

        if band_type == 'Red':
            mod_dir = self.config.DATA_PATHS['mod_red']
            file_name = f"MODIS_Red_nadir_{date_str}.nc"
            var_name = 'Red_nadir'
        elif band_type == 'NIR':
            mod_dir = self.config.DATA_PATHS['mod_nir']
            file_name = f"MODIS_NIR_nadir_{date_str}.nc"
            var_name = 'NIR_nadir'
        else:
            return {station: np.nan for station in stations}

        file_path = os.path.join(mod_dir, year, month, file_name)

        result = {station: np.nan for station in stations}

        if not os.path.exists(file_path):
            return result

        try:
            with nc.Dataset(file_path) as ds:
                modis_stations = ds.variables['Station'][:]
                modis_station_names = [''.join(s).strip() for s in modis_stations]
                modis_data = ds.variables[var_name][:]

                for station in stations:
                    if station in modis_station_names:
                        idx = modis_station_names.index(station)
                        val = modis_data[idx]
                        result[station] = val if val != -9999.0 and not np.isnan(val) else np.nan
        except Exception as e:
            print(f"Error reading MODIS {band_type} data: {e}")

        return result

    def load_ahi_data_for_period(self, stations, start_date, end_date, sample_days=None):
        """加载指定时间段内的AHI数据"""
        print(f"Loading AHI data from {start_date} to {end_date} for {len(stations)} stations")

        # 生成日期范围
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")

        if sample_days:
            all_dates = pd.date_range(start_dt, end_dt)
            sample_dates = pd.Series(all_dates).sample(n=min(sample_days, len(all_dates)), random_state=42)
            date_list = [d.to_pydatetime() for d in sample_dates]
            print(f"Sampling {len(date_list)} days")
        else:
            date_list = []
            current_dt = start_dt
            while current_dt <= end_dt:
                date_list.append(current_dt)
                current_dt += timedelta(days=1)

        all_data = []

        for date_obj in tqdm(date_list, desc="Loading daily data"):
            date_str = date_obj.strftime("%Y%m%d")

            # 加载全天数据（每天只有9个小时，根据H8卫星过境时间）
            # H8观测时间：02:00, 04:00, 05:00, 06:00, 07:00, 08:00, 09:00, 10:00, 11:00 (UTC时间)
            # 对应北京时间：10:00, 12:00, 13:00, 14:00, 15:00, 16:00, 17:00, 18:00, 19:00
            h8_observation_hours_utc = [2, 4, 5, 6, 7, 8, 9, 10, 11]

            for hour in h8_observation_hours_utc:
                df_hour = self.load_hourly_data(date_str, hour, stations)
                if not df_hour.empty:
                    all_data.append(df_hour)

        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            print(f"AHI data loaded: {len(final_df)} records from {len(stations)} stations")
            return final_df

        return pd.DataFrame()


# ==================== Cached Inversion Processor ====================
class CachedInversionProcessor:
    """带有缓存功能的并行反演处理器"""

    def __init__(self, config):
        self.config = config
        self.data_loader = AHIInversionDataLoader(config)

    def generate_cache_key(self, start_date, end_date, stations, params_hash=None):
        """生成缓存键"""
        # 使用日期范围和站点数量生成缓存键
        station_hash = hashlib.md5('_'.join(sorted(stations)).encode()).hexdigest()[:8]
        date_range = f"{start_date}_{end_date}"

        if params_hash:
            cache_key = f"lsr_cache_{date_range}_{station_hash}_{params_hash}"
        else:
            cache_key = f"lsr_cache_{date_range}_{station_hash}"

        return cache_key

    def check_cache_exists(self, cache_key):
        """检查缓存是否存在"""
        cache_dir = Path("lsr_cache")
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

    def save_to_cache(self, results_df, station_stats, cache_key):
        """保存到缓存"""
        cache_dir = Path("lsr_cache")
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
            'created_at': datetime.now().isoformat(),
            'total_records': len(results_df),
            'successful_inversions': int(results_df['success'].sum() if results_df['success'].dtype == bool else (
                        results_df['success'] == 1).sum())
        }
        with open(meta_path, 'w') as f:
            json.dump(meta_data, f, indent=2)

        return lsr_path, stats_path

    def load_from_cache(self, cache_key):
        """从缓存加载"""
        cache_dir = Path("lsr_cache")

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

            return results_df, station_stats

        except Exception as e:
            print(f"Error loading from cache: {e}")
            return None, None

    def prepare_inversion_params(self, ahi_data):
        """准备反演参数"""
        print("Preparing inversion parameters...")
        all_params = []

        for band_id in self.config.BAND_CONFIG.keys():
            band_wavelength = self.config.BAND_CONFIG[band_id]['wavelength']
            band_data = ahi_data.copy()

            # 筛选有TOA数据的记录
            toa_col = f'TOA_Albedo_{band_id}'
            if toa_col not in band_data.columns:
                continue

            band_data = band_data[band_data[toa_col].notna()].copy()

            if band_data.empty:
                continue

            print(f"  Band {band_id}: {len(band_data)} records")

            # 计算RAA
            band_data['raa'] = np.abs(band_data['SOA'] - band_data['SAA'])

            # 为每条记录创建参数
            for idx, row in band_data.iterrows():
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
                    'original_index': idx
                }
                all_params.append(params)

        print(f"Total inversion tasks: {len(all_params)}")
        return all_params

    def calculate_station_availability(self, ahi_data):
        """计算站点数据可用性"""
        print("\nCalculating station data availability...")

        station_stats = {}
        all_stations = ahi_data['station'].unique()

        for station in tqdm(all_stations, desc="Analyzing stations"):
            station_data = ahi_data[ahi_data['station'] == station]

            if not station_data.empty:
                # 计算波段3和4的数据可用性
                band3_available = 0
                band4_available = 0

                if 'TOA_Albedo_03' in station_data.columns:
                    band3_available = station_data['TOA_Albedo_03'].notna().sum()

                if 'TOA_Albedo_04' in station_data.columns:
                    band4_available = station_data['TOA_Albedo_04'].notna().sum()

                total_available = band3_available + band4_available

                # 修正：每天只有9个观测小时，而不是24个
                # 每个站点每天最多有9个观测时间 * 2个波段 = 18个观测值
                unique_dates = station_data['datetime_bj'].dt.date.nunique()
                total_possible = unique_dates * 9 * 2  # 9小时/天 * 2个波段

                availability = total_available / total_possible if total_possible > 0 else 0

                station_stats[station] = {
                    'availability': availability,
                    'records': len(station_data),
                    'band3_available': band3_available,
                    'band4_available': band4_available,
                    'unique_dates': unique_dates,
                    'total_possible': total_possible,
                    'total_score': total_available + unique_dates * 5
                }

        # 按可用性排序
        sorted_stats = sorted(station_stats.items(), key=lambda x: x[1]['availability'], reverse=True)

        print("\nTop 10 stations by availability:")
        for i, (station, stats) in enumerate(sorted_stats[:10]):
            print(f"  {i + 1}. {station}: {stats['availability'] * 100:5.1f}% "
                  f"({stats['records']:4d} records, {stats['unique_dates']:2d} days)")

        print(f"\nBottom 10 stations by availability:")
        for i, (station, stats) in enumerate(sorted_stats[-10:]):
            print(f"  {i + 1}. {station}: {stats['availability'] * 100:5.1f}% "
                  f"({stats['records']:4d} records, {stats['unique_dates']:2d} days)")

        # 整体统计
        all_availabilities = [stats['availability'] for stats in station_stats.values()]
        mean_availability = np.mean(all_availabilities) if all_availabilities else 0
        median_availability = np.median(all_availabilities) if all_availabilities else 0

        print(f"\nOverall statistics (considering 9 hours/day):")
        print(f"  Total stations: {len(station_stats)}")
        print(f"  Mean availability: {mean_availability * 100:.1f}%")
        print(f"  Median availability: {median_availability * 100:.1f}%")
        print(f"  Stations with availability >= 25%: {sum(1 for a in all_availabilities if a >= 0.25)}")
        print(f"  Stations with availability >= 30%: {sum(1 for a in all_availabilities if a >= 0.30)}")
        print(f"  Stations with availability >= 35%: {sum(1 for a in all_availabilities if a >= 0.35)}")

        return station_stats

    def filter_stations_by_availability(self, station_stats, min_availability=0.25, max_stations=None):
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
                print(f"  {i + 1}. {station}: {avail * 100:.1f}% "
                      f"({stats['records']} records, {stats['unique_dates']} days)")

        return filtered_station_names

    def run_parallel_inversion(self, inversion_params, output_path=None):
        """运行并行反演"""
        max_workers = self.config.PARALLEL_CONFIG['max_workers']
        chunk_size = self.config.PARALLEL_CONFIG['chunk_size']

        print(f"\nStarting parallel inversion with {max_workers} workers, chunk size: {chunk_size}")

        # 按波段分组
        band_groups = {}
        for params in inversion_params:
            band_id = params['band']
            if band_id not in band_groups:
                band_groups[band_id] = []
            band_groups[band_id].append(params)

        all_results = []
        start_time = time.time()

        # 为每个波段处理
        for band_id, band_params in band_groups.items():
            print(f"\nProcessing band {band_id} with {len(band_params)} tasks")

            band_wavelength = self.config.BAND_CONFIG[band_id]['wavelength']

            # 分块处理
            chunks = [band_params[i:i + chunk_size] for i in range(0, len(band_params), chunk_size)]
            total_chunks = len(chunks)

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

        # 统计成功率
        if 'success' in results_df.columns:
            success_count = results_df['success'].sum() if results_df['success'].dtype == bool else (
                        results_df['success'] == 1).sum()
            success_rate = (success_count / len(results_df)) * 100 if len(results_df) > 0 else 0
            print(f"\nInversion completed in {elapsed_time:.2f} seconds")
            print(f"Total tasks: {len(results_df)}")
            print(f"Successful inversions: {success_count}")
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Speed: {len(results_df) / elapsed_time:.2f} tasks/second")

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
                'total_records': len(results_df),
                'successful_inversions': int(success_count),
                'success_rate': float(success_rate),
                'processing_time_seconds': float(elapsed_time),
                'speed_tasks_per_second': float(len(results_df) / elapsed_time) if elapsed_time > 0 else 0
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

    def process_all_stations(self, start_date, end_date, output_path=None,
                             min_availability=0.25, max_stations=None,
                             lc_types=None, sample_days=None, use_cache=True):
        """处理所有站点的完整流程"""
        print("=" * 70)
        print("AHI TOA TO LSR 6S INVERSION WITH CACHE")
        print("=" * 70)

        # 1. 加载所有站点
        print("\nStep 1: Loading all stations...")
        stations_df = self.data_loader.load_stations()
        all_stations = stations_df['station'].tolist()
        print(f"Total stations available: {len(all_stations)}")

        # 2. 按LC类型过滤
        if lc_types:
            print(f"\nFiltering stations by LC types: {lc_types}")
            all_stations = self.data_loader.filter_stations_by_lc(all_stations, lc_types)

        # 3. 生成缓存键
        cache_key = self.generate_cache_key(start_date, end_date, all_stations)

        # 4. 检查缓存
        if use_cache:
            cache_exists, cache_files = self.check_cache_exists(cache_key)
            if cache_exists:
                print(f"\nCache found for key: {cache_key}")
                user_input = input("Do you want to use cached data? (y/n): ").strip().lower()
                if user_input == 'y':
                    # 从缓存加载
                    results_df, station_stats = self.load_from_cache(cache_key)
                    if results_df is not None:
                        print("\nUsing cached data. Skipping 6S inversion.")
                        return results_df, station_stats, cache_key
                else:
                    print("Proceeding with new 6S inversion...")
            else:
                print(f"\nNo cache found for key: {cache_key}")

        # 5. 加载AHI TOA数据
        print(f"\nStep 2: Loading AHI TOA data ({start_date} to {end_date})...")
        ahi_data = self.data_loader.load_ahi_data_for_period(
            all_stations, start_date, end_date, sample_days
        )

        if ahi_data.empty:
            print("Error: No AHI data loaded")
            return None, None, None

        print(f"AHI data loaded: {len(ahi_data)} records")

        # 6. 计算站点可用性（考虑每天只有9小时）
        print("\nStep 3: Calculating station data availability (9 hours/day)...")
        station_stats = self.calculate_station_availability(ahi_data)

        # 7. 按可用性过滤站点
        print("\nStep 4: Filtering stations by availability...")
        filtered_stations = self.filter_stations_by_availability(
            station_stats, min_availability, max_stations
        )

        if not filtered_stations:
            print("Warning: No stations meet the availability criteria")
            # 使用可用性最高的站点
            sorted_stations = sorted(station_stats.items(), key=lambda x: x[1]['availability'], reverse=True)
            filtered_stations = [s[0] for s in sorted_stations[:min(100, len(sorted_stations))]]
            print(f"Using top {len(filtered_stations)} stations by availability")

        # 8. 只处理过滤后的站点数据
        print(f"\nStep 5: Processing filtered data ({len(filtered_stations)} stations)...")
        ahi_data_filtered = ahi_data[ahi_data['station'].isin(filtered_stations)].copy()

        if ahi_data_filtered.empty:
            print("Error: No data after filtering")
            return None, None, None

        print(f"Filtered AHI data: {len(ahi_data_filtered)} records")

        # 9. 准备反演参数
        print("\nStep 6: Preparing inversion parameters...")
        inversion_params = self.prepare_inversion_params(ahi_data_filtered)

        if not inversion_params:
            print("Error: No inversion parameters prepared")
            return None, None, None

        # 10. 运行并行反演
        print("\nStep 7: Running parallel 6S inversion...")

        # 设置输出路径
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"./ahi_lsr_{start_date}_{end_date}_{timestamp}.parquet"

        results_df = self.run_parallel_inversion(inversion_params, output_path)

        if results_df is None or results_df.empty:
            print("Error: Inversion failed")
            return None, None, None

        # 11. 保存到缓存
        print("\nStep 8: Saving results to cache...")
        cache_lsr_path, cache_stats_path = self.save_to_cache(results_df, station_stats, cache_key)

        print("\n" + "=" * 70)
        print("PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Cache key: {cache_key}")
        print(f"Output file: {output_path}")
        print(f"Cached LSR data: {cache_lsr_path}")
        print(f"Stations processed: {len(filtered_stations)}")
        print(f"Total LSR records: {len(results_df)}")

        return results_df, station_stats, cache_key


# ==================== Main Function ====================
def main():
    parser = argparse.ArgumentParser(
        description='AHI TOA to LSR 6S Inversion with Cache Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 生成4月1-10日所有站点的LSR数据，使用缓存
  python ahi_6s_inversion_cached.py --start_date 20160401 --end_date 20160410

  # 不使用缓存，强制重新生成
  python ahi_6s_inversion_cached.py --start_date 20160401 --end_date 20160410 --no_cache

  # 只处理可用性>25%的站点（生长季，每天9小时）
  python ahi_6s_inversion_cached.py --min_availability 0.25 --start_date 20160401 --end_date 20160410

  # 只处理前500个高可用性站点
  python ahi_6s_inversion_cached.py --max_stations 500 --start_date 20160401 --end_date 20160410

  # 使用更多工作进程加速处理
  python ahi_6s_inversion_cached.py --max_workers 12 --chunk_size 2000 --start_date 20160401 --end_date 20160410

  # 只处理农田和城市站点（LC类型12,13）
  python ahi_6s_inversion_cached.py --lc_types 12 13 --start_date 20160401 --end_date 20160410
        """
    )

    parser.add_argument('--start_date', type=str, default='20160401',
                        help='Start date YYYYMMDD (default: 20160401 - April 1st, growing season)')
    parser.add_argument('--end_date', type=str, default='20160410',
                        help='End date YYYYMMDD (default: 20160410 - April 10th, growing season)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output file path (if not provided, auto-generated)')
    parser.add_argument('--max_workers', type=int, default=8,
                        help='Maximum number of parallel workers (default: 8)')
    parser.add_argument('--lc_types', type=int, nargs='+', default=None,
                        help='Filter stations by LC types (e.g., 12 13 for Croplands and Urban)')
    parser.add_argument('--sample_days', type=int, default=None,
                        help='Number of days to sample (default: all days)')
    parser.add_argument('--min_availability', type=float, default=0.25,
                        help='Minimum data availability ratio to include station (default: 0.25 for 9 hours/day)')
    parser.add_argument('--max_stations', type=int, default=None,
                        help='Maximum number of stations to process (default: all)')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Chunk size for parallel processing (default: 1000)')
    parser.add_argument('--no_cache', action='store_true',
                        help='Do not use cache, always run 6S inversion')

    args = parser.parse_args()

    print("=" * 70)
    print("AHI TOA TO LSR 6S INVERSION WITH CACHE")
    print("=" * 70)
    print(f"Time period: {args.start_date} to {args.end_date} (Growing season)")
    print(f"Max workers: {args.max_workers}")
    print(f"LC types filter: {args.lc_types if args.lc_types else 'All'}")
    print(f"Sample days: {args.sample_days if args.sample_days else 'All'}")
    print(f"Minimum availability: {args.min_availability * 100:.0f}% (adjusted for 9 hours/day)")
    print(f"Max stations: {args.max_stations if args.max_stations else 'All'}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Use cache: {not args.no_cache}")
    print("=" * 70)

    # 配置
    config = InversionConfig()
    config.PARALLEL_CONFIG['max_workers'] = args.max_workers
    config.PARALLEL_CONFIG['chunk_size'] = args.chunk_size

    # 创建处理器
    processor = CachedInversionProcessor(config)

    # 运行处理流程
    results_df, station_stats, cache_key = processor.process_all_stations(
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=args.output_path,
        min_availability=args.min_availability,
        max_stations=args.max_stations,
        lc_types=args.lc_types,
        sample_days=args.sample_days,
        use_cache=not args.no_cache
    )

    if results_df is not None:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Cache key: {cache_key}")

        if station_stats:
            # 统计信息
            availabilities = [stats.get('availability', 0) for stats in station_stats.values()]
            if availabilities:
                mean_avail = np.mean(availabilities)
                median_avail = np.median(availabilities)
                stations_above_25 = sum(1 for a in availabilities if a >= 0.25)
                stations_above_30 = sum(1 for a in availabilities if a >= 0.30)
                stations_above_35 = sum(1 for a in availabilities if a >= 0.35)

                print(f"Station availability statistics (9 hours/day):")
                print(f"  Total stations analyzed: {len(station_stats)}")
                print(f"  Mean availability: {mean_avail * 100:.1f}%")
                print(f"  Median availability: {median_avail * 100:.1f}%")
                print(f"  Stations with availability >= 25%: {stations_above_25}")
                print(f"  Stations with availability >= 30%: {stations_above_30}")
                print(f"  Stations with availability >= 35%: {stations_above_35}")

        if results_df is not None:
            print(f"LSR records generated: {len(results_df)}")
            if 'success' in results_df.columns:
                success_count = results_df['success'].sum() if results_df['success'].dtype == bool else (
                            results_df['success'] == 1).sum()
                success_rate = success_count / len(results_df) * 100 if len(results_df) > 0 else 0
                print(f"  Successful inversions: {success_count} ({success_rate:.1f}%)")

        print(f"\nTo use this data in validation, run:")
        print(
            f"python enhanced_validation_v0.7.py --lsr_data ./lsr_cache/{cache_key}.parquet --output_dir ./validation_results")

    print("=" * 70)

    return 0


if __name__ == "__main__":
    main()