# ==================== ahi_6s_inversion.py ====================
"""
AHI TOA到LSR的6S反演模块 - 多进程并行处理
第一步：加载AHI TOA数据和辅助数据，运行6S反演得到LSR
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

# Py6S for 6S simulations
from Py6S import *

warnings.filterwarnings('ignore')


# ==================== Configuration ====================
class InversionConfig:
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
        'max_workers': 4,
        'chunk_size': 1000
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

            # 清理实例
            del s
            gc.collect()

            return {
                'success': True,
                'rho_toa': rho_toa,
                'rho_lsr': rho_lsr,
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
                'datetime_bj': params.get('datetime_bj')
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

    def load_stations(self):
        """加载站点坐标"""
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

        return df

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

            # 添加大气参数（这里使用默认值，实际应用中应从MERRA2数据加载）
            df['AOD550'] = 0.1
            df['water'] = 2.0
            df['ozone'] = 0.3

            return df

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

    def load_ahi_toa_data(self, n_stations=50, start_date="20160101", end_date="20160110"):
        """加载AHI TOA数据"""
        print(f"Loading AHI TOA data: {start_date} to {end_date}")

        # 选择站点
        stations_df = self.load_stations()
        if len(stations_df) <= n_stations:
            stations = stations_df['station'].tolist()
        else:
            stations = stations_df.sample(n=n_stations, random_state=42)['station'].tolist()

        print(f"Selected {len(stations)} stations")

        # 生成日期范围
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")

        date_list = []
        current_dt = start_dt
        while current_dt <= end_dt:
            date_list.append(current_dt)
            current_dt += timedelta(days=1)

        all_data = []

        for date_obj in tqdm(date_list, desc="Loading daily data"):
            date_str = date_obj.strftime("%Y%m%d")

            # 加载MODIS数据
            modis_red_data = self.load_modis_data(date_obj, stations, 'Red')
            modis_nir_data = self.load_modis_data(date_obj, stations, 'NIR')

            # 加载全天数据
            for hour in range(0, 23):
                df_hour = self.load_hourly_data(date_str, hour, stations)

                if not df_hour.empty:
                    # 添加MODIS数据
                    df_hour['MODIS_Red'] = df_hour['station'].map(modis_red_data)
                    df_hour['MODIS_NIR'] = df_hour['station'].map(modis_nir_data)

                    all_data.append(df_hour)

        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            print(f"Data loading completed: {len(final_df)} records")
            print(f"Number of stations: {final_df['station'].nunique()}")

            return final_df

        return pd.DataFrame()


# ==================== Parallel Inversion Processor ====================
class ParallelInversionProcessor:
    """并行反演处理器"""

    def __init__(self, config):
        self.config = config
        self.data_loader = AHIInversionDataLoader(config)

    def prepare_inversion_params(self, ahi_data):
        """准备反演参数"""
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

            print(f"Preparing inversion for band {band_id}: {len(band_data)} records")

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

    def run_parallel_inversion(self, inversion_params, output_path=None):
        """运行并行反演"""
        max_workers = self.config.PARALLEL_CONFIG['max_workers']
        chunk_size = self.config.PARALLEL_CONFIG['chunk_size']

        print(f"Starting parallel inversion with {max_workers} workers...")

        # 按波段分组
        band_groups = {}
        for params in inversion_params:
            band_id = params['band']
            if band_id not in band_groups:
                band_groups[band_id] = []
            band_groups[band_id].append(params)

        all_results = []
        start_time = time.time()

        # 为每个波段创建进程池
        for band_id, band_params in band_groups.items():
            print(f"Processing band {band_id} with {len(band_params)} tasks")

            # 创建波段特定的工作器
            band_wavelength = self.config.BAND_CONFIG[band_id]['wavelength']
            worker = SixSInversionWorker(band_wavelength)

            # 分块处理
            chunks = [band_params[i:i + chunk_size] for i in range(0, len(band_params), chunk_size)]

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for chunk in chunks:
                    future = executor.submit(self._process_chunk, chunk, worker)
                    futures.append(future)

                # 收集结果
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"Band {band_id}"):
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
            print(f"Success rate: {success_rate:.1f}%")

        # 保存结果
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存为Parquet格式
            results_df.to_parquet(output_path)
            print(f"Results saved to: {output_path}")

            # 保存为NetCDF格式
            nc_path = output_path.with_suffix('.nc')
            self._save_to_netcdf(results_df, nc_path)
            print(f"Results also saved to: {nc_path}")

        return results_df

    def _process_chunk(self, chunk_params, worker):
        """处理一个数据块"""
        chunk_results = []
        for params in chunk_params:
            result = worker.run_inversion(params)
            chunk_results.append(result)
        return chunk_results

    def _save_to_netcdf(self, df, output_path):
        """保存为NetCDF格式"""
        # 转换为xarray Dataset
        ds = xr.Dataset.from_dataframe(df.reset_index())

        # 添加属性
        ds.attrs['creation_date'] = datetime.now().isoformat()
        ds.attrs['description'] = 'AHI TOA to LSR 6S Inversion Results'
        ds.attrs['version'] = '1.0'

        # 保存
        ds.to_netcdf(output_path)

    def process_ahi_data(self, n_stations=50, start_date="20160101",
                         end_date="20160110", output_path=None):
        """完整的数据处理流程"""
        # 1. 加载AHI TOA数据
        print("Step 1: Loading AHI TOA data...")
        ahi_data = self.data_loader.load_ahi_toa_data(
            n_stations=n_stations,
            start_date=start_date,
            end_date=end_date
        )

        if ahi_data.empty:
            print("Error: No AHI data loaded")
            return None

        # 2. 准备反演参数
        print("\nStep 2: Preparing inversion parameters...")
        inversion_params = self.prepare_inversion_params(ahi_data)

        if not inversion_params:
            print("Error: No inversion parameters prepared")
            return None

        # 3. 运行并行反演
        print("\nStep 3: Running parallel 6S inversion...")
        results_df = self.run_parallel_inversion(inversion_params, output_path)

        return results_df


# ==================== Main Function ====================
def main():
    parser = argparse.ArgumentParser(
        description='AHI TOA to LSR 6S Inversion - Parallel Processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 默认设置：50个站点，2016年1月1-10日
  python ahi_6s_inversion.py

  # 自定义站点数和日期范围
  python ahi_6s_inversion.py --stations 100 --start_date 20160101 --end_date 20160131

  # 指定输出路径
  python ahi_6s_inversion.py --output_path ./results/ahi_lsr_results.parquet
        """
    )

    parser.add_argument('--stations', type=int, default=50,
                        help='Number of stations to process (default: 50)')
    parser.add_argument('--start_date', type=str, default='20160101',
                        help='Start date YYYYMMDD (default: 20160101)')
    parser.add_argument('--end_date', type=str, default='20160110',
                        help='End date YYYYMMDD (default: 20160110)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output file path (default: ahi_lsr_results_<timestamp>.parquet)')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of parallel workers (default: 4)')

    args = parser.parse_args()

    print("=" * 70)
    print("AHI TOA to LSR 6S INVERSION PROCESSOR")
    print("=" * 70)
    print(f"Stations: {args.stations}")
    print(f"Time period: {args.start_date} to {args.end_date}")
    print(f"Max workers: {args.max_workers}")

    # 配置
    config = InversionConfig()
    config.PARALLEL_CONFIG['max_workers'] = args.max_workers

    # 设置输出路径
    if args.output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = f"./ahi_lsr_results_{timestamp}.parquet"

    # 创建处理器
    processor = ParallelInversionProcessor(config)

    # 运行处理流程
    results = processor.process_ahi_data(
        n_stations=args.stations,
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=args.output_path
    )

    if results is not None:
        print("\n" + "=" * 70)
        print("PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Output file: {args.output_path}")

        # 显示统计信息
        print("\nStatistics:")
        print(f"  Total records: {len(results)}")

        if 'success' in results.columns:
            success_count = results['success'].sum() if results['success'].dtype == bool else (
                        results['success'] == 1).sum()
            print(f"  Successful inversions: {success_count}")
            print(f"  Success rate: {success_count / len(results) * 100:.1f}%")

        if 'rho_lsr' in results.columns:
            valid_lsr = results['rho_lsr'].dropna()
            if len(valid_lsr) > 0:
                print(f"  LSR range: [{valid_lsr.min():.4f}, {valid_lsr.max():.4f}]")
                print(f"  LSR mean: {valid_lsr.mean():.4f}")

    print("=" * 70)

    return 0


if __name__ == "__main__":
    main()