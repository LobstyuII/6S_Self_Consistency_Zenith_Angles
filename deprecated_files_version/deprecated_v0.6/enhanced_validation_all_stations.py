# ==================== enhanced_validation_all_stations.py ====================
"""
处理所有站点的增强版验证工作流 - 完整版本
可以处理2000个站点，按可用性过滤，每5个站点一张图
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import argparse
import warnings
from tqdm import tqdm
import json
import seaborn as sns
import gc
import netCDF4 as nc

warnings.filterwarnings('ignore')


# ==================== Configuration ====================
class Config:
    """验证配置类"""

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
        '01': {'wavelength': 0.47, 'name': 'band1', 'modis_band': None},
        '02': {'wavelength': 0.51, 'name': 'band2', 'modis_band': None},
        '03': {'wavelength': 0.64, 'name': 'band3', 'modis_band': 'Red'},
        '04': {'wavelength': 0.86, 'name': 'band4', 'modis_band': 'NIR'},
        '05': {'wavelength': 1.60, 'name': 'band5', 'modis_band': None},
        '06': {'wavelength': 2.30, 'name': 'band6', 'modis_band': None}
    }

    # Model expected feature order (与训练时一致)
    EXPECTED_FEATURES = [
        'sza', 'vza', 'raa', 'aod550', 'h2o', 'o3', 'wavelength',
        'rho_toa', 'rho_retrieved', 'cos_sza', 'sin_sza', 'cos_vza',
        'sin_vza', 'cos_raa', 'sin_raa', 'scattering_angle',
        'airmass_sza', 'airmass_vza', 'total_airmass', 'vza_sza_ratio',
        'vza_minus_sza', 'vza_plus_sza', 'aod_airmass', 'aod_wavelength',
        'rho_ratio', 'rho_diff', 'rho_product', 'wavelength_cos_sza',
        'wavelength_cos_vza'
    ]


# ==================== All Stations Validator ====================
class AllStationsValidator:
    """处理所有站点的验证器"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.station_stats = None
        self.station_coords = None

    def load_model(self, model_path):
        """加载机器学习模型"""
        print(f"Loading model: {model_path}")
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def load_station_stats(self, stats_path):
        """加载站点统计信息"""
        print(f"Loading station statistics: {stats_path}")
        try:
            with open(stats_path, 'r') as f:
                self.station_stats = json.load(f)
            print(f"Loaded statistics for {len(self.station_stats)} stations")
            return True
        except Exception as e:
            print(f"Error loading station statistics: {e}")
            return False

    def load_station_coords(self):
        """加载站点坐标"""
        print("Loading station coordinates...")
        try:
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

                self.station_coords = pd.DataFrame({
                    'station': stations,
                    'lat': lats,
                    'lon': lons
                }).dropna(subset=['lat', 'lon']).drop_duplicates('station')

            print(f"Loaded coordinates for {len(self.station_coords)} stations")
            return True
        except Exception as e:
            print(f"Error loading station coordinates: {e}")
            return False

    def filter_stations_by_availability(self, min_availability=0.25, top_n=None):
        """按可用性过滤站点"""
        if self.station_stats is None:
            print("Error: No station statistics loaded")
            return []

        print(f"\nFiltering stations by availability (≥{min_availability * 100:.0f}%)...")

        # 过滤可用性>=min_availability的站点
        filtered_stations = []
        for station, stats in self.station_stats.items():
            if 'availability' in stats and stats['availability'] >= min_availability:
                filtered_stations.append((station, stats['availability']))

        # 按可用性排序（从高到低）
        filtered_stations.sort(key=lambda x: x[1], reverse=True)

        # 如果指定了top_n，只取前n个
        if top_n is not None and top_n > 0:
            filtered_stations = filtered_stations[:top_n]

        print(f"  Total stations in stats: {len(self.station_stats)}")
        print(f"  Stations meeting criteria: {len(filtered_stations)}")

        if len(filtered_stations) > 0:
            print("\nTop 10 stations by availability:")
            for i, (station, avail) in enumerate(filtered_stations[:10]):
                stats = self.station_stats.get(station, {})
                records = stats.get('records', 0)
                days = stats.get('unique_dates', 0)
                print(f"  {i + 1:2d}. {station:20s}: {avail * 100:5.1f}% "
                      f"({records} records, {days} days)")

        return [station for station, avail in filtered_stations]

    def load_lsr_data_for_stations(self, lsr_data_path, stations):
        """为指定站点加载LSR数据"""
        print(f"\nLoading LSR data for {len(stations)} stations...")

        lsr_path = Path(lsr_data_path)
        if not lsr_path.exists():
            print(f"Error: LSR data file not found: {lsr_data_path}")
            return pd.DataFrame()

        try:
            # 根据文件类型选择加载方法
            if lsr_path.suffix == '.parquet':
                # 使用pyarrow加载，支持谓词下推
                import pyarrow.parquet as pq

                # 创建过滤器
                filters = [('station', 'in', stations)]

                # 读取数据
                table = pq.read_table(lsr_path, filters=filters)
                lsr_data = table.to_pandas()

            elif lsr_path.suffix == '.nc':
                # 对于NetCDF，使用xarray
                import xarray as xr
                ds = xr.open_dataset(lsr_path)
                # 转换为DataFrame然后过滤
                lsr_data = ds.to_dataframe().reset_index()
                lsr_data = lsr_data[lsr_data['station'].isin(stations)]
                ds.close()

            elif lsr_path.suffix == '.csv':
                # CSV文件，使用pandas
                # 对于大文件，可以分块读取
                chunks = []
                for chunk in pd.read_csv(lsr_path, chunksize=100000):
                    chunk_filtered = chunk[chunk['station'].isin(stations)]
                    if not chunk_filtered.empty:
                        chunks.append(chunk_filtered)

                if chunks:
                    lsr_data = pd.concat(chunks, ignore_index=True)
                else:
                    lsr_data = pd.DataFrame()
            else:
                print(f"Unsupported file format: {lsr_path.suffix}")
                return pd.DataFrame()

            print(f"LSR data loaded: {len(lsr_data)} records")

            # 检查必要列
            required_cols = ['band', 'station', 'datetime_bj', 'rho_toa', 'rho_lsr', 'sza', 'vza', 'raa']
            missing_cols = [col for col in required_cols if col not in lsr_data.columns]

            if missing_cols:
                print(f"Warning: Missing columns in LSR data: {missing_cols}")
                # 尝试转换列名
                if 'datetime_utc' in lsr_data.columns and 'datetime_bj' not in lsr_data.columns:
                    lsr_data['datetime_bj'] = lsr_data['datetime_utc']

            return lsr_data

        except Exception as e:
            print(f"Error loading LSR data: {e}")
            return pd.DataFrame()

    def load_modis_data_for_stations(self, lsr_data):
        """为LSR数据中的站点和日期加载MODIS数据"""
        print("\nLoading MODIS data for AHI LSR records...")

        if lsr_data.empty:
            return lsr_data

        # 获取唯一的日期和站点
        lsr_data['date'] = pd.to_datetime(lsr_data['datetime_bj']).dt.date
        unique_dates = lsr_data['date'].unique()
        unique_stations = lsr_data['station'].unique()

        modis_data_dict = {}

        for date in tqdm(unique_dates, desc="Loading MODIS data"):
            date_obj = datetime.combine(date, datetime.min.time())
            date_str = date_obj.strftime("%Y%m%d")

            # 加载MODIS红波段
            modis_red_data = self._load_single_modis_data(date_obj, unique_stations, 'Red')
            # 加载MODIS近红外波段
            modis_nir_data = self._load_single_modis_data(date_obj, unique_stations, 'NIR')

            # 合并到字典
            for station in unique_stations:
                key = (date_str, station)
                modis_data_dict[key] = {
                    'MODIS_Red': modis_red_data.get(station, np.nan),
                    'MODIS_NIR': modis_nir_data.get(station, np.nan)
                }

        # 将MODIS数据添加到LSR数据
        lsr_data_with_modis = lsr_data.copy()
        lsr_data_with_modis['date_str'] = pd.to_datetime(lsr_data_with_modis['datetime_bj']).dt.strftime('%Y%m%d')

        # 添加MODIS列
        lsr_data_with_modis['MODIS_Red'] = np.nan
        lsr_data_with_modis['MODIS_NIR'] = np.nan

        for idx, row in lsr_data_with_modis.iterrows():
            key = (row['date_str'], row['station'])
            if key in modis_data_dict:
                lsr_data_with_modis.at[idx, 'MODIS_Red'] = modis_data_dict[key]['MODIS_Red']
                lsr_data_with_modis.at[idx, 'MODIS_NIR'] = modis_data_dict[key]['MODIS_NIR']

        print(f"MODIS data added: Red波段有效数据{lsr_data_with_modis['MODIS_Red'].notna().sum()}个, "
              f"NIR波段有效数据{lsr_data_with_modis['MODIS_NIR'].notna().sum()}个")

        return lsr_data_with_modis

    def _load_single_modis_data(self, date_obj, stations, band_type):
        """加载单个MODIS波段数据"""
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

    def prepare_features(self, lsr_data):
        """为模型预测准备特征"""
        print("\nPreparing features for model prediction...")

        if lsr_data.empty:
            return pd.DataFrame()

        # 确保有必要的列
        required_cols = ['sza', 'vza', 'raa', 'aod550', 'h2o', 'o3',
                         'wavelength', 'rho_toa', 'rho_lsr', 'band']

        for col in required_cols:
            if col not in lsr_data.columns:
                print(f"Warning: Missing column {col}, using default values")
                if col == 'aod550':
                    lsr_data[col] = 0.1
                elif col == 'h2o':
                    lsr_data[col] = 2.0
                elif col == 'o3':
                    lsr_data[col] = 0.3
                elif col == 'rho_lsr':
                    lsr_data[col] = lsr_data.get('rho_toa', 0.2) * 0.8
                else:
                    lsr_data[col] = 0.0

        all_features = []

        for band_id in self.config.BAND_CONFIG.keys():
            # 筛选当前波段的数据
            band_mask = lsr_data['band'] == band_id
            band_data = lsr_data[band_mask].copy()

            if band_data.empty:
                continue

            print(f"  Processing band {band_id}: {len(band_data)} records")

            # 计算特征
            features_df = self._calculate_features(band_data, band_id)
            all_features.append(features_df)

        if not all_features:
            print("Error: No features prepared")
            return pd.DataFrame()

        # 合并所有特征
        all_features_df = pd.concat(all_features, ignore_index=True)
        print(f"Total features prepared: {len(all_features_df)}")

        return all_features_df

    def _calculate_features(self, band_data, band_id):
        """计算特征"""
        features_df = band_data.copy()

        # 基本几何参数
        sza = features_df['sza'].values
        vza = features_df['vza'].values
        raa = features_df['raa'].values

        # 转换为弧度
        sza_rad = np.radians(sza)
        vza_rad = np.radians(vza)
        raa_rad = np.radians(raa)

        # 三角函数特征
        features_df['cos_sza'] = np.cos(sza_rad)
        features_df['sin_sza'] = np.sin(sza_rad)
        features_df['cos_vza'] = np.cos(vza_rad)
        features_df['sin_vza'] = np.sin(vza_rad)
        features_df['cos_raa'] = np.cos(raa_rad)
        features_df['sin_raa'] = np.sin(raa_rad)

        # 散射角
        cos_scat = -np.cos(sza_rad) * np.cos(vza_rad) + \
                   np.sin(sza_rad) * np.sin(vza_rad) * np.cos(raa_rad)
        cos_scat = np.clip(cos_scat, -1.0, 1.0)
        features_df['scattering_angle'] = np.degrees(np.arccos(cos_scat))

        # 空气质量
        cos_sza_safe = np.clip(features_df['cos_sza'], 0.001, 1.0)
        cos_vza_safe = np.clip(features_df['cos_vza'], 0.001, 1.0)
        features_df['airmass_sza'] = 1.0 / cos_sza_safe
        features_df['airmass_vza'] = 1.0 / cos_vza_safe
        features_df['total_airmass'] = features_df['airmass_sza'] + features_df['airmass_vza']

        # 角度关系
        features_df['vza_sza_ratio'] = features_df['vza'] / (features_df['sza'] + 1e-6)
        features_df['vza_minus_sza'] = features_df['vza'] - features_df['sza']
        features_df['vza_plus_sza'] = features_df['vza'] + features_df['sza']

        # 波长
        wavelength = self.config.BAND_CONFIG[band_id]['wavelength']
        features_df['wavelength'] = wavelength

        # 反射率特征
        features_df['rho_retrieved'] = features_df.get('rho_lsr', 0.2)  # 使用rho_lsr作为rho_retrieved

        # 交互特征
        features_df['aod_airmass'] = features_df['aod550'] * features_df['total_airmass']
        features_df['aod_wavelength'] = features_df['aod550'] / features_df['wavelength']

        features_df['rho_ratio'] = features_df['rho_toa'] / (features_df['rho_retrieved'] + 1e-6)
        features_df['rho_diff'] = features_df['rho_toa'] - features_df['rho_retrieved']
        features_df['rho_product'] = features_df['rho_toa'] * features_df['rho_retrieved']

        features_df['wavelength_cos_sza'] = features_df['wavelength'] * features_df['cos_sza']
        features_df['wavelength_cos_vza'] = features_df['wavelength'] * features_df['cos_vza']

        # 确保所有预期特征都存在
        for feat in self.config.EXPECTED_FEATURES:
            if feat not in features_df.columns:
                features_df[feat] = 0.0

        return features_df

    def predict_correction(self, features_df):
        """预测校正值"""
        if self.model is None:
            raise ValueError("Model not loaded")

        print("\nPredicting correction values...")

        # 确保特征顺序正确
        expected_features = self.config.EXPECTED_FEATURES
        missing_features = [feat for feat in expected_features if feat not in features_df.columns]

        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # 添加缺失的特征
            for feat in missing_features:
                features_df[feat] = 0.0

        # 按正确顺序排列特征
        X = features_df[expected_features]

        # 预测
        predictions = self.model.predict(X)

        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'original_index': features_df.index,
            'band': features_df['band'],
            'station': features_df['station'],
            'datetime_bj': features_df.get('datetime_bj', pd.NaT),
            'rho_toa': features_df['rho_toa'],
            'rho_lsr': features_df['rho_retrieved'],  # 原版6S反演的LSR
            'predicted_correction': predictions
        })

        # 计算校正后的LSR
        # 注意：模型预测的是 error_absolute = rho_retrieved - rho_true
        # 所以校正后的LSR = rho_lsr - predicted_correction
        results_df['rho_lsr_corrected'] = results_df['rho_lsr'] - results_df['predicted_correction']
        results_df['rho_lsr_corrected'] = results_df['rho_lsr_corrected'].clip(0, 1)

        print(f"Correction stats: mean={results_df['predicted_correction'].mean():.6f}, "
              f"std={results_df['predicted_correction'].std():.6f}")
        print(f"Corrected LSR stats: mean={results_df['rho_lsr_corrected'].mean():.6f}, "
              f"std={results_df['rho_lsr_corrected'].std():.6f}")

        return results_df

    def merge_results(self, original_data, predictions):
        """合并原始数据和预测结果"""
        print("\nMerging results...")

        # 确保有索引列
        original_data = original_data.reset_index(drop=True)

        # 初始化结果列
        for band_id in self.config.BAND_CONFIG.keys():
            original_data[f'correction_{band_id}'] = np.nan
            original_data[f'lsr_{band_id}'] = np.nan
            original_data[f'lsr_corrected_{band_id}'] = np.nan

        # 映射预测结果
        for _, pred_row in predictions.iterrows():
            idx = pred_row['original_index']
            band_id = pred_row['band']

            if idx < len(original_data):
                # 检查行是否匹配
                if 'band' in original_data.columns and 'station' in original_data.columns:
                    # 验证行匹配
                    if (original_data.at[idx, 'band'] == band_id and
                            original_data.at[idx, 'station'] == pred_row['station']):
                        original_data.at[idx, f'correction_{band_id}'] = pred_row['predicted_correction']
                        original_data.at[idx, f'lsr_{band_id}'] = pred_row['rho_lsr']
                        original_data.at[idx, f'lsr_corrected_{band_id}'] = pred_row['rho_lsr_corrected']
                else:
                    # 如果没有band和station列，直接按索引赋值
                    original_data.at[idx, f'correction_{band_id}'] = pred_row['predicted_correction']
                    original_data.at[idx, f'lsr_{band_id}'] = pred_row['rho_lsr']
                    original_data.at[idx, f'lsr_corrected_{band_id}'] = pred_row['rho_lsr_corrected']

        return original_data

    def group_stations_for_plots(self, stations, group_size=5):
        """将站点分组用于绘图"""
        groups = []
        for i in range(0, len(stations), group_size):
            group = stations[i:i + group_size]
            groups.append(group)

        print(f"Grouped {len(stations)} stations into {len(groups)} groups of {group_size}")
        return groups

    def _get_station_info(self, station):
        """获取站点信息：经纬度、LC类型、可用性"""
        # 经纬度
        lat, lon = np.nan, np.nan
        if self.station_coords is not None and not self.station_coords.empty:
            station_info = self.station_coords[self.station_coords['station'] == station]
            if not station_info.empty:
                lat = station_info['lat'].values[0]
                lon = station_info['lon'].values[0]

        # LC信息
        lc_info = "N/A"

        # 可用性
        availability = "N/A"
        if self.station_stats and station in self.station_stats:
            stats = self.station_stats[station]
            if 'availability' in stats:
                availability = f"{stats['availability'] * 100:.1f}%"

        return lat, lon, lc_info, availability

    def plot_timeseries_for_group(self, corrected_data, station_group, group_idx, output_dir):
        """为站点组绘制时间序列图"""
        import matplotlib.pyplot as plt

        # 设置颜色
        colors = {
            'ahi_toa': '#1f77b4',
            'ahi_lsr': '#ff7f0e',
            'ahi_lsr_corrected': '#2ca02c',
            'modis': '#d62728'
        }

        # 设置图形样式
        plt.style.use('seaborn-v0_8-whitegrid')

        # 为每个波段创建图形（只绘制波段3和4）
        bands_to_plot = ['03', '04']

        for band_id in bands_to_plot:
            if band_id not in self.config.BAND_CONFIG:
                continue

            wavelength = self.config.BAND_CONFIG[band_id]['wavelength']
            modis_band = self.config.BAND_CONFIG[band_id]['modis_band']

            # 创建图形
            n_stations = len(station_group)
            fig, axes = plt.subplots(n_stations, 1,
                                     figsize=(14, 3 * n_stations))
            if n_stations == 1:
                axes = [axes]

            title = f'Band {band_id} ({wavelength}µm) - Group {group_idx + 1} ({n_stations} stations)'
            fig.suptitle(title, fontsize=14, fontweight='bold')

            for idx, station in enumerate(station_group):
                ax = axes[idx]

                # 筛选站点数据
                station_data = corrected_data[corrected_data['station'] == station].copy()

                if station_data.empty:
                    ax.text(0.5, 0.5, f"No data for {station}",
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Station: {station} (No data)')
                    continue

                # 按时间排序
                station_data = station_data.sort_values('datetime_bj')

                # 获取站点信息
                lat, lon, lc_info, availability = self._get_station_info(station)

                # 绘制AHI TOA
                toa_col = f'TOA_Albedo_{band_id}' if f'TOA_Albedo_{band_id}' in station_data.columns else 'rho_toa'
                if toa_col in station_data.columns:
                    valid_mask = station_data[toa_col].notna()
                    if valid_mask.any():
                        ax.scatter(station_data.loc[valid_mask, 'datetime_bj'],
                                   station_data.loc[valid_mask, toa_col],
                                   color=colors['ahi_toa'], marker='o', s=20,
                                   label='AHI TOA', alpha=0.7, edgecolors='none')

                # 绘制AHI LSR
                lsr_col = f'lsr_{band_id}'
                if lsr_col in station_data.columns:
                    valid_mask = station_data[lsr_col].notna()
                    if valid_mask.any():
                        ax.scatter(station_data.loc[valid_mask, 'datetime_bj'],
                                   station_data.loc[valid_mask, lsr_col],
                                   color=colors['ahi_lsr'], marker='^', s=25,
                                   label='AHI LSR (6S)', alpha=0.7, edgecolors='none')

                # 绘制AHI LSR Corrected
                lsr_corr_col = f'lsr_corrected_{band_id}'
                if lsr_corr_col in station_data.columns:
                    valid_mask = station_data[lsr_corr_col].notna()
                    if valid_mask.any():
                        ax.scatter(station_data.loc[valid_mask, 'datetime_bj'],
                                   station_data.loc[valid_mask, lsr_corr_col],
                                   color=colors['ahi_lsr_corrected'], marker='s', s=20,
                                   label='AHI LSR Corrected', alpha=0.7, edgecolors='none')

                # 绘制MODIS数据
                if modis_band:
                    modis_col = f'MODIS_{modis_band}'
                    if modis_col in station_data.columns:
                        # MODIS是每日数据，需要按天处理
                        daily_data = station_data.copy()
                        daily_data['date'] = daily_data['datetime_bj'].dt.date

                        # 计算每日平均值
                        daily_avg = daily_data.groupby('date')[modis_col].mean().reset_index()
                        daily_avg['datetime'] = pd.to_datetime(daily_avg['date'])

                        valid_mask = daily_avg[modis_col].notna()
                        if valid_mask.any():
                            ax.scatter(daily_avg.loc[valid_mask, 'datetime'],
                                       daily_avg.loc[valid_mask, modis_col],
                                       color=colors['modis'], marker='D', s=40,
                                       label='MODIS', alpha=0.8, edgecolors='black', linewidth=0.5)

                # 设置轴标签和标题
                ax.set_ylabel('Reflectance', fontsize=10)

                # 构建站点标题
                station_title = f'Station: {station} | '
                if not np.isnan(lat) and not np.isnan(lon):
                    station_title += f'Lat: {lat:.2f}°, Lon: {lon:.2f}° | '
                station_title += f'Avail: {availability}'
                ax.set_title(station_title, fontsize=10, pad=10)

                # 设置网格和图例
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

                # 设置x轴标签旋转
                ax.tick_params(axis='x', rotation=45, labelsize=9)
                ax.tick_params(axis='y', labelsize=9)

                # 设置y轴范围
                ax.set_ylim(-0.02, 0.6)  # 反射率典型范围

                # 设置x轴日期格式
                if not station_data.empty:
                    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))

            # 调整布局
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # 保存图形
            output_path = Path(output_dir) / f'band{band_id}_group_{group_idx + 1:03d}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            print(f"  Saved band {band_id} group {group_idx + 1}: {output_path}")

            # 清理内存
            gc.collect()

    def plot_availability_distribution(self, output_dir):
        """绘制站点可用性分布图"""
        if self.station_stats is None:
            print("No station statistics available for distribution plot")
            return

        import matplotlib.pyplot as plt

        # 提取可用性数据
        availabilities = [stats.get('availability', 0) for stats in self.station_stats.values()]

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 直方图
        n_bins = 20
        counts, bins, patches = ax1.hist(availabilities, bins=n_bins, edgecolor='black',
                                         alpha=0.7, color='steelblue')
        ax1.set_xlabel('Data Availability', fontsize=12)
        ax1.set_ylabel('Number of Stations', fontsize=12)
        ax1.set_title('Distribution of Station Data Availability', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')

        # 添加统计信息
        total_stations = len(availabilities)
        mean_avail = np.mean(availabilities)
        median_avail = np.median(availabilities)
        std_avail = np.std(availabilities)

        ax1.axvline(mean_avail, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_avail:.3f}')
        ax1.axvline(median_avail, color='green', linestyle='--', linewidth=2,
                    label=f'Median: {median_avail:.3f}')
        ax1.axvline(0.5, color='orange', linestyle='-', linewidth=1.5, alpha=0.7,
                    label='50% threshold')
        ax1.legend(loc='upper right', fontsize=10)

        # 在直方图上添加文本
        stats_text = f"Total stations: {total_stations}\n"
        stats_text += f"Mean: {mean_avail:.3f}\n"
        stats_text += f"Median: {median_avail:.3f}\n"
        stats_text += f"Std: {std_avail:.3f}"

        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 累计分布图
        sorted_avail = np.sort(availabilities)
        cdf = np.arange(1, len(sorted_avail) + 1) / len(sorted_avail)

        ax2.plot(sorted_avail, cdf, 'b-', linewidth=3, alpha=0.8)
        ax2.set_xlabel('Data Availability', fontsize=12)
        ax2.set_ylabel('Cumulative Probability', fontsize=12)
        ax2.set_title('Cumulative Distribution of Station Availability',
                      fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')

        # 添加50%阈值线
        ax2.axvline(0.25, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.axhline(0.25, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

        # 计算有多少站点超过25%
        stations_above_50 = sum(1 for a in availabilities if a >= 0.25)
        proportion_above_50 = stations_above_50 / total_stations

        # 在CDF图上添加文本
        cdf_text = f'Stations ≥50%: {stations_above_50}/{total_stations}\n'
        cdf_text += f'({proportion_above_50 * 100:.1f}%)'

        ax2.text(0.02, 0.98, cdf_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        # 填充50%以上的区域
        mask_above_50 = sorted_avail >= 0.25
        if any(mask_above_50):
            ax2.fill_between(sorted_avail[mask_above_50], 0, cdf[mask_above_50],
                             alpha=0.3, color='green')

        # 调整布局
        plt.tight_layout()

        # 保存图形
        output_path = Path(output_dir) / 'station_availability_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print(f"Availability distribution plot saved: {output_path}")

        return output_path

    def generate_availability_report(self, filtered_stations, output_dir):
        """生成可用性报告"""
        if self.station_stats is None:
            return None

        print("\nGenerating availability report...")

        # 计算统计信息
        availabilities = []
        for station in filtered_stations:
            if station in self.station_stats:
                stats = self.station_stats[station]
                if 'availability' in stats:
                    availabilities.append(stats['availability'])

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("STATION AVAILABILITY REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total stations in statistics: {len(self.station_stats)}")
        report_lines.append(f"Filtered stations: {len(filtered_stations)}")
        report_lines.append("")

        if availabilities:
            report_lines.append("Availability statistics for filtered stations:")
            report_lines.append(f"  Mean: {np.mean(availabilities):.4f}")
            report_lines.append(f"  Median: {np.median(availabilities):.4f}")
            report_lines.append(f"  Minimum: {np.min(availabilities):.4f}")
            report_lines.append(f"  Maximum: {np.max(availabilities):.4f}")
            report_lines.append(f"  Standard deviation: {np.std(availabilities):.4f}")
            report_lines.append(f"  25th percentile: {np.percentile(availabilities, 25):.4f}")
            report_lines.append(f"  75th percentile: {np.percentile(availabilities, 75):.4f}")
            report_lines.append("")

        # 站点列表
        report_lines.append("Filtered stations (sorted by availability):")
        report_lines.append("-" * 80)

        for i, station in enumerate(filtered_stations, 1):
            stats = self.station_stats.get(station, {})
            availability = stats.get('availability', 0)
            records = stats.get('records', 0)
            days = stats.get('unique_dates', 0)
            band3 = stats.get('band3_available', 0)
            band4 = stats.get('band4_available', 0)

            report_lines.append(f"{i:4d}. {station:20s} "
                                f"Avail: {availability * 100:6.2f}% | "
                                f"Records: {records:6d} | "
                                f"Days: {days:3d} | "
                                f"B3: {band3:5d} | "
                                f"B4: {band4:5d}")

        # 保存报告
        report_path = Path(output_dir) / "station_availability_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"Availability report saved: {report_path}")
        return report_path

    def generate_validation_summary(self, output_dir, filtered_stations,
                                    total_groups_processed, total_plots_generated):
        """生成验证总结报告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("VALIDATION SUMMARY REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        report_lines.append("PROCESSING STATISTICS:")
        report_lines.append(f"  Total stations filtered: {len(filtered_stations)}")
        report_lines.append(f"  Station groups processed: {total_groups_processed}")
        report_lines.append(f"  Total plots generated: {total_plots_generated}")
        report_lines.append(f"  Output directory: {output_dir}")
        report_lines.append("")
        report_lines.append("PLOT INFORMATION:")
        report_lines.append("  Bands plotted: 03 (0.64µm, Red), 04 (0.86µm, NIR)")
        report_lines.append("  Data types shown: AHI TOA, AHI LSR (6S), AHI LSR Corrected, MODIS")
        report_lines.append("  MODIS bands: Red (band 3), NIR (band 4)")
        report_lines.append("")
        report_lines.append("FILE STRUCTURE:")
        report_lines.append("  Time series plots: band{band}_group_{group_number}.png")
        report_lines.append("  Availability distribution: station_availability_distribution.png")
        report_lines.append("  Availability report: station_availability_report.txt")
        report_lines.append("  This summary: validation_summary.txt")

        # 保存报告
        report_path = Path(output_dir) / "validation_summary.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"Validation summary saved: {report_path}")
        return report_path

    def run_all_stations_validation(self, lsr_data_path, stats_path, model_path,
                                    output_dir, min_availability=0.25,
                                    stations_per_plot=5, max_stations=None):
        """运行所有站点的验证流程"""
        print("=" * 70)
        print("ALL STATIONS VALIDATION WORKFLOW")
        print("=" * 70)
        print(f"LSR data: {lsr_data_path}")
        print(f"Station stats: {stats_path}")
        print(f"Model: {model_path}")
        print(f"Output directory: {output_dir}")
        print(f"Minimum availability: {min_availability * 100:.0f}%")
        print(f"Stations per plot: {stations_per_plot}")
        print(f"Max stations: {max_stations if max_stations else 'All'}")
        print("=" * 70)

        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 加载模型
        if not self.load_model(model_path):
            return False

        # 2. 加载站点统计
        if not self.load_station_stats(stats_path):
            return False

        # 3. 加载站点坐标
        self.load_station_coords()

        # 4. 绘制可用性分布图
        print("\nStep 1: Plotting station availability distribution...")
        self.plot_availability_distribution(output_dir)

        # 5. 按可用性过滤站点
        print("\nStep 2: Filtering stations by availability...")
        filtered_stations = self.filter_stations_by_availability(
            min_availability=min_availability,
            top_n=max_stations
        )

        if not filtered_stations:
            print("Error: No stations meet the availability criteria")
            return False

        # 6. 生成可用性报告
        print("\nStep 3: Generating availability report...")
        self.generate_availability_report(filtered_stations, output_dir)

        # 7. 分组站点用于绘图
        print(f"\nStep 4: Grouping {len(filtered_stations)} stations "
              f"into groups of {stations_per_plot}...")
        station_groups = self.group_stations_for_plots(filtered_stations, stations_per_plot)

        # 8. 处理每个组（分批处理以避免内存问题）
        total_groups_processed = 0
        total_plots_generated = 0

        for group_idx, station_group in enumerate(station_groups):
            print(f"\nProcessing group {group_idx + 1}/{len(station_groups)} "
                  f"({len(station_group)} stations)")

            # 8a. 加载该组的LSR数据
            lsr_data = self.load_lsr_data_for_stations(lsr_data_path, station_group)
            if lsr_data.empty:
                print(f"  No LSR data for group {group_idx + 1}, skipping...")
                continue

            # 8b. 加载MODIS数据
            lsr_data_with_modis = self.load_modis_data_for_stations(lsr_data)

            # 8c. 准备特征
            features_df = self.prepare_features(lsr_data_with_modis)
            if features_df.empty:
                print(f"  No features for group {group_idx + 1}, skipping...")
                continue

            # 8d. 预测校正值
            predictions = self.predict_correction(features_df)
            if predictions.empty:
                print(f"  No predictions for group {group_idx + 1}, skipping...")
                continue

            # 8e. 合并结果
            corrected_data = self.merge_results(lsr_data_with_modis, predictions)

            # 8f. 绘制时间序列图
            print(f"  Plotting time series for group {group_idx + 1}...")
            self.plot_timeseries_for_group(corrected_data, station_group, group_idx, output_dir)

            # 8g. 保存该组的结果
            group_output_path = output_dir / f"results_group_{group_idx + 1:03d}.parquet"
            corrected_data.to_parquet(group_output_path)
            print(f"  Group results saved: {group_output_path}")

            # 更新统计
            total_groups_processed += 1
            total_plots_generated += 2  # 波段3和4

            # 清理内存
            del lsr_data, lsr_data_with_modis, features_df, predictions, corrected_data
            gc.collect()

        # 9. 生成验证总结
        print("\nStep 5: Generating validation summary...")
        self.generate_validation_summary(
            output_dir, filtered_stations,
            total_groups_processed, total_plots_generated
        )

        print("\n" + "=" * 70)
        print("VALIDATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Station groups processed: {total_groups_processed}")
        print(f"Total plots generated: {total_plots_generated}")
        print(f"Output directory: {output_dir}")
        print("=" * 70)

        return True


# ==================== Main Function ====================
def main():
    parser = argparse.ArgumentParser(
        description='All Stations Validation - Process all stations with data availability > 50%',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 处理所有可用性>50%的站点，每5个站点一张图
  python enhanced_validation_all_stations.py --lsr_data ./ahi_lsr_all_stations.parquet --stats ./ahi_lsr_all_stations.station_stats.json

  # 只处理前200个高可用性站点
  python enhanced_validation_all_stations.py --lsr_data ./ahi_lsr_all_stations.parquet --stats ./ahi_lsr_all_stations.station_stats.json --max_stations 200

  # 提高可用性阈值到60%，每3个站点一张图
  python enhanced_validation_all_stations.py --lsr_data ./ahi_lsr_all_stations.parquet --stats ./ahi_lsr_all_stations.station_stats.json --min_availability 0.6 --stations_per_plot 3

  # 指定输出目录
  python enhanced_validation_all_stations.py --lsr_data ./ahi_lsr_all_stations.parquet --stats ./ahi_lsr_all_stations.station_stats.json --output_dir ./my_validation_results
        """
    )

    parser.add_argument('--lsr_data', type=str, required=True,
                        help='Path to LSR data file (Parquet format recommended)')
    parser.add_argument('--stats', type=str, required=True,
                        help='Path to station statistics JSON file')
    parser.add_argument('--model_path', type=str,
                        default=r'D:\6S_Self_Consistency_Zenith_Angles_v0.4\models\refactored\model_training_20260117_152723\models\XGBoost_model.pkl',
                        help='Path to trained model (default: XGBoost model)')
    parser.add_argument('--output_dir', type=str, default='./all_stations_validation',
                        help='Output directory for results')
    parser.add_argument('--min_availability', type=float, default=0.25,
                        help='Minimum data availability threshold (0.0-1.0, default: 0.25)')
    parser.add_argument('--stations_per_plot', type=int, default=5,
                        help='Number of stations per plot (default: 5)')
    parser.add_argument('--max_stations', type=int, default=None,
                        help='Maximum number of stations to process (default: all meeting criteria)')

    args = parser.parse_args()

    # 创建验证器
    config = Config()
    validator = AllStationsValidator(config)

    # 运行验证
    success = validator.run_all_stations_validation(
        lsr_data_path=args.lsr_data,
        stats_path=args.stats,
        model_path=args.model_path,
        output_dir=args.output_dir,
        min_availability=args.min_availability,
        stations_per_plot=args.stations_per_plot,
        max_stations=args.max_stations
    )

    if success:
        print("\nAll stations validation completed successfully!")
        return 0
    else:
        print("\nValidation failed!")
        return 1


if __name__ == "__main__":
    main()