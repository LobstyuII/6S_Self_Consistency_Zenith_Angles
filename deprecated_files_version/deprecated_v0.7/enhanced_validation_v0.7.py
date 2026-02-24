# ==================== enhanced_validation_v0.7.py ====================
"""
增强版验证工作流 V0.7 - 针对生长季数据优化
使用4月份数据，调整可用性阈值，优化缓存使用
"""

import os
import netCDF4 as nc
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
class EnhancedValidatorV0_7:
    """增强版验证器 V0.7 - 针对生长季数据优化"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.station_stats = None
        self.station_coords = None

    def load_model(self, model_path):
        """加载机器学习模型"""
        print(f"Loading model: {model_path}")
        try:
            # 尝试多种可能的路径格式
            possible_paths = [
                model_path,
                Path(model_path),
                Path.cwd() / model_path,
                Path.home() / model_path,
                Path("D:/6S_Self_Consistency_Zenith_Angles_v0.4/models/refactored") / model_path
            ]

            # 如果路径是目录，尝试查找里面的模型文件
            if os.path.isdir(model_path):
                possible_paths.extend([
                    Path(model_path) / "XGBoost_model.pkl",
                    Path(model_path) / "models" / "XGBoost_model.pkl",
                    Path(model_path) / "RandomForest_model.pkl"
                ])

            loaded = False
            for path in possible_paths:
                try:
                    if isinstance(path, Path):
                        path_str = str(path)
                    else:
                        path_str = path

                    if os.path.exists(path_str):
                        with open(path_str, 'rb') as f:
                            self.model = pickle.load(f)
                        print(f"Model loaded successfully from: {path_str}")
                        loaded = True
                        break
                except Exception as e:
                    continue

            if not loaded:
                # 尝试在常见位置搜索
                search_dirs = [
                    Path("../.."),
                    Path("./models"),
                    Path("D:/6S_Self_Consistency_Zenith_Angles_v0.4/models"),
                    Path("D:/6S_Self_Consistency_Zenith_Angles_v0.4/models/refactored")
                ]

                for search_dir in search_dirs:
                    if search_dir.exists():
                        for root, dirs, files in os.walk(search_dir):
                            for file in files:
                                if file.endswith('XGBoost_model.pkl') or file.endswith('.pkl'):
                                    model_file = Path(root) / file
                                    with open(model_file, 'rb') as f:
                                        self.model = pickle.load(f)
                                    print(f"Model loaded from auto-discovered location: {model_file}")
                                    return True

                print("Error: Could not find model file in any common location")
                return False
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_station_stats(self, stats_path):
        """加载站点统计信息"""
        print(f"Loading station statistics: {stats_path}")
        try:
            # 尝试多种可能的路径
            if not os.path.exists(stats_path):
                # 尝试从LSR数据同目录查找
                lsr_dir = Path(stats_path).parent
                possible_files = list(lsr_dir.glob("*stats*.json"))
                if possible_files:
                    stats_path = possible_files[0]
                    print(f"Found station stats at: {stats_path}")
                else:
                    print(f"Warning: Station stats file not found: {stats_path}")
                    return False

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
            import netCDF4 as nc
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
        """按可用性过滤站点（针对9小时/天的调整）"""
        if self.station_stats is None:
            print("Error: No station statistics loaded")
            return []

        print(f"\nFiltering stations by availability (≥{min_availability * 100:.0f}%, 9 hours/day)...")

        # 过滤可用性>=min_availability的站点
        filtered_stations = []
        for station, stats in self.station_stats.items():
            if 'availability' in stats:
                # 安全转换可用性值
                try:
                    availability = float(stats['availability'])
                except (ValueError, TypeError):
                    continue

                if availability >= min_availability:
                    filtered_stations.append((station, availability))

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

                # 安全转换函数
                def safe_int(value, default=0):
                    try:
                        return int(value)
                    except (ValueError, TypeError):
                        return default

                records = safe_int(stats.get('records', 0))
                days = safe_int(stats.get('unique_dates', 0))
                band3 = safe_int(stats.get('band3_available', 0))
                band4 = safe_int(stats.get('band4_available', 0))

                print(f"  {i + 1:2d}. {station:20s}: {avail * 100:5.1f}% "
                      f"({records} records, {days} days, B3:{band3}, B4:{band4})")

        return [station for station, avail in filtered_stations]

    def load_lsr_data(self, lsr_data_path):
        """加载LSR数据"""
        print(f"\nLoading LSR data from: {lsr_data_path}")

        lsr_path = Path(lsr_data_path)

        # 如果路径是通配符，查找匹配的文件
        if '*' in str(lsr_path):
            import glob
            files = glob.glob(str(lsr_path))
            if files:
                lsr_path = Path(files[0])
                print(f"Using file: {lsr_path}")
            else:
                print(f"Error: No files found matching pattern: {lsr_data_path}")
                return pd.DataFrame()

        if not lsr_path.exists():
            print(f"Error: LSR data file not found: {lsr_data_path}")
            # 尝试在缓存目录查找
            cache_path = Path("lsr_cache") / lsr_path.name
            if cache_path.exists():
                print(f"Found in cache directory: {cache_path}")
                lsr_path = cache_path
            else:
                # 查找任何包含日期的文件
                possible_files = list(Path("../..").glob("*lsr*.parquet"))
                if possible_files:
                    lsr_path = possible_files[0]
                    print(f"Found alternative file: {lsr_path}")
                else:
                    return pd.DataFrame()

        try:
            if lsr_path.suffix == '.parquet':
                lsr_data = pd.read_parquet(lsr_path)
            elif lsr_path.suffix == '.nc':
                import xarray as xr
                ds = xr.open_dataset(lsr_path)
                lsr_data = ds.to_dataframe().reset_index()
                ds.close()
            elif lsr_path.suffix == '.csv':
                lsr_data = pd.read_csv(lsr_path)
            else:
                print(f"Unsupported file format: {lsr_path.suffix}")
                return pd.DataFrame()

            print(f"LSR data loaded: {len(lsr_data)} records")

            # 检查数据列
            print(f"Columns in LSR data: {list(lsr_data.columns)}")

            # 检查是否有重复列
            duplicate_columns = lsr_data.columns[lsr_data.columns.duplicated()].tolist()
            if duplicate_columns:
                print(f"Warning: Found duplicate columns in LSR data: {duplicate_columns}")
                # 移除重复列，保留第一个出现的列
                lsr_data = lsr_data.loc[:, ~lsr_data.columns.duplicated()]

            # 重要：重命名列以确保一致性
            column_mapping = {}

            # 检查所有可能的LSR列名，统一映射到rho_retrieved（6S反演结果）
            possible_lsr_columns = [
                'rho_retrieved',  # 首选名称
                'rho_lsr',  # 常见变体
                'surface_reflectance',
                'lsr',
                'LSR',
                'reflectance_surface',
                'retrieved_reflectance'
            ]

            # 首先检查是否已经存在rho_retrieved列
            if 'rho_retrieved' in lsr_data.columns:
                print(f"Found 'rho_retrieved' column, no need to rename")
            else:
                # 查找第一个存在的LSR列并进行映射
                for col in possible_lsr_columns:
                    if col in lsr_data.columns and col != 'rho_retrieved':
                        column_mapping[col] = 'rho_retrieved'
                        print(f"Mapped column '{col}' to 'rho_retrieved' (6S LSR)")
                        break

            # 确保datetime列名正确
            if 'datetime_utc' in lsr_data.columns and 'datetime_bj' not in lsr_data.columns:
                # 复制而不是重命名，以避免列名冲突
                lsr_data['datetime_bj'] = lsr_data['datetime_utc']
                print("Created datetime_bj from datetime_utc")

            # 应用重命名映射
            if column_mapping:
                lsr_data = lsr_data.rename(columns=column_mapping)
                print(f"Renamed columns: {column_mapping}")

            # 再次检查重复列，重命名后可能产生重复列
            duplicate_columns = lsr_data.columns[lsr_data.columns.duplicated()].tolist()
            if duplicate_columns:
                print(f"Warning: Found duplicate columns after renaming: {duplicate_columns}")
                # 移除重复列，保留第一个出现的列
                lsr_data = lsr_data.loc[:, ~lsr_data.columns.duplicated()]

            # 检查必要列
            required_cols = ['band', 'station', 'datetime_bj', 'rho_toa', 'rho_retrieved', 'sza', 'vza', 'raa']
            available_cols = [col for col in required_cols if col in lsr_data.columns]
            missing_cols = [col for col in required_cols if col not in lsr_data.columns]

            if missing_cols:
                print(f"Warning: Missing columns in LSR data: {missing_cols}")
                print(f"Available columns: {available_cols}")

                # 尝试从其他列推断
                if 'rho_retrieved' not in lsr_data.columns:
                    # 如果还没有rho_retrieved，尝试创建
                    if 'rho_lsr' in lsr_data.columns:
                        lsr_data['rho_retrieved'] = lsr_data['rho_lsr']
                        print("Created rho_retrieved from rho_lsr")
                    else:
                        # 如果还是没有rho_retrieved，使用默认值
                        lsr_data['rho_retrieved'] = lsr_data.get('rho_toa', 0.2) * 0.8
                        print("Created default rho_retrieved column")

                if 'datetime_bj' not in lsr_data.columns and 'datetime_utc' in lsr_data.columns:
                    lsr_data['datetime_bj'] = lsr_data['datetime_utc']

                if 'raa' not in lsr_data.columns and 'SOA' in lsr_data.columns and 'SAA' in lsr_data.columns:
                    lsr_data['raa'] = np.abs(lsr_data['SOA'] - lsr_data['SAA'])
                    print("Calculated RAA from SOA and SAA")

            # 验证是否有6S反演结果
            if 'rho_retrieved' in lsr_data.columns:
                # 确保rho_retrieved是单列Series，而不是DataFrame
                if isinstance(lsr_data['rho_retrieved'], pd.DataFrame):
                    print("Warning: rho_retrieved is a DataFrame, selecting first column")
                    # 选择第一列
                    rho_retrieved_col = lsr_data['rho_retrieved'].iloc[:, 0]
                    # 删除原来的列
                    lsr_data = lsr_data.drop(columns=['rho_retrieved'])
                    # 添加单列
                    lsr_data['rho_retrieved'] = rho_retrieved_col

                valid_records = lsr_data['rho_retrieved'].notna().sum()
                print(f"✓ Found 6S LSR results (rho_retrieved): {valid_records} valid records")
                if valid_records > 0:
                    print(f"  Mean rho_retrieved: {lsr_data['rho_retrieved'].mean():.4f}")
            else:
                print("✗ Warning: No 6S LSR results found (rho_retrieved column missing)")

            # 检查TOA数据
            if 'rho_toa' in lsr_data.columns:
                valid_toa = lsr_data['rho_toa'].notna().sum()
                print(f"✓ TOA data available: {valid_toa} valid records")
            else:
                print("✗ Warning: TOA data (rho_toa) not found")

            return lsr_data

        except Exception as e:
            print(f"Error loading LSR data: {e}")
            import traceback
            traceback.print_exc()
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
            import netCDF4 as nc
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
                         'wavelength', 'rho_toa', 'rho_retrieved', 'band']

        for col in required_cols:
            if col not in lsr_data.columns:
                print(f"Warning: Missing column {col}, using default values")
                if col == 'aod550':
                    lsr_data[col] = 0.1
                elif col == 'h2o':
                    lsr_data[col] = 2.0
                elif col == 'o3':
                    lsr_data[col] = 0.3
                elif col == 'rho_retrieved':
                    lsr_data[col] = lsr_data.get('rho_toa', 0.2) * 0.8
                elif col == 'wavelength':
                    # 从波段ID推断波长
                    band_to_wavelength = {
                        '01': 0.47, '02': 0.51, '03': 0.64,
                        '04': 0.86, '05': 1.60, '06': 2.30
                    }
                    lsr_data[col] = lsr_data['band'].map(band_to_wavelength)
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

        # 检查特征列
        expected_features = self.config.EXPECTED_FEATURES
        missing_features = [feat for feat in expected_features if feat not in all_features_df.columns]
        if missing_features:
            print(f"Warning: Missing features in final dataset: {missing_features}")
            for feat in missing_features:
                all_features_df[feat] = 0.0

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

        # 波长（如果还没有）
        if 'wavelength' not in features_df.columns:
            wavelength = self.config.BAND_CONFIG[band_id]['wavelength']
            features_df['wavelength'] = wavelength

        # 反射率特征
        if 'rho_retrieved' not in features_df.columns:
            features_df['rho_retrieved'] = features_df.get('rho_lsr', 0.2)

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

        # 检查特征形状
        print(f"Feature matrix shape: {X.shape}")

        # 预测
        try:
            predictions = self.model.predict(X)
            print(f"Predictions shape: {predictions.shape}")
            print(
                f"Predictions stats - Min: {predictions.min():.4f}, Max: {predictions.max():.4f}, Mean: {predictions.mean():.4f}")
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            # 返回零预测值
            predictions = np.zeros(len(X))

        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'original_index': features_df.index,
            'band': features_df['band'],
            'station': features_df['station'],
            'datetime_bj': features_df.get('datetime_bj', pd.NaT),
            'rho_toa': features_df['rho_toa'],
            'rho_retrieved': features_df['rho_retrieved'],  # 原版6S反演的LSR
            'predicted_correction': predictions
        })

        # 计算校正后的LSR
        # 注意：模型预测的是 error_absolute = rho_retrieved - rho_true
        # 所以校正后的LSR = rho_retrieved - predicted_correction
        results_df['rho_lsr_corrected'] = results_df['rho_retrieved'] - results_df['predicted_correction']
        results_df['rho_lsr_corrected'] = results_df['rho_lsr_corrected'].clip(0, 1)

        print(
            f"Original LSR (6S) stats: mean={results_df['rho_retrieved'].mean():.6f}, std={results_df['rho_retrieved'].std():.6f}")
        print(
            f"Correction stats: mean={results_df['predicted_correction'].mean():.6f}, std={results_df['predicted_correction'].std():.6f}")
        print(
            f"Corrected LSR stats: mean={results_df['rho_lsr_corrected'].mean():.6f}, std={results_df['rho_lsr_corrected'].std():.6f}")

        return results_df

    def merge_results(self, original_data, predictions):
        """合并原始数据和预测结果"""
        print("\nMerging results...")

        # 确保有索引列
        original_data = original_data.reset_index(drop=True)

        # 添加模型预测结果列
        original_data['predicted_correction'] = np.nan
        original_data['rho_lsr_corrected'] = np.nan

        # 映射预测结果
        for _, pred_row in predictions.iterrows():
            idx = pred_row['original_index']

            if idx < len(original_data):
                # 检查行是否匹配
                if 'band' in original_data.columns and 'station' in original_data.columns:
                    # 验证行匹配
                    if (original_data.at[idx, 'band'] == pred_row['band'] and
                            original_data.at[idx, 'station'] == pred_row['station']):
                        original_data.at[idx, 'predicted_correction'] = pred_row['predicted_correction']
                        original_data.at[idx, 'rho_lsr_corrected'] = pred_row['rho_lsr_corrected']
                else:
                    # 如果没有band和station列，直接按索引赋值
                    original_data.at[idx, 'predicted_correction'] = pred_row['predicted_correction']
                    original_data.at[idx, 'rho_lsr_corrected'] = pred_row['rho_lsr_corrected']

        # 确保rho_retrieved列存在（6S LSR结果）
        if 'rho_retrieved' not in original_data.columns:
            print("Warning: rho_retrieved column not found in merged data")
            # 尝试从其他列推断
            if 'rho_lsr' in original_data.columns:
                original_data['rho_retrieved'] = original_data['rho_lsr']
                print("Created rho_retrieved from rho_lsr")

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
            'ahi_toa': '#1f77b4',  # 蓝色
            'ahi_lsr': '#ff7f0e',  # 橙色
            'ahi_lsr_corrected': '#2ca02c',  # 绿色
            'modis': '#d62728'  # 红色
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
            title += f'\nGrowing Season (April 1-10, 2016)'
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

                # 筛选当前波段的数据
                if 'band' in station_data.columns:
                    station_data_band = station_data[station_data['band'] == band_id].copy()
                else:
                    # 如果没有band列，假设所有数据都是当前波段的
                    station_data_band = station_data.copy()

                if station_data_band.empty:
                    ax.text(0.5, 0.5, f"No data for band {band_id} at {station}",
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Station: {station} (No band {band_id} data)')
                    continue

                # 获取站点信息
                lat, lon, lc_info, availability = self._get_station_info(station)

                # 跟踪绘制了哪些数据
                toa_plotted = False
                lsr_plotted = False
                lsr_corrected_plotted = False
                modis_plotted = False

                # 绘制AHI TOA
                toa_cols_to_try = [
                    f'TOA_Albedo_{band_id}',
                    'rho_toa',
                    'Reflectance_TOA'
                ]

                for toa_col in toa_cols_to_try:
                    if toa_col in station_data_band.columns:
                        valid_mask = station_data_band[toa_col].notna()
                        if valid_mask.any():
                            ax.scatter(station_data_band.loc[valid_mask, 'datetime_bj'],
                                       station_data_band.loc[valid_mask, toa_col],
                                       color=colors['ahi_toa'], marker='o', s=20,
                                       label='AHI TOA', alpha=0.7, edgecolors='none')
                            toa_plotted = True
                            print(f"    Station {station}: Plotted {valid_mask.sum()} TOA points from {toa_col}")
                            break

                # 绘制AHI LSR (6S反演结果) - 使用rho_retrieved
                if 'rho_retrieved' in station_data_band.columns:
                    valid_mask = station_data_band['rho_retrieved'].notna()
                    if valid_mask.any():
                        ax.scatter(station_data_band.loc[valid_mask, 'datetime_bj'],
                                   station_data_band.loc[valid_mask, 'rho_retrieved'],
                                   color=colors['ahi_lsr'], marker='^', s=25,
                                   label='AHI LSR (6S)', alpha=0.7, edgecolors='none')
                        lsr_plotted = True
                        print(f"    Station {station}: Plotted {valid_mask.sum()} 6S LSR points from rho_retrieved")

                # 绘制AHI LSR Corrected
                if 'rho_lsr_corrected' in station_data_band.columns:
                    valid_mask = station_data_band['rho_lsr_corrected'].notna()
                    if valid_mask.any():
                        ax.scatter(station_data_band.loc[valid_mask, 'datetime_bj'],
                                   station_data_band.loc[valid_mask, 'rho_lsr_corrected'],
                                   color=colors['ahi_lsr_corrected'], marker='s', s=20,
                                   label='AHI LSR Corrected', alpha=0.7, edgecolors='none')
                        lsr_corrected_plotted = True
                        print(
                            f"    Station {station}: Plotted {valid_mask.sum()} corrected LSR points from rho_lsr_corrected")

                # 绘制MODIS数据
                if modis_band:
                    modis_col = f'MODIS_{modis_band}'
                    if modis_col in station_data_band.columns:
                        # MODIS是每日数据，需要按天处理
                        daily_data = station_data_band.copy()
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
                            modis_plotted = True
                            print(f"    Station {station}: Plotted {valid_mask.sum()} MODIS points")

                # 如果没有绘制任何数据，显示消息
                if not (toa_plotted or lsr_plotted or lsr_corrected_plotted or modis_plotted):
                    ax.text(0.5, 0.5, f"No valid data for station {station}",
                            ha='center', va='center', transform=ax.transAxes,
                            fontsize=10, color='red')

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

                # 创建图例（只包含实际绘制的数据）
                from matplotlib.patches import Patch
                legend_handles = []
                legend_labels = []

                if toa_plotted:
                    legend_handles.append(Patch(color=colors['ahi_toa']))
                    legend_labels.append('AHI TOA')
                if lsr_plotted:
                    legend_handles.append(Patch(color=colors['ahi_lsr']))
                    legend_labels.append('AHI LSR (6S)')
                if lsr_corrected_plotted:
                    legend_handles.append(Patch(color=colors['ahi_lsr_corrected']))
                    legend_labels.append('AHI LSR Corrected')
                if modis_plotted:
                    legend_handles.append(Patch(color=colors['modis']))
                    legend_labels.append('MODIS')

                if legend_handles:
                    ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=8, framealpha=0.8)

                # 设置x轴标签旋转
                ax.tick_params(axis='x', rotation=45, labelsize=9)
                ax.tick_params(axis='y', labelsize=9)

                # 设置y轴范围（生长季反射率范围）
                ax.set_ylim(-0.02, 0.8)

                # 设置x轴日期格式
                if not station_data_band.empty:
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
        ax1.set_xlabel('Data Availability (9 hours/day)', fontsize=12)
        ax1.set_ylabel('Number of Stations', fontsize=12)
        ax1.set_title('Distribution of Station Data Availability\nGrowing Season (April 1-10, 2016)',
                      fontsize=14, fontweight='bold')
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
        ax1.axvline(0.25, color='orange', linestyle='-', linewidth=1.5, alpha=0.7,
                    label='25% threshold')
        ax1.legend(loc='upper right', fontsize=10)

        # 在直方图上添加文本
        stats_text = f"Total stations: {total_stations}\n"
        stats_text += f"Mean: {mean_avail:.3f}\n"
        stats_text += f"Median: {median_avail:.3f}\n"
        stats_text += f"Std: {std_avail:.3f}\n"
        stats_text += f"Max: {max(availabilities):.3f}"

        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 累计分布图
        sorted_avail = np.sort(availabilities)
        cdf = np.arange(1, len(sorted_avail) + 1) / len(sorted_avail)

        ax2.plot(sorted_avail, cdf, 'b-', linewidth=3, alpha=0.8)
        ax2.set_xlabel('Data Availability (9 hours/day)', fontsize=12)
        ax2.set_ylabel('Cumulative Probability', fontsize=12)
        ax2.set_title('Cumulative Distribution of Station Availability\nGrowing Season',
                      fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')

        # 添加阈值线
        thresholds = [0.25, 0.30, 0.35]
        colors = ['orange', 'red', 'purple']
        labels = ['25% threshold', '30% threshold', '35% threshold']

        for threshold, color, label in zip(thresholds, colors, labels):
            ax2.axvline(threshold, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
            ax2.axhline(0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

        # 计算有多少站点超过各个阈值
        threshold_info = []
        for threshold in thresholds:
            stations_above = sum(1 for a in availabilities if a >= threshold)
            proportion_above = stations_above / total_stations
            threshold_info.append(
                f"≥{threshold * 100:.0f}%: {stations_above}/{total_stations} ({proportion_above * 100:.1f}%)")

        # 在CDF图上添加文本
        cdf_text = '\n'.join(threshold_info)

        ax2.text(0.02, 0.98, cdf_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        # 填充25%以上的区域
        mask_above_25 = sorted_avail >= 0.25
        if any(mask_above_25):
            ax2.fill_between(sorted_avail[mask_above_25], 0, cdf[mask_above_25],
                             alpha=0.3, color='green')

        # 调整布局
        plt.tight_layout()

        # 保存图形
        output_path = Path(output_dir) / 'station_availability_distribution_growing_season.png'
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
        report_lines.append("=" * 80)
        report_lines.append("STATION AVAILABILITY REPORT - GROWING SEASON (April 1-10, 2016)")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Data period: April 1-10, 2016 (9 hours/day observation)")
        report_lines.append(f"Total stations in statistics: {len(self.station_stats)}")
        report_lines.append(f"Filtered stations: {len(filtered_stations)}")
        report_lines.append("")

        if availabilities:
            report_lines.append("Availability statistics for filtered stations (9 hours/day):")
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
        report_lines.append("-" * 100)

        for i, station in enumerate(filtered_stations, 1):
            stats = self.station_stats.get(station, {})
            availability = stats.get('availability', 0)

            # 安全获取并转换数值
            def safe_int(value, default=0):
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return default

            def safe_float(value, default=0.0):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default

            records = safe_int(stats.get('records', 0))
            days = safe_int(stats.get('unique_dates', 0))
            band3 = safe_int(stats.get('band3_available', 0))
            band4 = safe_int(stats.get('band4_available', 0))
            total_possible = safe_int(stats.get('total_possible', 0))
            availability_percent = safe_float(availability) * 100

            report_lines.append(f"{i:4d}. {station:20s} "
                                f"Avail: {availability_percent:6.2f}% | "
                                f"Records: {records:6d} | "
                                f"Days: {days:3d} | "
                                f"B3: {band3:5d} | "
                                f"B4: {band4:5d} | "
                                f"Possible: {total_possible:5d}")

        # 保存报告
        report_path = Path(output_dir) / "station_availability_report_growing_season.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"Availability report saved: {report_path}")
        return report_path

    def generate_validation_summary(self, output_dir, filtered_stations,
                                    total_groups_processed, total_plots_generated,
                                    lsr_data_path):
        """生成验证总结报告"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("VALIDATION SUMMARY REPORT - GROWING SEASON")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Data period: April 1-10, 2016")
        report_lines.append(f"LSR data source: {lsr_data_path}")
        report_lines.append("")
        report_lines.append("PROCESSING STATISTICS:")
        report_lines.append(f"  Total stations filtered: {len(filtered_stations)}")
        report_lines.append(f"  Station groups processed: {total_groups_processed}")
        report_lines.append(f"  Total plots generated: {total_plots_generated}")
        report_lines.append(f"  Output directory: {output_dir}")
        report_lines.append("")
        report_lines.append("DATA CHARACTERISTICS:")
        report_lines.append("  Observation hours: 9 hours/day (H8 satellite)")
        report_lines.append("  Season: Growing season (April)")
        report_lines.append("  Surface conditions: Vegetated surfaces")
        report_lines.append("")
        report_lines.append("PLOT INFORMATION:")
        report_lines.append("  Bands plotted: 03 (0.64µm, Red), 04 (0.86µm, NIR)")
        report_lines.append("  Data types shown: AHI TOA, AHI LSR (6S), AHI LSR Corrected, MODIS")
        report_lines.append("  MODIS bands: Red (band 3), NIR (band 4)")
        report_lines.append("  Stations per plot: 5")
        report_lines.append("")
        report_lines.append("FILE STRUCTURE:")
        report_lines.append("  Time series plots: band{band}_group_{group_number}.png")
        report_lines.append("  Availability distribution: station_availability_distribution_growing_season.png")
        report_lines.append("  Availability report: station_availability_report_growing_season.txt")
        report_lines.append("  This summary: validation_summary_growing_season.txt")

        # 保存报告
        report_path = Path(output_dir) / "validation_summary_growing_season.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"Validation summary saved: {report_path}")
        return report_path

    def run_validation(self, lsr_data_path, model_path=None, output_dir="./validation_results",
                       min_availability=0.25, stations_per_plot=5, max_stations=None):
        """运行验证流程"""
        print("=" * 80)
        print("ENHANCED VALIDATION WORKFLOW V0.7 - GROWING SEASON")
        print("=" * 80)
        print(f"LSR data: {lsr_data_path}")
        print(f"Model: {model_path if model_path else 'Using default model'}")
        print(f"Output directory: {output_dir}")
        print(f"Minimum availability: {min_availability * 100:.0f}% (adjusted for 9 hours/day)")
        print(f"Stations per plot: {stations_per_plot}")
        print(f"Max stations: {max_stations if max_stations else 'All meeting criteria'}")
        print(f"Season: Growing season (April 1-10, 2016)")
        print("=" * 80)

        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 加载模型（如果提供了模型路径）
        if model_path:
            if not self.load_model(model_path):
                print("Warning: Model loading failed, continuing without model prediction")
                self.model = None
        else:
            print("No model path provided, skipping correction prediction")
            print("Will plot: AHI TOA, AHI LSR (6S), and MODIS data")
            self.model = None

        # 2. 尝试加载站点统计信息（从LSR数据同级目录）
        lsr_path = Path(lsr_data_path)

        # 修正路径构建方法
        stats_path_candidates = []

        # 方法1: 使用 with_suffix() 的正确方式（只接受点号开头的后缀）
        stats_path_candidates.append(lsr_path.with_suffix('.station_stats.json'))

        # 方法2: 使用 parent / (stem + suffix) 的方式
        stats_path_candidates.append(lsr_path.parent / f"{lsr_path.stem}_stats.json")

        # 方法3: 在相同目录下查找任何包含 stats 的 JSON 文件
        if lsr_path.parent.exists():
            for file in lsr_path.parent.glob("*stats*.json"):
                stats_path_candidates.append(file)

        # 方法4: 在缓存目录查找
        cache_dir = Path("lsr_cache")
        if cache_dir.exists():
            # 查找与 LSR 文件同名的统计文件
            stats_path_candidates.append(cache_dir / f"{lsr_path.stem}_stats.json")

            # 查找任何包含缓存键的统计文件
            if "lsr_cache" in lsr_path.stem:
                cache_key = lsr_path.stem
                stats_path_candidates.append(cache_dir / f"{cache_key}_stats.json")

            # 查找任何统计文件
            for file in cache_dir.glob("*stats*.json"):
                stats_path_candidates.append(file)

        # 去重
        stats_path_candidates = list(set(stats_path_candidates))

        stats_loaded = False
        for stats_path in stats_path_candidates:
            try:
                if stats_path.exists():
                    print(f"Trying to load stats from: {stats_path}")
                    if self.load_station_stats(str(stats_path)):
                        stats_loaded = True
                        break
            except Exception as e:
                print(f"Error trying to load {stats_path}: {e}")
                continue

        if not stats_loaded:
            print(f"Warning: No station statistics found for {lsr_data_path}")
            print("Will load all stations from LSR data")
            self.station_stats = {}

        # 3. 加载站点坐标
        self.load_station_coords()

        # 4. 绘制可用性分布图
        if self.station_stats:
            print("\nStep 1: Plotting station availability distribution...")
            self.plot_availability_distribution(output_dir)

        # 5. 首先加载LSR数据来获取所有站点
        print("\nStep 2: Loading LSR data to get all stations...")
        lsr_data_sample = self.load_lsr_data(lsr_data_path)
        if lsr_data_sample.empty:
            print("Error: Cannot load LSR data")
            return False

        # 如果没有统计信息，从LSR数据中提取所有站点
        if not self.station_stats:
            print("No station statistics, using all stations from LSR data")
            filtered_stations = lsr_data_sample['station'].unique().tolist()
            print(f"Loaded {len(filtered_stations)} stations from LSR data")
        else:
            # 如果有统计信息，按可用性过滤站点
            print("Filtering stations by availability...")
            filtered_stations = self.filter_stations_by_availability(
                min_availability=min_availability,
                top_n=max_stations
            )

        if not filtered_stations:
            print("Error: No stations meet the criteria")
            return False

        # 6. 生成可用性报告
        if self.station_stats:
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
            lsr_data = self.load_lsr_data(lsr_data_path)
            if lsr_data.empty:
                print(f"  No LSR data for group {group_idx + 1}, skipping...")
                continue

            # 筛选当前组的站点
            lsr_data = lsr_data[lsr_data['station'].isin(station_group)].copy()
            if lsr_data.empty:
                print(f"  No LSR data for stations in group {group_idx + 1}, skipping...")
                continue

            # 检查数据质量
            print(f"  Data quality check for group {group_idx + 1}:")
            print(f"    Total records: {len(lsr_data)}")

            # 检查是否有6S反演结果
            if 'rho_retrieved' in lsr_data.columns:
                valid_6s = lsr_data['rho_retrieved'].notna().sum()
                print(f"    6S LSR records: {valid_6s} ({valid_6s / len(lsr_data) * 100:.1f}%)")
            else:
                print(f"    Warning: No 6S LSR results found in data")

            # 检查TOA数据
            if 'rho_toa' in lsr_data.columns:
                valid_toa = lsr_data['rho_toa'].notna().sum()
                print(f"    TOA records: {valid_toa} ({valid_toa / len(lsr_data) * 100:.1f}%)")

            # 8b. 加载MODIS数据
            lsr_data_with_modis = self.load_modis_data_for_stations(lsr_data)

            # 8c. 如果有模型，进行校正预测
            if self.model is not None:
                print(f"  Applying model correction...")

                # 准备特征
                features_df = self.prepare_features(lsr_data_with_modis)
                if features_df.empty:
                    print(f"  No features for group {group_idx + 1}, skipping...")
                    continue

                # 预测校正值
                predictions = self.predict_correction(features_df)
                if predictions.empty:
                    print(f"  No predictions for group {group_idx + 1}, skipping...")
                    continue

                # 合并结果
                corrected_data = self.merge_results(lsr_data_with_modis, predictions)

                # 检查校正后的数据
                if 'rho_lsr_corrected' in corrected_data.columns:
                    valid_corrected = corrected_data['rho_lsr_corrected'].notna().sum()
                    print(f"    Corrected LSR records: {valid_corrected}")
            else:
                # 没有模型，只使用原始数据
                print(f"  No model available, using original data (TOA + 6S LSR)")
                corrected_data = lsr_data_with_modis.copy()

                # 确保有rho_retrieved列（6S LSR结果）
                if 'rho_retrieved' not in corrected_data.columns:
                    # 尝试从其他列创建
                    if 'rho_lsr' in corrected_data.columns:
                        corrected_data['rho_retrieved'] = corrected_data['rho_lsr']
                        print(f"    Created rho_retrieved from rho_lsr")
                    else:
                        print(f"    Warning: No 6S LSR results found, cannot plot AHI LSR (6S)")

                # 添加校正列（虽然不使用，但为了保持数据结构一致）
                corrected_data['rho_lsr_corrected'] = np.nan
                corrected_data['predicted_correction'] = np.nan

            # 8d. 绘制时间序列图
            print(f"  Plotting time series for group {group_idx + 1}...")
            self.plot_timeseries_for_group(corrected_data, station_group, group_idx, output_dir)

            # 8e. 保存该组的结果
            group_output_path = output_dir / f"results_group_{group_idx + 1:03d}.parquet"
            corrected_data.to_parquet(group_output_path)
            print(f"  Group results saved: {group_output_path}")

            # 更新统计
            total_groups_processed += 1
            total_plots_generated += 2  # 波段3和4

            # 清理内存
            del lsr_data, lsr_data_with_modis, corrected_data
            if self.model is not None:
                del features_df, predictions
            gc.collect()

        # 9. 生成验证总结
        print("\nStep 5: Generating validation summary...")
        self.generate_validation_summary(
            output_dir, filtered_stations,
            total_groups_processed, total_plots_generated,
            lsr_data_path
        )

        print("\n" + "=" * 80)
        print("VALIDATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Season: Growing season (April 1-10, 2016)")
        print(f"Station groups processed: {total_groups_processed}")
        print(f"Total plots generated: {total_plots_generated}")
        print(f"Output directory: {output_dir}")
        print("=" * 80)

        return True


# ==================== Main Function ====================
def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Validation Workflow V0.7 - Growing Season (April 1-10, 2016)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # 使用默认设置运行验证（生长季数据）
  python enhanced_validation_v0.7.py --lsr_data ./ahi_lsr_20160401_20160410.parquet --output_dir ./validation_results

  # 使用自定义模型（绝对路径）
  python enhanced_validation_v0.7.py --lsr_data ./ahi_lsr_20160401_20160410.parquet --model_path "D:/6S_Self_Consistency_Zenith_Angles_v0.4/models/refactored/model_training_20260117_152723/models/XGBoost_model.pkl"

  # 使用模型目录（自动查找）
  python enhanced_validation_v0.7.py --lsr_data ./ahi_lsr_20160401_20160410.parquet --model_path "./models/"

  # 提高可用性阈值到30%
  python enhanced_validation_v0.7.py --lsr_data ./ahi_lsr_20160401_20160410.parquet --min_availability 0.30

  # 只处理前100个高可用性站点，每3个站点一张图
  python enhanced_validation_v0.7.py --lsr_data ./ahi_lsr_20160401_20160410.parquet --max_stations 100 --stations_per_plot 3

  # 使用缓存中的LSR数据
  python enhanced_validation_v0.7.py --lsr_data ./lsr_cache/lsr_cache_20160401_20160410_*.parquet
        """
    )

    parser.add_argument('--lsr_data', type=str, required=True,
                        help='Path to LSR data file (required)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model or directory containing models (optional)')
    parser.add_argument('--output_dir', type=str, default='./validation_results',
                        help='Output directory for results')
    parser.add_argument('--min_availability', type=float, default=0.25,
                        help='Minimum data availability threshold (default: 0.25 for 9 hours/day)')
    parser.add_argument('--stations_per_plot', type=int, default=5,
                        help='Number of stations per plot (default: 5)')
    parser.add_argument('--max_stations', type=int, default=None,
                        help='Maximum number of stations to process (default: all meeting criteria)')

    args = parser.parse_args()

    # 创建验证器
    config = Config()
    validator = EnhancedValidatorV0_7(config)

    # 运行验证
    success = validator.run_validation(
        lsr_data_path=args.lsr_data,
        model_path=args.model_path,
        output_dir=args.output_dir,
        min_availability=args.min_availability,
        stations_per_plot=args.stations_per_plot,
        max_stations=args.max_stations
    )

    if success:
        print("\nGrowing season validation completed successfully!")
        return 0
    else:
        print("\nValidation failed!")
        return 1


if __name__ == "__main__":
    main()