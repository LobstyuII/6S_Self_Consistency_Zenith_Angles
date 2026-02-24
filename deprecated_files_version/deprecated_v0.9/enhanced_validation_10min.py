# ==================== enhanced_validation_10min.py ====================
"""
增强版验证工作流 - 10分钟分辨率版本
针对10分钟分辨率数据优化，适配新的波段分开存储格式
"""

import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import warnings
from tqdm import tqdm
import json
import seaborn as sns
import gc
import glob
import traceback

warnings.filterwarnings('ignore')


# ==================== Configuration ====================
class Config:
    """验证配置类 - 10分钟版本"""

    # Data paths
    DATA_PATHS = {
        "h8l1_10min": "D:/H8_data/H8L1_10min/",
        "h8l2_10min": "D:/H8_data/H8L2ARP_10min/",
        "merra2_10min": "D:/H8_data/MERRA2_10min/combined/",
        "lucc": "D:/H8_data/LC_2015_2024.nc",
        "luts": "D:/H8_data/LUTs.nc",
        "mod_red": "D:/H8_Data/MODIS_Red_nadir/",
        "mod_nir": "D:/H8_Data/MODIS_NIR_nadir/",
        "output": "./validation"
    }

    # Band configuration - 包含所有6个波段
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
class EnhancedValidator10min:
    """增强版验证器 - 10分钟分辨率版本"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.station_stats = None
        self.station_coords = None
        self.model_loaded = False
        self.feature_check_passed = False

    def load_model(self, model_path, verbose=True):
        """加载机器学习模型 - 增强版，增加更多检查和诊断"""
        if verbose:
            print(f"\n🔍 加载模型: {model_path}")

        self.model_loaded = False
        model_files_tried = []

        try:
            # 尝试多种可能的路径格式
            possible_paths = [
                Path(model_path),
                Path.cwd() / model_path,
                Path.cwd() / "models" / model_path,
                Path.home() / model_path,
                Path("D:/6S_Self_Consistency_Zenith_Angles_v0.4/models") / model_path,
                Path("D:/6S_Self_Consistency_Zenith_Angles_v0.4/models/refactored") / model_path,
                Path("D:/6S_Self_Consistency_Zenith_Angles_v0.4/models") / "XGBoost_model.pkl",
                Path("D:/6S_Self_Consistency_Zenith_Angles_v0.4/models/refactored") / "XGBoost_model.pkl"
            ]

            # 如果路径是目录，查找里面的模型文件
            if os.path.isdir(model_path):
                model_dir = Path(model_path)
                possible_paths.extend([
                    model_dir / "XGBoost_model.pkl",
                    model_dir / "RandomForest_model.pkl",
                    model_dir / "models" / "XGBoost_model.pkl",
                    model_dir / "best_model.pkl"
                ])

            # 去重
            possible_paths = list(set([str(p) for p in possible_paths if p]))

            for path_str in possible_paths:
                try:
                    if os.path.exists(path_str):
                        if verbose:
                            print(f"  尝试: {path_str}")
                        model_files_tried.append(path_str)

                        with open(path_str, 'rb') as f:
                            self.model = pickle.load(f)

                        # 检查模型类型和属性
                        model_type = type(self.model).__name__
                        if verbose:
                            print(f"  ✅ 成功加载 {model_type} 模型")
                            print(f"     文件: {path_str}")

                        # 检查模型是否具有predict方法
                        if hasattr(self.model, 'predict'):
                            # 测试模型是否可以工作
                            test_features = np.zeros((1, len(self.config.EXPECTED_FEATURES)))
                            try:
                                test_pred = self.model.predict(test_features)
                                if verbose:
                                    print(f"  ✅ 模型预测测试通过")
                                    print(f"     期望特征数: {len(self.config.EXPECTED_FEATURES)}")
                                    print(f"     模型特征数: {test_features.shape[1]}")
                            except Exception as e:
                                if verbose:
                                    print(f"  ⚠️ 模型预测测试失败: {e}")

                        self.model_loaded = True
                        return True

                except Exception as e:
                    if verbose:
                        print(f"  ❌ 加载失败: {e}")
                    continue

            # 如果上面没找到，尝试在常见目录搜索
            if not self.model_loaded:
                if verbose:
                    print(f"\n🔍 在常见目录搜索模型文件...")

                search_dirs = [
                    Path("../.."),
                    Path("./models"),
                    Path("D:/6S_Self_Consistency_Zenith_Angles_v0.4/models"),
                    Path("D:/6S_Self_Consistency_Zenith_Angles_v0.4/models/refactored"),
                    Path.home() / "models"
                ]

                for search_dir in search_dirs:
                    if search_dir.exists():
                        for root, dirs, files in os.walk(search_dir):
                            for file in files:
                                if file.endswith(('.pkl', '.pickle', '.joblib')):
                                    model_file = Path(root) / file
                                    try:
                                        model_files_tried.append(str(model_file))
                                        with open(model_file, 'rb') as f:
                                            self.model = pickle.load(f)

                                        if hasattr(self.model, 'predict'):
                                            model_type = type(self.model).__name__
                                            if verbose:
                                                print(f"  ✅ 自动发现并加载 {model_type} 模型")
                                                print(f"     文件: {model_file}")

                                            # 测试模型
                                            test_features = np.zeros((1, len(self.config.EXPECTED_FEATURES)))
                                            try:
                                                test_pred = self.model.predict(test_features)
                                                if verbose:
                                                    print(f"  ✅ 模型预测测试通过")
                                            except Exception as e:
                                                if verbose:
                                                    print(f"  ⚠️ 模型预测测试失败: {e}")

                                            self.model_loaded = True
                                            return True

                                    except Exception as e:
                                        continue

            if not self.model_loaded:
                print(f"\n❌ 错误: 无法加载任何模型文件")
                print(f"尝试过的文件:")
                for f in model_files_tried:
                    print(f"  - {f}")
                print(f"\n请确保:")
                print(f"1. 模型文件存在且可读")
                print(f"2. 文件是有效的pickle格式")
                print(f"3. 包含有效的scikit-learn/XGBoost模型")
                return False

        except Exception as e:
            print(f"❌ 模型加载过程出错: {e}")
            traceback.print_exc()
            return False

    def validate_model_features(self, features_df):
        """验证特征是否与模型期望匹配"""
        print(f"\n🔍 验证特征匹配...")

        if self.model is None:
            print("  ❌ 错误: 模型未加载")
            return False

        # 检查特征列
        expected_features = self.config.EXPECTED_FEATURES
        actual_features = list(features_df.columns)

        print(f"  期望特征数: {len(expected_features)}")
        print(f"  实际特征数: {len(actual_features)}")

        # 检查缺失的特征
        missing_features = [feat for feat in expected_features if feat not in actual_features]
        if missing_features:
            print(f"  ❌ 缺失特征: {missing_features}")

            # 尝试计算缺失特征
            for feat in missing_features:
                if feat not in features_df.columns:
                    print(f"    创建默认值: {feat} = 0.0")
                    features_df[feat] = 0.0

        # 检查多余的特征
        extra_features = [feat for feat in actual_features if feat not in expected_features]
        if extra_features:
            print(f"  ⚠️ 多余特征: {extra_features}")

        # 确保特征顺序一致
        features_reordered = features_df[expected_features].copy()

        # 测试预测
        try:
            test_pred = self.model.predict(features_reordered.head(1))
            print(f"  ✅ 特征验证通过")
            print(f"     测试预测值: {test_pred[0]:.6f}")
            self.feature_check_passed = True
            return True
        except Exception as e:
            print(f"  ❌ 特征验证失败: {e}")
            print(f"     特征形状: {features_reordered.shape}")
            print(f"     特征列: {list(features_reordered.columns)}")
            return False

    def load_station_stats(self, stats_path):
        """加载站点统计信息"""
        print(f"Loading station statistics: {stats_path}")
        try:
            if not os.path.exists(stats_path):
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

    def filter_stations_by_availability(self, min_availability=0.10, top_n=None):
        """按可用性过滤站点（针对10分钟分辨率的调整）"""
        if self.station_stats is None:
            print("Error: No station statistics loaded")
            return []

        print(f"\nFiltering stations by availability (≥{min_availability * 100:.0f}%, 10min resolution)...")

        filtered_stations = []
        for station, stats in self.station_stats.items():
            if 'availability' in stats:
                try:
                    availability = float(stats['availability'])
                except (ValueError, TypeError):
                    continue

                if availability >= min_availability:
                    filtered_stations.append((station, availability))

        filtered_stations.sort(key=lambda x: x[1], reverse=True)

        if top_n is not None and top_n > 0:
            filtered_stations = filtered_stations[:top_n]

        print(f"  Total stations in stats: {len(self.station_stats)}")
        print(f"  Stations meeting criteria: {len(filtered_stations)}")

        if len(filtered_stations) > 0:
            print("\nTop 10 stations by availability:")
            for i, (station, avail) in enumerate(filtered_stations[:10]):
                stats = self.station_stats.get(station, {})
                records = int(stats.get('records', 0))
                days = int(stats.get('unique_dates', 0))
                print(f"  {i + 1:2d}. {station:20s}: {avail * 100:5.1f}% "
                      f"({records} records, {days} days)")

        return [station for station, avail in filtered_stations]

    def load_lsr_data(self, lsr_data_path, bands_to_load=None):
        """加载LSR数据 - 适配新的波段分开存储格式"""
        print(f"\nLoading LSR data from: {lsr_data_path}")

        if bands_to_load is None:
            bands_to_load = list(self.config.BAND_CONFIG.keys())

        print(f"Bands to load: {bands_to_load}")

        lsr_path = Path(lsr_data_path)
        all_lsr_data = []

        if '*' in str(lsr_path):
            files = glob.glob(str(lsr_path))
            for file in files:
                file_path = Path(file)
                band_found = None
                for band in bands_to_load:
                    if f'band{band}' in file_path.name.lower():
                        band_found = band
                        break

                if band_found:
                    print(f"Loading band {band_found} from: {file_path}")
                    band_data = self._load_single_lsr_file(file_path, band_found)
                    if not band_data.empty:
                        all_lsr_data.append(band_data)

        elif lsr_path.is_dir():
            print(f"Searching for band files in directory: {lsr_path}")
            for band in bands_to_load:
                band_files = list(lsr_path.glob(f"*band{band}*.parquet"))
                if not band_files:
                    band_files = list(lsr_path.glob(f"**/*band{band}*.parquet"))

                if band_files:
                    print(f"Found {len(band_files)} files for band {band}")
                    band_data = self._load_single_lsr_file(band_files[0], band)
                    if not band_data.empty:
                        all_lsr_data.append(band_data)
                else:
                    print(f"Warning: No files found for band {band}")

        elif lsr_path.exists() and lsr_path.is_file():
            print(f"Loading single file: {lsr_path}")
            band_found = None
            for band in bands_to_load:
                if f'band{band}' in lsr_path.name.lower():
                    band_found = band
                    break

            if band_found:
                band_data = self._load_single_lsr_file(lsr_path, band_found)
                if not band_data.empty:
                    all_lsr_data.append(band_data)
            else:
                print("File doesn't contain band info, loading as single band data")
                band_data = self._load_single_lsr_file(lsr_path, 'unknown')
                if not band_data.empty:
                    all_lsr_data.append(band_data)
        else:
            print(f"Error: LSR data path not found: {lsr_data_path}")
            return pd.DataFrame()

        if all_lsr_data:
            combined_data = pd.concat(all_lsr_data, ignore_index=True)
            print(f"\n📊 LSR数据统计:")
            print(f"  总记录数: {len(combined_data)}")
            print(f"  波段: {combined_data['band'].unique().tolist()}")
            print(f"  时间范围: {combined_data['datetime_bj'].min()} 到 {combined_data['datetime_bj'].max()}")
            print(f"  分辨率: 10分钟")

            # 检查关键列
            required_cols = ['rho_toa', 'rho_retrieved', 'sza', 'vza', 'raa']
            for col in required_cols:
                if col in combined_data.columns:
                    non_null = combined_data[col].notna().sum()
                    print(f"  {col}: {non_null} 个有效值")
                else:
                    print(f"  ⚠️ 缺失列: {col}")

            return combined_data
        else:
            print("Error: No LSR data loaded")
            return pd.DataFrame()

    def _load_single_lsr_file(self, file_path, band_id):
        """加载单个LSR文件"""
        try:
            if file_path.suffix == '.parquet':
                lsr_data = pd.read_parquet(file_path)
            elif file_path.suffix == '.nc':
                import xarray as xr
                ds = xr.open_dataset(file_path)
                lsr_data = ds.to_dataframe().reset_index()
                ds.close()
            elif file_path.suffix == '.csv':
                lsr_data = pd.read_csv(file_path)
            else:
                print(f"Unsupported file format: {file_path.suffix}")
                return pd.DataFrame()

            if 'band' not in lsr_data.columns:
                lsr_data['band'] = band_id

            # 确保datetime_bj列存在
            if 'datetime_bj' not in lsr_data.columns and 'datetime_utc' in lsr_data.columns:
                lsr_data['datetime_bj'] = lsr_data['datetime_utc'] + pd.Timedelta(hours=8)

            # 确保有rho_retrieved列（6S反演结果）
            if 'rho_retrieved' not in lsr_data.columns:
                if 'rho_lsr' in lsr_data.columns:
                    lsr_data['rho_retrieved'] = lsr_data['rho_lsr']
                elif 'surface_reflectance' in lsr_data.columns:
                    lsr_data['rho_retrieved'] = lsr_data['surface_reflectance']
                else:
                    print(f"  ⚠️ 警告: 没有找到反演反射率列，使用默认值")
                    lsr_data['rho_retrieved'] = lsr_data.get('rho_toa', 0.2) * 0.8

            # 确保大气参数存在
            for param in ['aod550', 'h2o', 'o3']:
                if param not in lsr_data.columns:
                    if param == 'aod550':
                        lsr_data[param] = 0.1
                    elif param == 'h2o':
                        lsr_data[param] = 2.0
                    elif param == 'o3':
                        lsr_data[param] = 0.3

            print(f"  Loaded {len(lsr_data)} records for band {band_id}")
            return lsr_data

        except Exception as e:
            print(f"Error loading LSR file {file_path}: {e}")
            return pd.DataFrame()

    def prepare_features(self, lsr_data):
        """为模型预测准备特征 - 增强版，确保与训练时完全一致"""
        print(f"\n🔧 准备模型特征...")

        if lsr_data.empty:
            print("  ❌ 错误: LSR数据为空")
            return pd.DataFrame()

        all_features = []

        for band_id in self.config.BAND_CONFIG.keys():
            band_mask = lsr_data['band'] == band_id
            band_data = lsr_data[band_mask].copy()

            if band_data.empty:
                continue

            print(f"  处理波段 {band_id}: {len(band_data)} 条记录")

            # 创建特征DataFrame
            features_df = pd.DataFrame(index=band_data.index)

            # 1. 基础特征
            features_df['sza'] = band_data['sza']
            features_df['vza'] = band_data['vza']
            features_df['raa'] = band_data['raa']
            features_df['aod550'] = band_data.get('aod550', 0.1)
            features_df['h2o'] = band_data.get('h2o', 2.0)
            features_df['o3'] = band_data.get('o3', 0.3)
            features_df['wavelength'] = self.config.BAND_CONFIG[band_id]['wavelength']
            features_df['rho_toa'] = band_data['rho_toa']
            features_df['rho_retrieved'] = band_data['rho_retrieved']

            # 2. 计算派生特征（必须与训练时完全一致）
            sza_rad = np.radians(features_df['sza'])
            vza_rad = np.radians(features_df['vza'])
            raa_rad = np.radians(features_df['raa'])

            # 三角函数
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

            # 大气质量因子
            cos_sza_safe = np.clip(features_df['cos_sza'], 0.001, 1.0)
            cos_vza_safe = np.clip(features_df['cos_vza'], 0.001, 1.0)
            features_df['airmass_sza'] = 1.0 / cos_sza_safe
            features_df['airmass_vza'] = 1.0 / cos_vza_safe
            features_df['total_airmass'] = features_df['airmass_sza'] + features_df['airmass_vza']

            # 角度关系
            features_df['vza_sza_ratio'] = features_df['vza'] / (features_df['sza'] + 1e-6)
            features_df['vza_minus_sza'] = features_df['vza'] - features_df['sza']
            features_df['vza_plus_sza'] = features_df['vza'] + features_df['sza']

            # 交互特征
            features_df['aod_airmass'] = features_df['aod550'] * features_df['total_airmass']
            features_df['aod_wavelength'] = features_df['aod550'] / features_df['wavelength']
            features_df['rho_ratio'] = features_df['rho_toa'] / (features_df['rho_retrieved'] + 1e-6)
            features_df['rho_diff'] = features_df['rho_toa'] - features_df['rho_retrieved']
            features_df['rho_product'] = features_df['rho_toa'] * features_df['rho_retrieved']
            features_df['wavelength_cos_sza'] = features_df['wavelength'] * features_df['cos_sza']
            features_df['wavelength_cos_vza'] = features_df['wavelength'] * features_df['cos_vza']

            # 3. 添加band和station信息用于后续合并
            features_df['band'] = band_id
            features_df['station'] = band_data['station']
            if 'datetime_bj' in band_data.columns:
                features_df['datetime_bj'] = band_data['datetime_bj']

            all_features.append(features_df)

        if not all_features:
            print("  ❌ 错误: 没有生成任何特征")
            return pd.DataFrame()

        # 合并所有特征
        combined_features = pd.concat(all_features, ignore_index=False)
        print(f"  总特征数: {len(combined_features)}")
        print(f"  特征列数: {combined_features.shape[1]}")

        # 验证特征完整性
        expected_features = self.config.EXPECTED_FEATURES
        missing_features = [f for f in expected_features if f not in combined_features.columns]

        if missing_features:
            print(f"  ⚠️ 警告: 缺失特征 {len(missing_features)} 个")
            for feat in missing_features[:5]:  # 只显示前5个
                print(f"    - {feat}")

            # 创建缺失特征
            for feat in missing_features:
                if feat not in combined_features.columns:
                    combined_features[feat] = 0.0
                    print(f"    创建默认值: {feat} = 0.0")

        # 确保特征顺序一致
        final_features = combined_features.copy()
        for feat in expected_features:
            if feat not in final_features.columns:
                final_features[feat] = 0.0

        # 重新排序特征
        final_features = final_features[expected_features + ['band', 'station', 'datetime_bj']]

        print(f"  ✅ 特征准备完成")
        print(f"     最终特征形状: {final_features.shape}")
        print(f"     特征列: {list(final_features.columns[:10])}...")

        return final_features

    def predict_correction(self, features_df):
        """预测校正值 - 增强版，增加更多诊断信息"""
        if self.model is None:
            print("❌ 错误: 模型未加载")
            return pd.DataFrame()

        if not self.model_loaded:
            print("❌ 错误: 模型加载失败")
            return pd.DataFrame()

        print(f"\n🤖 开始模型预测...")
        print(f"   输入数据形状: {features_df.shape}")

        # 分离特征和元数据
        expected_features = self.config.EXPECTED_FEATURES
        meta_cols = ['band', 'station', 'datetime_bj']

        # 检查特征列
        missing_features = [f for f in expected_features if f not in features_df.columns]
        if missing_features:
            print(f"  ❌ 错误: 缺失特征 {len(missing_features)} 个")
            for feat in missing_features[:5]:
                print(f"    - {feat}")
            return pd.DataFrame()

        # 提取特征矩阵
        X = features_df[expected_features].values
        print(f"   特征矩阵形状: {X.shape}")
        print(f"   特征范围: min={X.min():.4f}, max={X.max():.4f}, mean={X.mean():.4f}")

        try:
            # 进行预测
            print(f"   进行预测...")
            predictions = self.model.predict(X)
            print(f"   预测完成")
            print(f"   预测值形状: {predictions.shape}")
            print(f"   预测值统计: min={predictions.min():.6f}, max={predictions.max():.6f}, "
                  f"mean={predictions.mean():.6f}, std={predictions.std():.6f}")

            # 创建结果DataFrame
            results_df = pd.DataFrame({
                'original_index': features_df.index,
                'band': features_df['band'],
                'station': features_df['station'],
                'datetime_bj': features_df['datetime_bj'],
                'rho_toa': features_df['rho_toa'] if 'rho_toa' in features_df.columns else 0.0,
                'rho_retrieved': features_df['rho_retrieved'] if 'rho_retrieved' in features_df.columns else 0.0,
                'predicted_correction': predictions
            })

            # 计算校正后的LSR
            results_df['rho_lsr_corrected'] = results_df['rho_retrieved'] - results_df['predicted_correction']

            # 限制在合理范围
            results_df['rho_lsr_corrected'] = results_df['rho_lsr_corrected'].clip(0, 1)

            print(f"\n📊 校正结果统计:")
            print(f"   原始LSR (6S): mean={results_df['rho_retrieved'].mean():.6f}, "
                  f"std={results_df['rho_retrieved'].std():.6f}")
            print(f"   校正量: mean={results_df['predicted_correction'].mean():.6f}, "
                  f"std={results_df['predicted_correction'].std():.6f}")
            print(f"   校正后LSR: mean={results_df['rho_lsr_corrected'].mean():.6f}, "
                  f"std={results_df['rho_lsr_corrected'].std():.6f}")

            # 检查校正效果
            correction_magnitude = np.abs(results_df['predicted_correction']).mean()
            if correction_magnitude < 0.0001:
                print(f"  ⚠️ 警告: 校正量非常小 ({correction_magnitude:.6f})，可能模型未正确工作")

            return results_df

        except Exception as e:
            print(f"❌ 预测失败: {e}")
            print(f"   特征矩阵形状: {X.shape}")
            print(f"   特征列: {expected_features}")
            traceback.print_exc()
            return pd.DataFrame()

    def merge_results(self, original_data, predictions):
        """合并原始数据和预测结果"""
        print(f"\n🔄 合并结果...")

        if predictions.empty:
            print("  ⚠️ 警告: 预测结果为空，使用原始数据")
            original_data['predicted_correction'] = 0.0
            original_data['rho_lsr_corrected'] = original_data['rho_retrieved']
            return original_data

        # 重置索引以确保对齐
        original_data = original_data.reset_index(drop=True)

        # 创建合并后的DataFrame
        merged_data = original_data.copy()

        # 初始化校正列
        merged_data['predicted_correction'] = np.nan
        merged_data['rho_lsr_corrected'] = np.nan

        # 根据索引合并预测结果
        for idx, pred_row in predictions.iterrows():
            original_idx = pred_row['original_index']

            if original_idx in merged_data.index:
                # 验证行匹配
                if ('band' in merged_data.columns and 'station' in merged_data.columns and
                        'band' in pred_row and 'station' in pred_row):

                    if (merged_data.at[original_idx, 'band'] == pred_row['band'] and
                            merged_data.at[original_idx, 'station'] == pred_row['station']):
                        merged_data.at[original_idx, 'predicted_correction'] = pred_row['predicted_correction']
                        merged_data.at[original_idx, 'rho_lsr_corrected'] = pred_row['rho_lsr_corrected']

        # 处理未匹配的行
        nan_correction = merged_data['predicted_correction'].isna().sum()
        if nan_correction > 0:
            print(f"  ⚠️ 警告: {nan_correction} 行没有校正值，使用原始值")
            mask = merged_data['predicted_correction'].isna()
            merged_data.loc[mask, 'predicted_correction'] = 0.0
            merged_data.loc[mask, 'rho_lsr_corrected'] = merged_data.loc[mask, 'rho_retrieved']

        print(f"  合并完成:")
        print(f"    总行数: {len(merged_data)}")
        print(f"    有效校正: {merged_data['predicted_correction'].notna().sum()}")
        print(f"    校正后LSR范围: {merged_data['rho_lsr_corrected'].min():.4f} 到 "
              f"{merged_data['rho_lsr_corrected'].max():.4f}")

        return merged_data

    def plot_timeseries_for_group(self, corrected_data, station_group, group_idx, output_dir,
                                  is_representative=False, group_type="all", bands_to_plot=None):
        """为站点组绘制时间序列图 - 增强版，确保显示所有数据"""
        import matplotlib.pyplot as plt

        if bands_to_plot is None:
            bands_in_data = corrected_data['band'].unique().tolist()
            bands_to_plot = [band for band in bands_in_data if band in self.config.BAND_CONFIG]

        if not bands_to_plot:
            print(f"Warning: No valid bands to plot")
            return

        print(f"  绘制波段: {bands_to_plot}")

        colors = {
            'ahi_toa': '#1f77b4',  # 蓝色
            'ahi_lsr': '#ff7f0e',  # 橙色
            'ahi_lsr_corrected': '#2ca02c',  # 绿色
            'modis': '#d62728'  # 红色
        }

        plt.style.use('seaborn-v0_8-whitegrid')

        for band_id in bands_to_plot:
            if band_id not in self.config.BAND_CONFIG:
                continue

            wavelength = self.config.BAND_CONFIG[band_id]['wavelength']
            modis_band = self.config.BAND_CONFIG[band_id]['modis_band']

            n_stations = len(station_group)
            fig, axes = plt.subplots(n_stations, 1, figsize=(18, 4 * n_stations))
            if n_stations == 1:
                axes = [axes]

            title_prefix = f'REPRESENTATIVE STATIONS - ' if is_representative else f'Group {group_idx + 1} - '
            title = f'{title_prefix}Band {band_id} ({wavelength}µm)'
            if not is_representative:
                title += f' ({n_stations} stations)'

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

                station_data = station_data.sort_values('datetime_bj')

                # 筛选当前波段的数据
                station_data_band = station_data[station_data['band'] == band_id].copy()

                if station_data_band.empty:
                    ax.text(0.5, 0.5, f"No data for band {band_id} at {station}",
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Station: {station} (No band {band_id} data)')
                    continue

                # 检查数据可用性
                data_available = {
                    'toa': 'rho_toa' in station_data_band.columns and station_data_band['rho_toa'].notna().any(),
                    'lsr': 'rho_retrieved' in station_data_band.columns and station_data_band[
                        'rho_retrieved'].notna().any(),
                    'corrected': 'rho_lsr_corrected' in station_data_band.columns and station_data_band[
                        'rho_lsr_corrected'].notna().any(),
                    'modis': False
                }

                print(f"    Station {station}, Band {band_id}:")
                print(f"      TOA数据: {data_available['toa']}")
                print(f"      LSR数据: {data_available['lsr']}")
                print(f"      校正数据: {data_available['corrected']}")

                # 绘制AHI TOA
                if data_available['toa']:
                    valid_mask = station_data_band['rho_toa'].notna()
                    ax.scatter(station_data_band.loc[valid_mask, 'datetime_bj'],
                               station_data_band.loc[valid_mask, 'rho_toa'],
                               color=colors['ahi_toa'], marker='.', s=10,
                               label='AHI TOA', alpha=0.6, edgecolors='none')

                # 绘制AHI LSR (6S反演结果)
                if data_available['lsr']:
                    valid_mask = station_data_band['rho_retrieved'].notna()
                    ax.scatter(station_data_band.loc[valid_mask, 'datetime_bj'],
                               station_data_band.loc[valid_mask, 'rho_retrieved'],
                               color=colors['ahi_lsr'], marker='^', s=15,
                               label='AHI LSR (6S)', alpha=0.7, edgecolors='none')

                # 绘制AHI LSR Corrected - 关键检查点
                if data_available['corrected']:
                    valid_mask = station_data_band['rho_lsr_corrected'].notna()
                    if valid_mask.any():
                        corrected_values = station_data_band.loc[valid_mask, 'rho_lsr_corrected']
                        ax.scatter(station_data_band.loc[valid_mask, 'datetime_bj'],
                                   corrected_values,
                                   color=colors['ahi_lsr_corrected'], marker='s', s=12,
                                   label='AHI LSR Corrected', alpha=0.7, edgecolors='none')

                        # 检查校正是否有效
                        if 'rho_retrieved' in station_data_band.columns:
                            original_values = station_data_band.loc[valid_mask, 'rho_retrieved']
                            diff = (corrected_values - original_values).abs().mean()
                            print(f"      校正差异: mean_abs_diff={diff:.6f}")

                # 绘制MODIS数据（只针对波段3和4）
                if band_id in ['03', '04'] and modis_band:
                    modis_col = f'MODIS_{modis_band}'
                    if modis_col in station_data_band.columns:
                        daily_data = station_data_band.copy()
                        daily_data['date'] = daily_data['datetime_bj'].dt.date
                        daily_avg = daily_data.groupby('date')[modis_col].mean().reset_index()
                        daily_avg['datetime'] = pd.to_datetime(daily_avg['date'])

                        valid_mask = daily_avg[modis_col].notna()
                        if valid_mask.any():
                            ax.scatter(daily_avg.loc[valid_mask, 'datetime'],
                                       daily_avg.loc[valid_mask, modis_col],
                                       color=colors['modis'], marker='D', s=40,
                                       label='MODIS', alpha=0.8, edgecolors='black', linewidth=0.5)

                # 如果没有绘制任何数据
                plotted_any = data_available['toa'] or data_available['lsr'] or data_available['corrected']
                if not plotted_any:
                    ax.text(0.5, 0.5, f"No valid data for station {station}",
                            ha='center', va='center', transform=ax.transAxes,
                            fontsize=10, color='red')

                # 设置标题和标签
                ax.set_ylabel('Reflectance', fontsize=10)

                station_title = f'Station: {station} | '
                lat, lon, lc_info, availability = self._get_station_info(station)

                if not np.isnan(lat) and not np.isnan(lon):
                    station_title += f'Lat: {lat:.2f}°, Lon: {lon:.2f}° | '

                station_title += f'Points: {len(station_data_band)}'
                if data_available['corrected']:
                    station_title += ' | ✓ Corrected'

                ax.set_title(station_title, fontsize=11, pad=10, fontweight='bold')

                # 设置网格和图例
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

                # 创建图例
                from matplotlib.patches import Patch
                legend_handles = []
                legend_labels = []

                if data_available['toa']:
                    legend_handles.append(Patch(color=colors['ahi_toa']))
                    legend_labels.append('AHI TOA (10min)')
                if data_available['lsr']:
                    legend_handles.append(Patch(color=colors['ahi_lsr']))
                    legend_labels.append('AHI LSR (6S)')
                if data_available['corrected']:
                    legend_handles.append(Patch(color=colors['ahi_lsr_corrected']))
                    legend_labels.append('AHI LSR Corrected')
                if band_id in ['03', '04'] and modis_band and modis_col in station_data_band.columns:
                    legend_handles.append(Patch(color=colors['modis']))
                    legend_labels.append('MODIS (daily)')

                if legend_handles:
                    ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=8, framealpha=0.8)

                # 设置坐标轴
                ax.tick_params(axis='x', rotation=45, labelsize=9)
                ax.tick_params(axis='y', labelsize=9)

                # 设置y轴范围
                if band_id in ['01', '02', '03', '04']:
                    ax.set_ylim(-0.02, 0.8)
                else:
                    ax.set_ylim(-0.02, 0.4)

                # 设置日期格式
                if not station_data_band.empty:
                    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:%M'))

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # 保存图形
            if is_representative:
                output_path = Path(output_dir) / f'representative_band{band_id}.png'
            else:
                output_path = Path(output_dir) / f'band{band_id}_group_{group_idx + 1:03d}.png'

            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            print(f"  保存图形: {output_path}")
            gc.collect()

    def _get_station_info(self, station):
        """获取站点信息：经纬度、LC类型、可用性"""
        lat, lon = np.nan, np.nan
        if self.station_coords is not None and not self.station_coords.empty:
            station_info = self.station_coords[self.station_coords['station'] == station]
            if not station_info.empty:
                lat = station_info['lat'].values[0]
                lon = station_info['lon'].values[0]

        lc_info = "N/A"
        availability = "N/A"
        if self.station_stats and station in self.station_stats:
            stats = self.station_stats[station]
            if 'availability' in stats:
                availability = f"{stats['availability'] * 100:.1f}%"

        return lat, lon, lc_info, availability

    def select_representative_stations(self, lsr_data, top_n=10):
        """选择数据量最多的前N个代表性站点"""
        print(f"\n选择前 {top_n} 个代表性站点（按数据量）...")

        if lsr_data.empty:
            return []

        station_counts = lsr_data.groupby('station').size().reset_index(name='count')
        station_counts = station_counts.sort_values('count', ascending=False)

        representative_stations = station_counts.head(top_n)['station'].tolist()

        print(f"前 {top_n} 个代表性站点:")
        for i, station in enumerate(representative_stations, 1):
            count = station_counts[station_counts['station'] == station]['count'].values[0]
            availability = "N/A"
            if self.station_stats and station in self.station_stats:
                stats = self.station_stats[station]
                if 'availability' in stats:
                    availability = f"{stats['availability'] * 100:.1f}%"

            print(f"  {i:2d}. {station:20s}: {count:6d} 条记录, 可用性: {availability}")

        return representative_stations

    def run_validation(self, lsr_data_path, model_path=None, output_dir="./validation_results_10min",
                       min_availability=0.10, stations_per_plot=3, max_stations=None,
                       bands_to_process=None):
        """运行验证流程 - 10分钟版本（增强诊断）"""
        print("=" * 80)
        print("🔬 增强版验证工作流 - 10分钟分辨率（带诊断）")
        print("=" * 80)
        print(f"LSR数据: {lsr_data_path}")
        print(f"模型: {model_path if model_path else '无模型（仅绘图）'}")
        print(f"输出目录: {output_dir}")
        print(f"最小可用性: {min_availability * 100:.0f}%")
        print(f"每图站点数: {stations_per_plot}")
        print(f"最大站点数: {max_stations if max_stations else '全部'}")
        print(f"处理波段: {bands_to_process if bands_to_process else '全部'}")
        print("=" * 80)

        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        representative_dir = output_dir / "representative_stations"
        representative_dir.mkdir(exist_ok=True)

        # 1. 加载模型（如果有）
        if model_path:
            print(f"\n1️⃣ 加载模型...")
            model_success = self.load_model(model_path, verbose=True)
            if not model_success:
                print("❌ 模型加载失败！将只进行绘图，不进行校正。")
                self.model = None
                self.model_loaded = False
            else:
                print("✅ 模型加载成功")
        else:
            print(f"\nℹ️ 未提供模型路径，跳过校正步骤")
            self.model = None
            self.model_loaded = False

        # 2. 加载站点统计信息
        print(f"\n2️⃣ 加载站点统计...")
        lsr_path = Path(lsr_data_path)
        stats_path_candidates = [
            lsr_path.with_suffix('.station_stats.json'),
            lsr_path.parent / f"{lsr_path.stem}_stats.json",
        ]

        stats_loaded = False
        for stats_path in stats_path_candidates:
            if stats_path.exists():
                if self.load_station_stats(str(stats_path)):
                    stats_loaded = True
                    break

        if not stats_loaded:
            print("⚠️ 未找到站点统计文件，将使用LSR数据中的所有站点")
            self.station_stats = {}

        # 3. 加载站点坐标
        self.load_station_coords()

        # 4. 绘制可用性分布图
        if self.station_stats:
            print(f"\n3️⃣ 绘制可用性分布图...")
            self.plot_availability_distribution(output_dir)

        # 5. 加载LSR数据（只加载指定波段）
        print(f"\n4️⃣ 加载LSR数据...")
        lsr_data_sample = self.load_lsr_data(lsr_data_path, bands_to_load=bands_to_process)
        if lsr_data_sample.empty:
            print("❌ 错误: 无法加载LSR数据")
            return False

        bands_loaded = lsr_data_sample['band'].unique().tolist()
        print(f"✅ 加载的波段: {bands_loaded}")

        # 6. 过滤站点
        if not self.station_stats:
            filtered_stations = lsr_data_sample['station'].unique().tolist()
            print(f"使用LSR数据中的所有站点: {len(filtered_stations)} 个")
        else:
            filtered_stations = self.filter_stations_by_availability(
                min_availability=min_availability,
                top_n=max_stations
            )

        if not filtered_stations:
            print("❌ 错误: 没有符合条件的站点")
            return False

        # 7. 选择代表性站点
        print(f"\n5️⃣ 选择代表性站点...")
        representative_stations = self.select_representative_stations(lsr_data_sample, top_n=10)

        if not representative_stations:
            print("❌ 错误: 无法选择代表性站点")
            return False

        # 8. 处理代表性站点
        print(f"\n6️⃣ 处理 {len(representative_stations)} 个代表性站点...")
        lsr_data_representative = self.load_lsr_data(lsr_data_path, bands_to_load=bands_to_process)
        lsr_data_representative = lsr_data_representative[
            lsr_data_representative['station'].isin(representative_stations)
        ].copy()

        if lsr_data_representative.empty:
            print(f"❌ 错误: 代表性站点没有数据")
            return False

        # 加载MODIS数据
        lsr_data_with_modis = self.load_modis_data_for_stations(lsr_data_representative)

        # 如果有模型，进行校正
        if self.model_loaded and self.model is not None:
            print(f"\n7️⃣ 应用模型校正...")

            # 准备特征
            features_df = self.prepare_features(lsr_data_with_modis)
            if features_df.empty:
                print("❌ 错误: 特征准备失败")
                corrected_data = lsr_data_with_modis.copy()
            else:
                # 验证特征
                self.validate_model_features(features_df)

                # 预测校正值
                predictions = self.predict_correction(features_df)
                if predictions.empty:
                    print("❌ 错误: 预测失败，使用原始数据")
                    corrected_data = lsr_data_with_modis.copy()
                    corrected_data['predicted_correction'] = 0.0
                    corrected_data['rho_lsr_corrected'] = corrected_data['rho_retrieved']
                else:
                    # 合并结果
                    corrected_data = self.merge_results(lsr_data_with_modis, predictions)
        else:
            print(f"\nℹ️ 跳过校正步骤（无模型）")
            corrected_data = lsr_data_with_modis.copy()
            corrected_data['predicted_correction'] = 0.0
            corrected_data['rho_lsr_corrected'] = corrected_data['rho_retrieved']

        # 9. 保存数据用于检查
        print(f"\n8️⃣ 保存数据用于检查...")
        data_check_path = representative_dir / "corrected_data_check.parquet"
        corrected_data.to_parquet(data_check_path)
        print(f"   数据保存至: {data_check_path}")

        # 检查校正数据
        if 'rho_lsr_corrected' in corrected_data.columns:
            corrected_count = corrected_data['rho_lsr_corrected'].notna().sum()
            print(f"   校正数据有效行数: {corrected_count}/{len(corrected_data)}")

            if corrected_count > 0:
                mean_correction = corrected_data['predicted_correction'].abs().mean()
                print(f"   平均校正量: {mean_correction:.6f}")

                if mean_correction < 0.0001:
                    print(f"   ⚠️ 警告: 校正量非常小，可能模型未正确工作")
            else:
                print(f"   ⚠️ 警告: 没有有效的校正数据")

        # 10. 绘图
        print(f"\n9️⃣ 绘制图表...")
        self.plot_timeseries_for_group(
            corrected_data,
            representative_stations,
            0,
            representative_dir,
            is_representative=True,
            group_type="representative",
            bands_to_plot=bands_loaded
        )

        # 11. 生成总结报告
        print(f"\n🔟 生成总结报告...")
        total_plots_generated = len(bands_loaded)
        self.generate_validation_summary(
            output_dir, filtered_stations,
            1, total_plots_generated,
            lsr_data_path,
            representative_stations=representative_stations,
            bands_processed=bands_loaded
        )

        print("\n" + "=" * 80)
        print("🎉 验证完成！")
        print("=" * 80)
        print(f"输出目录: {output_dir}")
        print(f"代表性站点: {len(representative_stations)} 个")
        print(f"处理波段: {bands_loaded}")
        print(f"生成图表: {total_plots_generated} 张")
        print("=" * 80)

        return True

    # 以下方法保持原样，但需要确保在类中定义
    def load_modis_data_for_stations(self, lsr_data):
        """为LSR数据中的站点和日期加载MODIS数据"""
        print("\nLoading MODIS data for AHI LSR records...")
        if lsr_data.empty:
            return lsr_data

        lsr_data['date'] = pd.to_datetime(lsr_data['datetime_bj']).dt.date
        unique_dates = lsr_data['date'].unique()
        unique_stations = lsr_data['station'].unique()

        modis_data_dict = {}
        for date in tqdm(unique_dates, desc="Loading MODIS data"):
            date_obj = datetime.combine(date, datetime.min.time())
            date_str = date_obj.strftime("%Y%m%d")

            modis_red_data = self._load_single_modis_data(date_obj, unique_stations, 'Red')
            modis_nir_data = self._load_single_modis_data(date_obj, unique_stations, 'NIR')

            for station in unique_stations:
                key = (date_str, station)
                modis_data_dict[key] = {
                    'MODIS_Red': modis_red_data.get(station, np.nan),
                    'MODIS_NIR': modis_nir_data.get(station, np.nan)
                }

        lsr_data_with_modis = lsr_data.copy()
        lsr_data_with_modis['date_str'] = pd.to_datetime(lsr_data_with_modis['datetime_bj']).dt.strftime('%Y%m%d')

        lsr_data_with_modis['MODIS_Red'] = np.nan
        lsr_data_with_modis['MODIS_NIR'] = np.nan

        for idx, row in lsr_data_with_modis.iterrows():
            key = (row['date_str'], row['station'])
            if key in modis_data_dict:
                lsr_data_with_modis.at[idx, 'MODIS_Red'] = modis_data_dict[key]['MODIS_Red']
                lsr_data_with_modis.at[idx, 'MODIS_NIR'] = modis_data_dict[key]['MODIS_NIR']

        print(f"MODIS数据: Red波段 {lsr_data_with_modis['MODIS_Red'].notna().sum()} 个, "
              f"NIR波段 {lsr_data_with_modis['MODIS_NIR'].notna().sum()} 个")

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

    def plot_availability_distribution(self, output_dir):
        """绘制站点可用性分布图"""
        if self.station_stats is None:
            print("No station statistics available for distribution plot")
            return

        availabilities = [stats.get('availability', 0) for stats in self.station_stats.values()]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        n_bins = 20
        counts, bins, patches = ax1.hist(availabilities, bins=n_bins, edgecolor='black',
                                         alpha=0.7, color='steelblue')
        ax1.set_xlabel('Data Availability (10min resolution)', fontsize=12)
        ax1.set_ylabel('Number of Stations', fontsize=12)
        ax1.set_title('Distribution of Station Data Availability\n10min Resolution',
                      fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')

        mean_avail = np.mean(availabilities)
        median_avail = np.median(availabilities)
        std_avail = np.std(availabilities)

        ax1.axvline(mean_avail, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_avail:.3f}')
        ax1.axvline(median_avail, color='green', linestyle='--', linewidth=2,
                    label=f'Median: {median_avail:.3f}')
        ax1.axvline(0.10, color='orange', linestyle='-', linewidth=1.5, alpha=0.7,
                    label='10% threshold')
        ax1.legend(loc='upper right', fontsize=10)

        stats_text = f"Total stations: {len(availabilities)}\n"
        stats_text += f"Mean: {mean_avail:.3f}\n"
        stats_text += f"Median: {median_avail:.3f}\n"
        stats_text += f"Std: {std_avail:.3f}\n"
        stats_text += f"Max: {max(availabilities):.3f}"

        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        sorted_avail = np.sort(availabilities)
        cdf = np.arange(1, len(sorted_avail) + 1) / len(sorted_avail)

        ax2.plot(sorted_avail, cdf, 'b-', linewidth=3, alpha=0.8)
        ax2.set_xlabel('Data Availability (10min resolution)', fontsize=12)
        ax2.set_ylabel('Cumulative Probability', fontsize=12)
        ax2.set_title('Cumulative Distribution of Station Availability\n10min Resolution',
                      fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')

        thresholds = [0.10, 0.15, 0.20]
        for threshold in thresholds:
            ax2.axvline(threshold, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

        threshold_info = []
        for threshold in thresholds:
            stations_above = sum(1 for a in availabilities if a >= threshold)
            proportion_above = stations_above / len(availabilities)
            threshold_info.append(
                f"≥{threshold * 100:.0f}%: {stations_above}/{len(availabilities)} ({proportion_above * 100:.1f}%)")

        cdf_text = '\n'.join(threshold_info)
        ax2.text(0.02, 0.98, cdf_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        mask_above_10 = sorted_avail >= 0.10
        if any(mask_above_10):
            ax2.fill_between(sorted_avail[mask_above_10], 0, cdf[mask_above_10],
                             alpha=0.3, color='green')

        plt.tight_layout()
        output_path = Path(output_dir) / 'station_availability_distribution_10min.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print(f"可用性分布图保存: {output_path}")
        return output_path

    def generate_availability_report(self, filtered_stations, output_dir):
        """生成可用性报告"""
        if self.station_stats is None:
            return None

        print("\nGenerating availability report...")

        availabilities = []
        for station in filtered_stations:
            if station in self.station_stats:
                stats = self.station_stats[station]
                if 'availability' in stats:
                    availabilities.append(stats['availability'])

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("STATION AVAILABILITY REPORT - 10MIN RESOLUTION")
        report_lines.append("=" * 80)
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
            report_lines.append("")

        report_lines.append("Filtered stations (sorted by availability):")
        report_lines.append("-" * 100)

        for i, station in enumerate(filtered_stations, 1):
            stats = self.station_stats.get(station, {})
            availability = stats.get('availability', 0)
            records = int(stats.get('records', 0))
            days = int(stats.get('unique_dates', 0))

            report_lines.append(f"{i:4d}. {station:20s} "
                                f"Avail: {availability * 100:6.2f}% | "
                                f"Records: {records:6d} | "
                                f"Days: {days:3d}")

        report_path = Path(output_dir) / "station_availability_report_10min.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"可用性报告保存: {report_path}")
        return report_path

    def generate_validation_summary(self, output_dir, filtered_stations,
                                    total_groups_processed, total_plots_generated,
                                    lsr_data_path, representative_stations=None,
                                    bands_processed=None):
        """生成验证总结报告"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("VALIDATION SUMMARY REPORT - 10MIN RESOLUTION")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Time resolution: 10 minutes")
        report_lines.append(f"LSR data source: {lsr_data_path}")
        report_lines.append(f"Model loaded: {'Yes' if self.model_loaded else 'No'}")
        report_lines.append("")

        if bands_processed:
            report_lines.append(f"BANDS PROCESSED: {', '.join(bands_processed)}")
            report_lines.append("")

        if representative_stations:
            report_lines.append("REPRESENTATIVE STATIONS (Top 10 by data volume):")
            for i, station in enumerate(representative_stations, 1):
                report_lines.append(f"  {i:2d}. {station}")
            report_lines.append("")

        report_lines.append("PROCESSING STATISTICS:")
        report_lines.append(f"  Total stations filtered: {len(filtered_stations)}")
        if representative_stations:
            report_lines.append(f"  Representative stations: {len(representative_stations)}")
        report_lines.append(f"  Total plots generated: {total_plots_generated}")
        report_lines.append(f"  Output directory: {output_dir}")

        report_path = Path(output_dir) / "validation_summary_10min.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"验证总结保存: {report_path}")
        return report_path


# ==================== Main Function ====================
def main():
    parser = argparse.ArgumentParser(
        description='增强版验证工作流 - 10分钟分辨率（带诊断）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
使用示例:
  # 处理band01和band02，使用模型校正
  python enhanced_validation_10min.py --lsr_data ./lsr_results_10min/ --bands 01 02 --model_path "D:/6S_Self_Consistency_Zenith_Angles_v0.4/models/XGBoost_model.pkl" --output_dir ./validation_band12

  # 只处理band01
  python enhanced_validation_10min.py --lsr_data ./lsr_results_10min/band01/ --bands 01 --model_path "models/XGBoost_model.pkl"

  # 不使用模型，只绘图
  python enhanced_validation_10min.py --lsr_data ./lsr_results_10min/ --bands 01 02 --output_dir ./validation_no_model

  # 使用通配符
  python enhanced_validation_10min.py --lsr_data "./lsr_results_10min/*/lsr_band*.parquet" --bands 01 02
        """
    )

    parser.add_argument('--lsr_data', type=str, required=True,
                        help='LSR数据路径（文件、目录或通配符）')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型文件路径（可选）')
    parser.add_argument('--output_dir', type=str, default='./validation_results_10min',
                        help='输出目录')
    parser.add_argument('--min_availability', type=float, default=0.10,
                        help='最小数据可用性阈值（默认: 0.10）')
    parser.add_argument('--stations_per_plot', type=int, default=3,
                        help='每张图的站点数（默认: 3）')
    parser.add_argument('--max_stations', type=int, default=None,
                        help='最大处理站点数（默认: 全部）')
    parser.add_argument('--bands', type=str, nargs='+', default=None,
                        help='要处理的波段列表（例如: 01 02）')

    args = parser.parse_args()

    config = Config()
    validator = EnhancedValidator10min(config)

    success = validator.run_validation(
        lsr_data_path=args.lsr_data,
        model_path=args.model_path,
        output_dir=args.output_dir,
        min_availability=args.min_availability,
        stations_per_plot=args.stations_per_plot,
        max_stations=args.max_stations,
        bands_to_process=args.bands
    )

    if success:
        print("\n✅ 验证成功完成!")
        return 0
    else:
        print("\n❌ 验证失败!")
        return 1


if __name__ == "__main__":
    main()