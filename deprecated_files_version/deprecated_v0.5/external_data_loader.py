# ==================== external_data_loader.py ====================
"""
外部数据验证的数据读取与预处理模块 - 完整修复版本
"""

import os
import numpy as np
import pandas as pd
import netCDF4 as nc
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import random

from config import ExperimentConfig
from utils import setup_logger

warnings.filterwarnings('ignore')


class ExternalDataLoader:
    """
    外部数据验证的数据加载器 - 简化版本
    直接读取Himawari-AHI真实数据，选择测站进行验证
    """

    def __init__(self, config: ExperimentConfig = None, logger=None):
        self.config = config or ExperimentConfig
        self.logger = logger or setup_logger('ExternalDataLoader')

        # 定义数据路径（参考6S+BRDF_HimawariBRDF_SOZlessthan60_New.py）
        self.data_paths = {
            "hourly_sozSR": "D:/H8_data/Hourly_sozSR_Angles/",
            "merra2_slv": "D:/H8_data/MERRA2_slv/",
            "merra2_aer": "D:/H8_data/MERRA2_aer/",
            "lucc": "D:/H8_data/LC_2015_2024.nc",
            "luts": "D:/H8_data/LUTs.nc",
            "output_h8sr": "D:/H8_data/NBAR_SOZlessthan60_New/",
            "himawari_brdf": "D:/H8_data/Himawari_BRDF_Albedo"
        }

        # 波段配置
        self.band_wavelengths = [0.64, 0.86]
        self.band_names = ['Albedo_03', 'Albedo_04']
        self.angle_names = ['SAZ', 'SAA', 'SOZ', 'SOA']

        # 缓存坐标数据
        self.station_coords_cache = None

    def load_station_coordinates(self, force_reload: bool = False) -> pd.DataFrame:
        """
        加载所有测站的坐标信息
        """
        if self.station_coords_cache is not None and not force_reload:
            return self.station_coords_cache

        try:
            self.logger.info(f"加载测站坐标文件: {self.data_paths['luts']}")

            if not os.path.exists(self.data_paths["luts"]):
                self.logger.error(f"LUTs文件不存在: {self.data_paths['luts']}")
                return pd.DataFrame()

            with nc.Dataset(self.data_paths["luts"]) as ds:
                # 读取站点名称
                stations_raw = ds.variables['Station'][:]
                stations = []
                for s in stations_raw:
                    if isinstance(s, bytes):
                        try:
                            s = s.decode('utf-8')
                        except:
                            s = str(s)
                    stations.append(str(s).strip())

                # 读取坐标
                lats = ds.variables['Lat'][:]
                lons = ds.variables['Lon'][:]

                # 处理掩码数组
                if isinstance(lats, np.ma.MaskedArray):
                    lats = lats.filled(np.nan)
                if isinstance(lons, np.ma.MaskedArray):
                    lons = lons.filled(np.nan)

                # 创建DataFrame
                df = pd.DataFrame({
                    'station': stations,
                    'lat': lats,
                    'lon': lons
                })

                # 去除无效坐标
                df = df.dropna(subset=['lat', 'lon'])

                # 去除重复站点（保留第一个）
                initial_count = len(df)
                df = df.drop_duplicates(subset=['station'], keep='first')
                if len(df) < initial_count:
                    self.logger.info(f"移除了 {initial_count - len(df)} 个重复测站")

                self.station_coords_cache = df
                self.logger.info(f"成功加载了 {len(df)} 个测站的坐标信息")

                # 显示前几个测站作为示例
                self.logger.info(f"前5个测站示例: {df['station'].head().tolist()}")

                return df

        except Exception as e:
            self.logger.error(f"加载测站坐标失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def select_diverse_stations(self, n_stations: int = 100) -> List[str]:
        """
        选择空间分布相对均匀的测站（简化版本）
        """
        try:
            # 加载所有测站坐标
            stations_df = self.load_station_coordinates()

            if stations_df.empty:
                self.logger.error("没有可用的测站坐标")
                return []

            total_stations = len(stations_df)
            self.logger.info(f"总测站数: {total_stations}, 请求数: {n_stations}")

            if total_stations <= n_stations:
                self.logger.info(f"可用测站数({total_stations}) <= 请求数({n_stations})，返回所有测站")
                return stations_df['station'].tolist()

            # 简单的地理分块选择策略
            # 根据经纬度将区域分为网格，从每个网格中随机选择
            lat_min, lat_max = stations_df['lat'].min(), stations_df['lat'].max()
            lon_min, lon_max = stations_df['lon'].min(), stations_df['lon'].max()

            # 计算网格数量（取平方根近似）
            grid_size = int(np.sqrt(n_stations))
            if grid_size < 2:
                grid_size = 2

            # 创建网格
            lat_bins = np.linspace(lat_min, lat_max, grid_size + 1)
            lon_bins = np.linspace(lon_min, lon_max, grid_size + 1)

            selected_stations = []

            for i in range(grid_size):
                for j in range(grid_size):
                    # 当前网格的经纬度范围
                    lat_low, lat_high = lat_bins[i], lat_bins[i + 1]
                    lon_low, lon_high = lon_bins[j], lon_bins[j + 1]

                    # 在当前网格内的测站
                    mask = (
                            (stations_df['lat'] >= lat_low) &
                            (stations_df['lat'] <= lat_high) &
                            (stations_df['lon'] >= lon_low) &
                            (stations_df['lon'] <= lon_high)
                    )

                    grid_stations = stations_df[mask]

                    # 如果网格内有测站，随机选择一个
                    if len(grid_stations) > 0:
                        selected = grid_stations.sample(n=1, random_state=42).iloc[0]
                        selected_stations.append(selected['station'])

                    # 如果已经选择了足够的测站，就停止
                    if len(selected_stations) >= n_stations:
                        break

                if len(selected_stations) >= n_stations:
                    break

            # 如果网格选择不够，补充随机选择
            if len(selected_stations) < n_stations:
                remaining_needed = n_stations - len(selected_stations)
                remaining_stations = stations_df[~stations_df['station'].isin(selected_stations)]
                if len(remaining_stations) > 0:
                    additional = remaining_stations.sample(
                        n=min(remaining_needed, len(remaining_stations)),
                        random_state=42
                    )
                    selected_stations.extend(additional['station'].tolist())

            self.logger.info(f"成功选择了 {len(selected_stations)} 个测站")

            # 保存选择结果
            selection_df = stations_df[stations_df['station'].isin(selected_stations)].copy()
            selection_path = self.config.RESULTS_DIR / "selected_stations.csv"
            selection_df.to_csv(selection_path, index=False)
            self.logger.info(f"测站选择结果已保存: {selection_path}")

            return selected_stations

        except Exception as e:
            self.logger.error(f"选择测站失败: {str(e)}")
            import traceback
            traceback.print_exc()

            # 如果失败，随机选择测站
            stations_df = self.load_station_coordinates()
            if not stations_df.empty:
                n_to_select = min(n_stations, len(stations_df))
                return stations_df.sample(n=n_to_select, random_state=42)['station'].tolist()
            return []

    def load_single_file_data(self, file_path: str, variables: List[str], station_indices: List[int]) -> Dict:
        """
        加载单个NetCDF文件的数据
        """
        try:
            if not os.path.exists(file_path):
                self.logger.debug(f"文件不存在: {file_path}")
                return {}

            with nc.Dataset(file_path) as ds:
                data = {}
                for var in variables:
                    if var not in ds.variables:
                        continue

                    var_data = ds.variables[var][:][station_indices]

                    # 处理掩码数组
                    if isinstance(var_data, np.ma.MaskedArray):
                        var_data = var_data.filled(np.nan)

                    # 处理特殊变量
                    if var == 'Station':
                        # 转换为字符串列表
                        stations = []
                        for s in var_data:
                            if isinstance(s, bytes):
                                try:
                                    s = s.decode('utf-8')
                                except:
                                    s = str(s)
                            stations.append(str(s).strip())
                        data[var] = stations
                    elif var == 'AOT550':
                        # 处理无效值
                        var_data = np.where(var_data == -9999.0, np.nan, var_data)
                        data[var] = var_data
                    else:
                        data[var] = var_data

                return data

        except Exception as e:
            self.logger.error(f"加载文件失败 {file_path}: {str(e)}")
            return {}

    def load_hourly_data_simple(self, date: datetime, hour: int, stations: List[str]) -> pd.DataFrame:
        """
        简化版本：加载指定日期和小时的单个文件数据
        """
        try:
            date_str = date.strftime("%Y%m%d")
            hour_str = f"{hour * 100:04d}"
            time_key = f"{date_str}_{hour_str}"

            # 构建文件路径
            sozSR_file = os.path.join(
                self.data_paths["hourly_sozSR"],
                date_str[:4],
                date_str[4:6],
                f"H8_hourly_sozSR_angles_{time_key}.nc"
            )

            self.logger.debug(f"加载文件: {sozSR_file}")

            if not os.path.exists(sozSR_file):
                self.logger.debug(f"文件不存在，跳过: {sozSR_file}")
                return pd.DataFrame()

            # 先加载角度文件获取文件中的站点列表
            with nc.Dataset(sozSR_file) as ds:
                # 读取文件中的站点
                file_stations_raw = ds.variables['Station'][:]
                file_stations = []
                for s in file_stations_raw:
                    if isinstance(s, bytes):
                        try:
                            s = s.decode('utf-8')
                        except:
                            s = str(s)
                    file_stations.append(str(s).strip())

                # 找到我们需要的测站索引
                station_indices = []
                valid_stations = []
                for station in stations:
                    if station in file_stations:
                        idx = file_stations.index(station)
                        station_indices.append(idx)
                        valid_stations.append(station)

                if not station_indices:
                    self.logger.debug(f"没有找到指定测站: {stations}")
                    return pd.DataFrame()

                # 提取数据
                data_dict = {'station': valid_stations}

                # 提取角度和反射率
                for var_name in self.angle_names + self.band_names:
                    if var_name in ds.variables:
                        var_data = ds.variables[var_name][:][station_indices]
                        if isinstance(var_data, np.ma.MaskedArray):
                            var_data = var_data.filled(np.nan)

                        # 如果是反射率，转换为0-1范围
                        if var_name in self.band_names:
                            var_data = var_data / 100.0
                            data_dict[f'TOA_{var_name}'] = var_data
                        else:
                            data_dict[var_name] = var_data

            # 转换为DataFrame
            df = pd.DataFrame(data_dict)

            if df.empty:
                return df

            # 添加时间和坐标信息
            df['date'] = date
            df['hour'] = hour
            df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')

            # 添加坐标信息
            if self.station_coords_cache is None:
                self.load_station_coordinates()

            if self.station_coords_cache is not None:
                # 使用merge添加坐标
                coords_df = self.station_coords_cache[['station', 'lat', 'lon']].copy()
                df = pd.merge(df, coords_df, on='station', how='left')

            # 尝试加载大气数据（如果文件存在）
            merra_slv_file = os.path.join(
                self.data_paths["merra2_slv"],
                date_str[:4],
                date_str[4:6],
                f"MERRA2_{time_key}_TO3_TQV.nc"
            )

            if os.path.exists(merra_slv_file):
                try:
                    with nc.Dataset(merra_slv_file) as ds:
                        merra_stations_raw = ds.variables['Station'][:]
                        merra_stations = []
                        for s in merra_stations_raw:
                            if isinstance(s, bytes):
                                try:
                                    s = s.decode('utf-8')
                                except:
                                    s = str(s)
                            merra_stations.append(str(s).strip())

                        # 为每个测站获取大气数据
                        ozone_vals = []
                        water_vals = []

                        for station in df['station']:
                            if station in merra_stations:
                                idx = merra_stations.index(station)
                                to3 = ds.variables['TO3'][:][idx] if 'TO3' in ds.variables else np.nan
                                tqv = ds.variables['TQV'][:][idx] if 'TQV' in ds.variables else np.nan

                                if isinstance(to3, np.ma.MaskedArray):
                                    to3 = to3.filled(np.nan)
                                if isinstance(tqv, np.ma.MaskedArray):
                                    tqv = tqv.filled(np.nan)

                                ozone_vals.append(to3 * 0.001)  # Dobson -> cm-atm
                                water_vals.append(tqv * 0.1)  # kg/m² -> g/cm²
                            else:
                                ozone_vals.append(np.nan)
                                water_vals.append(np.nan)

                        df['ozone'] = ozone_vals
                        df['water'] = water_vals
                except Exception as e:
                    self.logger.warning(f"加载大气数据失败: {str(e)}")
                    df['ozone'] = np.nan
                    df['water'] = np.nan

            # 尝试加载气溶胶数据
            merra_aer_file = os.path.join(
                self.data_paths["merra2_aer"],
                date_str[:4],
                date_str[4:6],
                f"MERRA2_{time_key}_AOT550.nc"
            )

            if os.path.exists(merra_aer_file):
                try:
                    with nc.Dataset(merra_aer_file) as ds:
                        aer_stations_raw = ds.variables['Station'][:]
                        aer_stations = []
                        for s in aer_stations_raw:
                            if isinstance(s, bytes):
                                try:
                                    s = s.decode('utf-8')
                                except:
                                    s = str(s)
                            aer_stations.append(str(s).strip())

                        aod_vals = []
                        for station in df['station']:
                            if station in aer_stations:
                                idx = aer_stations.index(station)
                                aod = ds.variables['AOT550'][:][idx] if 'AOT550' in ds.variables else np.nan
                                if isinstance(aod, np.ma.MaskedArray):
                                    aod = aod.filled(np.nan)
                                if aod == -9999.0:
                                    aod = np.nan
                                aod_vals.append(aod)
                            else:
                                aod_vals.append(np.nan)

                        df['AOD550'] = aod_vals
                except Exception as e:
                    self.logger.warning(f"加载气溶胶数据失败: {str(e)}")
                    df['AOD550'] = np.nan

            return df

        except Exception as e:
            self.logger.error(f"加载数据失败 {date_str} {hour_str}: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def generate_validation_dataset(self,
                                    n_stations: int = 10,
                                    start_date: str = "20160101",
                                    end_date: str = "20160110",
                                    output_path: Optional[Path] = None) -> pd.DataFrame:
        """
        生成外部验证数据集 - 简化版本
        """
        self.logger.info(f"开始生成外部验证数据集")
        self.logger.info(f"测站数量: {n_stations}")
        self.logger.info(f"时间范围: {start_date} 到 {end_date}")

        try:
            # 选择测站
            selected_stations = self.select_diverse_stations(n_stations)

            if not selected_stations:
                self.logger.error("无法选择测站")
                return pd.DataFrame()

            self.logger.info(f"已选择测站: {selected_stations}")

            # 解析日期
            start_dt = datetime.strptime(start_date, "%Y%m%d")
            end_dt = datetime.strptime(end_date, "%Y%m%d")

            # 生成日期列表
            dates = []
            current = start_dt
            while current <= end_dt:
                dates.append(current)
                current += timedelta(days=1)

            self.logger.info(f"将处理 {len(dates)} 天的数据")

            # 白天小时
            day_hours = list(range(8, 16))  # 8:00到16:00

            all_data = []

            for i, date in enumerate(dates):
                date_str = date.strftime("%Y-%m-%d")

                # 每处理一天输出一次进度
                if i % 5 == 0 or i == len(dates) - 1:
                    self.logger.info(f"处理进度: {i + 1}/{len(dates)} 天 ({date_str})")

                for hour in day_hours:
                    df_hour = self.load_hourly_data_simple(date, hour, selected_stations)

                    if not df_hour.empty:
                        # 基本数据验证
                        required_cols = ['SOZ', 'TOA_Albedo_03', 'TOA_Albedo_04']
                        if all(col in df_hour.columns for col in required_cols):
                            # 过滤SOZ太大的数据
                            df_hour = df_hour[df_hour['SOZ'] <= 85].copy()

                            # 添加时间特征
                            df_hour['doy'] = df_hour['datetime'].dt.dayofyear
                            df_hour['month'] = df_hour['datetime'].dt.month
                            df_hour['weekday'] = df_hour['datetime'].dt.weekday
                            df_hour['hour_of_day'] = df_hour['datetime'].dt.hour

                            all_data.append(df_hour)

            if all_data:
                # 合并所有数据
                final_df = pd.concat(all_data, ignore_index=True)

                self.logger.info(f"数据集生成完成，共 {len(final_df)} 条记录")
                self.logger.info(f"数据列: {list(final_df.columns)}")

                # 基本统计
                self.logger.info(f"测站数: {final_df['station'].nunique()}")
                self.logger.info(f"时间范围: {final_df['datetime'].min()} 到 {final_df['datetime'].max()}")

                # 保存数据
                if output_path:
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # 保存为Parquet格式
                    final_df.to_parquet(output_path, index=False)
                    self.logger.info(f"数据集已保存: {output_path}")

                    # 同时保存CSV以便查看
                    csv_path = output_path.with_suffix('.csv')
                    final_df.to_csv(csv_path, index=False)
                    self.logger.info(f"CSV版本已保存: {csv_path}")

                return final_df
            else:
                self.logger.warning("没有生成有效数据")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"生成数据集失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()