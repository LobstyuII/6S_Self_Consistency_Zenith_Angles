import os
import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import glob
from pathlib import Path
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


class MERRA2_10min_Interpolator:
    """将MERRA2辅助数据从1小时插值到10分钟"""

    def __init__(self, base_path="D:/H8_data"):
        self.base_path = base_path
        self.merra2_slv_path = os.path.join(base_path, "MERRA2_slv")
        self.merra2_aer_path = os.path.join(base_path, "MERRA2_aer")
        self.output_path = os.path.join(base_path, "MERRA2_10min")

        # 创建输出目录
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "AOD550"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "H2O"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "O3"), exist_ok=True)

    def interpolate_aer_hourly_to_10min(self, start_date="20200501", end_date="20200507"):
        """处理气溶胶数据 (AOT550 -> AOD550)"""
        logger.info("Processing AOD550 from AOT550...")

        # 生成日期范围
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")

        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)

        for date in tqdm(dates, desc="Processing AOD550"):
            year = date.strftime("%Y")
            month = date.strftime("%m")
            day = date.strftime("%d")

            # 查找该日期的小时文件
            pattern = os.path.join(self.merra2_aer_path, year, month,
                                   f"MERRA2_{year}{month}{day}_*_AOT550.nc")
            hour_files = glob.glob(pattern)

            if not hour_files:
                logger.warning(f"No AOT550 files found on {year}{month}{day}")
                continue

            # 读取所有小时数据
            hour_data = {}
            stations = None
            for file_path in sorted(hour_files):
                try:
                    # 从文件名中提取小时部分 (格式: MERRA2_YYYYMMDD_HHMM_AOT550.nc)
                    hour_str = os.path.basename(file_path).split('_')[2]  # 获取HHMM部分
                    hour = int(hour_str[:2])  # 转换为小时整数

                    with nc.Dataset(file_path, 'r') as ds:
                        # 读取AOT550数据
                        data = ds.variables['AOT550'][:]
                        hour_data[hour] = data

                        # 获取站点信息
                        if stations is None:
                            stations = ds.variables['Station'][:]
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
                    continue

            if not hour_data:
                continue

            # 创建10分钟时间序列
            n_stations = len(hour_data[list(hour_data.keys())[0]])
            all_10min_data = {}

            # 生成所有10分钟时间点
            for hour in range(24):
                for minute in [0, 10, 20, 30, 40, 50]:
                    time_key = f"{hour:02d}{minute:02d}"

                    # 计算插值权重
                    if hour in hour_data:
                        # 整点数据
                        if minute == 0:
                            # 直接使用整点数据
                            all_10min_data[time_key] = hour_data[hour]
                        else:
                            # 线性插值
                            if hour + 1 in hour_data:
                                # 当前小时和下一小时都有数据
                                weight = minute / 60.0
                                all_10min_data[time_key] = (
                                        hour_data[hour] * (1 - weight) +
                                        hour_data[hour + 1] * weight
                                )
                            else:
                                # 只有当前小时有数据，使用最近邻
                                all_10min_data[time_key] = hour_data[hour]
                    else:
                        # 当前小时没有数据，尝试插值
                        prev_hour = max([h for h in hour_data.keys() if h < hour], default=None)
                        next_hour = min([h for h in hour_data.keys() if h > hour], default=None)

                        if prev_hour is not None and next_hour is not None:
                            # 在两个有数据的整点之间插值
                            total_diff = next_hour - prev_hour
                            current_hour = hour + minute / 60.0
                            weight = (current_hour - prev_hour) / total_diff

                            # 线性插值
                            all_10min_data[time_key] = (
                                    hour_data[prev_hour] * (1 - weight) +
                                    hour_data[next_hour] * weight
                            )
                        elif prev_hour is not None:
                            # 只有前一个整点有数据，使用最近邻
                            all_10min_data[time_key] = hour_data[prev_hour]
                        elif next_hour is not None:
                            # 只有后一个整点有数据，使用最近邻
                            all_10min_data[time_key] = hour_data[next_hour]
                        else:
                            # 没有数据，使用NaN
                            all_10min_data[time_key] = np.full(n_stations, np.nan)

            # 保存10分钟数据
            output_year_dir = os.path.join(self.output_path, "AOD550", year)
            output_month_dir = os.path.join(output_year_dir, month)
            os.makedirs(output_month_dir, exist_ok=True)

            for time_key, data in all_10min_data.items():
                output_file = os.path.join(
                    output_month_dir,
                    f"MERRA2_AOD550_{year}{month}{day}_{time_key}.nc"
                )

                try:
                    with nc.Dataset(output_file, 'w', format='NETCDF4') as ds_out:
                        # 创建维度
                        ds_out.createDimension('Station', n_stations)

                        # 创建变量
                        station_var = ds_out.createVariable('Station', str, ('Station',))
                        data_var = ds_out.createVariable('AOD550', 'f4', ('Station',))

                        # 写入数据
                        station_var[:] = np.array([s.decode() if isinstance(s, bytes) else s for s in stations],
                                                  dtype='S')
                        data_var[:] = data.astype(np.float32)

                        # 添加属性
                        data_var.units = "1"
                        data_var.long_name = "Aerosol Optical Depth at 550nm"

                        ds_out.setncattr('time', f"{year}{month}{day}_{time_key}")
                        ds_out.setncattr('variable', 'AOD550')
                        ds_out.setncattr('resolution', '10min')
                        ds_out.setncattr('interpolation_method', 'linear')
                        ds_out.setncattr('source_variable', 'AOT550')

                except Exception as e:
                    logger.error(f"Error writing {output_file}: {e}")

        logger.info("Completed processing AOD550")

    def interpolate_slv_hourly_to_10min(self, start_date="20200501", end_date="20200507"):
        """处理地表数据 (TQV -> H2O, TO3 -> O3)"""
        logger.info("Processing H2O and O3 from TQV and TO3...")

        # 生成日期范围
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")

        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)

        for date in tqdm(dates, desc="Processing H2O and O3"):
            year = date.strftime("%Y")
            month = date.strftime("%m")
            day = date.strftime("%d")

            # 查找该日期的小时文件
            pattern = os.path.join(self.merra2_slv_path, year, month,
                                   f"MERRA2_{year}{month}{day}_*_TO3_TQV.nc")
            hour_files = glob.glob(pattern)

            if not hour_files:
                logger.warning(f"No TO3_TQV files found on {year}{month}{day}")
                continue

            # 读取所有小时数据
            hour_data_h2o = {}
            hour_data_o3 = {}
            stations = None

            for file_path in sorted(hour_files):
                try:
                    # 从文件名中提取小时部分
                    hour_str = os.path.basename(file_path).split('_')[2]  # 获取HHMM部分
                    hour = int(hour_str[:2])  # 转换为小时整数

                    with nc.Dataset(file_path, 'r') as ds:
                        # 读取TQV和TO3数据
                        h2o_data = ds.variables['TQV'][:]  # TQV -> H2O
                        o3_data = ds.variables['TO3'][:]  # TO3 -> O3

                        hour_data_h2o[hour] = h2o_data
                        hour_data_o3[hour] = o3_data

                        # 获取站点信息
                        if stations is None:
                            stations = ds.variables['Station'][:]
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
                    continue

            if not hour_data_h2o or not hour_data_o3:
                continue

            # 为H2O和O3分别创建10分钟数据
            n_stations = len(hour_data_h2o[list(hour_data_h2o.keys())[0]])

            # 处理H2O
            all_10min_data_h2o = self._interpolate_data(hour_data_h2o, n_stations)
            # 处理O3
            all_10min_data_o3 = self._interpolate_data(hour_data_o3, n_stations)

            # 保存H2O数据
            self._save_slv_data(year, month, day, stations, all_10min_data_h2o,
                                "H2O", "Total Precipitable Water Vapor", "kg/m^2")

            # 保存O3数据
            self._save_slv_data(year, month, day, stations, all_10min_data_o3,
                                "O3", "Total Ozone Column", "Dobson Units")

        logger.info("Completed processing H2O and O3")

    def _interpolate_data(self, hour_data, n_stations):
        """通用插值函数"""
        all_10min_data = {}

        # 生成所有10分钟时间点
        for hour in range(24):
            for minute in [0, 10, 20, 30, 40, 50]:
                time_key = f"{hour:02d}{minute:02d}"

                # 计算插值权重
                if hour in hour_data:
                    # 整点数据
                    if minute == 0:
                        # 直接使用整点数据
                        all_10min_data[time_key] = hour_data[hour]
                    else:
                        # 线性插值
                        if hour + 1 in hour_data:
                            # 当前小时和下一小时都有数据
                            weight = minute / 60.0
                            all_10min_data[time_key] = (
                                    hour_data[hour] * (1 - weight) +
                                    hour_data[hour + 1] * weight
                            )
                        else:
                            # 只有当前小时有数据，使用最近邻
                            all_10min_data[time_key] = hour_data[hour]
                else:
                    # 当前小时没有数据，尝试插值
                    prev_hour = max([h for h in hour_data.keys() if h < hour], default=None)
                    next_hour = min([h for h in hour_data.keys() if h > hour], default=None)

                    if prev_hour is not None and next_hour is not None:
                        # 在两个有数据的整点之间插值
                        total_diff = next_hour - prev_hour
                        current_hour = hour + minute / 60.0
                        weight = (current_hour - prev_hour) / total_diff

                        # 线性插值
                        all_10min_data[time_key] = (
                                hour_data[prev_hour] * (1 - weight) +
                                hour_data[next_hour] * weight
                        )
                    elif prev_hour is not None:
                        # 只有前一个整点有数据，使用最近邻
                        all_10min_data[time_key] = hour_data[prev_hour]
                    elif next_hour is not None:
                        # 只有后一个整点有数据，使用最近邻
                        all_10min_data[time_key] = hour_data[next_hour]
                    else:
                        # 没有数据，使用NaN
                        all_10min_data[time_key] = np.full(n_stations, np.nan)

        return all_10min_data

    def _save_slv_data(self, year, month, day, stations, all_10min_data,
                       variable_name, long_name, units):
        """保存SLV数据"""
        if variable_name == "H2O":
            output_dir = os.path.join(self.output_path, "H2O")
        else:
            output_dir = os.path.join(self.output_path, "O3")

        output_year_dir = os.path.join(output_dir, year)
        output_month_dir = os.path.join(output_year_dir, month)
        os.makedirs(output_month_dir, exist_ok=True)

        for time_key, data in all_10min_data.items():
            output_file = os.path.join(
                output_month_dir,
                f"MERRA2_{variable_name}_{year}{month}{day}_{time_key}.nc"
            )

            try:
                with nc.Dataset(output_file, 'w', format='NETCDF4') as ds_out:
                    # 创建维度
                    ds_out.createDimension('Station', len(stations))

                    # 创建变量
                    station_var = ds_out.createVariable('Station', str, ('Station',))
                    data_var = ds_out.createVariable(variable_name, 'f4', ('Station',))

                    # 写入数据
                    station_var[:] = np.array([s.decode() if isinstance(s, bytes) else s for s in stations],
                                              dtype='S')
                    data_var[:] = data.astype(np.float32)

                    # 添加属性
                    data_var.units = units
                    data_var.long_name = long_name

                    ds_out.setncattr('time', f"{year}{month}{day}_{time_key}")
                    ds_out.setncattr('variable', variable_name)
                    ds_out.setncattr('resolution', '10min')
                    ds_out.setncattr('interpolation_method', 'linear')

                    if variable_name == "H2O":
                        ds_out.setncattr('source_variable', 'TQV')
                    else:
                        ds_out.setncattr('source_variable', 'TO3')

            except Exception as e:
                logger.error(f"Error writing {output_file}: {e}")

    def process_all_variables(self, start_date="20200501", end_date="20200507"):
        """处理所有变量"""

        # AOD550 (气溶胶光学厚度) - 从AOT550转换
        self.interpolate_aer_hourly_to_10min(start_date, end_date)

        # H2O (水汽)和O3 (臭氧) - 从TO3_TQV文件转换
        self.interpolate_slv_hourly_to_10min(start_date, end_date)

        # 创建合并的数据集（可选）
        self.create_combined_dataset(start_date, end_date)

    def create_combined_dataset(self, start_date="20200501", end_date="20200507"):
        """创建合并的10分钟数据集"""

        # 生成日期范围
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")

        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)

        for date in tqdm(dates, desc="Creating combined dataset"):
            year = date.strftime("%Y")
            month = date.strftime("%m")
            day = date.strftime("%d")

            # 查找10分钟文件
            time_patterns = []
            for hour in range(24):
                for minute in [0, 10, 20, 30, 40, 50]:
                    time_patterns.append(f"{hour:02d}{minute:02d}")

            for time_pattern in time_patterns:
                # 读取所有变量
                aod_file = os.path.join(
                    self.output_path, "AOD550", year, month,
                    f"MERRA2_AOD550_{year}{month}{day}_{time_pattern}.nc"
                )
                h2o_file = os.path.join(
                    self.output_path, "H2O", year, month,
                    f"MERRA2_H2O_{year}{month}{day}_{time_pattern}.nc"
                )
                o3_file = os.path.join(
                    self.output_path, "O3", year, month,
                    f"MERRA2_O3_{year}{month}{day}_{time_pattern}.nc"
                )

                if not all(os.path.exists(f) for f in [aod_file, h2o_file, o3_file]):
                    continue

                # 读取数据
                with nc.Dataset(aod_file, 'r') as aod_ds:
                    aod_data = aod_ds.variables['AOD550'][:]
                    stations = aod_ds.variables['Station'][:]

                with nc.Dataset(h2o_file, 'r') as h2o_ds:
                    h2o_data = h2o_ds.variables['H2O'][:]

                with nc.Dataset(o3_file, 'r') as o3_ds:
                    o3_data = o3_ds.variables['O3'][:]

                # 创建合并文件
                output_dir = os.path.join(self.output_path, "combined", year, month)
                os.makedirs(output_dir, exist_ok=True)

                output_file = os.path.join(
                    output_dir,
                    f"MERRA2_combined_{year}{month}{day}_{time_pattern}.nc"
                )

                with nc.Dataset(output_file, 'w', format='NETCDF4') as ds_out:
                    # 创建维度
                    ds_out.createDimension('Station', len(stations))

                    # 创建变量
                    station_var = ds_out.createVariable('Station', str, ('Station',))
                    aod_var = ds_out.createVariable('AOD550', 'f4', ('Station',))
                    h2o_var = ds_out.createVariable('H2O', 'f4', ('Station',))
                    o3_var = ds_out.createVariable('O3', 'f4', ('Station',))

                    # 写入数据
                    station_var[:] = np.array([s.decode() if isinstance(s, bytes) else s for s in stations], dtype='S')
                    aod_var[:] = aod_data.astype(np.float32)
                    h2o_var[:] = h2o_data.astype(np.float32)
                    o3_var[:] = o3_data.astype(np.float32)

                    # 添加属性
                    aod_var.units = "1"
                    aod_var.long_name = "Aerosol Optical Depth at 550nm"
                    h2o_var.units = "kg/m^2"
                    h2o_var.long_name = "Total Precipitable Water Vapor"
                    o3_var.units = "Dobson Units"
                    o3_var.long_name = "Total Ozone Column"

                    ds_out.setncattr('time', f"{year}{month}{day}_{time_pattern}")
                    ds_out.setncattr('resolution', '10min')
                    ds_out.setncattr('variables', 'AOD550, H2O, O3')

        logger.info("Combined dataset created successfully")


def main():
    """主函数"""
    print("=" * 60)
    print("MERRA2 1小时到10分钟插值器")
    print("=" * 60)

    interpolator = MERRA2_10min_Interpolator()

    print("开始插值处理...")
    interpolator.process_all_variables(
        start_date="20200501",
        end_date="20200507"
    )

    print("\n处理完成!")
    print(f"10分钟数据保存至: {interpolator.output_path}")


if __name__ == "__main__":
    main()