# prepare_local_brdf.py
"""
将本地 MCD43A1 影像（NetCDF 或 GeoTIFF）聚合到 AHI 5km 网格，
输出包含每日 ISO、VOL、GEO 的 NetCDF 文件。
"""
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
from utils.spatial import aggregate_to_grid
import rioxarray
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_modis_brdf(file_path, band_idx=3):
    """
    读取 MCD43A1 的指定波段参数。
    支持 NetCDF 或 GeoTIFF。
    返回: lat_flat, lon_flat, iso_flat, vol_flat, geo_flat, qa_flat
    """
    if file_path.suffix == '.nc':
        ds = xr.open_dataset(file_path)
    else:
        ds = rioxarray.open_rasterio(file_path)

    # 假设变量名规则：Band{band_idx}_iso, Band{band_idx}_vol, Band{band_idx}_geo, Band{band_idx}_qa
    iso = ds[f'Band{band_idx}_iso'].values
    vol = ds[f'Band{band_idx}_vol'].values
    geo = ds[f'Band{band_idx}_geo'].values
    qa = ds[f'Band{band_idx}_qa'].values

    # 获取经纬度坐标
    if 'lat' in ds.coords and 'lon' in ds.coords:
        lat2d = ds['lat'].values
        lon2d = ds['lon'].values
    else:
        # 对于 GeoTIFF，需要从 transform 计算
        # 这里简化处理：假设有 x, y 坐标，需通过坐标转换（实际使用时建议先转换为 NetCDF）
        raise NotImplementedError("GeoTIFF 经纬度提取需实现，建议先将 GeoTIFF 转换为 NetCDF")

    lat_flat = lat2d.ravel()
    lon_flat = lon2d.ravel()
    iso_flat = iso.ravel()
    vol_flat = vol.ravel()
    geo_flat = geo.ravel()
    qa_flat = qa.ravel()

    # 质量筛选（QA=0 最佳，1 好）
    valid = (qa_flat <= 1) & ~np.isnan(iso_flat)
    return lat_flat[valid], lon_flat[valid], iso_flat[valid], vol_flat[valid], geo_flat[valid]


def main():
    parser = argparse.ArgumentParser(description='Aggregate local MCD43A1 files to AHI 5km grid')
    parser.add_argument('--ahi_angle', required=True, help='AHI angle file (from extract_ahi_angles.py)')
    parser.add_argument('--modis_dir', required=True, help='Directory containing MCD43A1 files')
    parser.add_argument('--band', type=int, default=3, help='MODIS band index (1-7)')
    parser.add_argument('--start_date', required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end_date', required=True, help='End date YYYY-MM-DD (exclusive)')
    parser.add_argument('--output', required=True, help='Output NetCDF file with daily BRDF parameters')
    args = parser.parse_args()

    # 加载 AHI 网格
    ahi = xr.open_dataset(args.ahi_angle)
    ahi_lat = ahi['lat'].values.ravel()
    ahi_lon = ahi['lon'].values.ravel()
    shape = ahi.dims['y'], ahi.dims['x']
    n_ahi = len(ahi_lat)

    # 生成日期列表
    dates = pd.date_range(start=args.start_date, end=args.end_date, freq='D', inclusive='left')
    n_days = len(dates)
    logger.info(f"Processing {n_days} days")

    # 初始化每日结果数组 (days, n_ahi)
    iso_daily = np.full((n_days, n_ahi), np.nan)
    vol_daily = np.full((n_days, n_ahi), np.nan)
    geo_daily = np.full((n_days, n_ahi), np.nan)

    # 遍历每一天
    for day_idx, date in enumerate(tqdm(dates, desc="Processing days")):
        date_str = date.strftime('%Y%m%d')
        # 在 modis_dir 中查找对应日期的文件（可根据实际命名调整）
        possible_files = list(Path(args.modis_dir).glob(f'*{date_str}*.nc')) + \
                         list(Path(args.modis_dir).glob(f'*{date_str}*.tif'))
        if not possible_files:
            logger.warning(f"No file found for {date_str}, skipping")
            continue

        file = possible_files[0]  # 取第一个匹配文件
        try:
            mlat, mlon, miso, mvol, mgeo = read_modis_brdf(file, args.band)
            # 聚合到 AHI 网格（使用 nearest 或 average，这里用 average 以匹配 reduceRegions 的 mean）
            agg_iso, _ = aggregate_to_grid(mlat, mlon, miso, ahi_lat, ahi_lon, method='average')
            agg_vol, _ = aggregate_to_grid(mlat, mlon, mvol, ahi_lat, ahi_lon, method='average')
            agg_geo, _ = aggregate_to_grid(mlat, mlon, mgeo, ahi_lat, ahi_lon, method='average')
            iso_daily[day_idx] = agg_iso
            vol_daily[day_idx] = agg_vol
            geo_daily[day_idx] = agg_geo
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")

    # 输出数据集
    out_ds = xr.Dataset(
        {
            'iso': (('time', 'y', 'x'), iso_daily.reshape((n_days, shape[0], shape[1]))),
            'vol': (('time', 'y', 'x'), vol_daily.reshape((n_days, shape[0], shape[1]))),
            'geo': (('time', 'y', 'x'), geo_daily.reshape((n_days, shape[0], shape[1]))),
        },
        coords={
            'time': dates,
            'y': ahi['y'].values,
            'x': ahi['x'].values,
            'lat': (('y', 'x'), ahi['lat'].values),
            'lon': (('y', 'x'), ahi['lon'].values),
        }
    )
    out_ds['iso'].attrs['long_name'] = 'Isotropic parameter'
    out_ds['vol'].attrs['long_name'] = 'Volumetric parameter'
    out_ds['geo'].attrs['long_name'] = 'Geometric parameter'
    out_ds.to_netcdf(args.output)
    logger.info(f"Saved daily BRDF to {args.output}")


if __name__ == '__main__':
    main()