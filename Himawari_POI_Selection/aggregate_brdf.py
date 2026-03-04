# aggregate_brdf.py
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
from utils.spatial import aggregate_to_grid
import rioxarray  # 用于读取GeoTIFF

def read_modis_brdf(file_path, band_idx=3):
    """
    读取MCD43A1的指定波段参数和质量。
    支持NetCDF或GeoTIFF。
    返回: lat_flat, lon_flat, iso_flat, vol_flat, geo_flat, qa_flat
    """
    ds = xr.open_dataset(file_path, engine='netcdf4') if file_path.suffix == '.nc' else rioxarray.open_rasterio(file_path)
    # 假设变量名：Band{band_idx}_iso, Band{band_idx}_vol, Band{band_idx}_geo, Band{band_idx}_qa
    iso = ds[f'Band{band_idx}_iso'].values
    vol = ds[f'Band{band_idx}_vol'].values
    geo = ds[f'Band{band_idx}_geo'].values
    qa = ds[f'Band{band_idx}_qa'].values
    # 经纬度坐标
    if 'lat' in ds.coords and 'lon' in ds.coords:
        lat2d = ds['lat'].values
        lon2d = ds['lon'].values
    else:
        # 对于GeoTIFF，需要从transform计算
        # 这里简化：假设有x,y坐标，通过坐标转换（需完善）
        # 实际建议将GeoTIFF转换为NetCDF
        raise NotImplementedError("GeoTIFF经纬度提取需实现")
    lat_flat = lat2d.ravel()
    lon_flat = lon2d.ravel()
    iso_flat = iso.ravel()
    vol_flat = vol.ravel()
    geo_flat = geo.ravel()
    qa_flat = qa.ravel()
    # 质量筛选（QA=0最佳，1好）
    valid = (qa_flat <= 1) & ~np.isnan(iso_flat)
    return lat_flat[valid], lon_flat[valid], iso_flat[valid], vol_flat[valid], geo_flat[valid]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ahi_angle', required=True)
    parser.add_argument('--modis_dir', required=True)
    parser.add_argument('--band', type=int, default=3, help='MODIS band index (1-7)')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    # 加载AHI网格
    ahi = xr.open_dataset(args.ahi_angle)
    ahi_lat = ahi['lat'].values.ravel()
    ahi_lon = ahi['lon'].values.ravel()
    shape = ahi.dims['y'], ahi.dims['x']
    n_ahi = len(ahi_lat)

    # 获取所有MODIS文件
    modis_files = sorted(Path(args.modis_dir).glob('MCD43A1_2016*.nc'))  # 根据实际命名调整
    if not modis_files:
        modis_files = sorted(Path(args.modis_dir).glob('*.tif'))

    # 初始化每日结果数组 (days, n_ahi)
    n_days = len(modis_files)
    iso_daily = np.full((n_days, n_ahi), np.nan)
    vol_daily = np.full((n_days, n_ahi), np.nan)
    geo_daily = np.full((n_days, n_ahi), np.nan)

    for day_idx, file in enumerate(tqdm(modis_files, desc="Processing days")):
        try:
            mlat, mlon, miso, mvol, mgeo = read_modis_brdf(file, args.band)
            agg_iso, _ = aggregate_to_grid(mlat, mlon, miso, ahi_lat, ahi_lon)
            agg_vol, _ = aggregate_to_grid(mlat, mlon, mvol, ahi_lat, ahi_lon)
            agg_geo, _ = aggregate_to_grid(mlat, mlon, mgeo, ahi_lat, ahi_lon)
            iso_daily[day_idx] = agg_iso
            vol_daily[day_idx] = agg_vol
            geo_daily[day_idx] = agg_geo
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # 计算每日AI
    ai_daily = (vol_daily + geo_daily) / iso_daily
    ai_daily[iso_daily <= 0] = np.nan

    # 统计
    ai_mean = np.nanmean(ai_daily, axis=0)
    ai_std = np.nanstd(ai_daily, axis=0)
    ai_cv = ai_std / ai_mean
    valid_days = np.sum(~np.isnan(ai_daily), axis=0)

    # 输出
    out_ds = xr.Dataset(
        {
            'ai_mean': (('y','x'), np.full(shape, np.nan)),
            'ai_std': (('y','x'), np.full(shape, np.nan)),
            'ai_cv': (('y','x'), np.full(shape, np.nan)),
            'valid_days': (('y','x'), np.zeros(shape, dtype=int)),
        },
        coords={'y': ahi['y'].values, 'x': ahi['x'].values,
                'lat': ahi['lat'].values, 'lon': ahi['lon'].values}
    )
    out_ds['ai_mean'].values.ravel()[:] = ai_mean
    out_ds['ai_std'].values.ravel()[:] = ai_std
    out_ds['ai_cv'].values.ravel()[:] = ai_cv
    out_ds['valid_days'].values.ravel()[:] = valid_days
    out_ds.to_netcdf(args.output)
    print(f"Saved aggregated BRDF to {args.output}")

if __name__ == '__main__':
    main()