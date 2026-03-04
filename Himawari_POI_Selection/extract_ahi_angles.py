# extract_ahi_angles.py
"""
从任意一个AHI L1 NetCDF文件中提取卫星天顶角、卫星方位角及经纬度，保存为NetCDF。
支持常见的变量名变体（latitude/longitude/SAZ/SAA等），并自动处理一维或二维坐标。
"""

import xarray as xr
import argparse
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--l1_file', required=True, help='Example AHI L1 NetCDF file')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    # 打开数据集，禁用 timedelta 解码以避免 FutureWarning
    ds = xr.open_dataset(args.l1_file, decode_timedelta=False)

    # ----- 查找卫星天顶角变量 -----
    vza_var = None
    for name in ['SAZ', 'saz', 'SatelliteZenithAngle']:
        if name in ds:
            vza_var = name
            break
    if vza_var is None:
        raise KeyError("No satellite zenith angle variable found. Tried: SAZ, saz, SatelliteZenithAngle")

    # ----- 查找卫星方位角变量 -----
    saa_var = None
    for name in ['SAA', 'saa', 'SatelliteAzimuthAngle']:
        if name in ds:
            saa_var = name
            break
    if saa_var is None:
        raise KeyError("No satellite azimuth angle variable found. Tried: SAA, saa, SatelliteAzimuthAngle")

    # ----- 查找纬度变量 -----
    lat_var = None
    for name in ['latitude', 'Latitude', 'lat', 'Lat']:
        if name in ds:
            lat_var = name
            break
    if lat_var is None:
        raise KeyError("No latitude variable found. Tried: latitude, Latitude, lat, Lat")

    # ----- 查找经度变量 -----
    lon_var = None
    for name in ['longitude', 'Longitude', 'lon', 'Lon']:
        if name in ds:
            lon_var = name
            break
    if lon_var is None:
        raise KeyError("No longitude variable found. Tried: longitude, Longitude, lon, Lon")

    # 提取数据
    vza = ds[vza_var]
    saa = ds[saa_var]
    lat = ds[lat_var]
    lon = ds[lon_var]

    # 处理经纬度维度：如果是一维则广播为二维，否则直接使用
    if lat.ndim == 1 and lon.ndim == 1:
        # 假设一维坐标对应网格的行列顺序，广播生成二维网格
        lon2d, lat2d = xr.broadcast(lon, lat)
    else:
        # 如果已经是二维，直接赋值（假设维度顺序为 (y, x)）
        lat2d = lat
        lon2d = lon

    # 检查形状是否与角度数据一致
    if lat2d.shape != vza.shape:
        raise ValueError(
            f"Shape mismatch: latitude {lat2d.shape} vs VZA {vza.shape}. "
            "Please check the coordinate dimensions."
        )

    # 创建输出数据集
    out_ds = xr.Dataset(
        {
            'vza': (('y', 'x'), vza.values),
            'saa': (('y', 'x'), saa.values),
            'lat': (('y', 'x'), lat2d.values),
            'lon': (('y', 'x'), lon2d.values),
        },
        coords={'y': np.arange(vza.shape[0]), 'x': np.arange(vza.shape[1])}
    )

    # 写入属性（可选）
    out_ds['vza'].attrs['long_name'] = 'Satellite Zenith Angle'
    out_ds['vza'].attrs['units'] = 'degrees'
    out_ds['saa'].attrs['long_name'] = 'Satellite Azimuth Angle'
    out_ds['saa'].attrs['units'] = 'degrees'
    out_ds['lat'].attrs['long_name'] = 'Latitude'
    out_ds['lat'].attrs['units'] = 'degrees_north'
    out_ds['lon'].attrs['long_name'] = 'Longitude'
    out_ds['lon'].attrs['units'] = 'degrees_east'

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(args.output)
    print(f"Saved angles to {args.output}")

if __name__ == '__main__':
    main()