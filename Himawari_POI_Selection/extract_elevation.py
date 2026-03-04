# extract_elevation.py
import xarray as xr
import numpy as np
from utils.spatial import aggregate_to_grid
import argparse
import rioxarray

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poi_nc', required=True)
    parser.add_argument('--dem_file', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    poi = xr.open_dataset(args.poi_nc)
    lats = poi['Lat'].values
    lons = poi['Lon'].values

    # 读取DEM
    dem_ds = xr.open_dataset(args.dem_file) if args.dem_file.endswith('.nc') else rioxarray.open_rasterio(args.dem_file)
    # 获取经纬度
    if 'lat' in dem_ds.coords and 'lon' in dem_ds.coords:
        dem_lat = dem_ds['lat'].values.ravel()
        dem_lon = dem_ds['lon'].values.ravel()
        dem_ele = dem_ds['elevation'].values.ravel()
    else:
        raise NotImplementedError

    # 对每个POI，取其所在AHI像素内的DEM像元平均
    # 由于POI的经纬度对应AHI网格中心，我们需要该像素范围（约5km）内的DEM点平均
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([dem_lat, dem_lon]))
    elevations = []
    for lat, lon in zip(lats, lons):
        # 查询半径0.045°内的DEM点
        indices = tree.query_ball_point([lat, lon], 0.045)
        if indices:
            vals = dem_ele[indices]
            elevations.append(np.nanmean(vals))
        else:
            elevations.append(np.nan)

    # 添加到POI数据集
    poi['Elevation'] = ('Station', elevations)
    poi['Elevation'].attrs['units'] = 'm'
    poi.to_netcdf(args.output)
    print(f"Added elevation to {args.output}")

if __name__ == '__main__':
    main()