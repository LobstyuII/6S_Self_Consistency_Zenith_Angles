# aggregate_lc.py
import xarray as xr
import numpy as np
from scipy import stats
from utils.spatial import aggregate_to_grid
import argparse
import rioxarray

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ahi_angle', required=True)
    parser.add_argument('--lc_file', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    ahi = xr.open_dataset(args.ahi_angle)
    ahi_lat = ahi['lat'].values.ravel()
    ahi_lon = ahi['lon'].values.ravel()
    shape = ahi.dims['y'], ahi.dims['x']

    # 读取LC数据
    lc_ds = xr.open_dataset(args.lc_file) if args.lc_file.endswith('.nc') else rioxarray.open_rasterio(args.lc_file)
    # 获取经纬度
    if 'lat' in lc_ds.coords and 'lon' in lc_ds.coords:
        lc_lat = lc_ds['lat'].values.ravel()
        lc_lon = lc_ds['lon'].values.ravel()
        lc_val = lc_ds['LC_Type1'].values.ravel()
    else:
        # 处理GeoTIFF
        raise NotImplementedError

    # 对于每个AHI像素，收集其范围内的LC值，取众数
    # 使用KD树找到每个MODIS像元所属的AHI像素
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([ahi_lat, ahi_lon]))
    modis_points = np.column_stack([lc_lat, lc_lon])
    dist, idx = tree.query(modis_points, distance_upper_bound=0.045)
    valid = dist < 0.045
    ahi_indices = idx[valid]
    lc_vals = lc_val[valid]

    # 分组众数
    from collections import defaultdict, Counter
    mode_dict = {}
    for ahi_idx, lc in zip(ahi_indices, lc_vals):
        if not np.isnan(lc):
            if ahi_idx not in mode_dict:
                mode_dict[ahi_idx] = []
            mode_dict[ahi_idx].append(lc)
    lc_mode = np.full(len(ahi_lat), np.nan)
    for ahi_idx, vals in mode_dict.items():
        if vals:
            mode_result = stats.mode(vals, keepdims=True)
            lc_mode[ahi_idx] = mode_result.mode[0]

    out_ds = xr.Dataset(
        {'lc': (('y','x'), np.full(shape, np.nan))},
        coords={'y': ahi['y'].values, 'x': ahi['x'].values,
                'lat': ahi['lat'].values, 'lon': ahi['lon'].values}
    )
    out_ds['lc'].values.ravel()[:] = lc_mode
    out_ds.to_netcdf(args.output)
    print(f"Saved aggregated LC to {args.output}")

if __name__ == '__main__':
    main()