# utils/spatial.py
import numpy as np
from scipy.spatial import cKDTree

def build_kdtree(lat_flat, lon_flat):
    """构建经纬度KD树"""
    points = np.column_stack([lat_flat, lon_flat])
    return cKDTree(points)

def aggregate_to_grid(modis_lat, modis_lon, modis_values,
                      grid_lat, grid_lon, max_dist_deg=0.045):
    """
    将MODIS点聚合到网格点（取平均）。
    modis_lat, modis_lon, modis_values: 一维数组
    grid_lat, grid_lon: 网格中心点经纬度（一维）
    max_dist_deg: 距离阈值（度）
    返回: (agg_values, count) 长度等于网格点数
    """
    tree = build_kdtree(grid_lat, grid_lon)
    modis_points = np.column_stack([modis_lat, modis_lon])
    distances, indices = tree.query(modis_points, distance_upper_bound=max_dist_deg)
    valid = distances < max_dist_deg
    grid_indices = indices[valid]
    vals = modis_values[valid]

    # 按网格索引分组平均
    from collections import defaultdict
    sum_dict = defaultdict(float)
    count_dict = defaultdict(int)
    for idx, val in zip(grid_indices, vals):
        sum_dict[idx] += val
        count_dict[idx] += 1
    n_grid = len(grid_lat)
    agg = np.full(n_grid, np.nan)
    cnt = np.zeros(n_grid, dtype=int)
    for idx in sum_dict:
        agg[idx] = sum_dict[idx] / count_dict[idx]
        cnt[idx] = count_dict[idx]
    return agg, cnt