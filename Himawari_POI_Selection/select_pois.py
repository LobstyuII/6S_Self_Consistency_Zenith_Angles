# select_pois.py
import xarray as xr
import pandas as pd
import numpy as np
import argparse
from config import LC_LAND_CODES, AI_THRESHOLD, CV_THRESHOLD, MIN_VALID_DAYS, SAA_BINS, VZA_BINS, POIS_PER_CELL

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ai_stats_nc', required=True, help='Input NetCDF file with AI statistics for candidate points')
    parser.add_argument('--output_csv', required=True, help='Output CSV file for final POI list')
    parser.add_argument('--output_nc', required=True, help='Output NetCDF file for final POI list')
    parser.add_argument('--ai_threshold', type=float, default=None,
                        help='Override AI_THRESHOLD from config (default: use config value)')
    args = parser.parse_args()

    # 确定使用的 AI 阈值
    ai_threshold = args.ai_threshold if args.ai_threshold is not None else AI_THRESHOLD

    ds = xr.open_dataset(args.ai_stats_nc)
    df = pd.DataFrame({
        'station_id': np.arange(len(ds['Station'])),
        'row': ds['Row'].values,
        'col': ds['Col'].values,
        'lat': ds['Lat'].values,
        'lon': ds['Lon'].values,
        'lc': ds['LC'].values,
        'vza': ds['VZA'].values,
        'saa': ds['SAA'].values,
        'ai_mean': ds['AI_mean'].values,
        'ai_cv': ds['AI_cv'].values,
        'valid_days': ds['ValidDays'].values,
        'elevation': ds['Elevation'].values,
    })
    if 'ValidCount' in ds:
        df['valid_count'] = ds['ValidCount'].values

    # 打印所有候选点 AI_mean 的统计信息（分位数）
    print("\n===== AI_mean statistics for all candidate points =====")
    print(df['ai_mean'].describe(percentiles=[.1, .25, .5, .75, .9]))
    print("=======================================================\n")

    land_mask = np.isin(df['lc'], LC_LAND_CODES)
    ai_mask = (df['ai_mean'] < ai_threshold) & (df['ai_cv'] < CV_THRESHOLD) & (df['valid_days'] >= MIN_VALID_DAYS)
    mask = land_mask & ai_mask & ~np.isnan(df['vza']) & ~np.isnan(df['saa'])
    df_filtered = df[mask].copy()
    if df_filtered.empty:
        print("No points meet criteria.")
        return

    df_filtered['saa_bin'] = pd.cut(df_filtered['saa'], bins=SAA_BINS, labels=False, right=False)
    df_filtered['vza_bin'] = pd.cut(df_filtered['vza'], bins=VZA_BINS, labels=False, right=False)

    selected = []
    grouped = df_filtered.groupby(['lc', 'saa_bin', 'vza_bin'])
    for (lc_code, saa_bin, vza_bin), group in grouped:
        group_sorted = group.sort_values('ai_mean')
        n_take = min(POIS_PER_CELL, len(group_sorted))
        selected.append(group_sorted.head(n_take))

    result = pd.concat(selected, ignore_index=True)
    result.to_csv(args.output_csv, index=False)

    n_poi = len(result)
    station_names = [f"POI_{i:04d}" for i in range(n_poi)]
    out_ds = xr.Dataset(
        {
            'Station': ('Station', np.array(station_names, dtype='S')),
            'Lat': ('Station', result['lat'].values),
            'Lon': ('Station', result['lon'].values),
            'LC': ('Station', result['lc'].values),
            'VZA': ('Station', result['vza'].values),
            'SAA': ('Station', result['saa'].values),
            'AI_mean': ('Station', result['ai_mean'].values),
            'AI_cv': ('Station', result['ai_cv'].values),
            'ValidDays': ('Station', result['valid_days'].values),
            'Row': ('Station', result['row'].values),
            'Col': ('Station', result['col'].values),
            'Elevation': ('Station', result['elevation'].values),
        }
    )
    out_ds.to_netcdf(args.output_nc)
    print(f"Saved POI list to {args.output_nc}")

if __name__ == '__main__':
    main()