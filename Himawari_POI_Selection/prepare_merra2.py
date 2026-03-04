# prepare_merra2.py
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def extract_for_poi(poi_lat, poi_lon, merra2_dir, times, out_file):
    """提取单个POI的时间序列"""
    n_times = len(times)
    aod = np.full(n_times, np.nan)
    h2o = np.full(n_times, np.nan)
    o3 = np.full(n_times, np.nan)

    for i, t in enumerate(tqdm(times, desc=f"POI ({poi_lat:.2f},{poi_lon:.2f})", leave=False)):
        year = t.strftime('%Y')
        month = t.strftime('%m')
        day = t.strftime('%d')
        hour_min = t.strftime('%H%M')
        f = merra2_dir / year / month / f"MERRA2_combined_{year}{month}{day}_{hour_min}.nc"
        if not f.exists():
            continue
        ds = xr.open_dataset(f)
        # 找到最近的站点
        stations_lat = ds['Lat'].values
        stations_lon = ds['Lon'].values
        dist = np.hypot(stations_lat - poi_lat, stations_lon - poi_lon)
        idx = np.argmin(dist)
        aod[i] = ds['AOD550'].isel(Station=idx).values
        h2o[i] = ds['H2O'].isel(Station=idx).values
        o3[i] = ds['O3'].isel(Station=idx).values
        ds.close()

    out_ds = xr.Dataset(
        {
            'AOD550': ('time', aod),
            'H2O': ('time', h2o),
            'O3': ('time', o3),
        },
        coords={'time': times}
    )
    out_ds.to_netcdf(out_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poi_nc', required=True)
    parser.add_argument('--merra2_dir', required=True)
    parser.add_argument('--start_date', default='2016-05-01')
    parser.add_argument('--end_date', default='2016-05-08')
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    poi = xr.open_dataset(args.poi_nc)
    lats = poi['Lat'].values
    lons = poi['Lon'].values
    stations = poi['Station'].values

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    times = pd.date_range(start=args.start_date, end=args.end_date, freq='10min', inclusive='left')

    for i, (lat, lon, name) in enumerate(zip(lats, lons, stations)):
        out_file = Path(args.output_dir) / f"{name.decode() if isinstance(name, bytes) else name}_merra2.nc"
        extract_for_poi(lat, lon, Path(args.merra2_dir), times, out_file)

if __name__ == '__main__':
    main()