# extract_modis_at_pois.py
"""
从 GEE 提取 MODIS BRDF、LC、DEM 数据到候选点列表。
每个点对应 AHI 5km 像素，BRDF 提取使用精确像素矩形（0.05°），LC/DEM 仍使用缓冲区。
支持可选 QA 掩膜，默认不掩膜。
"""
import ee
import xarray as xr
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging
import concurrent.futures
from tqdm import tqdm
import sys
import time
import math
import random

sys.path.append(str(Path(__file__).parent))
from config import GEE_CREDENTIALS, START_DATE, END_DATE, BAND_MATCH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def initialize_gee(credentials_path):
    """初始化 GEE，使用服务账号"""
    try:
        credentials = ee.ServiceAccountCredentials(None, credentials_path)
        ee.Initialize(credentials)
        logger.info("GEE initialized successfully.")
    except Exception as e:
        logger.error(f"GEE initialization failed: {e}")
        raise


def create_point_features(df_batch, buffer_m=2500):
    """
    创建点缓冲区要素（用于 LC/DEM 提取）
    """
    features = []
    for _, row in df_batch.iterrows():
        point = ee.Geometry.Point([row['lon'], row['lat']])
        feature = ee.Feature(point, {'station_id': int(row['station_id'])})
        features.append(feature)
    fc = ee.FeatureCollection(features)
    return fc.map(lambda f: f.buffer(buffer_m))


def create_pixel_rectangle(lat, lon):
    """
    根据 Himawari 5km 网格定义计算像素矩形边界。
    网格公式: longitude = 80 + x * 0.05, latitude = 60 - y * 0.05
    """
    x_index = int(math.floor((lon - 80) / 0.05))
    y_index = int(math.floor((60 - lat) / 0.05))
    west = 80 + x_index * 0.05
    east = west + 0.05
    north = 60 - y_index * 0.05
    south = north - 0.05
    return west, south, east, north


def create_pixel_rectangle_features(df_batch):
    """
    为每个候选点创建 Himawari 像素矩形 FeatureCollection（用于 BRDF 提取）
    """
    features = []
    for _, row in df_batch.iterrows():
        west, south, east, north = create_pixel_rectangle(row['lat'], row['lon'])
        rect = ee.Geometry.Rectangle([west, south, east, north])
        feature = ee.Feature(rect, {'station_id': int(row['station_id'])})
        features.append(feature)
    return ee.FeatureCollection(features)


def batch_extract_lc(df, buffer_m=2500, lc_year=2016, batch_size=1000):
    """
    分批提取土地覆盖类型（众数）
    """
    logger.info("Starting land cover extraction (mode reducer on MCD12Q1 LC_Type1)...")
    start_time = time.time()
    total_points = len(df)
    num_batches = math.ceil(total_points / batch_size)
    logger.info(f"Total points: {total_points}, batch size: {batch_size}, batches: {num_batches}")

    lc_img = ee.Image(f"MODIS/061/MCD12Q1/{lc_year}_01_01").select('LC_Type1')
    lc_values = {}

    for batch_idx in tqdm(range(num_batches), desc="LC batches"):
        start = batch_idx * batch_size
        batch_df = df.iloc[start: start + batch_size]
        batch_fc = create_point_features(batch_df, buffer_m)

        for attempt in range(5):
            try:
                reduced = lc_img.reduceRegions(
                    collection=batch_fc,
                    reducer=ee.Reducer.mode(),
                    scale=500,
                    crs='EPSG:4326'
                )
                batch_info = reduced.getInfo()
                break
            except Exception as e:
                wait = (2 ** attempt) + random.random()
                logger.warning(f"LC batch {batch_idx} attempt {attempt+1} failed: {e}. Retrying in {wait:.2f}s")
                time.sleep(wait)
        else:
            logger.error(f"LC batch {batch_idx} failed after 5 attempts, skipping.")
            continue

        for feat in batch_info['features']:
            props = feat['properties']
            sid = props['station_id']
            lc = props.get('mode', -9999)
            if lc is None:
                lc = -9999
            lc_values[sid] = lc

    elapsed = time.time() - start_time
    logger.info(f"Land cover extraction completed in {elapsed:.2f} seconds. Retrieved {len(lc_values)} points.")
    return lc_values


def batch_extract_dem(df, buffer_m=2500, batch_size=1000):
    """
    分批提取 DEM（平均值）
    """
    logger.info("Starting DEM extraction (mean reducer on SRTM elevation)...")
    start_time = time.time()
    total_points = len(df)
    num_batches = math.ceil(total_points / batch_size)
    logger.info(f"Total points: {total_points}, batch size: {batch_size}, batches: {num_batches}")

    dem_img = ee.Image("USGS/SRTMGL1_003").select('elevation')
    dem_values = {}

    for batch_idx in tqdm(range(num_batches), desc="DEM batches"):
        start = batch_idx * batch_size
        batch_df = df.iloc[start: start + batch_size]
        batch_fc = create_point_features(batch_df, buffer_m)

        for attempt in range(5):
            try:
                reduced = dem_img.reduceRegions(
                    collection=batch_fc,
                    reducer=ee.Reducer.mean(),
                    scale=90,
                    crs='EPSG:4326'
                )
                batch_info = reduced.getInfo()
                break
            except Exception as e:
                wait = (2 ** attempt) + random.random()
                logger.warning(f"DEM batch {batch_idx} attempt {attempt+1} failed: {e}. Retrying in {wait:.2f}s")
                time.sleep(wait)
        else:
            logger.error(f"DEM batch {batch_idx} failed after 5 attempts, skipping.")
            continue

        for feat in batch_info['features']:
            props = feat['properties']
            sid = props['station_id']
            dem = props.get('mean', -9999)
            if dem is None:
                dem = -9999
            dem_values[sid] = dem

    elapsed = time.time() - start_time
    logger.info(f"DEM extraction completed in {elapsed:.2f} seconds. Retrieved {len(dem_values)} points.")
    return dem_values


def batch_process_daily_brdf(date, df, band_idx, use_qa=False, batch_size=100):
    """
    分批处理单日 BRDF 数据。
    如果 use_qa=True，则选择 QA 波段并应用掩膜 (qa <= 1)；
    否则不进行 QA 筛选，直接提取均值。
    返回三个数组 (iso_vals, vol_vals, geo_vals)，长度与 df 相同，未成功点为 NaN。
    """
    band_iso = f'BRDF_Albedo_Parameters_Band{band_idx}_iso'
    band_vol = f'BRDF_Albedo_Parameters_Band{band_idx}_vol'
    band_geo = f'BRDF_Albedo_Parameters_Band{band_idx}_geo'

    start = ee.Date(date.strftime('%Y-%m-%d'))
    end = start.advance(1, 'day')

    if use_qa:
        band_qa = f'BRDF_Albedo_Band_Mandatory_Quality_Band{band_idx}'
        col = ee.ImageCollection("MODIS/061/MCD43A1") \
            .filterDate(start, end) \
            .select([band_iso, band_vol, band_geo, band_qa])
    else:
        col = ee.ImageCollection("MODIS/061/MCD43A1") \
            .filterDate(start, end) \
            .select([band_iso, band_vol, band_geo])

    try:
        img = col.first()
        if img is None:
            logger.warning(f"No image for {date.strftime('%Y-%m-%d')}")
            return None
    except Exception as e:
        logger.warning(f"Error accessing image for {date.strftime('%Y-%m-%d')}: {e}")
        return None

    if use_qa:
        qa = img.select(band_qa)
        mask = qa.lte(1)
        img_masked = img.updateMask(mask)
        img_to_use = img_masked
    else:
        img_to_use = img

    total_points = len(df)
    num_batches = math.ceil(total_points / batch_size)

    iso_vals = np.full(total_points, np.nan)
    vol_vals = np.full(total_points, np.nan)
    geo_vals = np.full(total_points, np.nan)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_points)
        batch_df = df.iloc[start_idx:end_idx]
        batch_fc = create_pixel_rectangle_features(batch_df)

        for attempt in range(5):
            try:
                reduced = img_to_use.select([band_iso, band_vol, band_geo]).reduceRegions(
                    collection=batch_fc,
                    reducer=ee.Reducer.mean(),
                    scale=500,
                    crs='EPSG:4326'
                )
                batch_info = reduced.getInfo()
                break
            except Exception as e:
                wait = (2 ** attempt) + random.random()
                logger.warning(f"Date {date.strftime('%Y-%m-%d')} batch {batch_idx} attempt {attempt+1} failed: {e}. Retrying in {wait:.2f}s")
                time.sleep(wait)
        else:
            logger.error(f"Date {date.strftime('%Y-%m-%d')} batch {batch_idx} failed after 5 attempts, skipping.")
            continue

        for feat in batch_info['features']:
            props = feat['properties']
            sid = props['station_id']
            iso = props.get(f'{band_iso}_mean', np.nan)
            vol = props.get(f'{band_vol}_mean', np.nan)
            geo = props.get(f'{band_geo}_mean', np.nan)

            if start_idx <= sid < end_idx:
                iso_vals[sid] = iso if iso is not None else np.nan
                vol_vals[sid] = vol if vol is not None else np.nan
                geo_vals[sid] = geo if geo is not None else np.nan
            else:
                logger.warning(f"sid {sid} out of batch range [{start_idx}, {end_idx})")

        time.sleep(0.2)  # 批次间短暂休眠

    n_success = np.sum(~np.isnan(iso_vals))
    logger.info(f"Date {date.strftime('%Y-%m-%d')}: {n_success}/{total_points} points retrieved.")
    return iso_vals, vol_vals, geo_vals


def save_brdf_cache(cache_file, iso, vol, geo, station_ids):
    """
    将一天的 BRDF 数据保存为 NetCDF 文件。
    """
    ds = xr.Dataset(
        {
            'iso': ('station', iso),
            'vol': ('station', vol),
            'geo': ('station', geo),
            'station_id': ('station', station_ids),
        }
    )
    ds.to_netcdf(cache_file)
    logger.info(f"Saved BRDF cache to {cache_file}")


def load_brdf_cache(cache_file, n_points):
    """
    从缓存文件加载 BRDF 数据，返回 (iso, vol, geo) 数组。
    如果文件不存在或 station_id 不匹配，返回 None。
    """
    if not Path(cache_file).exists():
        return None
    try:
        ds = xr.open_dataset(cache_file)
        cached_ids = ds['station_id'].values
        expected_ids = np.arange(n_points)
        if not np.array_equal(cached_ids, expected_ids):
            logger.warning(f"Cache file {cache_file} station_id mismatch, ignoring.")
            return None
        iso = ds['iso'].values
        vol = ds['vol'].values
        geo = ds['geo'].values
        ds.close()
        logger.info(f"Loaded BRDF cache from {cache_file}")
        return iso, vol, geo
    except Exception as e:
        logger.warning(f"Failed to load cache {cache_file}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates_nc', required=True, help='NetCDF file with candidate points (from select_top_pixels.py)')
    parser.add_argument('--output_nc', required=True, help='Output NetCDF file with AI statistics')
    parser.add_argument('--gee_key', default=GEE_CREDENTIALS, help='GEE service account JSON key')
    parser.add_argument('--start_date', default=START_DATE, help='Start date YYYY-MM-DD')
    parser.add_argument('--end_date', default=END_DATE, help='End date YYYY-MM-DD (exclusive)')
    parser.add_argument('--modis_band', type=int, default=3, help='MODIS band index (1-7)')
    parser.add_argument('--buffer_m', type=float, default=2500, help='Buffer radius in meters for LC/DEM extraction (ignored for BRDF)')
    parser.add_argument('--max_workers', type=int, default=2, help='Parallel threads (recommended ≤2 for stability)')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of points per batch (50-200)')
    parser.add_argument('--brdf_cache_dir', help='Directory to cache daily BRDF data (optional)')
    parser.add_argument('--use_qa', action='store_true', help='Apply QA mask (qa <= 1) for BRDF extraction (default: False)')
    args = parser.parse_args()

    # 初始化 GEE
    logger.info("Initializing GEE...")
    initialize_gee(args.gee_key)

    logger.info(f"Loading candidate points from {args.candidates_nc}")
    cand_ds = xr.open_dataset(args.candidates_nc)
    df = pd.DataFrame({
        'station_id': np.arange(len(cand_ds['Station'])),
        'row': cand_ds['Row'].values,
        'col': cand_ds['Col'].values,
        'lat': cand_ds['Lat'].values,
        'lon': cand_ds['Lon'].values,
    })
    logger.info(f"Loaded {len(df)} candidate points")

    # 提取 LC 和 DEM（使用缓冲区）
    lc_values = batch_extract_lc(df, buffer_m=args.buffer_m, lc_year=2016, batch_size=args.batch_size)
    dem_values = batch_extract_dem(df, buffer_m=args.buffer_m, batch_size=args.batch_size)

    # 准备日期列表
    dates = pd.date_range(start=args.start_date, end=args.end_date, freq='D', inclusive='left')
    n_days = len(dates)
    n_points = len(df)

    # 初始化每日数组
    iso_daily = np.full((n_days, n_points), np.nan)
    vol_daily = np.full((n_days, n_points), np.nan)
    geo_daily = np.full((n_days, n_points), np.nan)

    if args.brdf_cache_dir:
        cache_dir = Path(args.brdf_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    else:
        cache_dir = None

    logger.info(f"Processing {n_days} days of BRDF data with up to {args.max_workers} workers...")

    def process_day(date_idx):
        date, idx = date_idx
        date_str = date.strftime('%Y-%m-%d')

        # 检查缓存
        if cache_dir:
            cache_file = cache_dir / f"brdf_{date_str}.nc"
            cached = load_brdf_cache(cache_file, n_points)
            if cached is not None:
                return idx, cached[0], cached[1], cached[2]

        # 在线提取
        try:
            result = batch_process_daily_brdf(
                date, df, args.modis_band,
                use_qa=args.use_qa,
                batch_size=args.batch_size
            )
            if result is None:
                return idx, None, None, None
            iso_vals, vol_vals, geo_vals = result

            # 保存缓存
            if cache_dir:
                save_brdf_cache(cache_file, iso_vals, vol_vals, geo_vals, df['station_id'].values)

            return idx, iso_vals, vol_vals, geo_vals
        except Exception as e:
            logger.error(f"Error processing {date_str}: {e}")
            return idx, None, None, None

    # 使用线程池并发处理每日数据
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_day, (date, i)): i for i, date in enumerate(dates)}
        for future in tqdm(concurrent.futures.as_completed(futures), total=n_days, desc="Processing BRDF days"):
            i, iso_vals, vol_vals, geo_vals = future.result()
            if iso_vals is not None:
                iso_daily[i, :] = iso_vals
                vol_daily[i, :] = vol_vals
                geo_daily[i, :] = geo_vals

    logger.info("Computing anisotropy index (AI) statistics...")
    ai_daily = (vol_daily + geo_daily) / iso_daily
    ai_daily[iso_daily <= 0] = np.nan

    ai_mean = np.nanmean(ai_daily, axis=0)
    ai_std = np.nanstd(ai_daily, axis=0)
    ai_cv = ai_std / ai_mean
    valid_days = np.sum(~np.isnan(ai_daily), axis=0)

    n_valid_ai = np.sum(~np.isnan(ai_mean))
    logger.info(f"Points with valid AI_mean: {n_valid_ai}/{n_points}")

    logger.info("Assembling output dataset...")
    out_ds = xr.Dataset(
        {
            'Station': ('Station', cand_ds['Station'].values),
            'Row': ('Station', df['row'].values),
            'Col': ('Station', df['col'].values),
            'Lat': ('Station', df['lat'].values),
            'Lon': ('Station', df['lon'].values),
            'VZA': ('Station', cand_ds['VZA'].values if 'VZA' in cand_ds else np.nan),
            'SAA': ('Station', cand_ds['SAA'].values if 'SAA' in cand_ds else np.nan),
            'LC': ('Station', [lc_values.get(i, -9999) for i in range(n_points)]),
            'Elevation': ('Station', [dem_values.get(i, np.nan) for i in range(n_points)]),
            'AI_mean': ('Station', ai_mean),
            'AI_std': ('Station', ai_std),
            'AI_cv': ('Station', ai_cv),
            'ValidDays': ('Station', valid_days),
        }
    )
    if 'ValidCount' in cand_ds:
        out_ds['ValidCount'] = ('Station', cand_ds['ValidCount'].values)

    logger.info(f"Saving results to {args.output_nc}")
    out_ds.to_netcdf(args.output_nc)
    logger.info("All done.")


if __name__ == '__main__':
    main()