# select_top_pixels.py
"""
分析AHI L2ARP文件，使用QA_flag解码每个像素的晴空陆地有效观测次数，
选择前10%的像素作为候选点。
输出候选点列表（包含行列、经纬度、角度、有效次数）。
"""
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def decode_qa_flag(qa_value):
    """
    根据Downloader_Himawari.py中的比特位定义解码QA_flag
    返回一个包含各标志的字典
    """
    return {
        'Data_Availability': qa_value & 1,                     # 比特0
        'Land_Water_Flag': (qa_value >> 1) & 1,                # 比特1
        'Cloud_Flag': (qa_value >> 2) & 1,                      # 比特2
        'Retrieval_Status': (qa_value >> 3) & 1,                # 比特3
        'AOT_Confidence': (qa_value >> 4) & 0x03,               # 比特4-5
        'AE_Confidence': (qa_value >> 6) & 0x03,                # 比特6-7
        'SSA_Confidence': (qa_value >> 8) & 0x03,               # 比特8-9
        'Additional_Cloud_Flag': (qa_value >> 10) & 1,          # 比特10
        'Sunglint': (qa_value >> 11) & 1,                        # 比特11
        'Angle_Threshold': (qa_value >> 12) & 1,                 # 比特12
        'Surface_Reflectance_Confidence': (qa_value >> 13) & 1, # 比特13
        'Snow_Ice': (qa_value >> 14) & 1,                        # 比特14
        'Turbid_Water': (qa_value >> 15) & 1                     # 比特15
    }

def is_valid_pixel(flags):
    """
    根据标志判断是否为有效晴空陆地像素
    条件：
    - Data_Availability == 0 (数据有效)
    - Land_Water_Flag == 0 (陆地, 假设0为陆地，1为水体)
    - Cloud_Flag == 0 (晴空)
    - Retrieval_Status == 0 (反演成功)
    - Additional_Cloud_Flag == 0 (无额外云)
    - Turbid_Water == 0 (非浑浊水体)
    - Snow_Ice == 0 (无雪/冰)
    - Angle_Threshold == 0 (太阳/卫星角未超过70°)
    - Sunglint == 0 (无太阳耀斑)
    """
    return (flags['Data_Availability'] == 0 and
            flags['Land_Water_Flag'] == 0 and
            flags['Cloud_Flag'] == 0 and
            flags['Retrieval_Status'] == 0 and
            flags['Additional_Cloud_Flag'] == 0 and
            flags['Turbid_Water'] == 0 and
            flags['Snow_Ice'] == 0 and
            flags['Angle_Threshold'] == 0 and
            flags['Sunglint'] == 0)

def find_common_varname(ds, possible_names, var_type='variable'):
    """在数据集中查找可能的变量名，返回第一个存在的"""
    for name in possible_names:
        if name in ds.variables:
            return name
    raise KeyError(f"None of {possible_names} found in dataset for {var_type}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--l2_dir', required=True, help='Directory containing H8L2ARP_10min files (with subfolders year/month)')
    parser.add_argument('--angle_file', required=True, help='AHI angle file (from extract_ahi_angles.py)')
    parser.add_argument('--start_date', required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end_date', required=True, help='End date YYYY-MM-DD (exclusive)')
    parser.add_argument('--top_percent', type=float, default=10, help='Percentage of top valid pixels to select')
    parser.add_argument('--max_points', type=int, default=None, help='Maximum number of points to select (overrides top_percent)')
    parser.add_argument('--output_csv', required=True, help='Output CSV file with candidate points')
    parser.add_argument('--output_nc', required=True, help='Output NetCDF file with candidate points')
    parser.add_argument('--qa_var', default='QA_flag', help='Variable name for QA flag in L2 files')
    args = parser.parse_args()

    # 加载角度文件获取网格信息
    angle_ds = xr.open_dataset(args.angle_file)
    lat = angle_ds['lat'].values
    lon = angle_ds['lon'].values
    vza = angle_ds['vza'].values
    saa = angle_ds['saa'].values
    shape = vza.shape
    rows, cols = shape
    logger.info(f"Angle grid shape: {shape}")

    # 初始化有效计数数组
    valid_count = np.zeros(shape, dtype=np.int32)

    # 获取日期范围内的所有L2文件
    l2_dir = Path(args.l2_dir)
    pattern = re.compile(r'.*_(\d{8})_(\d{4})\.nc$')
    files_in_range = []
    for f in l2_dir.rglob('*.nc'):
        m = pattern.search(f.name)
        if m:
            ymd = m.group(1)
            if args.start_date.replace('-','') <= ymd < args.end_date.replace('-',''):
                files_in_range.append(f)
    logger.info(f"Found {len(files_in_range)} L2ARP files in date range")

    # 遍历文件，累计有效计数
    for f in tqdm(files_in_range, desc="Scanning L2 files"):
        try:
            # 添加 decode_timedelta=False 消除警告，并指定 engine 为 'netcdf4' 避免自动检测失败
            ds = xr.open_dataset(f, decode_timedelta=False, engine='netcdf4')
            # 查找QA_flag变量
            qa_var = find_common_varname(ds, [args.qa_var, 'QA_flag', 'QA_Flag', 'QA'])
            qa = ds[qa_var].values
            if qa.shape != shape:
                logger.warning(f"File {f} has shape {qa.shape}, expected {shape}. Skipping.")
                ds.close()
                continue

            # 逐像素解码并判断有效
            # 为加速，使用向量化操作（但比特位操作仍需循环，可尝试优化）
            valid_mask = np.zeros(shape, dtype=bool)
            # 使用numpy的向量化操作比较耗时，但可接受
            for i in range(rows):
                for j in range(cols):
                    flags = decode_qa_flag(qa[i, j])
                    valid_mask[i, j] = is_valid_pixel(flags)

            valid_count += valid_mask.astype(np.int32)
            ds.close()
        except Exception as e:
            logger.error(f"Error processing {f}: {e}")
            continue

    logger.info(f"Valid count stats: min={valid_count.min()}, max={valid_count.max()}, mean={valid_count.mean():.2f}")

    # 只考虑有观测的陆地像素
    land_pixels = np.where(valid_count > 0)
    n_land = len(land_pixels[0])
    logger.info(f"Number of land pixels with at least one valid observation: {n_land}")

    if n_land == 0:
        logger.error("No valid land pixels found.")
        return

    # 确定阈值
    valid_flat = valid_count[valid_count > 0].flatten()
    if args.max_points is not None:
        threshold = np.sort(valid_flat)[-args.max_points]  # 第N大的值
        selected = valid_count >= threshold
    else:
        percentile = 100 - args.top_percent
        threshold = np.percentile(valid_flat, percentile)
        selected = valid_count >= threshold

    if not np.any(selected):
        max_val = valid_count.max()
        selected = valid_count == max_val
        logger.warning(f"No points above threshold, using max value {max_val}")

    rows_sel, cols_sel = np.where(selected)
    n_sel = len(rows_sel)
    logger.info(f"Selected {n_sel} candidate pixels ({n_sel/n_land*100:.2f}% of land pixels)")

    # 提取对应数据
    data = {
        'row': rows_sel,
        'col': cols_sel,
        'lat': lat[rows_sel, cols_sel],
        'lon': lon[rows_sel, cols_sel],
        'vza': vza[rows_sel, cols_sel],
        'saa': saa[rows_sel, cols_sel],
        'valid_count': valid_count[rows_sel, cols_sel]
    }
    df = pd.DataFrame(data)
    df.to_csv(args.output_csv, index=False)
    logger.info(f"Saved candidate points to {args.output_csv}")

    # 保存NetCDF
    out_ds = xr.Dataset(
        {
            'Station': ('Station', [f'CAND_{i:06d}' for i in range(n_sel)]),
            'Row': ('Station', rows_sel),
            'Col': ('Station', cols_sel),
            'Lat': ('Station', data['lat']),
            'Lon': ('Station', data['lon']),
            'VZA': ('Station', data['vza']),
            'SAA': ('Station', data['saa']),
            'ValidCount': ('Station', data['valid_count']),
        }
    )
    out_ds.to_netcdf(args.output_nc)
    logger.info(f"Saved candidate points to {args.output_nc}")

if __name__ == '__main__':
    main()