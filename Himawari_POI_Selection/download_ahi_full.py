# download_ahi_full.py
"""
下载Himawari AHI L1和L2原始NetCDF文件（完整网格），用于后续提取角度场和质量信息。
支持从JSON配置文件读取FTP账号，也可通过命令行参数直接指定。
"""
import os
import ftplib
import logging
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_ftp_config(config_path, config_key=None, config_index=0):
    """
    从JSON文件加载FTP配置。
    如果指定config_key（如'config1'），则返回该配置；
    否则按索引返回（默认0）。
    """
    with open(config_path, 'r') as f:
        configs = json.load(f)
    if not configs:
        raise ValueError("No FTP configurations found in JSON file.")
    if config_key:
        if config_key not in configs:
            raise KeyError(f"Configuration key '{config_key}' not found. Available: {list(configs.keys())}")
        return configs[config_key]
    else:
        keys = list(configs.keys())
        if config_index < 0 or config_index >= len(keys):
            raise IndexError(f"config_index {config_index} out of range. Available indices: 0-{len(keys)-1}")
        return configs[keys[config_index]]

def download_file(ftp, remote_path, local_path, retries=3):
    """下载单个文件，支持重试"""
    for attempt in range(retries):
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                ftp.retrbinary(f'RETR {remote_path}', f.write)
            logger.info(f"Downloaded: {local_path}")
            return True
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {remote_path}: {e}")
            if attempt == retries-1:
                logger.error(f"Failed to download {remote_path}")
                return False

def generate_time_list(start_date, end_date):
    """生成10分钟时间点列表"""
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    current = start
    times = []
    while current <= end:
        for hour in range(24):
            for minute in [0,10,20,30,40,50]:
                times.append(current.replace(hour=hour, minute=minute))
        current += timedelta(days=1)
    return times

def get_l1_remote_path(date, satellite='H08'):
    """L1文件FTP路径（JAXA格式）"""
    return f"/jma/netcdf/{date.year:04d}{date.month:02d}/{date.day:02d}/NC_{satellite}_{date.year:04d}{date.month:02d}{date.day:02d}_{date.hour:02d}{date.minute:02d}_R21_FLDK.02401_02401.nc"

def get_l2_remote_path(date, satellite='H08'):
    """L2 ARP文件FTP路径（JAXA格式）"""
    version = "031" if date >= datetime(2022,12,13) else "030"
    return f"/pub/himawari/L2/ARP/{version}/{date.year:04d}{date.month:02d}/{date.day:02d}/{date.hour:02d}/NC_{satellite}_{date.year:04d}{date.month:02d}{date.day:02d}_{date.hour:02d}{date.minute:02d}_L2ARP{version}_FLDK.02401_02401.nc"

def main():
    parser = argparse.ArgumentParser(description='Download full-resolution Himawari AHI L1/L2 data')
    parser.add_argument('--start_date', default='20160501', help='Start date YYYYMMDD')
    parser.add_argument('--end_date', default='20160507', help='End date YYYYMMDD')
    parser.add_argument('--output_dir', required=True, help='Base output directory')
    parser.add_argument('--satellite', default='H08', choices=['H08','H09'], help='Satellite identifier')

    # FTP 配置方式1：直接从配置文件读取
    parser.add_argument('--ftp_config', default='./ftp_configs.json', help='Path to FTP JSON config file')
    parser.add_argument('--config_name', help='Specific config key in JSON (e.g., config1)')
    parser.add_argument('--config_index', type=int, default=0, help='Index of config in JSON (0-based)')

    # FTP 配置方式2：手动指定（兼容旧方式）
    parser.add_argument('--ftp_host', help='FTP host (if not using config file)')
    parser.add_argument('--ftp_user', help='FTP username')
    parser.add_argument('--ftp_pass', help='FTP password')

    parser.add_argument('--max_workers', type=int, default=4, help='Number of parallel downloads (not used in this version)')
    args = parser.parse_args()

    # 确定FTP配置
    ftp_host = args.ftp_host
    ftp_user = args.ftp_user
    ftp_pass = args.ftp_pass

    if ftp_host and ftp_user and ftp_pass:
        # 使用命令行提供的参数
        logger.info("Using FTP credentials from command line.")
    else:
        # 尝试从配置文件读取
        config_path = Path(args.ftp_config)
        if not config_path.exists():
            logger.error(f"FTP config file not found: {config_path}. Please provide --ftp_host, --ftp_user, --ftp_pass or ensure the config file exists.")
            sys.exit(1)
        try:
            cfg = load_ftp_config(config_path, args.config_name, args.config_index)
            ftp_host = cfg['FTP_ADDRESS']
            ftp_user = cfg['FTP_UID']
            ftp_pass = cfg['FTP_PW']
            logger.info(f"Loaded FTP configuration: {ftp_user} from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load FTP config: {e}")
            sys.exit(1)

    out_dir = Path(args.output_dir)
    l1_dir = out_dir / "H8L1_10min"
    l2_dir = out_dir / "H8L2ARP_10min"
    l1_dir.mkdir(parents=True, exist_ok=True)
    l2_dir.mkdir(parents=True, exist_ok=True)

    times = generate_time_list(args.start_date, args.end_date)
    logger.info(f"Total time slots: {len(times)}")

    # 连接FTP
    try:
        ftp = ftplib.FTP(ftp_host)
        ftp.login(ftp_user, ftp_pass)
        ftp.voidcmd("TYPE I")
        logger.info(f"Connected to FTP server {ftp_host}")
    except Exception as e:
        logger.error(f"FTP login failed: {e}")
        sys.exit(1)

    for dt in tqdm(times, desc="Downloading"):
        year = dt.strftime("%Y")
        month = dt.strftime("%m")
        day = dt.strftime("%d")
        hour_min = dt.strftime("%H%M")

        # L1
        l1_remote = get_l1_remote_path(dt, args.satellite)
        l1_local = l1_dir / year / month / f"H8_{dt.strftime('%Y%m%d_%H%M')}.nc"
        # 检查本地文件是否已存在
        if l1_local.exists():
            logger.info(f"File already exists, skipping: {l1_local}")
        else:
            download_file(ftp, l1_remote, l1_local)

        # L2
        l2_remote = get_l2_remote_path(dt, args.satellite)
        l2_local = l2_dir / year / month / f"H8L2ARP_{dt.strftime('%Y%m%d_%H%M')}.nc"
        if l2_local.exists():
            logger.info(f"File already exists, skipping: {l2_local}")
        else:
            download_file(ftp, l2_remote, l2_local)

    ftp.quit()
    logger.info("All downloads completed.")

if __name__ == '__main__':
    main()