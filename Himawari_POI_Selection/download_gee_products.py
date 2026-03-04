# download_gee_products.py
"""
通过GEE下载MCD43A1、MCD12Q1、SRTM数据。
需先认证。
修改为直接遍历ImageCollection中的每个影像，避免日期边界问题。
"""
import ee
import argparse
from pathlib import Path
import logging
from utils.gee_utils import initialize_gee, export_image_to_drive

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_roi():
    """定义感兴趣区域（AHI圆盘覆盖范围）"""
    return ee.Geometry.Rectangle([100, -10, 160, 50])

def export_mcd43a1_images(start_date, end_date, folder, region):
    """导出MCD43A1中每个影像的指定波段，直接遍历ImageCollection"""
    bands = ['BRDF_Albedo_Parameters_Band3_iso', 'BRDF_Albedo_Parameters_Band3_vol',
             'BRDF_Albedo_Parameters_Band3_geo', 'BRDF_Albedo_Band_Mandatory_Quality_Band3']

    col = ee.ImageCollection("MODIS/061/MCD43A1") \
        .filterDate(start_date, end_date) \
        .select(bands)

    # 获取影像数量
    try:
        count = col.size().getInfo()
        logger.info(f"Total images in collection: {count}")
    except Exception as e:
        logger.error(f"Failed to get collection size: {e}")
        return

    # 将ImageCollection转为列表以便遍历
    img_list = col.toList(count)
    for i in range(count):
        try:
            img = ee.Image(img_list.get(i))
            # 获取影像日期（用于描述）
            date = ee.Date(img.get('system:time_start')).format('yyyyMMdd').getInfo()
            logger.info(f"Processing image for date {date}")

            # 检查波段是否存在（理论上已经select，但可再确认）
            band_names = img.bandNames().getInfo()
            missing = [b for b in bands if b not in band_names]
            if missing:
                logger.warning(f"Image {date} missing bands: {missing}, skipping.")
                continue

            # 重投影
            img_geo = img.reproject(crs='EPSG:4326', scale=500)

            task = export_image_to_drive(
                image=img_geo,
                description=f"MCD43A1_{date}",
                folder=folder,
                region=region,
                scale=500
            )
            logger.info(f"Export task for {date} started (ID: {task.id}).")
        except Exception as e:
            logger.error(f"Error processing image at index {i}: {e}")
            continue

def export_mcd12q1(year, folder, region):
    """导出MCD12Q1 IGBP分类"""
    try:
        img = ee.Image(f"MODIS/061/MCD12Q1/{year}_01_01").select('LC_Type1')
        if img is None:
            logger.error(f"MCD12Q1 image for year {year} is null.")
            return
        img_geo = img.reproject(crs='EPSG:4326', scale=500)
        task = export_image_to_drive(
            image=img_geo,
            description=f"MCD12Q1_{year}_LC_Type1",
            folder=folder,
            region=region,
            scale=500
        )
        logger.info(f"Export task for MCD12Q1 {year} started (ID: {task.id}).")
    except Exception as e:
        logger.error(f"Failed to export MCD12Q1: {e}")

def export_srtm(folder, region):
    """导出SRTM DEM"""
    try:
        img = ee.Image("USGS/SRTMGL1_003")
        if img is None:
            logger.error("SRTM image is null.")
            return
        img_geo = img.reproject(crs='EPSG:4326', scale=90)
        task = export_image_to_drive(
            image=img_geo,
            description="SRTM_DEM",
            folder=folder,
            region=region,
            scale=90
        )
        logger.info(f"Export task for SRTM started (ID: {task.id}).")
    except Exception as e:
        logger.error(f"Failed to export SRTM: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gee_key', required=True, help='Path to GEE service account JSON key')
    parser.add_argument('--start_date', default='2016-05-01', help='Start date YYYY-MM-DD')
    parser.add_argument('--end_date', default='2016-05-08', help='End date YYYY-MM-DD (exclusive)')
    parser.add_argument('--drive_folder', default='Himawari_validation', help='Google Drive folder name')
    args = parser.parse_args()

    # 初始化GEE
    initialize_gee(args.gee_key)

    # 定义区域
    region = get_roi()
    logger.info(f"Region: {region.getInfo()}")

    # 导出MCD43A1每日数据（直接遍历影像）
    logger.info("Starting MCD43A1 exports...")
    export_mcd43a1_images(args.start_date, args.end_date, args.drive_folder, region)

    # 导出MCD12Q1（2016年）
    logger.info("Starting MCD12Q1 export...")
    export_mcd12q1(2016, args.drive_folder, region)

    # 导出SRTM
    logger.info("Starting SRTM export...")
    export_srtm(args.drive_folder, region)

    logger.info("All export tasks submitted. Please check Google Drive and download files manually.")

if __name__ == '__main__':
    main()