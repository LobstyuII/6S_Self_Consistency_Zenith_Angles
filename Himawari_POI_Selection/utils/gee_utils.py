import ee
import logging

logger = logging.getLogger(__name__)

def initialize_gee(credentials_path):
    """初始化GEE，使用服务账号"""
    try:
        credentials = ee.ServiceAccountCredentials(
            email=None,
            key_file=credentials_path
        )
        ee.Initialize(credentials)
        logger.info("GEE initialized successfully.")
    except Exception as e:
        logger.error(f"GEE initialization failed: {e}")
        raise

def export_image_to_drive(image, description, folder, region, scale, crs='EPSG:4326'):
    """导出图像到Google Drive"""
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=region,
        scale=scale,
        crs=crs,
        maxPixels=1e10
    )
    task.start()
    logger.info(f"Started export task: {description}")
    return task