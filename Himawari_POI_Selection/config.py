# config.py
from pathlib import Path

# 根目录（所有数据存放于此）
BASE_DIR = Path("/path/to/your/data")  # 请修改为实际路径

# 日期范围
START_DATE = "2016-05-01"
END_DATE = "2016-05-08"  # 不包含当天

# Himawari FTP 配置
HIMAWARI_FTP = {
    "address": "ftp.ptree.jaxa.jp",
    "user": "your_username",
    "password": "your_password"
}

# GEE 认证文件路径
GEE_CREDENTIALS = "D:\\Users\\lobst\\PycharmProjects\\6S_Self_Consistency_Zenith_Angles\\premium-cipher-424203-d0-18b67488cea9.json"

# 数据子目录
AHI_L1_DIR = BASE_DIR / "H8L1_10min"
AHI_L2_DIR = BASE_DIR / "H8L2ARP_10min"
MODIS_BRDF_DIR = BASE_DIR / "MCD43A1"
MODIS_LC_DIR = BASE_DIR / "MCD12Q1"
DEM_DIR = BASE_DIR / "SRTM"
MERRA2_10MIN_DIR = BASE_DIR / "MERRA2_10min" / "combined"
POI_OUTPUT_DIR = BASE_DIR / "poi_output"

# 创建目录
for d in [AHI_L1_DIR, AHI_L2_DIR, MODIS_BRDF_DIR, MODIS_LC_DIR, DEM_DIR,
          MERRA2_10MIN_DIR.parent, POI_OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# AHI 网格大小
AHI_SHAPE = (6000, 6000)  # 假设，实际从文件读取时会自动获取

# 波段匹配（MODIS波段索引与AHI波段对应）
# 键为AHI波段号，值为MODIS波段号（1-7）
BAND_MATCH = {
    "01": 3,  # 0.47µm
    "02": 4,  # 0.51µm
    "03": 1,  # 0.64µm
    "04": 2,  # 0.86µm
    "05": 6,  # 1.60µm
    "06": 7,  # 2.30µm
}

# 土地覆盖类型（IGBP）
LC_LAND_CODES = [1,2,3,4,5,8,10,12,13,14]  # 森林、灌木、草原、农田、城市等

# AI 筛选阈值
AI_THRESHOLD = 0.1
CV_THRESHOLD = 0.2
MIN_VALID_DAYS = 3

# 分层采样参数
SAA_BINS = [0,20,40,60,80,100,120,140,160,180]
VZA_BINS = [0,10,20,30,40,50,60,70,80,90]
POIS_PER_CELL = 5

# 输出文件
AHI_ANGLE_FILE = POI_OUTPUT_DIR / "ahi_angles_5km.nc"
AHI_LC_FILE = POI_OUTPUT_DIR / "ahi_lc_5km.nc"
AHI_AI_FILE = POI_OUTPUT_DIR / "ahi_brdf_ai.nc"
POI_NC_FILE = POI_OUTPUT_DIR / "poi_list.nc"
POI_CSV_FILE = POI_OUTPUT_DIR / "poi_list.csv"
POI_ELEV_FILE = POI_OUTPUT_DIR / "poi_list_with_elev.nc"
MERRA2_POI_DIR = POI_OUTPUT_DIR / "merra2_poi"


L2_QA_VAR = 'QA_flag'                # QA_flag 变量名
L2_VALID_CONDITIONS = {               # 定义有效像素条件（可选）
    'Data_Availability': 0,
    'Land_Water_Flag': 0,
    'Cloud_Flag': 0,
    'Retrieval_Status': 0,
    'Additional_Cloud_Flag': 0,
    'Turbid_Water': 0,
    'Snow_Ice': 0,
    'Angle_Threshold': 0,
    'Sunglint': 0
}



