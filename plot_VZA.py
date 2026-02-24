import ee
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.patches as mpatches
import rasterio
from rasterio.plot import show as rio_show
import os
from datetime import datetime
import time
import requests
import shutil

# ====================== 1. 配置参数 ======================
LON_0 = 140.7
LAT_0 = 0.0

# 标准全盘范围（东西经80°E~160°W，南北60°）
LON_MIN = 80
LON_MAX = 200   # 160°W 对应 200°E
LAT_MIN = -60
LAT_MAX = 60

# 用户指定的数据获取范围（70°E–150°W, 60°S–60°N）
RANGE_LON_START = 70      # 70°E
RANGE_LON_END = 210       # 150°W 对应 210°E（连续经度）
RANGE_LAT_MIN = -60
RANGE_LAT_MAX = 60

ECLIPTIC_OFFSET = 23.4

# 本地数据路径（脚本所在目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GEE_CREDENTIALS = os.path.join(SCRIPT_DIR, 'premium-cipher-424203-d0-18b67488cea9.json')
MCD12Q1_TIF = os.path.join(SCRIPT_DIR, 'MCD12Q1_2016_LC_Type1_Global_5km.tif')

FIGURE_TITLE = 'Himawari-8/9 AHI Full Disk with Data Range & VZA Contours'
OUTPUT_FILENAME = os.path.join(SCRIPT_DIR, 'Himawari_AHI_FullDisk_DataRange_VZA.png')
DPI = 300

# ====================== 2. GEE 认证 ======================
def initialize_gee(credentials_path):
    try:
        credentials = ee.ServiceAccountCredentials(
            email=None,
            key_file=credentials_path
        )
        ee.Initialize(credentials)
        print("GEE 初始化成功（服务账号）")
    except Exception as e:
        print(f"GEE 初始化失败: {e}")
        raise

initialize_gee(GEE_CREDENTIALS)

# ====================== 3. 自动下载全球MCD12Q1数据（若本地不存在）======================
def download_mcd12q1_global():
    print("本地未找到全球MCD12Q1数据，开始从GEE自动下载（分辨率5000米，全球范围，请稍候）...")
    region = ee.Geometry.Rectangle([-180, -90, 180, 90])
    lc_image = ee.Image('MODIS/006/MCD12Q1/2016_01_01').select('LC_Type1').clip(region)
    params = {
        'name': 'MCD12Q1_2016_LC_Type1_Global_5km',
        'scale': 5000,
        'crs': 'EPSG:4326',
        'region': region,
        'format': 'GEO_TIFF'
    }
    try:
        url = lc_image.getDownloadURL(params)
        print("获取下载链接成功，开始下载...")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        with open(MCD12Q1_TIF, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        print(f"下载完成，文件保存至: {MCD12Q1_TIF}")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False

def ensure_mcd12q1_local():
    if os.path.exists(MCD12Q1_TIF):
        print(f"本地全球MCD12Q1已存在: {MCD12Q1_TIF}")
        return True
    else:
        success = download_mcd12q1_global()
        if success:
            return True
        else:
            print("自动下载失败，将使用半透明陆地填充作为替代。")
            return False

USE_LOCAL_LC = ensure_mcd12q1_local()

# ====================== 4. 加载或准备土地覆盖数据 ======================
lc_data_for_plot = None
if USE_LOCAL_LC:
    try:
        with rasterio.open(MCD12Q1_TIF) as src:
            # 只读取元数据，后续用 add_raster 直接绘制
            lc_src = src
            print("成功打开本地MCD12Q1数据，准备用 add_raster 绘制")
    except Exception as e:
        print(f"加载本地LC数据失败: {e}，将使用半透明陆地填充")
        USE_LOCAL_LC = False

# ====================== 5. 计算全球VZA网格（1°间隔，使用正确公式）======================
print("正在计算VZA等高线数据（全球网格，1°间隔）...")
lons_global = np.arange(-180, 180.1, 1)
lats_global = np.arange(-90, 90.1, 1)
LON_G, LAT_G = np.meshgrid(lons_global, lats_global)

R_EARTH = 6371
SAT_HEIGHT = 35800

lon0_rad = np.radians(LON_0)
lat0_rad = np.radians(LAT_0)
lon_rad = np.radians(LON_G)
lat_rad = np.radians(LAT_G)

# 地心角 theta
cos_theta = np.sin(lat0_rad) * np.sin(lat_rad) + np.cos(lat0_rad) * np.cos(lat_rad) * np.cos(lon_rad - lon0_rad)
cos_theta = np.clip(cos_theta, -1, 1)
theta = np.arccos(cos_theta)

# 地面天顶角 VZA
R_plus_H = R_EARTH + SAT_HEIGHT
denom = R_plus_H * np.cos(theta) - R_EARTH
valid_mask = denom > 0
vza = np.full_like(theta, np.nan)
vza[valid_mask] = np.arctan(R_plus_H * np.sin(theta[valid_mask]) / denom[valid_mask])
vza = np.degrees(vza)
print(f"VZA 数组形状: {vza.shape}")
print(f"VZA 有效值范围: {np.nanmin(vza):.2f} ~ {np.nanmax(vza):.2f}")
print(f"VZA NaN 比例: {np.sum(np.isnan(vza)) / vza.size * 100:.2f}%")
print("VZA 计算完成。")

# ====================== 6. 绘图 ======================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8

proj = ccrs.Orthographic(central_longitude=LON_0, central_latitude=LAT_0)
fig = plt.figure(figsize=(12, 10))
ax = plt.axes(projection=proj)
ax.set_global()

# --- 添加底图要素 ---
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5, edgecolor='black')
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', linewidth=0.4, edgecolor='black')
ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#E0F0FF', alpha=0.3)
ax.add_feature(cfeature.LAKES.with_scale('50m'), facecolor='#E0F0FF', edgecolor='black', linewidth=0.2, alpha=0.5)

# --- 绘制土地覆盖（使用 add_raster 避免 imshow 问题）---
if USE_LOCAL_LC:
    try:
        with rasterio.open(MCD12Q1_TIF) as src:
            # 直接添加栅格，自动重投影到当前投影
            ax.add_raster(src, cmap='tab20', alpha=0.7)
        print("成功添加 MCD12Q1 土地覆盖数据")
    except Exception as e:
        print(f"add_raster 失败: {e}，回退到半透明陆地填充")
        ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#F0EAD6', alpha=0.4)
else:
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#F0EAD6', alpha=0.4)

# --- 绘制VZA等高线 ---
levels = np.arange(10, 91, 10)
print("开始绘制VZA等高线...")
contour = ax.contour(LON_G, LAT_G, vza, levels, transform=ccrs.PlateCarree(),
                     colors='darkred', linewidths=0.8, linestyles='--', alpha=0.7)
if contour.allsegs and any(len(seg) > 0 for seg in contour.allsegs):
    print("VZA等高线绘制成功，正在添加标注...")
    ax.clabel(contour, inline=True, fontsize=7, fmt='%d°')
else:
    print("警告：VZA等高线为空，请检查数据范围或网格设置。")
# 强调VZA=90°圆盘边界
ax.contour(LON_G, LAT_G, vza, [90], transform=ccrs.PlateCarree(),
           colors='darkred', linewidths=1.5, linestyles='-', alpha=0.9)

# --- 绘制数据获取范围 ---
lon_range = np.linspace(RANGE_LON_START, RANGE_LON_END, 200)
# 北纬60°
ax.plot(lon_range, np.full_like(lon_range, RANGE_LAT_MAX),
        transform=ccrs.PlateCarree(), color='black', linewidth=2, linestyle='--', alpha=0.8)
# 南纬60°
ax.plot(lon_range, np.full_like(lon_range, RANGE_LAT_MIN),
        transform=ccrs.PlateCarree(), color='black', linewidth=2, linestyle='--', alpha=0.8)

lat_range = np.linspace(RANGE_LAT_MIN, RANGE_LAT_MAX, 200)
# 70°E
ax.plot(np.full_like(lat_range, RANGE_LON_START), lat_range,
        transform=ccrs.PlateCarree(), color='black', linewidth=2, linestyle='--', alpha=0.8)
# 150°W
ax.plot(np.full_like(lat_range, RANGE_LON_END), lat_range,
        transform=ccrs.PlateCarree(), color='black', linewidth=2, linestyle='--', alpha=0.8)

# 范围标签
ax.text(RANGE_LON_START+2, RANGE_LAT_MAX-2, '60°N', transform=ccrs.PlateCarree(),
        fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
ax.text(RANGE_LON_START+2, RANGE_LAT_MIN+2, '60°S', transform=ccrs.PlateCarree(),
        fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
ax.text(RANGE_LON_START+2, RANGE_LAT_MIN+5, '70°E', transform=ccrs.PlateCarree(),
        fontsize=8, bbox=dict(facecolor='white', alpha=0.7), rotation=20)
ax.text(RANGE_LON_END-2, RANGE_LAT_MIN+5, '150°W', transform=ccrs.PlateCarree(),
        fontsize=8, bbox=dict(facecolor='white', alpha=0.7), rotation=-20)

# --- 绘制黄道面边界（完整纬圈，虚线）---
lon_full = np.linspace(0, 360, 360)
ax.plot(lon_full, np.full_like(lon_full, ECLIPTIC_OFFSET),
        transform=ccrs.PlateCarree(), color='goldenrod', linewidth=1.5, linestyle=':', alpha=0.8)
ax.plot(lon_full, np.full_like(lon_full, -ECLIPTIC_OFFSET),
        transform=ccrs.PlateCarree(), color='goldenrod', linewidth=1.5, linestyle=':', alpha=0.8)

# --- 标注星下点 ---
ax.plot(LON_0, LAT_0, marker='*', markersize=10, color='red', transform=ccrs.PlateCarree(),
        markeredgecolor='black', markeredgewidth=0.3)
ax.text(LON_0+2, LAT_0+2, 'Himawari-8/9 Nadir\n(140.7°E, 0°N)', transform=ccrs.PlateCarree(),
        fontsize=8, verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# --- 设置坐标轴网格 ---
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlocator = plt.FixedLocator(np.arange(80, 201, 20))
gl.ylocator = plt.FixedLocator(np.arange(-60, 61, 20))
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}

# --- 添加图例和标题 ---
handles = [
    mpatches.Patch(color='#F0EAD6', label='Land (MCD12Q1)', alpha=0.7),  # 简化土地覆盖图例
    plt.Line2D([], [], color='darkred', linestyle='--', label='VZA Contours (10° interval)'),
    plt.Line2D([], [], color='goldenrod', linestyle=':', label='Ecliptic boundaries'),
    plt.Line2D([], [], color='black', linestyle='--', label='Data acquisition range'),
    plt.Line2D([], [], color='black', linestyle=':', label='National borders'),
    plt.Line2D([], [], marker='*', markersize=8, color='red', label='Satellite Nadir', linestyle='None')
]
ax.legend(handles=handles, loc='lower left', frameon=True, fontsize=7, framealpha=0.9)

plt.title(FIGURE_TITLE, fontsize=12, fontweight='bold', pad=20)
plt.savefig(OUTPUT_FILENAME, dpi=DPI, bbox_inches='tight')
print(f"图形已保存为 {OUTPUT_FILENAME}")
plt.show()