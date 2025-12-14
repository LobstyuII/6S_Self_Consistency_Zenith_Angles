# ==================== clear_cache.py ====================
"""
清除缓存文件
"""
from pathlib import Path
import shutil


def clear_cache():
    """清除缓存目录"""
    cache_dir = Path("D:/PythonProject2_data_band1/data/cache")

    if cache_dir.exists():
        print(f"正在清除缓存目录: {cache_dir}")
        shutil.rmtree(cache_dir)
        print("缓存已清除")
    else:
        print("缓存目录不存在")

    # 也可以清除旧的模拟结果
    data_dir = Path("D:/PythonProject2_data_band1/data")
    for file in data_dir.glob("simulation_results_*.nc"):
        print(f"删除: {file}")
        file.unlink()


if __name__ == "__main__":
    clear_cache()