# quick_check.py
from pathlib import Path
import xarray as xr

data_dir = Path("D:/6S_Self_Consistency_Zenith_Angles/data")

for band in range(1, 7):
    file = data_dir / f"simulation_results_band{band}_parallel.nc"
    print(f"\n波段 band{band}:")

    if file.exists():
        try:
            ds = xr.open_dataset(file)
            df = ds.to_dataframe().reset_index()
            ds.close()

            print(f"  ✓ 样本数: {len(df):,}")
            print(f"  ✓ 列数: {len(df.columns)}")

            if 'success' in df.columns:
                success_rate = df['success'].mean() * 100 if df['success'].dtype == bool else (df[
                                                                                                   'success'] == 1).mean() * 100
                print(f"  ✓ 成功率: {success_rate:.1f}%")

            if 'error_absolute' in df.columns:
                errors = df['error_absolute'].dropna()
                print(f"  ✓ 有效误差: {len(errors):,}")
                print(f"  ✓ 误差均值: {errors.mean():.6f}")

        except Exception as e:
            print(f"  ✗ 读取失败: {e}")
    else:
        print(f"  ✗ 文件不存在")