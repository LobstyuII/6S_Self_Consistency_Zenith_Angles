# ==================== convert_old_lsr_to_band_files.py ====================
"""
将旧代码生成的LSR parquet文件转换为新代码格式的单独波段文件
只提取波段4、5、6的正确数据
"""

import pandas as pd
import numpy as np
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class LSRBandExtractor:
    """LSR波段数据提取器"""

    def __init__(self, old_lsr_path, output_base_dir):
        """
        初始化提取器

        Parameters:
        -----------
        old_lsr_path : str
            旧LSR数据文件路径（包含所有波段）
        output_base_dir : str
            输出基础目录
        """
        self.old_lsr_path = Path(old_lsr_path)
        self.output_base_dir = Path(output_base_dir)

        # 波段映射（旧格式->新格式）
        self.band_mapping = {
            '04': '04',  # NIR
            '05': '05',  # SWIR1
            '06': '06'  # SWIR2
        }

        # 新代码需要的列
        self.required_columns = [
            'success', 'rho_toa', 'rho_lsr', 'rho_retrieved', 'sza', 'vza', 'raa',
            'aod550', 'h2o', 'o3', 'wavelength', 'band', 'station', 'datetime_utc',
            'datetime_bj', 'original_index', 'atmos_profile', 'is_default_merra2',
            'merra2_warning', 'lat', 'lon', 'date'
        ]

    def load_old_lsr_data(self):
        """加载旧LSR数据"""
        print(f"Loading old LSR data from: {self.old_lsr_path}")

        try:
            # 读取parquet文件
            df = pd.read_parquet(self.old_lsr_path)
            print(f"Loaded {len(df)} records")
            print(f"Columns: {list(df.columns)}")
            print(f"Unique bands: {df['band'].unique() if 'band' in df.columns else 'band column not found'}")

            # 显示波段分布
            if 'band' in df.columns:
                band_counts = df['band'].value_counts().sort_index()
                print(f"\nBand distribution:")
                for band, count in band_counts.items():
                    print(f"  Band {band}: {count} records ({count / len(df) * 100:.1f}%)")

            return df
        except Exception as e:
            print(f"Error loading old LSR data: {e}")
            return None

    def extract_band_data(self, df, target_bands):
        """提取指定波段的数据"""
        if df is None or df.empty:
            print("No data to extract")
            return {}

        band_data = {}

        for old_band, new_band in target_bands.items():
            print(f"\nProcessing band {old_band} -> {new_band}")

            # 过滤出当前波段的数据
            band_mask = df['band'] == old_band
            band_df = df[band_mask].copy()

            if band_df.empty:
                print(f"  No data for band {old_band}")
                continue

            print(f"  Found {len(band_df)} records for band {old_band}")

            # 添加缺失的列
            band_df = self._add_missing_columns(band_df, new_band)

            # 保存到字典
            band_data[new_band] = band_df

            # 显示统计信息
            self._show_band_stats(band_df, new_band)

        return band_data

    def _add_missing_columns(self, df, band_id):
        """添加新代码需要的缺失列"""

        # 波长配置（与新代码一致）
        wavelength_map = {
            '01': 0.47,  # band1
            '02': 0.51,  # band2
            '03': 0.64,  # band3 (Red)
            '04': 0.86,  # band4 (NIR)
            '05': 1.60,  # band5 (SWIR1)
            '06': 2.30  # band6 (SWIR2)
        }

        # 添加波长（如果已经存在但可能不准确，我们重新设置）
        df['wavelength'] = wavelength_map.get(band_id, 0.86)

        # 检查大气廓线信息
        if 'atmos_profile' not in df.columns or df['atmos_profile'].isna().all():
            df['atmos_profile'] = 'MidlatitudeSummer'
        else:
            # 确保atmos_profile是字符串
            df['atmos_profile'] = df['atmos_profile'].astype(str)

        # 添加MERRA2默认标记
        if 'is_default_merra2' not in df.columns:
            df['is_default_merra2'] = False

        # 添加MERRA2警告信息
        if 'merra2_warning' not in df.columns:
            df['merra2_warning'] = ''

        # 添加经纬度（默认值，后续可以更新）
        if 'lat' not in df.columns:
            df['lat'] = 0.0

        if 'lon' not in df.columns:
            df['lon'] = 0.0

        # 添加日期（从datetime_bj推断）
        if 'date' not in df.columns and 'datetime_bj' in df.columns:
            df['date'] = pd.to_datetime(df['datetime_bj']).dt.date

        # 确保band列是正确的值
        df['band'] = band_id

        # 重命名rho_retrieved为rho_lsr（如果需要）
        if 'rho_retrieved' in df.columns and 'rho_lsr' not in df.columns:
            df['rho_lsr'] = df['rho_retrieved']

        # 确保rho_lsr存在
        if 'rho_lsr' not in df.columns and 'rho_retrieved' in df.columns:
            df['rho_lsr'] = df['rho_retrieved']

        # 检查成功标记
        if 'success' in df.columns:
            # 确保success是布尔类型
            if df['success'].dtype != bool:
                df['success'] = df['success'].astype(bool)

        return df

    def _show_band_stats(self, df, band_id):
        """显示波段统计信息"""
        if df.empty:
            return

        print(f"  Statistics for band {band_id}:")
        print(f"    Total records: {len(df)}")

        if 'success' in df.columns:
            success_count = df['success'].sum()
            success_rate = success_count / len(df) * 100 if len(df) > 0 else 0
            print(f"    Successful inversions: {success_count} ({success_rate:.1f}%)")

        if 'rho_lsr' in df.columns:
            valid_lsr = df['rho_lsr'].dropna()
            if len(valid_lsr) > 0:
                print(
                    f"    LSR reflectance - Min: {valid_lsr.min():.4f}, Max: {valid_lsr.max():.4f}, Mean: {valid_lsr.mean():.4f}")

        if 'rho_toa' in df.columns:
            valid_toa = df['rho_toa'].dropna()
            if len(valid_toa) > 0:
                print(
                    f"    TOA reflectance - Min: {valid_toa.min():.4f}, Max: {valid_toa.max():.4f}, Mean: {valid_toa.mean():.4f}")

        if 'datetime_bj' in df.columns:
            print(f"    Time range: {df['datetime_bj'].min()} to {df['datetime_bj'].max()}")

        if 'station' in df.columns:
            print(f"    Unique stations: {df['station'].nunique()}")

        if 'aod550' in df.columns:
            valid_aod = df['aod550'].dropna()
            if len(valid_aod) > 0:
                print(
                    f"    AOD550 - Min: {valid_aod.min():.3f}, Max: {valid_aod.max():.3f}, Mean: {valid_aod.mean():.3f}")

    def _convert_to_serializable(self, obj):
        """将对象转换为可JSON序列化的类型"""
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj

    def save_band_files(self, band_data):
        """保存波段数据到单独文件"""
        if not band_data:
            print("No band data to save")
            return []

        saved_files = []

        for band_id, df in band_data.items():
            # 创建波段子目录
            band_dir = self.output_base_dir / f"band{band_id}"
            band_dir.mkdir(parents=True, exist_ok=True)

            # 生成文件名（从原文件名提取日期范围）
            old_filename = self.old_lsr_path.stem

            # 尝试提取日期范围
            date_range = self._extract_date_range(old_filename)

            if date_range:
                filename = f"lsr_band{band_id}_{date_range}.parquet"
            else:
                # 使用当前时间戳
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"lsr_band{band_id}_{timestamp}.parquet"

            # 完整文件路径
            filepath = band_dir / filename

            # 保存为parquet
            df.to_parquet(filepath)

            print(f"\nSaved band {band_id} data to: {filepath}")
            print(f"  File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
            print(f"  Records: {len(df)}")
            print(f"  Columns: {len(df.columns)}")

            saved_files.append(str(filepath))

            # 保存统计信息
            self._save_band_stats(df, band_id, filepath)

        return saved_files

    def _extract_date_range(self, filename):
        """从文件名提取日期范围"""
        import re

        # 查找日期模式 YYYYMMDD
        date_pattern = r'(\d{8})[_.-](\d{8})'
        match = re.search(date_pattern, filename)

        if match:
            start_date = match.group(1)
            end_date = match.group(2)
            return f"{start_date}_{end_date}"

        # 查找单个日期
        single_date_pattern = r'(\d{8})'
        match = re.search(single_date_pattern, filename)

        if match:
            single_date = match.group(1)
            return f"{single_date}_{single_date}"

        return None

    def _save_band_stats(self, df, band_id, filepath):
        """保存波段统计信息"""
        stats_path = filepath.with_suffix('.stats.json')

        # 准备统计信息，确保所有值都是可序列化的
        stats = {
            'band_id': band_id,
            'total_records': int(len(df)),
            'unique_stations': int(df['station'].nunique()) if 'station' in df.columns else 0,
            'time_range': {
                'start': str(df['datetime_bj'].min()) if 'datetime_bj' in df.columns else None,
                'end': str(df['datetime_bj'].max()) if 'datetime_bj' in df.columns else None
            },
            'lsr_reflectance': {
                'min': float(df['rho_lsr'].min()) if 'rho_lsr' in df.columns and not pd.isna(
                    df['rho_lsr'].min()) else None,
                'max': float(df['rho_lsr'].max()) if 'rho_lsr' in df.columns and not pd.isna(
                    df['rho_lsr'].max()) else None,
                'mean': float(df['rho_lsr'].mean()) if 'rho_lsr' in df.columns and not pd.isna(
                    df['rho_lsr'].mean()) else None,
                'std': float(df['rho_lsr'].std()) if 'rho_lsr' in df.columns and not pd.isna(
                    df['rho_lsr'].std()) else None
            },
            'toa_reflectance': {
                'min': float(df['rho_toa'].min()) if 'rho_toa' in df.columns and not pd.isna(
                    df['rho_toa'].min()) else None,
                'max': float(df['rho_toa'].max()) if 'rho_toa' in df.columns and not pd.isna(
                    df['rho_toa'].max()) else None,
                'mean': float(df['rho_toa'].mean()) if 'rho_toa' in df.columns and not pd.isna(
                    df['rho_toa'].mean()) else None,
                'std': float(df['rho_toa'].std()) if 'rho_toa' in df.columns and not pd.isna(
                    df['rho_toa'].std()) else None
            },
            'aod550': {
                'min': float(df['aod550'].min()) if 'aod550' in df.columns and not pd.isna(
                    df['aod550'].min()) else None,
                'max': float(df['aod550'].max()) if 'aod550' in df.columns and not pd.isna(
                    df['aod550'].max()) else None,
                'mean': float(df['aod550'].mean()) if 'aod550' in df.columns and not pd.isna(
                    df['aod550'].mean()) else None
            },
            'success_rate': float(df['success'].mean() * 100) if 'success' in df.columns else None,
            'generated_at': datetime.now().isoformat(),
            'source_file': str(self.old_lsr_path)
        }

        # 添加大气廓线分布（如果存在）
        if 'atmos_profile' in df.columns:
            profile_counts = df['atmos_profile'].value_counts().to_dict()
            # 确保值是整数
            profile_counts = {str(k): int(v) for k, v in profile_counts.items()}
            stats['atmos_profile_distribution'] = profile_counts

        # 转换所有值为可序列化类型
        stats = self._convert_to_serializable(stats)

        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"  Statistics saved to: {stats_path}")

    def create_summary_report(self, band_data, saved_files):
        """创建总结报告"""
        report_path = self.output_base_dir / "conversion_summary.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("LSR DATA CONVERSION SUMMARY\n")
            f.write("=" * 70 + "\n")
            f.write(f"Source file: {self.old_lsr_path}\n")
            f.write(f"Output directory: {self.output_base_dir}\n")
            f.write(f"Conversion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("BAND DATA SUMMARY:\n")
            f.write("-" * 70 + "\n")

            total_records = 0
            for band_id, df in band_data.items():
                f.write(f"\nBand {band_id}:\n")
                f.write(f"  Records: {len(df)}\n")

                if 'success' in df.columns:
                    success_count = df['success'].sum()
                    success_rate = success_count / len(df) * 100 if len(df) > 0 else 0
                    f.write(f"  Successful inversions: {success_count} ({success_rate:.1f}%)\n")

                if 'station' in df.columns:
                    f.write(f"  Unique stations: {df['station'].nunique()}\n")

                if 'datetime_bj' in df.columns:
                    f.write(f"  Time range: {df['datetime_bj'].min()} to {df['datetime_bj'].max()}\n")

                if 'rho_lsr' in df.columns:
                    valid_lsr = df['rho_lsr'].dropna()
                    if len(valid_lsr) > 0:
                        f.write(f"  LSR reflectance range: [{valid_lsr.min():.4f}, {valid_lsr.max():.4f}]\n")

                total_records += len(df)

            f.write(f"\n" + "-" * 70 + "\n")
            f.write(f"TOTAL RECORDS: {total_records}\n")
            f.write(f"BANDS PROCESSED: {len(band_data)}\n")
            f.write(f"BAND LIST: {', '.join(sorted(band_data.keys()))}\n\n")

            f.write("OUTPUT FILES:\n")
            f.write("-" * 70 + "\n")
            for filepath in saved_files:
                f.write(f"  {filepath}\n")

        print(f"\nSummary report saved to: {report_path}")

    def convert(self):
        """执行转换"""
        print("=" * 70)
        print("CONVERTING OLD LSR DATA TO NEW BAND FORMAT")
        print("=" * 70)

        # 1. 加载旧数据
        old_df = self.load_old_lsr_data()
        if old_df is None:
            print("Failed to load old LSR data")
            return False

        # 2. 提取波段数据
        print("\n" + "=" * 70)
        print("EXTRACTING BAND DATA")
        print("=" * 70)
        band_data = self.extract_band_data(old_df, self.band_mapping)

        if not band_data:
            print("No band data extracted")
            return False

        # 3. 保存波段文件
        print("\n" + "=" * 70)
        print("SAVING BAND FILES")
        print("=" * 70)
        saved_files = self.save_band_files(band_data)

        # 4. 创建总结报告
        print("\n" + "=" * 70)
        print("GENERATING SUMMARY")
        print("=" * 70)
        self.create_summary_report(band_data, saved_files)

        print("\n" + "=" * 70)
        print("CONVERSION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Total bands converted: {len(band_data)}")
        print(f"Total records: {sum(len(df) for df in band_data.values())}")
        print(f"Output directory: {self.output_base_dir}")

        return True


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='将旧LSR数据转换为新格式的波段文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换单个文件
  python convert_old_lsr_to_band_files.py --input ./old_lsr_data.parquet --output ./converted_bands

  # 转换多个文件
  python convert_old_lsr_to_band_files.py --input "./path/to/*.parquet" --output ./converted_bands --batch

  # 指定特定波段（默认是4,5,6）
  python convert_old_lsr_to_band_files.py --input ./old_lsr_data.parquet --output ./converted_bands --bands 04 05 06
        """
    )

    parser.add_argument('--input', type=str, required=True,
                        help='旧LSR数据文件路径（支持通配符）')
    parser.add_argument('--output', type=str, default='./converted_bands',
                        help='输出目录（默认: ./converted_bands）')
    parser.add_argument('--bands', type=str, nargs='+', default=['04', '05', '06'],
                        help='要提取的波段（默认: 04 05 06）')
    parser.add_argument('--batch', action='store_true',
                        help='批量处理多个文件')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理单个或多个文件
    if args.batch:
        # 批量处理
        import glob
        input_files = glob.glob(args.input)

        if not input_files:
            print(f"No files found matching pattern: {args.input}")
            return 1

        print(f"Found {len(input_files)} files to process")

        success_count = 0
        for i, input_file in enumerate(input_files, 1):
            print(f"\n{'=' * 70}")
            print(f"Processing file {i}/{len(input_files)}: {input_file}")
            print(f"{'=' * 70}")

            # 为每个文件创建子目录
            file_stem = Path(input_file).stem
            file_output_dir = output_dir / file_stem
            file_output_dir.mkdir(parents=True, exist_ok=True)

            # 创建波段映射
            band_mapping = {band: band for band in args.bands}

            # 执行转换
            extractor = LSRBandExtractor(input_file, file_output_dir)
            extractor.band_mapping = band_mapping

            if extractor.convert():
                success_count += 1

        print(f"\n{'=' * 70}")
        print(f"BATCH PROCESSING COMPLETED")
        print(f"{'=' * 70}")
        print(f"Files processed: {len(input_files)}")
        print(f"Successful conversions: {success_count}")
        print(f"Failed conversions: {len(input_files) - success_count}")

    else:
        # 处理单个文件
        # 创建波段映射
        band_mapping = {band: band for band in args.bands}

        extractor = LSRBandExtractor(args.input, output_dir)
        extractor.band_mapping = band_mapping

        if not extractor.convert():
            return 1

    return 0


# ==================== 直接转换示例（无需命令行） ====================
def convert_specific_file():
    """
    直接转换特定文件的示例函数
    如果你不想使用命令行，可以直接修改这里的路径
    """
    # 旧LSR文件路径
    old_lsr_file = "/lsr_cache_10min/lsr_cache_10min_20200501_20200507_d2a71af0.parquet"

    # 输出目录
    output_dir = "D:/H8_data/LSR_10min/converted_bands"

    # 要提取的波段
    bands_to_extract = {
        '04': '04',  # NIR
        '05': '05',  # SWIR1
        '06': '06'  # SWIR2
    }

    # 创建提取器并执行转换
    extractor = LSRBandExtractor(old_lsr_file, output_dir)
    extractor.band_mapping = bands_to_extract

    success = extractor.convert()

    if success:
        print("\n转换完成！")
        print(f"输出目录: {output_dir}")
        print("新代码可以直接使用这些文件，格式为:")
        print(f"  {output_dir}/band04/lsr_band04_*.parquet")
        print(f"  {output_dir}/band05/lsr_band05_*.parquet")
        print(f"  {output_dir}/band06/lsr_band06_*.parquet")
    else:
        print("转换失败！")


if __name__ == "__main__":
    # 使用方式1: 通过命令行参数运行
    # import sys
    # sys.exit(main())

    # 使用方式2: 直接运行示例函数（修改上面的文件路径）
    convert_specific_file()