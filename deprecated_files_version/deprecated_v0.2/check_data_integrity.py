# ==================== check_data_integrity.py ====================
"""
数据完整性检查脚本 - 验证批量并行生成的数据
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import ExperimentConfig
from utils import setup_logger


def check_file_integrity(file_path: Path, band_id: str) -> Dict:
    """检查单个文件的完整性"""
    results = {
        'file_exists': False,
        'can_read': False,
        'file_size': 0,
        'n_samples': 0,
        'missing_columns': [],
        'nan_counts': {},
        'param_ranges': {},
        'success_rate': 0.0,
        'error_stats': {}
    }

    # 1. 检查文件是否存在
    if not file_path.exists():
        return {**results, 'error': f"文件不存在: {file_path}"}

    results['file_exists'] = True
    results['file_size'] = file_path.stat().st_size / (1024 * 1024)  # MB

    # 2. 尝试读取文件
    try:
        ds = xr.open_dataset(file_path)
        df = ds.to_dataframe().reset_index()
        ds.close()
        results['can_read'] = True
    except Exception as e:
        return {**results, 'error': f"读取文件失败: {e}"}

    # 3. 检查样本数量
    results['n_samples'] = len(df)

    # 4. 检查必要的列
    required_columns = [
        'sza', 'vza', 'aod550', 'rho_true', 'h2o', 'o3',
        'rho_toa', 'rho_retrieved', 'error_absolute'
    ]

    available_columns = df.columns.tolist()
    results['missing_columns'] = [col for col in required_columns if col not in available_columns]

    # 5. 检查NaN值
    for col in required_columns:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            results['nan_counts'][col] = {
                'count': nan_count,
                'percentage': (nan_count / len(df)) * 100 if len(df) > 0 else 0
            }

    # 6. 检查参数范围
    param_columns = ['sza', 'vza', 'aod550', 'rho_true', 'h2o', 'o3']
    for col in param_columns:
        if col in df.columns:
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                results['param_ranges'][col] = {
                    'min': float(valid_values.min()),
                    'max': float(valid_values.max()),
                    'mean': float(valid_values.mean()),
                    'n_unique': valid_values.nunique()
                }

    # 7. 检查成功率
    success_columns = ['success', 'closed_loop_success']
    for col in success_columns:
        if col in df.columns:
            if df[col].dtype == bool:
                success_count = df[col].sum()
            else:
                success_count = (df[col] == 1).sum()
            success_rate = (success_count / len(df)) * 100 if len(df) > 0 else 0
            results['success_rate'] = success_rate
            results['success_count'] = int(success_count)
            break

    # 8. 检查误差统计
    if 'error_absolute' in df.columns:
        errors = df['error_absolute'].dropna()
        if len(errors) > 0:
            results['error_stats'] = {
                'mean': float(errors.mean()),
                'std': float(errors.std()),
                'min': float(errors.min()),
                'max': float(errors.max()),
                'rmse': float(np.sqrt(np.mean(errors ** 2))),
                'n_valid': len(errors)
            }

    # 9. 检查波段标识
    if 'band' in df.columns:
        unique_bands = df['band'].unique()
        results['band_check'] = {
            'unique_bands': [str(b) for b in unique_bands],
            'expected_band': band_id,
            'matches': any(band_id in str(b) for b in unique_bands)
        }

    return results


def check_all_bands(data_dir: Path, logger) -> Dict:
    """检查所有波段的数据完整性"""
    bands = ['band1', 'band2', 'band3', 'band4', 'band5', 'band6']
    results = {}

    logger.info("=" * 60)
    logger.info("开始检查数据完整性")
    logger.info(f"数据目录: {data_dir}")
    logger.info("=" * 60)

    for band_id in tqdm(bands, desc="检查波段数据"):
        file_name = f"simulation_results_{band_id}_parallel.nc"
        file_path = data_dir / file_name

        logger.info(f"\n检查波段: {band_id}")
        logger.info(f"文件: {file_path}")

        result = check_file_integrity(file_path, band_id)
        results[band_id] = result

        # 打印简要结果
        if result['file_exists'] and result['can_read']:
            logger.info(f"  ✓ 文件存在 ({result['file_size']:.1f} MB)")
            logger.info(f"  ✓ 样本数量: {result['n_samples']:,}")

            if result['missing_columns']:
                logger.warning(f"  ⚠ 缺失列: {result['missing_columns']}")
            else:
                logger.info(f"  ✓ 所有必要列都存在")

            # 检查NaN值
            nan_problems = []
            for col, nan_info in result['nan_counts'].items():
                if nan_info['percentage'] > 50:  # 超过50%的NaN
                    nan_problems.append(f"{col}: {nan_info['percentage']:.1f}%")

            if nan_problems:
                logger.warning(f"  ⚠ 高NaN比例: {nan_problems}")

            # 成功率
            logger.info(
                f"  ✓ 成功率: {result.get('success_rate', 0):.1f}% ({result.get('success_count', 0)}/{result['n_samples']})")

            # 参数范围检查
            expected_ranges = {
                'sza': (0, 85),
                'vza': (0, 75),
                'aod550': (0.05, 1.0),
                'rho_true': (0.05, 0.5),
                'h2o': (0.5, 5.0),
                'o3': (0.2, 0.4)
            }

            range_problems = []
            for param, expected in expected_ranges.items():
                if param in result['param_ranges']:
                    actual = result['param_ranges'][param]
                    if actual['min'] < expected[0] or actual['max'] > expected[1]:
                        range_problems.append(
                            f"{param}: {actual['min']}-{actual['max']} (期望: {expected[0]}-{expected[1]})")

            if range_problems:
                logger.warning(f"  ⚠ 参数范围问题: {range_problems}")

        else:
            error_msg = result.get('error', '未知错误')
            logger.error(f"  ✗ 文件检查失败: {error_msg}")

    return results


def generate_summary_report(results: Dict, output_dir: Path, logger):
    """生成完整性检查报告"""
    report_file = output_dir / "data_integrity_report.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("数据完整性检查报告\n")
        f.write(f"生成时间: {pd.Timestamp.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")

        total_samples = 0
        total_success = 0
        all_missing_columns = set()

        for band_id, result in results.items():
            f.write(f"\n波段: {band_id}\n")
            f.write("-" * 40 + "\n")

            if not result['file_exists']:
                f.write(f"  ✗ 文件不存在\n")
                continue

            f.write(f"  文件大小: {result['file_size']:.1f} MB\n")
            f.write(f"  样本数量: {result['n_samples']:,}\n")
            total_samples += result['n_samples']

            if result['missing_columns']:
                f.write(f"  缺失列: {result['missing_columns']}\n")
                all_missing_columns.update(result['missing_columns'])
            else:
                f.write(f"  ✓ 所有列都存在\n")

            # NaN统计
            f.write(f"  NaN值统计:\n")
            for col, nan_info in result['nan_counts'].items():
                f.write(f"    {col}: {nan_info['count']} ({nan_info['percentage']:.1f}%)\n")

            # 参数范围
            f.write(f"  参数范围:\n")
            for param, ranges in result['param_ranges'].items():
                f.write(f"    {param}: {ranges['min']:.2f}-{ranges['max']:.2f} "
                        f"(均值: {ranges['mean']:.3f}, 唯一值: {ranges['n_unique']})\n")

            # 成功率
            success_rate = result.get('success_rate', 0)
            success_count = result.get('success_count', 0)
            f.write(f"  成功率: {success_rate:.1f}% ({success_count}/{result['n_samples']})\n")
            total_success += success_count

            # 误差统计
            if result['error_stats']:
                stats = result['error_stats']
                f.write(f"  误差统计:\n")
                f.write(f"    均值: {stats['mean']:.6f}\n")
                f.write(f"    标准差: {stats['std']:.6f}\n")
                f.write(f"    范围: [{stats['min']:.6f}, {stats['max']:.6f}]\n")
                f.write(f"    RMSE: {stats['rmse']:.6f}\n")
                f.write(f"    有效误差样本: {stats['n_valid']}\n")

            f.write("\n")

        # 汇总统计
        f.write("\n" + "=" * 60 + "\n")
        f.write("汇总统计\n")
        f.write("=" * 60 + "\n")
        f.write(f"总样本数: {total_samples:,}\n")
        overall_success_rate = (total_success / total_samples * 100) if total_samples > 0 else 0
        f.write(f"总成功率: {overall_success_rate:.1f}% ({total_success}/{total_samples})\n")

        if all_missing_columns:
            f.write(f"\n所有波段共缺失的列: {list(all_missing_columns)}\n")

        # 评估结果
        f.write("\n" + "=" * 60 + "\n")
        f.write("完整性评估\n")
        f.write("=" * 60 + "\n")

        issues = []
        for band_id, result in results.items():
            if not result['file_exists']:
                issues.append(f"{band_id}: 文件不存在")
            elif not result['can_read']:
                issues.append(f"{band_id}: 无法读取")
            elif result['missing_columns']:
                issues.append(f"{band_id}: 缺失列 {result['missing_columns']}")
            elif result['n_samples'] == 0:
                issues.append(f"{band_id}: 无样本数据")
            elif result.get('success_rate', 0) < 90:
                issues.append(f"{band_id}: 成功率低 ({result.get('success_rate', 0):.1f}%)")

        if issues:
            f.write("发现以下问题:\n")
            for issue in issues:
                f.write(f"  - {issue}\n")

            severity = "严重" if any("文件不存在" in i or "无法读取" in i for i in issues) else "警告"
            f.write(f"\n评估: {severity}问题，建议检查数据处理流程\n")
        else:
            f.write("✓ 所有检查通过，数据完整性良好\n")

    logger.info(f"完整性报告已保存: {report_file}")

    # 生成可视化报告
    generate_visual_report(results, output_dir, logger)


def generate_visual_report(results: Dict, output_dir: Path, logger):
    """生成可视化报告"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()

    # 1. 样本数量柱状图
    band_ids = list(results.keys())
    sample_counts = [results[band]['n_samples'] for band in band_ids]

    axes[0].bar(band_ids, sample_counts, color='skyblue')
    axes[0].set_title('各波段样本数量', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('样本数', fontsize=10)
    axes[0].tick_params(axis='x', rotation=45)

    # 在柱子上标注数量
    for i, count in enumerate(sample_counts):
        axes[0].text(i, count + max(sample_counts) * 0.01, f'{count:,}',
                     ha='center', va='bottom', fontsize=9)

    # 2. 成功率柱状图
    success_rates = [results[band].get('success_rate', 0) for band in band_ids]
    colors = ['green' if rate > 90 else 'orange' if rate > 70 else 'red' for rate in success_rates]

    axes[1].bar(band_ids, success_rates, color=colors)
    axes[1].set_title('各波段成功率', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('成功率 (%)', fontsize=10)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90%阈值')
    axes[1].legend()

    # 在柱子上标注百分比
    for i, rate in enumerate(success_rates):
        axes[1].text(i, rate + 1, f'{rate:.1f}%',
                     ha='center', va='bottom', fontsize=9)

    # 3. 文件大小柱状图
    file_sizes = [results[band].get('file_size', 0) for band in band_ids]

    axes[2].bar(band_ids, file_sizes, color='lightcoral')
    axes[2].set_title('各波段文件大小', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('文件大小 (MB)', fontsize=10)
    axes[2].tick_params(axis='x', rotation=45)

    for i, size in enumerate(file_sizes):
        axes[2].text(i, size + max(file_sizes) * 0.01, f'{size:.1f}MB',
                     ha='center', va='bottom', fontsize=9)

    # 4. 误差均值箱线图
    error_means = []
    valid_bands = []

    for band_id in band_ids:
        if 'error_stats' in results[band_id] and results[band_id]['error_stats']:
            error_means.append(results[band_id]['error_stats']['mean'])
            valid_bands.append(band_id)

    if error_means:
        axes[3].boxplot(error_means)
        axes[3].scatter(range(1, len(error_means) + 1), error_means, color='red', zorder=3)
        axes[3].set_xticklabels(valid_bands)
        axes[3].set_title('误差均值分布', fontsize=12, fontweight='bold')
        axes[3].set_ylabel('误差均值', fontsize=10)
        axes[3].tick_params(axis='x', rotation=45)
        axes[3].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 5. SZA参数范围
    sza_ranges = []
    for band_id in band_ids:
        if 'param_ranges' in results[band_id] and 'sza' in results[band_id]['param_ranges']:
            sza_ranges.append((
                results[band_id]['param_ranges']['sza']['min'],
                results[band_id]['param_ranges']['sza']['max']
            ))

    if sza_ranges:
        for i, (min_val, max_val) in enumerate(sza_ranges):
            axes[4].plot([i + 1, i + 1], [min_val, max_val], 'b-', linewidth=3)
            axes[4].scatter(i + 1, (min_val + max_val) / 2, color='red', s=50)

        axes[4].set_xticks(range(1, len(sza_ranges) + 1))
        axes[4].set_xticklabels([f'band{i + 1}' for i in range(len(sza_ranges))])
        axes[4].set_title('SZA参数范围', fontsize=12, fontweight='bold')
        axes[4].set_ylabel('SZA (°)', fontsize=10)
        axes[4].axhline(y=85, color='r', linestyle='--', alpha=0.5, label='最大85°')
        axes[4].legend()

    # 6. 缺失列统计
    missing_data = []
    for band_id in band_ids:
        if results[band_id].get('missing_columns'):
            missing_data.append(len(results[band_id]['missing_columns']))
        else:
            missing_data.append(0)

    axes[5].bar(band_ids, missing_data, color=['red' if count > 0 else 'green' for count in missing_data])
    axes[5].set_title('缺失列数量', fontsize=12, fontweight='bold')
    axes[5].set_ylabel('缺失列数', fontsize=10)
    axes[5].tick_params(axis='x', rotation=45)

    for i, count in enumerate(missing_data):
        if count > 0:
            axes[5].text(i, count + 0.1, str(count),
                         ha='center', va='bottom', fontsize=9, color='red')

    plt.suptitle('数据完整性检查可视化报告', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    visual_report_file = output_dir / "data_integrity_visual_report.png"
    plt.savefig(visual_report_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"可视化报告已保存: {visual_report_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='检查批量并行生成的数据完整性')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据目录路径（默认使用配置中的DATA_DIR）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出报告目录')
    parser.add_argument('--detailed', action='store_true',
                        help='显示详细检查信息')

    args = parser.parse_args()

    # 设置日志
    logger = setup_logger('DataIntegrityChecker', level='INFO')

    # 确定数据目录
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = ExperimentConfig.DATA_DIR

    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / "integrity_checks"

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("批量并行数据完整性检查工具")
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)

    # 检查所有波段
    results = check_all_bands(data_dir, logger)

    # 生成报告
    generate_summary_report(results, output_dir, logger)

    logger.info("=" * 60)
    logger.info("完整性检查完成!")
    logger.info(f"详细报告保存在: {output_dir}")
    logger.info("=" * 60)

    # 打印简要总结
    total_samples = sum(results[band]['n_samples'] for band in results)
    total_success = sum(results[band].get('success_count', 0) for band in results)
    overall_rate = (total_success / total_samples * 100) if total_samples > 0 else 0

    logger.info(f"总样本数: {total_samples:,}")
    logger.info(f"总成功率: {overall_rate:.1f}%")

    # 检查问题
    problems = []
    for band_id, result in results.items():
        if not result['file_exists']:
            problems.append(f"{band_id}: 文件不存在")
        elif not result['can_read']:
            problems.append(f"{band_id}: 无法读取")
        elif result['missing_columns']:
            problems.append(f"{band_id}: 缺失列 {result['missing_columns']}")
        elif result.get('success_rate', 0) < 90:
            problems.append(f"{band_id}: 成功率低 ({result.get('success_rate', 0):.1f}%)")

    if problems:
        logger.warning("发现以下问题:")
        for problem in problems:
            logger.warning(f"  - {problem}")
    else:
        logger.info("✓ 所有检查通过，数据完整性良好")


if __name__ == "__main__":
    main()