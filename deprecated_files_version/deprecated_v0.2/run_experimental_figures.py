# ==================== run_experimental_figures.py ====================
"""
运行实验性图表生成的脚本
"""
import argparse
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from deprecated.experimental_figures import main as generate_figures

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行实验性图表生成')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据目录路径')
    parser.add_argument('--figures', type=str, default='all',
                        help='要生成的图表编号（逗号分隔，如1,3,5 或 all）')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')

    args = parser.parse_args()

    # 设置命令行参数
    sys.argv = ['experimental_figures.py']
    if args.data_dir:
        sys.argv.extend(['--data_dir', args.data_dir])
    if args.figures:
        sys.argv.extend(['--figures', args.figures])
    if args.debug:
        sys.argv.append('--debug')

    # 运行主函数
    generate_figures()