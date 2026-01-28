# ==================== test_integrated_workflow.py ====================
"""
测试整合的工作流程
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from integrated_validation_workflow import run_integrated_validation


def quick_test():
    """快速测试整合的工作流程"""
    # 设置测试参数
    test_args = [
        "--n_stations", "10",
        "--start_date", "20160101",
        "--end_date", "20160103",
        "--model_path", r"D:\6S_Self_Consistency_Zenith_Angles_v0.4\models\refactored\model_training_20260117_152723\models\XGBoost_model.pkl",
        "--output_suffix", "test_run"
    ]

    # 保存原始参数
    original_argv = sys.argv

    try:
        # 设置测试参数
        sys.argv = ["test_integrated_workflow.py"] + test_args

        # 运行工作流程
        result = run_integrated_validation()

        if result == 0:
            print("\n测试成功！")
        else:
            print("\n测试失败！")

    finally:
        # 恢复原始参数
        sys.argv = original_argv


if __name__ == "__main__":
    quick_test()