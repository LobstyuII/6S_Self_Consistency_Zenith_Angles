# run_pipeline.py
import subprocess
import argparse
from config import *

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--l1_example', required=True, help='Path to an example AHI L1 file for angle extraction')
    parser.add_argument('--l2_dir', required=True, help='Directory containing H8L2ARP_10min files')
    parser.add_argument('--gee_key', required=True, help='GEE service account JSON key')
    parser.add_argument('--merra2_dir', required=True, help='Directory with MERRA2 10min combined files')
    parser.add_argument('--start_date', default=START_DATE, help='Start date YYYY-MM-DD')
    parser.add_argument('--end_date', default=END_DATE, help='End date YYYY-MM-DD (exclusive)')
    parser.add_argument('--top_percent', type=float, default=10, help='Percentage of top valid pixels to select')
    parser.add_argument('--max_points', type=int, default=None, help='Maximum number of candidate points')
    # L2变量名（按需修改）
    parser.add_argument('--cloud_var', default='CloudMask', help='Variable name for cloud mask in L2 files')
    parser.add_argument('--land_var', default='LandSeaMask', help='Variable name for land mask in L2 files')
    args = parser.parse_args()

    # 步骤1：提取AHI角度
    run_cmd(f"python extract_ahi_angles.py --l1_file {args.l1_example} --output {AHI_ANGLE_FILE}")

    # 步骤2：选择候选点
    run_cmd(f"python select_top_pixels.py --l2_dir {args.l2_dir} --angle_file {AHI_ANGLE_FILE} "
            f"--start_date {args.start_date} --end_date {args.end_date} "
            f"--top_percent {args.top_percent} --max_points {args.max_points} "
            f"--qa_var {L2_QA_VAR} "
            f"--output_csv {POI_OUTPUT_DIR / 'candidates.csv'} --output_nc {POI_OUTPUT_DIR / 'candidates.nc'}")

    # 步骤3：提取MODIS数据，计算AI统计
    run_cmd(f"python extract_modis_at_pois.py --candidates_nc {POI_OUTPUT_DIR / 'candidates.nc'} "
            f"--output_nc {AHI_AI_FILE} --gee_key {args.gee_key} "
            f"--start_date {args.start_date} --end_date {args.end_date}")

    # 步骤4：选择最终POI
    run_cmd(f"python select_pois.py --ai_stats_nc {AHI_AI_FILE} "
            f"--output_csv {POI_CSV_FILE} --output_nc {POI_NC_FILE}")

    # 步骤5：提取MERRA2
    run_cmd(f"python prepare_merra2.py --poi_nc {POI_NC_FILE} --merra2_dir {args.merra2_dir} "
            f"--start_date {args.start_date} --end_date {args.end_date} --output_dir {MERRA2_POI_DIR}")

    print("Pipeline completed successfully.")

if __name__ == '__main__':
    main()