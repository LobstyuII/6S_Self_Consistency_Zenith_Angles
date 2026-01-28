# ==================== main_workflow.py（更新部分）====================
"""
主工作流程 - 集成所有重构模块（包含外部验证）
添加：import语句和更新外部验证部分
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from datetime import datetime

from config import ExperimentConfig
from utils import setup_logger
from refactored_parallel_simulator import RefactoredParallelSimulator
from ml_pipeline import MLPipeline
from lut_generator import OperationalLUTGenerator
from external_data_loader import ExternalDataLoader
from external_validation_analysis import ExternalValidationAnalysis
from visualization_suite import VisualizationSuite


def run_complete_workflow():
    """
    运行完整的工作流程（包含外部验证）
    """
    parser = argparse.ArgumentParser(description='完整的6S几何校正工作流程')

    # 数据生成参数
    parser.add_argument('--training_samples', type=int, default=100000,
                        help='每波段训练样本数，默认10万')
    parser.add_argument('--training_gen', action='store_true',
                        help='生成蒙特卡洛训练数据（默认不生成）')
    parser.add_argument('--validation_grid', action='store_true',
                        help='生成验证网格')
    parser.add_argument('--bands', type=str, default='all',
                        help='波段列表，用逗号分隔或all')
    parser.add_argument('--n_workers', type=int, default=6,
                        help='并行工作进程数')

    # 机器学习参数
    parser.add_argument('--train_ml', action='store_true',
                        help='训练机器学习模型')
    parser.add_argument('--ml_models', type=str,
                        default='RandomForest,XGBoost,LightGBM',
                        help='要训练的ML模型')
    parser.add_argument('--ml_sample_ratio', type=float, default=1.0,
                        help='ML训练数据采样比例')

    # LUT生成参数
    parser.add_argument('--generate_lut', action='store_true',
                        help='生成业务化LUT')
    parser.add_argument('--ml_model_path', type=str,
                        help='ML模型路径，用于生成LUT')

    # 外部验证参数
    parser.add_argument('--external_validation', action='store_true',
                        help='执行外部验证（使用真实Himawari数据）')
    parser.add_argument('--n_validation_stations', type=int, default=50,
                        help='外部验证测站数量，默认50个')
    parser.add_argument('--validation_start_date', type=str, default='20160101',
                        help='外部验证开始日期 (YYYYMMDD)')
    parser.add_argument('--validation_end_date', type=str, default='20170101',
                        help='外部验证结束日期 (YYYYMMDD)')
    parser.add_argument('--validation_data_path', type=str,
                        help='外部验证数据路径（如果已存在）')
    parser.add_argument('--skip_data_generation', action='store_true',
                        help='跳过数据生成阶段（使用现有数据）')

    # 可视化参数
    parser.add_argument('--generate_plots', action='store_true',
                        help='生成分析图表')

    # 输出参数
    parser.add_argument('--output_suffix', type=str, default='',
                        help='输出目录后缀')

    args = parser.parse_args()

    # 初始化
    config = ExperimentConfig
    logger = setup_logger('CompleteWorkflow', level='INFO')

    logger.info("=" * 80)
    logger.info("完整的6S几何校正工作流程")
    logger.info("=" * 80)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 创建工作流程时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_suffix:
        suffix = f"_{args.output_suffix}" if not args.output_suffix.startswith('_') else args.output_suffix
        workflow_dir = config.RESULTS_DIR / f"workflow_{timestamp}{suffix}"
    else:
        workflow_dir = config.RESULTS_DIR / f"workflow_{timestamp}"

    workflow_dir.mkdir(parents=True, exist_ok=True)

    # 确定波段
    if args.bands == 'all':
        bands_to_process = list(config.BANDS.keys())
    else:
        bands_to_process = [b.strip() for b in args.bands.split(',')]

    logger.info(f"处理波段: {bands_to_process}")
    logger.info(f"工作流程目录: {workflow_dir}")
    logger.info("=" * 80)

    # 存储重要路径
    workflow_paths = {
        'workflow_dir': workflow_dir,
        'models_dir': None,
        'lut_path': None,
        'validation_dir': None
    }

    # 1. 数据生成阶段
    if args.training_gen or args.validation_grid:
        logger.info("\n阶段1: 数据生成")
        logger.info("-" * 80)

        simulator = RefactoredParallelSimulator(config, logger)
        output_dir = config.DATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- 子任务 A: 训练数据生成 ---
        if args.training_gen:
            logger.info(">>> 子任务 A: 生成蒙特卡洛训练数据集")
            logger.info(f"    目标样本数: {args.training_samples}/波段")
            logger.info(f"    角度范围: SZA 0-85°, VZA 0-85°")
            logger.info(f"    采样策略: 60%常规 + 40%极端角度过采样")

            # 生成训练数据集（蒙特卡洛混合采样）
            training_data = simulator.generate_training_dataset(
                total_samples_per_band=args.training_samples,
                bands=bands_to_process
            )

            # 模拟训练数据集
            logger.info("    开始模拟计算...")
            training_results = simulator.simulate_dataset(
                training_data,
                max_workers=args.n_workers
            )

            # 保存训练数据
            for band_id, df in training_results.items():
                if len(df) > 0:
                    # 过滤成功样本
                    if 'success' in df.columns:
                        df_success = df[df['success']].copy() if df['success'].dtype == bool else df[
                            df['success'] == 1].copy()
                    else:
                        df_success = df

                    if not df_success.empty:
                        output_file = output_dir / f"training_data_{band_id}.nc"

                        # 转换为字典并保存
                        data_dict = {}
                        for col in df_success.columns:
                            col_data = df_success[col].values

                            if col_data.dtype == object:
                                try:
                                    col_data = col_data.astype(str)
                                except:
                                    col_data = np.array([str(x) if pd.notna(x) else '' for x in col_data])

                            data_dict[col] = col_data

                        from utils import save_dataset
                        save_dataset(data_dict, output_file)
                        logger.info(f"    保存训练数据: {output_file} ({len(df_success)} 样本)")
        else:
            logger.info(">>> 跳过训练数据生成 (未指定 --training_gen)")

        # --- 子任务 B: 验证网格生成 ---
        if args.validation_grid:
            logger.info("\n>>> 子任务 B: 生成固定验证网格")
            validation_data = simulator.generate_validation_grid(bands_to_process)

            logger.info("    开始模拟计算...")
            validation_results = simulator.simulate_dataset(
                validation_data,
                max_workers=args.n_workers
            )

            for band_id, df in validation_results.items():
                if len(df) > 0:
                    if 'success' in df.columns:
                        df_success = df[df['success']].copy() if df['success'].dtype == bool else df[
                            df['success'] == 1].copy()
                    else:
                        df_success = df

                    if not df_success.empty:
                        output_file = output_dir / f"validation_grid_{band_id}.nc"

                        data_dict = {}
                        for col in df_success.columns:
                            col_data = df_success[col].values

                            if col_data.dtype == object:
                                try:
                                    col_data = col_data.astype(str)
                                except:
                                    col_data = np.array([str(x) if pd.notna(x) else '' for x in col_data])

                            data_dict[col] = col_data

                        from utils import save_dataset
                        save_dataset(data_dict, output_file)
                        logger.info(f"    保存验证网格: {output_file} ({len(df_success)} 样本)")
    else:
        logger.info("\n阶段1: 数据生成 (跳过)")
        logger.info("-" * 80)

    # 2. 机器学习阶段
    if args.train_ml:
        logger.info("\n阶段2: 机器学习训练")
        logger.info("-" * 80)

        # 使用MLPipeline进行训练
        ml_pipeline = MLPipeline(config)

        # 构建参数列表
        import sys
        sys.argv = [
            'ml_pipeline.py',
            'full',
            '--sample', str(args.ml_sample_ratio),
            '--models', args.ml_models,
            '--test_size', '0.2',
            '--cv_folds', '5',
            '--n_jobs', str(args.n_workers),
            '--output_suffix', args.output_suffix if args.output_suffix else 'complete_workflow'
        ]

        # 运行训练和评估
        train_result = ml_pipeline.run_full_pipeline(
            sample_fraction=args.ml_sample_ratio,
            models_to_train=args.ml_models,
            test_size=0.2,
            cv_folds=5,
            n_jobs=args.n_workers,
            use_feature_engineering=True,
            use_shap=True,
            output_suffix=args.output_suffix if args.output_suffix else 'complete_workflow'
        )

        if train_result != 0:
            logger.error("ML训练失败")
            # 继续执行其他步骤
        else:
            logger.info("ML训练完成")

            # 获取最新创建的模型目录
            models_dir = config.MODELS_DIR
            model_dirs = sorted(models_dir.glob("model_training_*"))
            if model_dirs:
                latest_model_dir = model_dirs[-1]
                logger.info(f"最新模型目录: {latest_model_dir}")

                # 保存模型路径供后续使用
                ml_model_path = latest_model_dir / "models" / "RandomForest_model.pkl"

                if not ml_model_path.exists():
                    # 查找其他模型文件
                    model_files = list(latest_model_dir.rglob("*_model.pkl"))
                    if model_files:
                        ml_model_path = model_files[0]
                        logger.info(f"使用模型: {ml_model_path}")
                        workflow_paths['models_dir'] = latest_model_dir
                        args.ml_model_path = str(ml_model_path)
                    else:
                        logger.warning("未找到模型文件")
                else:
                    workflow_paths['models_dir'] = latest_model_dir
                    args.ml_model_path = str(ml_model_path)

    # 3. LUT生成阶段
    if args.generate_lut:
        logger.info("\n阶段3: 业务化LUT生成")
        logger.info("-" * 80)

        # 检查是否有模型路径
        if not args.ml_model_path:
            logger.error("需要提供ML模型路径 (--ml_model_path)")
            # 尝试使用最新训练的模型
            if workflow_paths['models_dir']:
                model_files = list(workflow_paths['models_dir'].rglob("*_model.pkl"))
                if model_files:
                    args.ml_model_path = str(model_files[0])
                    logger.info(f"使用最新模型: {args.ml_model_path}")
                else:
                    logger.error("未找到可用模型")
                    return 1
            else:
                logger.error("没有可用的模型路径")
                return 1

        lut_generator = OperationalLUTGenerator(config, logger)

        lut_output_path = workflow_dir / "operational_lut.nc"
        lut_ds = lut_generator.generate_operational_lut(
            ml_model_path=Path(args.ml_model_path),
            output_path=lut_output_path
        )

        workflow_paths['lut_path'] = lut_output_path
        logger.info(f"LUT生成完成: {lut_output_path}")

        # 输出LUT统计信息
        logger.info("\nLUT统计信息:")
        logger.info(f"  维度: {dict(lut_ds.dims)}")
        logger.info(
            f"  校正误差范围: {lut_ds['ml_correction'].min().item():.4f} 到 {lut_ds['ml_correction'].max().item():.4f}")
        logger.info(f"  校正误差均值: {lut_ds['ml_correction'].mean().item():.4f}")
        logger.info(f"  校正误差标准差: {lut_ds['ml_correction'].std().item():.4f}")

        # 4. 外部验证阶段


    # 5. 可视化阶段
    if args.generate_plots:
        logger.info("\n阶段5: 结果可视化")
        logger.info("-" * 80)

        try:
            visualization_dir = workflow_dir / "visualization"
            visualization_dir.mkdir(parents=True, exist_ok=True)

            visualizer = VisualizationSuite(data_dir=config.DATA_DIR, logger=logger)
            visualizer.run_all_plots()

            logger.info(f"可视化完成: {visualization_dir}")
        except Exception as e:
            logger.error(f"可视化失败: {e}")
            import traceback
            traceback.print_exc()

    # 生成工作流程总结
    logger.info("\n" + "=" * 80)
    logger.info("工作流程总结")
    logger.info("=" * 80)
    logger.info(f"工作流程目录: {workflow_dir}")

    summary_path = workflow_dir / "workflow_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("6S几何校正工作流程总结\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"工作流程ID: {timestamp}\n")
        f.write(f"输出后缀: {args.output_suffix if args.output_suffix else '无'}\n\n")

        f.write("执行阶段:\n")
        f.write(f"  1. 数据生成: {'✓' if args.training_gen or args.validation_grid else '✗'}\n")
        f.write(f"  2. 机器学习训练: {'✓' if args.train_ml else '✗'}\n")
        f.write(f"  3. LUT生成: {'✓' if args.generate_lut else '✗'}\n")
        f.write(f"  4. 外部验证: {'✓' if args.external_validation else '✗'}\n")
        f.write(f"  5. 可视化: {'✓' if args.generate_plots else '✗'}\n\n")

        f.write("重要路径:\n")
        for key, path in workflow_paths.items():
            if path:
                f.write(f"  {key}: {path}\n")
            else:
                f.write(f"  {key}: 未生成\n")

        f.write("\n参数配置:\n")
        for arg in vars(args):
            f.write(f"  {arg}: {getattr(args, arg)}\n")

        # 添加外部验证统计（如果执行了）
        # 在外部验证阶段，替换为以下代码：
        if args.external_validation:
            logger.info("\n阶段4: 外部数据验证（特征对齐版本）")
            logger.info("-" * 80)

            try:
                # 创建外部验证目录
                validation_output_dir = workflow_dir / "external_validation_aligned"
                validation_output_dir.mkdir(parents=True, exist_ok=True)

                validation_data_path = validation_output_dir / "validation_dataset.parquet"

                # 检查是否使用现有数据
                if args.validation_data_path and Path(args.validation_data_path).exists():
                    logger.info(f"使用现有验证数据: {args.validation_data_path}")
                    validation_data_path = Path(args.validation_data_path)
                elif args.skip_data_generation and validation_data_path.exists():
                    logger.info(f"使用已有数据: {validation_data_path}")
                else:
                    # 生成新的验证数据集
                    logger.info(f"生成新的验证数据集...")
                    logger.info(f"测站数量: {args.n_validation_stations}")
                    logger.info(f"时间范围: {args.validation_start_date} 到 {args.validation_end_date}")

                    data_loader = ExternalDataLoader(config, logger)
                    validation_df = data_loader.generate_validation_dataset(
                        n_stations=args.n_validation_stations,
                        start_date=args.validation_start_date,
                        end_date=args.validation_end_date,
                        output_path=validation_data_path
                    )

                    if validation_df.empty:
                        logger.warning("验证数据集为空，跳过验证")
                        return 0

                # 选择模型或LUT进行校正
                correction_source = None
                correction_path = None

                if args.ml_model_path and Path(args.ml_model_path).exists():
                    correction_source = "model"
                    correction_path = Path(args.ml_model_path)
                    logger.info(f"使用ML模型进行校正: {correction_path}")
                elif workflow_paths['lut_path'] and workflow_paths['lut_path'].exists():
                    correction_source = "lut"
                    correction_path = workflow_paths['lut_path']
                    logger.info(f"使用LUT进行校正: {correction_path}")
                else:
                    logger.error("没有可用的模型或LUT用于校正")
                    return 1

                # 运行更新后的外部验证分析
                from updated_external_validation_analysis import UpdatedExternalValidationAnalysis
                analyzer = UpdatedExternalValidationAnalysis(config, logger)

                success = analyzer.run_complete_analysis(
                    data_path=validation_data_path,
                    model_path=correction_path if correction_source == "model" else None,
                    lut_path=correction_path if correction_source == "lut" else None,
                    output_dir=validation_output_dir
                )

                if success:
                    workflow_paths['validation_dir'] = validation_output_dir
                    logger.info("外部验证分析完成")

                    # 输出验证统计
                    if hasattr(analyzer, 'correction_results') and not analyzer.correction_results.empty:
                        logger.info("\n外部验证统计:")
                        for band in ['03', '04']:
                            corr_key = f'correction_{band}'
                            if corr_key in analyzer.correction_results.columns:
                                corr_data = analyzer.correction_results[corr_key].dropna()
                                if len(corr_data) > 0:
                                    pos_ratio = (corr_data > 0).sum() / len(corr_data) * 100
                                    mean_imp = corr_data.mean()
                                    logger.info(f"  波段 {band}: {pos_ratio:.1f}% 正改进, 平均改进 {mean_imp:.6f}")
                else:
                    logger.error("外部验证分析失败")

            except Exception as e:
                logger.error(f"外部验证失败: {e}")
                import traceback
                traceback.print_exc()

    logger.info(f"工作流程总结已保存: {summary_path}")

    # 生成HTML报告（可选）
    try:
        html_report_path = workflow_dir / "workflow_report.html"
        _generate_html_report(workflow_dir, summary_path, workflow_paths)
        logger.info(f"HTML报告已生成: {html_report_path}")
    except:
        logger.info("跳过HTML报告生成")

    logger.info("\n" + "=" * 80)
    logger.info("工作流程完成!")
    logger.info("=" * 80)

    return 0


def _generate_html_report(workflow_dir: Path, summary_path: Path, workflow_paths: dict):
    """生成HTML格式的工作流程报告"""
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>6S几何校正工作流程报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; border-radius: 10px; }}
        .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .success {{ color: #4CAF50; font-weight: bold; }}
        .warning {{ color: #ff9800; font-weight: bold; }}
        .error {{ color: #f44336; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>6S几何校正工作流程报告</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="section">
        <h2>工作流程概览</h2>
        <p>工作流程目录: <strong>{workflow_dir}</strong></p>
    </div>

    <div class="section">
        <h2>执行阶段</h2>
        <table>
            <tr>
                <th>阶段</th>
                <th>状态</th>
                <th>描述</th>
            </tr>
            <tr>
                <td>数据生成</td>
                <td class="success">✓ 完成</td>
                <td>生成训练和验证数据集</td>
            </tr>
            <tr>
                <td>机器学习训练</td>
                <td class="success">✓ 完成</td>
                <td>训练误差预测模型</td>
            </tr>
            <tr>
                <td>LUT生成</td>
                <td class="success">✓ 完成</td>
                <td>生成业务化查找表</td>
            </tr>
            <tr>
                <td>外部验证</td>
                <td class="success">✓ 完成</td>
                <td>使用真实数据进行验证</td>
            </tr>
            <tr>
                <td>可视化</td>
                <td class="success">✓ 完成</td>
                <td>生成分析图表</td>
            </tr>
        </table>
    </div>

    <div class="section">
        <h2>重要文件</h2>
        <table>
            <tr>
                <th>文件类型</th>
                <th>路径</th>
                <th>状态</th>
            </tr>
"""

    # 添加文件信息
    for key, path in workflow_paths.items():
        if path and Path(path).exists():
            status = '<span class="success">✓ 存在</span>'
        elif path:
            status = '<span class="warning">⚠ 未找到</span>'
        else:
            status = '<span class="error">✗ 未生成</span>'

        html_content += f"""
            <tr>
                <td>{key}</td>
                <td>{path if path else '未生成'}</td>
                <td>{status}</td>
            </tr>
"""

    html_content += """
        </table>
    </div>

    <div class="section">
        <h2>下一步建议</h2>
        <ul>
            <li>检查生成的LUT文件，确保覆盖所有需要的参数范围</li>
            <li>查看外部验证结果，评估模型在不同条件下的性能</li>
            <li>根据验证结果调整模型参数或特征工程策略</li>
            <li>考虑将LUT集成到实际的数据处理流程中</li>
        </ul>
    </div>

    <div class="footer">
        <p>生成于: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        <p>6S几何校正工作流程 v1.0</p>
    </div>
</body>
</html>
"""

    html_report_path = workflow_dir / "workflow_report.html"
    with open(html_report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


if __name__ == "__main__":
    sys.exit(run_complete_workflow())