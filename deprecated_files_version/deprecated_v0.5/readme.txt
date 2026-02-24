# ==================== 6S几何校正工作流程命令列表 v2.1 ====================
# 版本：v2.1 - 完整工作流程，包含外部验证
# 基础设置：每波段10万样本，极端角度过采样40%

# ==================== 环境配置与检查 ====================

# 1.1 检查Python环境和依赖
python -c "import sys; print(f'Python版本: {sys.version}')"
python -c "import numpy, pandas, xarray, sklearn, matplotlib; print('核心依赖检查通过')"

# 1.2 测试6S安装
python -c "from Py6S import *; s = SixS(); s.run(); print('6S安装正常')"

# 1.3 检查项目配置
python -c "from config import ExperimentConfig; print(f'配置加载成功'); print(f'波段数: {len(ExperimentConfig.BANDS)}'); print(f'数据目录: {ExperimentConfig.DATA_DIR}')"

# ==================== 2. 完整端到端工作流程 ====================

# 2.1 完整工作流程：从数据生成到外部验证
# 注意：这将需要很长时间（预计24-48小时），建议在服务器上运行
python main_workflow.py \
  --training_gen \
  --validation_grid \
  --train_ml \
  --generate_lut \
  --external_validation \
  --generate_plots \
  --training_samples 100000 \
  --n_workers 8 \
  --ml_models "RandomForest,XGBoost,LightGBM" \
  --n_validation_stations 100 \
  --validation_start_date "20160101" \
  --validation_end_date "20170101" \
  --output_suffix "full_workflow"

# 2.2 快速完整流程（测试用，小样本量）
python main_workflow.py \
  --training_gen \
  --validation_grid \
  --train_ml \
  --external_validation \
  --generate_plots \
  --training_samples 10000 \
  --n_workers 4 \
  --ml_models "RandomForest" \
  --n_validation_stations 20 \
  --validation_start_date "20160101" \
  --validation_end_date "20160201" \
  --output_suffix "quick_test"

# ==================== 3. 分阶段工作流程 ====================

# 3.1 阶段1：仅生成训练数据（10万样本/波段）
python main_workflow.py \
  --training_gen \
  --training_samples 100000 \
  --n_workers 8 \
  --output_suffix "training_data_only"

# 3.2 阶段2：仅生成验证网格
python main_workflow.py \
  --validation_grid \
  --n_workers 8 \
  --output_suffix "validation_grid_only"

# 3.3 阶段3：仅训练机器学习模型（使用现有数据）
python main_workflow.py \
  --train_ml \
  --ml_models "RandomForest,XGBoost,LightGBM" \
  --ml_sample_ratio 0.8 \
  --output_suffix "ml_training"

# 3.4 阶段4：仅生成LUT（需要指定模型路径）
python main_workflow.py --generate_lut --ml_model_path "D:\6S_Self_Consistency_Zenith_Angles_v0.4\models\refactored\model_training_20260117_152723\models\XGBoost_model.pkl" --output_suffix "lut_generation"

# 3.5 阶段5：仅执行外部验证（需要指定模型路径）
python main_workflow.py \
  --external_validation \
  --n_validation_stations 50 \
  --validation_start_date "20160101" \
  --validation_end_date "20170101" \
  --ml_model_path "D:/6S_Self_Consistency_Zenith_Angles_v0.4/models/refactored/best_model_RandomForest_refactored.pkl" \
  --output_suffix "external_validation"

# 3.6 阶段6：仅生成可视化图表
python main_workflow.py \
  --generate_plots \
  --output_suffix "visualization"

# ==================== 4. 数据生成专项命令 ====================

# 4.1 生成所有波段的数据（默认）
python main_workflow.py --training_gen --training_samples 100000 --n_workers 8

# 4.2 生成特定波段的数据（例如只处理band1和band4）
python main_workflow.py --training_gen --bands "band1,band4" --training_samples 100000 --n_workers 8

# 4.3 生成验证网格并指定波段
python main_workflow.py --validation_grid --bands "band3,band4" --n_workers 8

# 4.4 同时生成训练数据和验证网格
python main_workflow.py --training_gen --validation_grid --training_samples 100000 --n_workers 8

# ==================== 5. 机器学习训练专项命令 ====================

# 5.1 训练所有支持的ML模型
python main_workflow.py --train_ml --ml_models "RandomForest,XGBoost,LightGBM,GradientBoosting,ExtraTrees,SVR_RBF,MLP,Ridge,Lasso,ElasticNet"

# 5.2 训练单个ML模型（速度最快）
python main_workflow.py --train_ml --ml_models "RandomForest" --ml_sample_ratio 1.0

# 5.3 训练多个ML模型（中等复杂度）
python main_workflow.py --train_ml --ml_models "RandomForest,XGBoost,LightGBM,GradientBoosting" --ml_sample_ratio 0.5

# 5.4 训练并启用SHAP分析
python ml_pipeline.py full \
  --sample 0.5 \
  --models "RandomForest,XGBoost" \
  --n_jobs 8 \
  --no_shap

# ==================== 6. LUT生成专项命令 ====================

# 6.1 使用RandomForest模型生成LUT
python main_workflow.py --generate_lut \
  --ml_model_path "D:/6S_Self_Consistency_Zenith_Angles_v0.4/models/refactored/best_model_RandomForest_refactored.pkl"

# 6.2 使用XGBoost模型生成LUT
python main_workflow.py --generate_lut \
  --ml_model_path "D:/6S_Self_Consistency_Zenith_Angles_v0.4/models/refactored/best_model_XGBoost_refactored.pkl"

# 6.3 使用LightGBM模型生成LUT
python main_workflow.py --generate_lut \
  --ml_model_path "D:/6S_Self_Consistency_Zenith_Angles_v0.4/models/refactored/best_model_LightGBM_refactored.pkl"

# ==================== 7. 外部验证专项命令 ====================

# 7.1 完整外部验证流程（含数据准备）
python external_validation.py \
  --n_stations 100 \
  --start_date "20160101" \
  --end_date "20170101" \
  --ml_model "D:/6S_Self_Consistency_Zenith_Angles_v0.4/models/refactored/best_model_RandomForest_refactored.pkl"

# 7.2 使用现有数据集进行验证
python external_validation.py \
  --skip_data_prep \
  --data_path "D:/6S_Self_Consistency_Zenith_Angles_v0.4/data/external_validation_dataset.nc" \
  --ml_model "D:/6S_Self_Consistency_Zenith_Angles_v0.4/models/refactored/best_model_RandomForest_refactored.pkl"

# 7.3 快速外部验证（少量测站）
python external_validation.py \
  --n_stations 20 \
  --start_date "20160101" \
  --end_date "20160201" \
  --ml_model "D:/6S_Self_Consistency_Zenith_Angles_v0.4/models/refactored/best_model_RandomForest_refactored.pkl"

# ==================== 8. 模型评估与比较命令 ====================

# 8.1 评估已保存的模型目录
python ml_pipeline.py evaluate \
  --saved_dir "D:/6S_Self_Consistency_Zenith_Angles_v0.4/models/refactored/model_training_20250117_152723" \
  --sample 0.3

# 8.2 只训练模型（不评估）
python ml_pipeline.py train \
  --sample 0.5 \
  --models "RandomForest,XGBoost" \
  --n_jobs 8

# 8.3 只运行完整流水线（训练+评估）
python ml_pipeline.py full \
  --sample 0.5 \
  --models "RandomForest,XGBoost" \
  --n_jobs 8

# ==================== 9. 可视化专项命令 ====================

# 9.1 生成所有可视化图表
python main_workflow.py --generate_plots

# 9.2 使用可视化套件生成图表
python -c "from visualization_suite import VisualizationSuite; vs = VisualizationSuite(); vs.run_all_plots()"

# 9.3 生成验证网格并自动绘图
python main_workflow.py --validation_grid --generate_plots --n_workers 8

# ==================== 10. 调试与测试命令 ====================

# 10.1 最小化测试：验证整个流程
python main_workflow.py \
  --training_gen \
  --validation_grid \
  --train_ml \
  --generate_plots \
  --training_samples 1000 \
  --n_workers 2 \
  --ml_sample_ratio 0.1 \
  --output_suffix "debug_test"

# 10.2 测试数据加载器
python -c "from data_loader import DataLoader; dl = DataLoader(); data = dl.load_data(sample_fraction=0.01); print(f'测试数据加载成功: {len(data)} 条记录')"

# 10.3 测试LUT生成器
python -c "from lut_generator import OperationalLUTGenerator; gen = OperationalLUTGenerator(); axes = gen.generate_lut_axes(); print(f'LUT轴生成成功: {list(axes.keys())}')"

# 10.4 测试外部验证数据加载
python -c "from external_validation import HimawariDataLoader; loader = HimawariDataLoader(); stations = loader.load_station_coordinates(); print(f'加载了 {len(stations)} 个测站坐标')"

# ==================== 11. 实用工具命令 ====================

# 11.1 清理临时文件和缓存
python -c "import shutil, os; dirs = ['__pycache__', '.ipynb_checkpoints']; [shutil.rmtree(d, ignore_errors=True) for d in dirs]; print('缓存清理完成')"

# 11.2 查看数据目录内容
python -c "from pathlib import Path; data_dir = Path('D:/6S_Self_Consistency_Zenith_Angles_v0.4/data'); files = list(data_dir.glob('*.nc')); print(f'数据目录: {len(files)} 个文件'); [print(f'  {f.name}') for f in files[:10]]"

# 11.3 查看模型目录内容
python -c "from pathlib import Path; model_dir = Path('D:/6S_Self_Consistency_Zenith_Angles_v0.4/models/refactored'); dirs = [d for d in model_dir.iterdir() if d.is_dir()]; print(f'模型目录: {len(dirs)} 个子目录'); [print(f'  {d.name}') for d in sorted(dirs)[-5:]]"

# ==================== 12. 高级配置命令 ====================

# 12.1 自定义工作流程输出目录
python main_workflow.py --training_gen --training_samples 50000 --output_suffix "custom_experiment_01"

# 12.2 禁用特征工程（对比实验）
python ml_pipeline.py full \
  --sample 0.5 \
  --models "RandomForest" \
  --no_feature_engineering \
  --output_suffix "no_feature_eng"

# 12.3 使用不同采样比例
python main_workflow.py --train_ml --ml_sample_ratio 0.3 --output_suffix "sample_30pct"

# ==================== 使用说明与最佳实践 ====================

# 最佳实践1：首次运行使用快速测试
# python main_workflow.py --training_gen --validation_grid --train_ml --generate_plots --training_samples 10000 --n_workers 4 --output_suffix "first_run"

# 最佳实践2：生产环境数据生成
# python main_workflow.py --training_gen --training_samples 100000 --n_workers 12 --output_suffix "production_data"

# 最佳实践3：模型比较实验
# python main_workflow.py --train_ml --ml_models "RandomForest,XGBoost,LightGBM" --ml_sample_ratio 0.8 --output_suffix "model_comparison"

# 最佳实践4：外部验证（使用最佳模型）
# python main_workflow.py --external_validation --n_validation_stations 100 --ml_model_path "最佳模型路径" --output_suffix "final_validation"

# 结束