# ==================== quick_evaluate.py ====================
"""
快速评估现有模型的工具
专门用于您已有的4个pkl模型文件
"""
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
import gc

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

from config import ExperimentConfig
from data_loader import DataLoader, AdvancedFeatureEngineering
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# 设置专业科研字体（RSE期刊风格）- 关键修改！
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 11,
    'figure.dpi': 300,
})

# 自定义颜色映射（与plot2相同风格）
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# 使用更专业的密度颜色映射
DENSITY_COLORS = [
    (1.0, 1.0, 1.0),  # 白色
    (0.9, 0.95, 1.0),  # 极浅蓝
    (0.7, 0.85, 1.0),  # 浅蓝色
    (0.5, 0.7, 1.0),  # 蓝色
    (0.3, 0.5, 0.9),  # 中蓝色
    (0.1, 0.3, 0.8),  # 深蓝色
    (0.0, 0.15, 0.7),  # 深蓝
    (0.0, 0.1, 0.6),  # 更深蓝
]
DENSITY_CMAP = LinearSegmentedColormap.from_list('density_cmap', DENSITY_COLORS, N=256)

ERROR_COLORS = [
    (0.95, 0.98, 1.0),  # 近白色
    (0.7, 0.85, 1.0),  # 更浅的蓝色
    (0.5, 0.7, 1.0),  # 浅蓝色
    (0.3, 0.5, 0.9),  # 中蓝色
    (0.1, 0.3, 0.8),  # 蓝色
    (0.0, 0.15, 0.7),  # 深蓝色
]
ERROR_CMAP = LinearSegmentedColormap.from_list('error_cmap', ERROR_COLORS, N=256)

MODEL_COLORS = {
    'RandomForest': '#1f77b4',  # 蓝色
    'XGBoost': '#ff7f0e',  # 橙色
    'LightGBM': '#2ca02c',  # 绿色
    'GradientBoosting': '#d62728',  # 红色
    'SVR_RBF': '#9467bd',  # 紫色
    'MLP': '#8c564b',  # 棕色
    'Ridge': '#e377c2',  # 粉色
    'Lasso': '#7f7f7f',  # 灰色
    'ElasticNet': '#bcbd22',  # 黄绿色
}


def load_models_from_directory(model_dir: Path):
    """从目录加载所有pkl模型"""
    models = {}

    # 查找所有pkl文件
    pkl_files = list(model_dir.glob("*.pkl"))

    for pkl_file in pkl_files:
        model_name = pkl_file.stem

        # 跳过标准化器等非模型文件
        if 'scaler' in model_name or 'artifacts' in model_name or 'feature' in model_name:
            continue

        try:
            with open(pkl_file, 'rb') as f:
                model = pickle.load(f)
            models[model_name] = model
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")

    return models


def create_hexbin_matrix(y, predictions_dict, model_names, plots_dir):
    """创建2×2 hexbin密度图矩阵，每个图都有颜色条"""
    print(f"Creating hexbin density matrix for {len(model_names)} models...")

    # 只取前4个模型（如果超过4个）
    selected_models = model_names[:4]
    n_models = len(selected_models)

    if n_models == 0:
        print("No models selected for hexbin matrix")
        return

    # 根据模型数量确定子图布局
    if n_models == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))  # 正方形图形
        axes = np.array([axes])
    elif n_models == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))  # 宽高比2:1
    elif n_models == 3:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # 正方形
        axes.flat[3].set_visible(False)
    else:  # n_models == 4
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # 正方形

    # 展平axes数组以便迭代
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    # 确定全局数据范围
    all_y = []
    all_preds = []
    for model_name in selected_models:
        y_pred = predictions_dict[model_name]
        all_y.extend(y.values)
        all_preds.extend(y_pred)

    x_min, x_max = min(all_y), max(all_y)
    y_min, y_max = min(all_preds), max(all_preds)

    # 确保坐标范围一致，使用相同的min和max
    data_min = min(x_min, y_min)
    data_max = max(x_max, y_max)

    # 扩展一点边界
    data_range = data_max - data_min
    data_min -= 0.02 * data_range
    data_max += 0.02 * data_range

    for i, model_name in enumerate(selected_models):
        if i >= len(axes_flat):
            break

        ax = axes_flat[i]
        if not ax.get_visible():
            continue

        try:
            y_pred = predictions_dict[model_name]

            # 创建hexbin图 - 使用密度颜色映射
            hexbin = ax.hexbin(y, y_pred, gridsize=40, cmap=DENSITY_CMAP,
                               mincnt=1, bins='log', edgecolors='none',
                               alpha=0.9)

            # 关键修改：设置轴比为1:1
            ax.set_aspect('equal', adjustable='box')

            # 设置相同的轴范围
            ax.set_xlim(data_min, data_max)
            ax.set_ylim(data_min, data_max)

            # 添加对角线
            ax.plot([data_min, data_max], [data_min, data_max], 'r--', lw=1.5, alpha=0.8)

            # 设置标签和标题
            ax.set_xlabel('True Value', fontsize=9)
            ax.set_ylabel('Predicted Value', fontsize=9)
            ax.set_title(f'{model_name}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.2, linestyle='--')

            # 设置坐标轴边框
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_linewidth(0.75)
                ax.spines[spine].set_color('black')

            # 添加颜色条
            cbar = plt.colorbar(hexbin, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('log10(Density)', fontsize=8)
            cbar.ax.tick_params(labelsize=7)

            # 添加统计信息
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            stats_text = f'RMSE: {rmse:.3f}\nR²: {r2:.3f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            print(f"  Created hexbin density plot for {model_name}")

        except Exception as e:
            print(f"  Error creating hexbin plot for {model_name}: {e}")
            ax.text(0.5, 0.5, f"Error: {str(e)[:50]}...",
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=9, color='red')
            ax.set_title(f'{model_name} - Error', fontsize=10)

    # 调整布局
    plt.tight_layout()
    hexbin_matrix_path = plots_dir / "model_hexbin_density_matrix.png"
    plt.savefig(hexbin_matrix_path, dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    gc.collect()  # 清理内存

    print(f"  Saved hexbin density matrix to: {hexbin_matrix_path}")


def quick_evaluate(models_dir: str, data_sample: float = 0.2,
                   output_dir: str = None):
    """快速评估模型"""
    print("\n" + "=" * 80)
    print("Quick Model Evaluation Tool")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model Directory: {models_dir}")
    print(f"Data Sampling: {data_sample * 100:.1f}%")
    print("=" * 80)

    model_dir = Path(models_dir)

    if not model_dir.exists():
        print(f"Directory does not exist: {model_dir}")
        return 1

    # 创建输出目录
    if output_dir:
        output_path = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = model_dir.parent / f"quick_eval_{timestamp}"

    output_path.mkdir(parents=True, exist_ok=True)
    plots_dir = output_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 加载模型
    print("\nLoading models...")
    models = load_models_from_directory(model_dir)

    if not models:
        print("No available model files found")
        return 1

    print(f"Successfully loaded {len(models)} models")

    # 加载数据
    print("\nLoading data...")
    dl = DataLoader()
    data = dl.load_data(sample_fraction=data_sample)

    # 准备特征（使用与训练时相同的逻辑）
    print("\nPreparing features...")
    feature_engineer = AdvancedFeatureEngineering()
    X = feature_engineer.create_features(data)
    y = data['error_absolute']

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # 评估模型
    print("\nEvaluating models...")
    results = []
    all_predictions = {}

    for model_name, model in models.items():
        try:
            print(f"  Evaluating {model_name}...")

            # 预测
            y_pred = model.predict(X_scaled)
            all_predictions[model_name] = y_pred

            # 计算指标
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            results.append({
                'Model': model_name,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'Samples': len(y)
            })

            print(f"    RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}")

        except Exception as e:
            print(f"   {model_name} evaluation failed: {e}")

    # 创建结果DataFrame
    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No model evaluated successfully")
        return 1

    # 排序并保存结果
    results_df = results_df.sort_values('RMSE')
    results_path = output_path / "evaluation_results.csv"
    results_df.to_csv(results_path, index=False)

    # 生成可视化 - 使用RSE风格
    print("\nGenerating visualizations...")

    # 1. 模型比较图 - 更新为RSE风格
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 为每个模型分配颜色
        bar_colors = [MODEL_COLORS.get(m, '#1f77b4') for m in results_df['Model']]

        # RMSE比较
        axes[0].bar(results_df['Model'], results_df['RMSE'], alpha=0.8, color=bar_colors)
        axes[0].set_xlabel('Model', fontsize=9)
        axes[0].set_ylabel('RMSE', fontsize=9)
        axes[0].set_title('Model RMSE Comparison', fontsize=10, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45, labelsize=8)
        axes[0].grid(True, alpha=0.2, linestyle='--')
        # 设置坐标轴边框
        for spine in ['top', 'right']:
            axes[0].spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            axes[0].spines[spine].set_linewidth(0.75)
            axes[0].spines[spine].set_color('black')

        # MAE比较
        axes[1].bar(results_df['Model'], results_df['MAE'], alpha=0.8, color=bar_colors)
        axes[1].set_xlabel('Model', fontsize=9)
        axes[1].set_ylabel('MAE', fontsize=9)
        axes[1].set_title('Model MAE Comparison', fontsize=10, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45, labelsize=8)
        axes[1].grid(True, alpha=0.2, linestyle='--')
        for spine in ['top', 'right']:
            axes[1].spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            axes[1].spines[spine].set_linewidth(0.75)
            axes[1].spines[spine].set_color('black')

        # R²比较
        axes[2].bar(results_df['Model'], results_df['R2'], alpha=0.8, color=bar_colors)
        axes[2].set_xlabel('Model', fontsize=9)
        axes[2].set_ylabel('R²', fontsize=9)
        axes[2].set_title('Model R² Comparison', fontsize=10, fontweight='bold')
        axes[2].tick_params(axis='x', rotation=45, labelsize=8)
        axes[2].grid(True, alpha=0.2, linestyle='--')
        for spine in ['top', 'right']:
            axes[2].spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            axes[2].spines[spine].set_linewidth(0.75)
            axes[2].spines[spine].set_color('black')

        plt.tight_layout()
        plt.savefig(plots_dir / "model_comparison.png", dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print("  Created model comparison plot")
        gc.collect()
    except Exception as e:
        print(f"  Error creating model comparison plot: {e}")

    # 2. 最佳模型详细分析 - 将散点图也改为hexbin
    try:
        best_model_name = results_df.iloc[0]['Model']
        best_model = models[best_model_name]
        y_pred_best = all_predictions[best_model_name]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 改为hexbin图 - 使用密度颜色映射
        # 在最佳模型详细分析部分，找到hexbin图代码，修改如下：

        # 改为hexbin图 - 使用密度颜色映射
        hexbin0 = axes[0, 0].hexbin(y, y_pred_best, gridsize=40, cmap=DENSITY_CMAP,
                                    mincnt=1, bins='log', edgecolors='none',
                                    alpha=0.9)

        # 关键修改：设置轴比为1:1
        axes[0, 0].set_aspect('equal', adjustable='box')

        # 设置相同的轴范围
        data_min = min(y.min(), y_pred_best.min())
        data_max = max(y.max(), y_pred_best.max())
        data_range = data_max - data_min
        data_min -= 0.02 * data_range
        data_max += 0.02 * data_range

        axes[0, 0].set_xlim(data_min, data_max)
        axes[0, 0].set_ylim(data_min, data_max)
        axes[0, 0].plot([data_min, data_max], [data_min, data_max], 'r--', lw=1.5, alpha=0.8)
        axes[0, 0].set_xlabel('True Value', fontsize=9)
        axes[0, 0].set_ylabel('Predicted Value', fontsize=9)
        axes[0, 0].set_title(f'{best_model_name} - Prediction vs True', fontsize=10, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.2, linestyle='--')
        for spine in ['top', 'right']:
            axes[0, 0].spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            axes[0, 0].spines[spine].set_linewidth(0.75)
            axes[0, 0].spines[spine].set_color('black')

        # 添加颜色条
        cbar0 = plt.colorbar(hexbin0, ax=axes[0, 0], shrink=0.8)
        cbar0.set_label('log10(Density)', fontsize=8)
        cbar0.ax.tick_params(labelsize=7)

        # 残差图 - 保留散点图
        residuals = y - y_pred_best
        scatter1 = axes[0, 1].scatter(y_pred_best, residuals, alpha=0.6, s=10,
                                      c=np.abs(residuals), cmap=ERROR_CMAP, edgecolors='none')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=1.5, alpha=0.8)
        axes[0, 1].set_xlabel('Predicted Value', fontsize=9)
        axes[0, 1].set_ylabel('Residual', fontsize=9)
        axes[0, 1].set_title(f'{best_model_name} - Residual Plot', fontsize=10, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.2, linestyle='--')
        for spine in ['top', 'right']:
            axes[0, 1].spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            axes[0, 1].spines[spine].set_linewidth(0.75)
            axes[0, 1].spines[spine].set_color('black')

        cbar1 = plt.colorbar(scatter1, ax=axes[0, 1], shrink=0.8)
        cbar1.set_label('|Residual|', fontsize=8)
        cbar1.ax.tick_params(labelsize=7)

        # 误差分布
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black',
                        color=MODEL_COLORS.get(best_model_name, '#1f77b4'))
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=1.5, alpha=0.8)
        axes[1, 0].set_xlabel('Error', fontsize=9)
        axes[1, 0].set_ylabel('Frequency', fontsize=9)
        axes[1, 0].set_title(f'{best_model_name} - Error Distribution', fontsize=10, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.2, linestyle='--')
        for spine in ['top', 'right']:
            axes[1, 0].spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            axes[1, 0].spines[spine].set_linewidth(0.75)
            axes[1, 0].spines[spine].set_color('black')

        # QQ图
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f'{best_model_name} - QQ Plot', fontsize=10, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.2, linestyle='--')
        axes[1, 1].set_xlabel('Theoretical Quantiles', fontsize=9)
        axes[1, 1].set_ylabel('Sample Quantiles', fontsize=9)
        for spine in ['top', 'right']:
            axes[1, 1].spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            axes[1, 1].spines[spine].set_linewidth(0.75)
            axes[1, 1].spines[spine].set_color('black')

        plt.tight_layout()
        plt.savefig(plots_dir / f"{best_model_name}_analysis.png", dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f"  Created detailed analysis for best model: {best_model_name}")
        gc.collect()
    except Exception as e:
        print(f"  Error creating best model analysis: {e}")

    # 3. 新增：2×2 hexbin密度图矩阵（每个模型一个图，带颜色条）
    try:
        create_hexbin_matrix(y, all_predictions, results_df['Model'].tolist(), plots_dir)
    except Exception as e:
        print(f"  Error creating hexbin density matrix: {e}")

    # 4. 所有模型的预测对比 - 更新为RSE风格
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        # 随机选择一些样本进行可视化
        n_samples = min(100, len(y))
        sample_indices = np.random.choice(len(y), n_samples, replace=False)
        y_sample = y.iloc[sample_indices]

        x_pos = np.arange(n_samples)
        width = 0.8 / len(models)

        # 为每个模型创建柱状图
        for i, (model_name, y_pred) in enumerate(all_predictions.items()):
            y_pred_sample = y_pred[sample_indices]
            errors = np.abs(y_sample - y_pred_sample)
            color = MODEL_COLORS.get(model_name, f'C{i}')
            ax.bar(x_pos + i * width, errors, width, label=model_name, alpha=0.7, color=color)

        ax.set_xlabel('Sample Index', fontsize=9)
        ax.set_ylabel('Absolute Error', fontsize=9)
        ax.set_title('Prediction Error Comparison Across Models', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, alpha=0.2, linestyle='--')

        # 设置坐标轴边框
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(0.75)
            ax.spines[spine].set_color('black')

        ax.tick_params(axis='both', which='both', length=4, width=0.75, direction='out', labelsize=8)

        plt.tight_layout()
        plt.savefig(plots_dir / "all_models_error_comparison.png", dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print("  Created all models error comparison plot")
        gc.collect()
    except Exception as e:
        print(f"  Error creating error comparison plot: {e}")

    # 生成报告
    print("\nGenerating evaluation report...")

    report = {
        'quick_evaluation': {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'models_evaluated': list(models.keys()),
            'data_samples': len(y),
            'output_dir': str(output_path)
        },
        'results': results_df.to_dict('records'),
        'best_model': {
            'name': results_df.iloc[0]['Model'] if not results_df.empty else None,
            'rmse': float(results_df.iloc[0]['RMSE']) if not results_df.empty else None,
            'mae': float(results_df.iloc[0]['MAE']) if not results_df.empty else None,
            'r2': float(results_df.iloc[0]['R2']) if not results_df.empty else None
        }
    }

    report_path = output_path / "quick_evaluation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 打印总结
    print("\n" + "=" * 80)
    print("Quick Evaluation Complete!")
    print("=" * 80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Directory: {output_path}")

    print(f"\nModel Performance Ranking:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Model':<20} {'RMSE':<12} {'MAE':<12} {'R²':<10}")
    print("-" * 80)
    for i, (_, row) in enumerate(results_df.iterrows()):
        print(f"{i + 1:<4} {row['Model']:<20} {row['RMSE']:<12.6f} "
              f"{row['MAE']:<12.6f} {row['R2']:<10.4f}")

    print(f"\nGenerated Files:")
    print(f"  Evaluation Results: {results_path}")
    print(f"  Plots Directory: {plots_dir}/")
    print(f"  Evaluation Report: {report_path}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Quick Model Evaluation Tool')
    parser.add_argument('--models_dir', type=str, required=True,
                        help='Directory containing pkl model files')
    parser.add_argument('--sample', type=float, default=0.2,
                        help='Data sampling ratio (0.01-1.0)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: create sibling directory)')

    args = parser.parse_args()

    quick_evaluate(
        models_dir=args.models_dir,
        data_sample=args.sample,
        output_dir=args.output_dir
    )