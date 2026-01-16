# ==================== model_evaluator.py ====================
"""
模型评估和可视化模块
负责加载已保存的模型进行评估、生成图表和报告
"""
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Any
from datetime import datetime
import warnings

# 机器学习库
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, explained_variance_score)

# 高级分析工具
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed, skipping SHAP analysis")

# 可视化库
import plotly.graph_objects as go

# 统计库
from scipy import stats

# 项目模块
from config import ExperimentConfig
from utils import setup_logger
from data_loader import DataLoader

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
from matplotlib.colors import LinearSegmentedColormap

ERROR_COLORS = [
    (0.0, 0.15, 0.7),  # 深蓝色
    (0.1, 0.3, 0.8),  # 蓝色
    (0.3, 0.5, 0.9),  # 中蓝色
    (0.5, 0.7, 1.0),  # 浅蓝色
    (0.7, 0.85, 1.0),  # 更浅的蓝色
    (0.95, 0.98, 1.0),  # 近白色
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
    'ExtraTrees': '#17becf',  # 青色
}

# 散点图颜色映射
SCATTER_COLORS = [
    (0.9, 0.95, 1.0),  # 浅蓝
    (0.7, 0.8, 1.0),  # 淡蓝
    (0.4, 0.6, 1.0),  # 中蓝
    (0.2, 0.4, 0.9),  # 蓝色
    (0.1, 0.2, 0.8),  # 深蓝
    (0.3, 0.1, 0.7),  # 蓝紫
    (0.5, 0.1, 0.6),  # 紫色
]
SCATTER_CMAP = LinearSegmentedColormap.from_list('scatter_cmap', SCATTER_COLORS, N=256)


class ModelEvaluator:
    """模型评估器类"""

    def __init__(self, saved_dir: Path, config: ExperimentConfig = None,
                 logger=None, use_shap: bool = True):

        self.saved_dir = Path(saved_dir)
        self.config = config or ExperimentConfig
        self.logger = logger or setup_logger('ModelEvaluator')
        self.use_shap = use_shap and SHAP_AVAILABLE

        # 验证目录结构
        if not self.saved_dir.exists():
            raise ValueError(f"Saved model directory does not exist: {self.saved_dir}")

        # 确定子目录
        self.models_dir = self.saved_dir / "models"
        if not self.models_dir.exists():
            # 如果models子目录不存在，假设模型直接保存在saved_dir中
            self.models_dir = self.saved_dir

        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.saved_dir / f"evaluation_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        self.plots_dir = self.output_dir / "plots"
        self.tables_dir = self.output_dir / "tables"
        for d in [self.plots_dir, self.tables_dir]:
            d.mkdir(exist_ok=True)

        # 加载模型和配置
        self.models = self._load_models()
        self.model_info = self._load_model_info()
        self.results = {}

        print(f"Loaded {len(self.models)} models from {self.saved_dir}")

    def _load_models(self) -> Dict[str, Any]:
        """加载所有保存的模型"""
        models = {}

        # 查找所有pkl文件
        model_files = list(self.models_dir.glob("*_model.pkl"))
        if not model_files:
            # 也查找普通的pkl文件
            model_files = list(self.models_dir.glob("*.pkl"))

        for model_file in model_files:
            # 提取模型名称
            if "_model.pkl" in model_file.name:
                model_name = model_file.stem.replace("_model", "")
            else:
                model_name = model_file.stem

            try:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                models[model_name] = model
                print(f"   Loaded model: {model_name}")
            except Exception as e:
                print(f"   Warning: Failed to load model {model_name}: {e}")

        return models

    def _load_model_info(self) -> Dict[str, Dict]:
        """加载模型信息"""
        info_path = self.saved_dir / "training_summary.json"
        if info_path.exists():
            try:
                with open(info_path, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                print("   Loaded model training information")
                return info
            except Exception as e:
                print(f"   Warning: Failed to load model info: {e}")

        # 创建基本的模型信息
        info = {}
        for model_name in self.models.keys():
            info[model_name] = {
                'description': f'{model_name} model',
                'color': MODEL_COLORS.get(model_name, '#1f77b4')
            }
        return info

    def load_training_artifacts(self) -> Dict:
        """加载训练时的标准化器和特征名称"""
        artifacts_path = self.saved_dir / "training_artifacts.pkl"
        if artifacts_path.exists():
            try:
                with open(artifacts_path, 'rb') as f:
                    artifacts = pickle.load(f)
                print("   Loaded training scaler and feature names")
                return artifacts
            except Exception as e:
                print(f"   Warning: Failed to load training artifacts: {e}")

        return None

    def prepare_test_data(self, data: pd.DataFrame,
                          use_feature_engineering: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """准备测试数据（与训练时一致）"""
        print("=" * 70)
        print("Preparing test data...")

        # 加载训练时的标准化器
        artifacts = self.load_training_artifacts()

        # 创建特征（使用与训练时相同的逻辑）
        from data_loader import AdvancedFeatureEngineering
        feature_engineer = AdvancedFeatureEngineering()

        if use_feature_engineering:
            X = feature_engineer.create_features(data)
            print(f"   Feature engineering generated {X.shape[1]} features")
        else:
            # 使用基础特征
            base_features = ['sza', 'vza', 'raa', 'aod550', 'h2o', 'o3',
                             'wavelength', 'rho_toa']
            X = data[base_features].copy()

            # 添加反演反射率
            if 'rho_true' in data.columns and 'error_absolute' in data.columns:
                X['rho_retrieved'] = data['rho_true'] + data['error_absolute']
            else:
                X['rho_retrieved'] = data.get('rho_true', 0.2)

        # 目标变量
        y = data['error_absolute']

        # 标准化（使用训练时的标准化器）
        if artifacts and 'scaler' in artifacts:
            scaler = artifacts['scaler']
            X_scaled = scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            print(f"   Applied training scaler")
        else:
            print(f"   Warning: Training scaler not found, creating new one")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        print(f"   Number of features: {X_scaled.shape[1]}")
        print(f"   Number of samples: {X_scaled.shape[0]}")

        return X_scaled, y

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """评估所有已加载的模型"""
        print("=" * 70)
        print("Evaluating models...")

        results = []

        for model_name, model in self.models.items():
            try:
                print(f"   Evaluating {model_name}...")

                # 预测
                y_pred = model.predict(X_test)

                # 计算指标
                metrics = self._compute_metrics(y_test, y_pred, model_name)
                self.results[model_name] = metrics

                # 保存结果
                result_row = {
                    'Model': model_name,
                    'Description': self.model_info.get(model_name, {}).get('description', f'{model_name} model'),
                    **metrics
                }
                results.append(result_row)

                print(f"    RMSE: {metrics['RMSE']:.6f}")
                print(f"    MAE: {metrics['MAE']:.6f}")
                print(f"    R²: {metrics['R2']:.4f}")

            except Exception as e:
                print(f"   {model_name} evaluation failed: {e}")

        # 创建结果DataFrame
        results_df = pd.DataFrame(results)

        if not results_df.empty:
            # 按RMSE排序
            results_df = results_df.sort_values('RMSE')

            # 保存结果
            results_path = self.tables_dir / "evaluation_results.csv"
            results_df.to_csv(results_path, index=False)

            # 打印排名
            print(f"\nModel Performance Ranking:")
            print("-" * 70)
            print(f"{'Rank':<4} {'Model':<15} {'RMSE':<10} {'R²':<8} {'MAE':<10}")
            print("-" * 70)
            for i, (_, row) in enumerate(results_df.iterrows()):
                print(f"{i + 1:<4} {row['Model']:<15} {row['RMSE']:.6f}  {row['R2']:.4f}  {row['MAE']:.6f}")

        return results_df

    def _compute_metrics(self, y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> Dict:
        """计算评估指标"""
        metrics = {}

        # 基础指标
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['R2'] = r2_score(y_true, y_pred)
        metrics['Explained_Variance'] = explained_variance_score(y_true, y_pred)

        # 相对误差
        nonzero_mask = y_true != 0
        if nonzero_mask.any():
            relative_errors = np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])
            metrics['MAPE'] = np.mean(relative_errors) * 100
            metrics['Median_APE'] = np.median(relative_errors) * 100
        else:
            metrics['MAPE'] = np.nan
            metrics['Median_APE'] = np.nan

        # 偏差和精度
        residuals = y_true - y_pred
        metrics['Bias'] = np.mean(residuals)
        metrics['Std_Residuals'] = np.std(residuals)

        # 分位数误差
        metrics['Q10_Error'] = np.percentile(np.abs(residuals), 10)
        metrics['Q90_Error'] = np.percentile(np.abs(residuals), 90)

        # 保存预测值
        metrics['y_true'] = y_true.values
        metrics['y_pred'] = y_pred

        return metrics

    def generate_all_visualizations(self, X_test: pd.DataFrame, y_test: pd.Series,
                                    results_df: pd.DataFrame):
        """生成所有可视化图表"""
        print("=" * 70)
        print("Generating visualizations...")

        # 1. 模型比较图
        self._plot_model_comparison(results_df)

        # 2. 散点图矩阵
        self._plot_scatter_matrix(X_test, y_test)

        # 3. 残差分析图
        self._plot_residual_analysis(X_test, y_test)

        # 4. 预测误差分布图
        self._plot_error_distribution()

        # 5. SHAP分析（如果可用）
        if self.use_shap and self.models:
            self._perform_shap_analysis(X_test)

        # 6. 交互式3D图
        self._plot_interactive_3d()

        print(f"All plots saved to: {self.plots_dir}")

    def _plot_model_comparison(self, results_df: pd.DataFrame):
        """绘制模型比较图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. RMSE和MAE比较
        ax = axes[0, 0]
        x = np.arange(len(results_df))
        width = 0.35

        # 获取颜色
        colors = [MODEL_COLORS.get(m, '#1f77b4') for m in results_df['Model']]

        ax.bar(x - width / 2, results_df['RMSE'], width, label='RMSE',
               alpha=0.8, color=[c for c in colors])
        ax.bar(x + width / 2, results_df['MAE'], width, label='MAE',
               alpha=0.8, color=[c for c in colors])
        ax.set_xlabel('Model', fontsize=9)
        ax.set_ylabel('Error', fontsize=9)
        ax.set_title('Model Error Comparison', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, alpha=0.2, linestyle='--')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(0.75)
            ax.spines[spine].set_color('black')

        # 2. R²比较
        ax = axes[0, 1]
        bars = ax.bar(results_df['Model'], results_df['R2'], color=colors, alpha=0.8)
        ax.set_xlabel('Model', fontsize=9)
        ax.set_ylabel('R²', fontsize=9)
        ax.set_title('Model Coefficient of Determination (R²) Comparison',
                     fontsize=10, fontweight='bold')
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.2, linestyle='--')

        # 添加数值标签
        for bar, value in zip(bars, results_df['R2']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(0.75)
            ax.spines[spine].set_color('black')

        # 3. 预测偏差
        ax = axes[0, 2]
        ax.bar(results_df['Model'], results_df['Bias'], alpha=0.8, color=colors)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, linewidth=1.2)
        ax.set_xlabel('Model', fontsize=9)
        ax.set_ylabel('Prediction Bias', fontsize=9)
        ax.set_title('Model Prediction Bias', fontsize=10, fontweight='bold')
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.2, linestyle='--')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(0.75)
            ax.spines[spine].set_color('black')

        # 4. MAPE比较
        ax = axes[1, 0]
        ax.bar(results_df['Model'], results_df['MAPE'], alpha=0.8, color=colors)
        ax.set_xlabel('Model', fontsize=9)
        ax.set_ylabel('MAPE (%)', fontsize=9)
        ax.set_title('Mean Absolute Percentage Error (MAPE)',
                     fontsize=10, fontweight='bold')
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.2, linestyle='--')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(0.75)
            ax.spines[spine].set_color('black')

        # 5. 解释方差
        ax = axes[1, 1]
        ax.bar(results_df['Model'], results_df['Explained_Variance'], alpha=0.8, color=colors)
        ax.set_xlabel('Model', fontsize=9)
        ax.set_ylabel('Explained Variance', fontsize=9)
        ax.set_title('Model Explained Variance', fontsize=10, fontweight='bold')
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.2, linestyle='--')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(0.75)
            ax.spines[spine].set_color('black')

        # 6. 标准差
        ax = axes[1, 2]
        ax.bar(results_df['Model'], results_df['Std_Residuals'], alpha=0.8, color=colors)
        ax.set_xlabel('Model', fontsize=9)
        ax.set_ylabel('Std of Residuals', fontsize=9)
        ax.set_title('Standard Deviation of Residuals', fontsize=10, fontweight='bold')
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.2, linestyle='--')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(0.75)
            ax.spines[spine].set_color('black')

        plt.tight_layout()
        plt.savefig(self.plots_dir / "model_comparison.png", dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

    def _plot_scatter_matrix(self, X_test: pd.DataFrame, y_test: pd.Series):
        """绘制散点图矩阵"""
        n_models = len(self.models)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (model_name, model) in enumerate(list(self.models.items())[:len(axes)]):
            ax = axes[idx]
            y_pred = model.predict(X_test)

            # 散点图 - 使用颜色映射显示误差
            abs_errors = np.abs(y_test - y_pred)
            scatter = ax.scatter(y_test, y_pred, alpha=0.6, s=10,
                                 c=abs_errors, cmap=SCATTER_CMAP, edgecolors='none')

            # 对角线
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, alpha=0.8)

            # 统计信息
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            ax.text(0.05, 0.95, f'RMSE = {rmse:.4f}\nR² = {r2:.3f}\nMAE = {mae:.4f}',
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                              edgecolor='gray', linewidth=0.5))

            ax.set_xlabel('True Value', fontsize=9)
            ax.set_ylabel('Predicted Value', fontsize=9)
            ax.set_title(f'{model_name} - Prediction vs True', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.2, linestyle='--')

            # 设置坐标轴边框
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_linewidth(0.75)
                ax.spines[spine].set_color('black')

            ax.tick_params(axis='both', which='both', length=4, width=0.75,
                           direction='out', labelsize=8)

            # 添加颜色条
            if idx == 0:
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
                cbar.set_label('Absolute Error', fontsize=8)
                cbar.ax.tick_params(labelsize=7)

        # 隐藏多余的子图
        for idx in range(len(self.models), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "prediction_scatter_plots.png", dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

    def _plot_residual_analysis(self, X_test: pd.DataFrame, y_test: pd.Series):
        """绘制残差分析图"""
        if not self.results:
            return

        # 获取最佳模型
        best_model_name = list(self.results.keys())[0]
        model = self.models[best_model_name]
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. 残差分布
        ax = axes[0, 0]
        ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black',
                color=MODEL_COLORS.get(best_model_name, '#1f77b4'))
        ax.axvline(x=0, color='r', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Residual', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title('Residual Distribution', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(0.75)
            ax.spines[spine].set_color('black')

        # 2. 残差vs预测值
        ax = axes[0, 1]
        scatter = ax.scatter(y_pred, residuals, alpha=0.6, s=10,
                             c=np.abs(residuals), cmap=SCATTER_CMAP, edgecolors='none')
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Predicted Value', fontsize=9)
        ax.set_ylabel('Residual', fontsize=9)
        ax.set_title('Residuals vs Predicted Values', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--')

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('|Residual|', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(0.75)
            ax.spines[spine].set_color('black')

        # 3. QQ图
        ax = axes[0, 2]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot of Residuals', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_xlabel('Theoretical Quantiles', fontsize=9)
        ax.set_ylabel('Sample Quantiles', fontsize=9)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(0.75)
            ax.spines[spine].set_color('black')

        # 4. 累计残差
        ax = axes[1, 0]
        cum_residuals = np.cumsum(residuals)
        ax.plot(cum_residuals, color=MODEL_COLORS.get(best_model_name, '#1f77b4'), linewidth=1.5)
        ax.set_xlabel('Sample Index', fontsize=9)
        ax.set_ylabel('Cumulative Residual', fontsize=9)
        ax.set_title('Cumulative Residual Plot', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(0.75)
            ax.spines[spine].set_color('black')

        # 5. 误差vs目标值
        ax = axes[1, 1]
        ax.scatter(y_test, np.abs(residuals), alpha=0.6, s=10,
                   c=np.abs(residuals), cmap=SCATTER_CMAP, edgecolors='none')
        ax.set_xlabel('True Value', fontsize=9)
        ax.set_ylabel('Absolute Error', fontsize=9)
        ax.set_title('Absolute Error vs True Value', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(0.75)
            ax.spines[spine].set_color('black')

        # 6. 残差自相关
        ax = axes[1, 2]
        pd.plotting.autocorrelation_plot(residuals, ax=ax)
        ax.set_title('Residual Autocorrelation', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(0.75)
            ax.spines[spine].set_color('black')

        # 设置所有子图的刻度标签大小
        for ax in axes.flatten():
            ax.tick_params(axis='both', which='both', labelsize=8)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "residual_analysis.png", dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

    def _plot_error_distribution(self):
        """绘制误差分布图"""
        if not self.results:
            return

        n_models = len(self.results)
        n_cols = min(2, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (model_name, metrics) in enumerate(list(self.results.items())[:len(axes)]):
            ax = axes[idx]

            y_true = metrics['y_true']
            y_pred = metrics['y_pred']
            errors = y_true - y_pred

            # 误差直方图
            ax.hist(errors, bins=50, alpha=0.7, density=True, edgecolor='black',
                    color=MODEL_COLORS.get(model_name, '#1f77b4'))

            # 拟合正态分布
            mu, sigma = stats.norm.fit(errors)
            x = np.linspace(errors.min(), errors.max(), 100)
            p = stats.norm.pdf(x, mu, sigma)
            ax.plot(x, p, 'r-', linewidth=1.5, alpha=0.8,
                    label=f'Normal fit\nμ={mu:.4f}, σ={sigma:.4f}')

            ax.set_xlabel('Error', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.set_title(f'{model_name} - Error Distribution', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8, frameon=False)
            ax.grid(True, alpha=0.2, linestyle='--')

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_linewidth(0.75)
                ax.spines[spine].set_color('black')

            ax.tick_params(axis='both', which='both', length=4, width=0.75,
                           direction='out', labelsize=8)

        # 隐藏多余的子图
        for idx in range(len(self.results), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "error_distributions.png", dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

    def _perform_shap_analysis(self, X_test: pd.DataFrame):
        """执行SHAP分析"""
        if not self.models:
            return

        print("   Performing SHAP analysis...")

        # 选择最佳模型进行SHAP分析
        best_model_name = list(self.results.keys())[0]
        model = self.models[best_model_name]

        # 采样以减少计算时间
        if len(X_test) > 1000:
            X_sample = X_test.sample(n=min(500, len(X_test)), random_state=self.config.RANDOM_SEED)
        else:
            X_sample = X_test

        try:
            # 创建SHAP解释器
            if best_model_name in ['RandomForest', 'GradientBoosting', 'ExtraTrees']:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            elif best_model_name in ['XGBoost', 'LightGBM']:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            else:
                # 对于其他模型，使用KernelExplainer
                explainer = shap.KernelExplainer(model.predict, X_sample[:100])
                shap_values = explainer.shap_values(X_sample)

            # 1. 特征重要性摘要图
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.title(f'{best_model_name} - SHAP Feature Importance',
                      fontsize=11, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"{best_model_name}_shap_summary.png",
                        dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()

            # 2. 条形图
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            plt.title(f'{best_model_name} - SHAP Feature Importance (Bar Plot)',
                      fontsize=11, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"{best_model_name}_shap_bar.png",
                        dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()

            print("   SHAP analysis completed")

        except Exception as e:
            print(f"   SHAP analysis failed: {e}")

    def _plot_interactive_3d(self):
        """绘制交互式3D图"""
        try:
            import plotly.offline as pyo

            # 创建3D散点图
            if self.results:
                best_model_name = list(self.results.keys())[0]
                metrics = self.results[best_model_name]

                fig = go.Figure()

                # 添加预测vs实际散点
                fig.add_trace(go.Scatter3d(
                    x=metrics['y_true'],
                    y=metrics['y_pred'],
                    z=np.abs(metrics['y_true'] - metrics['y_pred']),
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=np.abs(metrics['y_true'] - metrics['y_pred']),
                        colorscale='Viridis',
                        opacity=0.6,
                        colorbar=dict(title="Absolute Error")
                    ),
                    name='Predicted points'
                ))

                # 添加理想预测平面
                x_range = np.linspace(min(metrics['y_true']), max(metrics['y_true']), 10)
                y_range = np.linspace(min(metrics['y_pred']), max(metrics['y_pred']), 10)
                X, Y = np.meshgrid(x_range, y_range)
                Z = np.zeros_like(X)

                fig.add_trace(go.Surface(
                    x=X, y=Y, z=Z,
                    colorscale='Greys',
                    opacity=0.2,
                    showscale=False,
                    name='Ideal prediction plane'
                ))

                fig.update_layout(
                    title=f'{best_model_name} - 3D Prediction Visualization',
                    scene=dict(
                        xaxis_title='True Value',
                        yaxis_title='Predicted Value',
                        zaxis_title='Absolute Error'
                    ),
                    width=900,
                    height=700
                )

                # 保存为HTML文件
                plot_path = self.plots_dir / f"{best_model_name}_3d_visualization.html"
                fig.write_html(str(plot_path))

                print(f"   3D interactive plot saved: {plot_path}")

        except Exception as e:
            print(f"   3D plot generation failed: {e}")

    def generate_report(self, results_df: pd.DataFrame):
        """生成评估报告"""
        print("=" * 70)
        print("Generating evaluation report...")

        report = {
            'evaluation_summary': {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'models_evaluated': list(self.models.keys()),
                'total_samples': len(self.results.get(list(self.models.keys())[0], {}).get('y_true', []))
                if self.results else 0,
                'saved_dir': str(self.saved_dir),
                'output_dir': str(self.output_dir),
                'shap_analysis': self.use_shap
            },
            'model_performance': results_df.to_dict('records'),
            'best_model': {
                'name': results_df.iloc[0]['Model'] if not results_df.empty else None,
                'rmse': float(results_df.iloc[0]['RMSE']) if not results_df.empty else None,
                'r2': float(results_df.iloc[0]['R2']) if not results_df.empty else None,
                'mae': float(results_df.iloc[0]['MAE']) if not results_df.empty else None
            }
        }

        # 保存JSON报告
        report_path = self.output_dir / "evaluation_summary.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 生成文本报告
        text_report = self._generate_text_report(results_df)
        text_path = self.output_dir / "evaluation_report.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_report)

        print(f"Evaluation reports saved:")
        print(f"  JSON report: {report_path}")
        print(f"  Text report: {text_path}")

    def _generate_text_report(self, results_df: pd.DataFrame) -> str:
        """生成文本格式的报告"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("Machine Learning Model Evaluation Report")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Model Directory: {self.saved_dir}")
        report_lines.append(f"Models Evaluated: {len(self.models)}")
        report_lines.append(f"SHAP Analysis: {'Enabled' if self.use_shap else 'Disabled'}")
        report_lines.append("")

        report_lines.append("Model Performance Ranking:")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Rank':<4} {'Model':<15} {'RMSE':<12} {'R²':<10} {'MAE':<12}")
        report_lines.append("-" * 80)

        for i, (_, row) in enumerate(results_df.iterrows()):
            report_lines.append(f"{i + 1:<4} {row['Model']:<15} {row['RMSE']:<12.6f} "
                                f"{row['R2']:<10.4f} {row['MAE']:<12.6f}")

        report_lines.append("")
        report_lines.append("Best Model Details:")
        report_lines.append("-" * 80)
        if not results_df.empty:
            best_model = results_df.iloc[0]
            report_lines.append(f"Model Name: {best_model['Model']}")
            report_lines.append(f"Description: {best_model['Description']}")
            report_lines.append(f"RMSE: {best_model['RMSE']:.6f}")
            report_lines.append(f"MAE: {best_model['MAE']:.6f}")
            report_lines.append(f"R²: {best_model['R2']:.4f}")
            report_lines.append(f"MAPE: {best_model['MAPE']:.2f}%")
            report_lines.append(f"Bias: {best_model['Bias']:.6f}")

        report_lines.append("")
        report_lines.append("Output Files:")
        report_lines.append(f"  Evaluation Results: {self.tables_dir}/evaluation_results.csv")
        report_lines.append(f"  Plots Directory: {self.plots_dir}/")
        report_lines.append(f"  Evaluation Report: {self.output_dir}/evaluation_summary.json")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)


def evaluate_main():
    """评估主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Machine Learning Model Evaluation System')
    parser.add_argument('--saved_dir', type=str, required=True,
                        help='Directory path of saved models')
    parser.add_argument('--sample', type=float, default=0.3,
                        help='Test data sampling ratio (0.01-1.0)')
    parser.add_argument('--no_feature_engineering', action='store_true',
                        help='Disable feature engineering')
    parser.add_argument('--no_shap', action='store_true',
                        help='Disable SHAP analysis')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Test data directory (default: use config.DATA_DIR)')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("Machine Learning Model Evaluation System")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model Directory: {args.saved_dir}")
    print(f"Data Sampling: {args.sample * 100:.1f}%")
    print(f"Feature Engineering: {'Disabled' if args.no_feature_engineering else 'Enabled'}")
    print(f"SHAP Analysis: {'Disabled' if args.no_shap else 'Enabled'}")
    print("=" * 80)

    try:
        # 修改配置的数据目录（如果需要）
        config = ExperimentConfig
        if args.data_dir:
            config.DATA_DIR = Path(args.data_dir)

        # 创建模型评估器
        evaluator = ModelEvaluator(
            saved_dir=Path(args.saved_dir),
            config=config,
            use_shap=not args.no_shap
        )

        # 加载测试数据
        dl = DataLoader(config=config)
        test_data = dl.load_data(sample_fraction=args.sample)

        # 准备特征
        X_test, y_test = evaluator.prepare_test_data(
            test_data,
            use_feature_engineering=not args.no_feature_engineering
        )

        # 评估模型
        results_df = evaluator.evaluate_models(X_test, y_test)

        # 生成可视化
        evaluator.generate_all_visualizations(X_test, y_test, results_df)

        # 生成报告
        evaluator.generate_report(results_df)

        print("\n" + "=" * 80)
        print("Model Evaluation Complete!")
        print("=" * 80)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if not results_df.empty:
            print(f"\nModel Performance Ranking:")
            print("-" * 80)
            print(f"{'Rank':<4} {'Model':<15} {'RMSE':<12} {'R²':<10} {'MAE':<12}")
            print("-" * 80)
            for i, (_, row) in enumerate(results_df.iterrows()):
                print(f"{i + 1:<4} {row['Model']:<15} {row['RMSE']:<12.6f} "
                      f"{row['R2']:<10.4f} {row['MAE']:<12.6f}")

        print("\nGenerated Files:")
        print(f"  Evaluation Results: {evaluator.tables_dir}/evaluation_results.csv")
        print(f"  Plots Directory: {evaluator.plots_dir}/")
        print(f"  Evaluation Report: {evaluator.output_dir}/evaluation_summary.json")
        print("=" * 80)

    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    evaluate_main()