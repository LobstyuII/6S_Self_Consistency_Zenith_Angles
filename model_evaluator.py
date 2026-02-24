# ==================== model_evaluator.py ====================
"""
模型评估和可视化模块（新框架：目标变量 ΔTOA）
负责加载已保存的模型进行评估、生成图表和报告
"""
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Any, List
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
from scipy.stats import gaussian_kde

# 项目模块
from config import ExperimentConfig
from utils import setup_logger
from data_loader import DataLoader

warnings.filterwarnings('ignore')

# 设置专业科研字体（RSE期刊风格）
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

# 自定义颜色映射
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

# Hexbin图颜色映射 - 密度相关
HEXBIN_COLORS = [
    (0.95, 0.95, 0.95),  # 浅灰色
    (0.8, 0.8, 0.8),  # 浅中灰
    (0.6, 0.6, 0.6),  # 中灰色
    (0.4, 0.4, 0.4),  # 深灰色
    (0.2, 0.2, 0.2),  # 深灰色
    (0.1, 0.1, 0.1),  # 近黑色
]
HEXBIN_CMAP = LinearSegmentedColormap.from_list('hexbin_cmap', HEXBIN_COLORS, N=256)


class ModelEvaluator:
    """模型评估器类（新框架：目标变量为 delta_toa）"""

    def __init__(self, saved_dir: Path, config: ExperimentConfig = None,
                 logger=None, use_shap: bool = True):

        self.saved_dir = Path(saved_dir)
        self.config = config or ExperimentConfig
        self.logger = logger or setup_logger('ModelEvaluator')
        self.use_shap = use_shap and SHAP_AVAILABLE

        if not self.saved_dir.exists():
            raise ValueError(f"Saved model directory does not exist: {self.saved_dir}")

        self.models_dir = self.saved_dir / "models"
        if not self.models_dir.exists():
            self.models_dir = self.saved_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.saved_dir / f"evaluation_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.plots_dir = self.output_dir / "plots"
        self.tables_dir = self.output_dir / "tables"
        for d in [self.plots_dir, self.tables_dir]:
            d.mkdir(exist_ok=True)

        self.models = self._load_models()
        self.model_info = self._load_model_info()
        self.results = {}

        print(f"Loaded {len(self.models)} models from {self.saved_dir}")

    def _load_models(self) -> Dict[str, Any]:
        """加载所有保存的模型"""
        models = {}
        model_files = list(self.models_dir.glob("*_model.pkl"))
        if not model_files:
            model_files = list(self.models_dir.glob("*.pkl"))

        for model_file in model_files:
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
                          use_feature_engineering: bool = True,
                          target_column: str = 'delta_toa') -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备测试数据（与训练时一致）
        参数:
            data: 测试数据
            use_feature_engineering: 是否使用特征工程
            target_column: 目标变量列名（默认 delta_toa）
        """
        print("=" * 70)
        print("Preparing test data...")

        artifacts = self.load_training_artifacts()

        from data_loader import AdvancedFeatureEngineering
        feature_engineer = AdvancedFeatureEngineering()

        if use_feature_engineering:
            X = feature_engineer.create_features(data)
            print(f"   Feature engineering generated {X.shape[1]} features")
        else:
            base_features = ['sza', 'vza', 'raa', 'aod550', 'h2o', 'o3',
                             'wavelength', 'rho_toa']
            X = data[base_features].copy()
            # 若数据中包含 rho_toa_sa，则优先使用
            if 'rho_toa_sa' in data.columns:
                X['rho_toa'] = data['rho_toa_sa']
            elif 'rho_toa' in data.columns:
                X['rho_toa'] = data['rho_toa']
            else:
                X['rho_toa'] = 0.2
                print("   Warning: No TOA reflectance column found, using default 0.2")

            # 添加反演反射率（可选，新框架可能不需要）
            if 'rho_retrieved' in data.columns:
                X['rho_retrieved'] = data['rho_retrieved']
            else:
                X['rho_retrieved'] = 0.0

        # 目标变量
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        y = data[target_column]

        if artifacts and 'scaler' in artifacts:
            scaler = artifacts['scaler']
            # 确保特征顺序与训练时一致
            expected_features = artifacts.get('feature_names', X.columns.tolist())
            X = X[expected_features]  # 重新排序
            X_scaled = scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=expected_features, index=X.index)
            print(f"   Applied training scaler with {len(expected_features)} features")
        else:
            print(f"   Warning: Training scaler not found, creating new one")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

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
                y_pred = model.predict(X_test)

                metrics = self._compute_metrics(y_test, y_pred, model_name)
                self.results[model_name] = metrics

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

        results_df = pd.DataFrame(results)

        if not results_df.empty:
            results_df = results_df.sort_values('RMSE')
            results_path = self.tables_dir / "evaluation_results.csv"
            results_df.to_csv(results_path, index=False)

            print(f"\nModel Performance Ranking:")
            print("-" * 70)
            print(f"{'Rank':<4} {'Model':<15} {'RMSE':<10} {'R²':<8} {'MAE':<10}")
            print("-" * 70)
            for i, (_, row) in enumerate(results_df.iterrows()):
                print(f"{i+1:<4} {row['Model']:<15} {row['RMSE']:.6f}  {row['R2']:.4f}  {row['MAE']:.6f}")

        return results_df

    def _compute_metrics(self, y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> Dict:
        """计算评估指标"""
        metrics = {}
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['R2'] = r2_score(y_true, y_pred)
        metrics['Explained_Variance'] = explained_variance_score(y_true, y_pred)

        nonzero_mask = y_true != 0
        if nonzero_mask.any():
            relative_errors = np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])
            metrics['MAPE'] = np.mean(relative_errors) * 100
            metrics['Median_APE'] = np.median(relative_errors) * 100
        else:
            metrics['MAPE'] = np.nan
            metrics['Median_APE'] = np.nan

        residuals = y_true - y_pred
        metrics['Bias'] = np.mean(residuals)
        metrics['Std_Residuals'] = np.std(residuals)

        metrics['Q10_Error'] = np.percentile(np.abs(residuals), 10)
        metrics['Q90_Error'] = np.percentile(np.abs(residuals), 90)

        metrics['y_true'] = y_true.values.tolist()
        metrics['y_pred'] = y_pred.tolist()

        return metrics

    def generate_all_visualizations(self, X_test: pd.DataFrame, y_test: pd.Series,
                                    results_df: pd.DataFrame):
        """生成所有可视化图表"""
        print("=" * 70)
        print("Generating visualizations...")

        self._plot_model_comparison(results_df)
        self._plot_scatter_matrix(X_test, y_test)
        self._plot_residual_analysis(X_test, y_test)
        self._plot_error_distribution()
        self._plot_hexbin_scatter(X_test, y_test)
        if self.use_shap and self.models:
            self._perform_shap_analysis(X_test)
        self._plot_interactive_3d()

        print(f"All plots saved to: {self.plots_dir}")

    # ==================== 绘图方法 ====================

    def _set_custom_ticks(self, ax, axis_limit):
        """
        根据轴范围设置非均匀刻度：
        - 若 axis_limit > 2.0：内部密集（-2~2 步长0.2），外部稀疏（步长10）
        - 否则：采用均匀刻度，根据范围自动选择步长
        """
        if axis_limit > 2.0:
            # 内部密集区域（-2 ~ 2）
            inner_ticks = np.arange(-2.0, 2.01, 0.2)   # 包含 2.0
            # 左侧稀疏区域（小于 -2 的部分），步长 10
            left_ticks = np.arange(-axis_limit, -2.0, 10) if -axis_limit < -2.0 else np.array([])
            # 右侧稀疏区域（大于 2 的部分），步长 10（从 2+10 开始避免重复）
            right_ticks = np.arange(2.0+10, axis_limit+0.1, 10) if axis_limit > 2.0 else np.array([])

            # 合并所有刻度，并限制在轴范围内
            all_ticks = np.unique(np.concatenate([left_ticks, inner_ticks, right_ticks]))
            all_ticks = all_ticks[(all_ticks >= -axis_limit) & (all_ticks <= axis_limit)]

            # 设置主刻度位置
            ax.set_xticks(all_ticks)
            ax.set_yticks(all_ticks)

            # 自定义格式化：内部显示一位小数，外部显示整数
            def tick_formatter(x, pos):
                if abs(x) < 2.1:   # 容差，覆盖所有内部刻度
                    return f'{x:.1f}'
                else:
                    return f'{x:.0f}'
            ax.xaxis.set_major_formatter(plt.FuncFormatter(tick_formatter))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(tick_formatter))

            # 旋转刻度标签避免重叠
            ax.tick_params(axis='x', rotation=45)
        else:
            # 范围较小时，使用均匀刻度
            if axis_limit <= 0.2:
                tick_step = 0.05
            elif axis_limit <= 0.5:
                tick_step = 0.1
            else:
                tick_step = 0.2
            major_ticks = np.arange(-axis_limit, axis_limit + tick_step/2, tick_step)
            ax.set_xticks(major_ticks)
            ax.set_yticks(major_ticks)

    def _plot_model_comparison(self, results_df: pd.DataFrame):
        """绘制模型比较图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. RMSE和MAE比较
        ax = axes[0, 0]
        x = np.arange(len(results_df))
        width = 0.35
        rmse_color = '#1f77b4'
        mae_color = '#ff7f0e'

        for i, (_, row) in enumerate(results_df.iterrows()):
            ax.bar(i - width/2, row['RMSE'], width, color=rmse_color, alpha=0.7, edgecolor='black')
            ax.bar(i + width/2, row['MAE'], width, color=mae_color, alpha=0.7, edgecolor='black')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=rmse_color, alpha=0.7, edgecolor='black', label='RMSE'),
            Patch(facecolor=mae_color, alpha=0.7, edgecolor='black', label='MAE')
        ]
        ax.legend(handles=legend_elements, fontsize=8, frameon=False)

        ax.set_xlabel('Model', fontsize=9)
        ax.set_ylabel('Error', fontsize=9)
        ax.set_title('Model Error Comparison', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.2, linestyle='--')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        # 2. R²比较
        ax = axes[0, 1]
        colors = [MODEL_COLORS.get(m, '#1f77b4') for m in results_df['Model']]
        bars = ax.bar(results_df['Model'], results_df['R2'], color=colors, alpha=0.8, edgecolor='black')
        ax.set_xlabel('Model', fontsize=9)
        ax.set_ylabel('R²', fontsize=9)
        ax.set_title('Model Coefficient of Determination (R²)', fontsize=10, fontweight='bold')
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.2, linestyle='--')
        for bar, value in zip(bars, results_df['R2']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        # 3. 预测偏差
        ax = axes[0, 2]
        ax.bar(results_df['Model'], results_df['Bias'], alpha=0.8, color=colors, edgecolor='black')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, linewidth=1.2)
        ax.set_xlabel('Model', fontsize=9)
        ax.set_ylabel('Prediction Bias', fontsize=9)
        ax.set_title('Model Prediction Bias', fontsize=10, fontweight='bold')
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.2, linestyle='--')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        # 4. MAPE比较
        ax = axes[1, 0]
        ax.bar(results_df['Model'], results_df['MAPE'], alpha=0.8, color=colors, edgecolor='black')
        ax.set_xlabel('Model', fontsize=9)
        ax.set_ylabel('MAPE (%)', fontsize=9)
        ax.set_title('Mean Absolute Percentage Error (MAPE)', fontsize=10, fontweight='bold')
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.2, linestyle='--')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        # 5. 解释方差
        ax = axes[1, 1]
        ax.bar(results_df['Model'], results_df['Explained_Variance'], alpha=0.8, color=colors, edgecolor='black')
        ax.set_xlabel('Model', fontsize=9)
        ax.set_ylabel('Explained Variance', fontsize=9)
        ax.set_title('Explained Variance', fontsize=10, fontweight='bold')
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.2, linestyle='--')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        # 6. 标准差
        ax = axes[1, 2]
        ax.bar(results_df['Model'], results_df['Std_Residuals'], alpha=0.8, color=colors, edgecolor='black')
        ax.set_xlabel('Model', fontsize=9)
        ax.set_ylabel('Std of Residuals', fontsize=9)
        ax.set_title('Standard Deviation of Residuals', fontsize=10, fontweight='bold')
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.2, linestyle='--')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

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

            # 计算对称的轴范围
            data_min = min(y_test.min(), y_pred.min())
            data_max = max(y_test.max(), y_pred.max())
            data_range = max(abs(data_min), abs(data_max))
            axis_limit = max(0.10, np.ceil(data_range * 10) / 10)  # 确保至少0.1
            ax.set_xlim(-axis_limit, axis_limit)
            ax.set_ylim(-axis_limit, axis_limit)
            ax.set_aspect('equal')

            abs_errors = np.abs(y_test - y_pred)
            scatter = ax.scatter(y_test, y_pred, alpha=0.6, s=10,
                                 c=abs_errors, cmap=SCATTER_CMAP, edgecolors='none')

            # 1:1线
            ax.plot([-axis_limit, axis_limit], [-axis_limit, axis_limit], 'r--', lw=1.5, alpha=0.8)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            ax.text(0.05, 0.95, f'RMSE = {rmse:.4f}\nR² = {r2:.3f}\nMAE = {mae:.4f}',
                    transform=ax.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                              edgecolor='gray', linewidth=0.5))

            ax.set_xlabel('True Value', fontsize=9)
            ax.set_ylabel('Predicted Value', fontsize=9)
            ax.set_title(f'{model_name} - Prediction vs True', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.2, linestyle='--')
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.tick_params(axis='both', which='both', length=4, width=0.75, direction='out', labelsize=8)

            # 应用自定义刻度
            self._set_custom_ticks(ax, axis_limit)

            if idx == 0:
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
                cbar.set_label('Absolute Error', fontsize=8)
                cbar.ax.tick_params(labelsize=7)

        for idx in range(len(self.models), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "prediction_scatter_plots.png", dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

    def _plot_hexbin_scatter(self, X_test: pd.DataFrame, y_test: pd.Series):
        """绘制专业的Hexbin散点图 - 科研红蓝配色版"""
        print("   Generating professional hexbin scatter plots...")

        model_performance = []
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            model_performance.append((model_name, model, r2, y_pred))

        model_performance.sort(key=lambda x: x[2], reverse=True)

        n_models = len(model_performance)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        try:
            import seaborn as sns
            red_blue_cmap = sns.diverging_palette(240, 10, as_cmap=True)
            print("   Using seaborn red-blue diverging color palette")
        except ImportError:
            red_blue_cmap = plt.cm.RdBu_r
            print("   Using matplotlib RdBu_r color palette")

        for idx, (model_name, model, r2_value, y_pred) in enumerate(model_performance[:len(axes)]):
            ax = axes[idx]

            residuals = y_pred - y_test
            data_min = min(y_test.min(), y_pred.min())
            data_max = max(y_test.max(), y_pred.max())
            data_range = max(abs(data_min), abs(data_max))
            axis_limit = max(0.10, np.ceil(data_range * 10) / 10)
            ax.set_xlim(-axis_limit, axis_limit)
            ax.set_ylim(-axis_limit, axis_limit)
            ax.set_aspect('equal')

            n_samples = len(y_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            hexbin = ax.hexbin(y_test, y_pred, gridsize=60, cmap=red_blue_cmap,
                               mincnt=1, edgecolors='none', alpha=0.9, zorder=4)

            counts = hexbin.get_array()

            x_range = np.array([-axis_limit, axis_limit])
            ax.plot(x_range, x_range, 'k-', linewidth=1.5, label='1:1 Line', zorder=5)

            if len(y_test) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred)
                fit_line = slope * x_range + intercept
                ax.plot(x_range, fit_line, 'k--', linewidth=1.5,
                        label=f'Fit (slope={slope:.3f})', zorder=6)

            ax.set_xlabel('True Value', fontsize=11, fontweight='bold')
            ax.set_ylabel('Predicted Value', fontsize=11, fontweight='bold')
            ax.set_title(f'{model_name} (R²={r2_value:.3f})\nn={n_samples:,}, RMSE={rmse:.4f}, MAE={mae:.4f}',
                         fontsize=12, fontweight='bold', pad=12)
            ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5, zorder=0)
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color('black')
            ax.tick_params(axis='both', which='major', length=6, width=0.8, direction='out', labelsize=9)
            ax.tick_params(axis='both', which='minor', length=3, width=0.5, direction='out', labelsize=7)

            # 应用自定义刻度
            self._set_custom_ticks(ax, axis_limit)

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles=handles, labels=labels, fontsize=9, frameon=True,
                          framealpha=0.9, edgecolor='black', loc='upper left',
                          bbox_to_anchor=(0.02, 0.98), borderaxespad=0.5, ncol=1)

            if counts is not None and len(counts) > 0:
                log_counts = np.log10(counts)
                min_log = np.min(log_counts)
                max_log = np.max(log_counts)

                from matplotlib.cm import ScalarMappable
                from matplotlib.colors import Normalize
                norm = Normalize(vmin=min_log, vmax=max_log)
                sm = ScalarMappable(cmap=red_blue_cmap, norm=norm)
                sm.set_array([])

                cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.03)
                cbar.set_label('log₁₀(Count)', fontsize=10, fontweight='bold')
                cbar.ax.tick_params(labelsize=8)

                log_range = max_log - min_log
                if log_range <= 1.0:
                    tick_step = 0.2
                elif log_range <= 2.0:
                    tick_step = 0.5
                else:
                    tick_step = 1.0
                start_tick = np.floor(min_log / tick_step) * tick_step
                end_tick = np.ceil(max_log / tick_step) * tick_step
                ticks = np.arange(start_tick, end_tick + tick_step/2, tick_step)
                ticks = ticks[(ticks >= min_log - 0.1) & (ticks <= max_log + 0.1)]

                if len(ticks) >= 2:
                    cbar.set_ticks(ticks)
                    tick_labels = []
                    for tick in ticks:
                        if tick.is_integer():
                            tick_labels.append(f'{int(tick)}')
                        elif abs(tick * 10 - round(tick * 10)) < 0.01:
                            tick_labels.append(f'{tick:.1f}')
                        else:
                            tick_labels.append(f'{tick:.2f}')
                    cbar.set_ticklabels(tick_labels)
                else:
                    n_ticks = min(5, int(log_range * 2) + 1)
                    if n_ticks >= 2:
                        ticks = np.linspace(min_log, max_log, n_ticks)
                        cbar.set_ticks(ticks)
                        tick_labels = [f'{tick:.1f}' for tick in ticks]
                        cbar.set_ticklabels(tick_labels)

        for idx in range(len(model_performance), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "hexbin_density_plots_red_blue.png",
                    dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print("   Professional hexbin plots saved successfully")

    def _plot_residual_analysis(self, X_test: pd.DataFrame, y_test: pd.Series):
        """绘制残差分析图"""
        if not self.results:
            return

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

        # 2. 残差vs预测值
        ax = axes[0, 1]
        scatter = ax.scatter(y_pred, residuals, alpha=0.6, s=10,
                             c=np.abs(residuals), cmap=SCATTER_CMAP, edgecolors='none')
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Predicted Value', fontsize=9)
        ax.set_ylabel('Residual', fontsize=9)
        ax.set_title('Residuals vs Predicted Values', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--')
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('|Residual|', fontsize=8)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        # 3. QQ图
        ax = axes[0, 2]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot of Residuals', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_xlabel('Theoretical Quantiles', fontsize=9)
        ax.set_ylabel('Sample Quantiles', fontsize=9)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

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

        # 6. 残差自相关
        ax = axes[1, 2]
        pd.plotting.autocorrelation_plot(residuals, ax=ax)
        ax.set_title('Residual Autocorrelation', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

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

            y_true = np.array(metrics['y_true'])
            y_pred = np.array(metrics['y_pred'])
            errors = y_true - y_pred

            ax.hist(errors, bins=50, alpha=0.7, density=True, edgecolor='black',
                    color=MODEL_COLORS.get(model_name, '#1f77b4'))

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
            ax.tick_params(axis='both', which='both', length=4, width=0.75, direction='out', labelsize=8)

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

        best_model_name = list(self.results.keys())[0]
        model = self.models[best_model_name]

        if len(X_test) > 1000:
            X_sample = X_test.sample(n=min(500, len(X_test)), random_state=self.config.RANDOM_SEED)
        else:
            X_sample = X_test

        try:
            if best_model_name in ['RandomForest', 'GradientBoosting', 'ExtraTrees', 'XGBoost', 'LightGBM']:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            else:
                explainer = shap.KernelExplainer(model.predict, X_sample[:100])
                shap_values = explainer.shap_values(X_sample)

            # 摘要图（Top 20）
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
            plt.title(f'{best_model_name} - SHAP Feature Importance (Top 20)', fontsize=11, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"{best_model_name}_shap_summary_top20.png",
                        dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()

            # 摘要图（所有特征）
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_sample, show=False, max_display=min(50, X_sample.shape[1]))
            plt.title(f'{best_model_name} - SHAP Feature Importance (All Features)', fontsize=11, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"{best_model_name}_shap_summary_all.png",
                        dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()

            # 条形图（Top 20）
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
            plt.title(f'{best_model_name} - SHAP Feature Importance (Bar - Top 20)', fontsize=11, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"{best_model_name}_shap_bar_top20.png",
                        dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()

            # 条形图（所有特征）
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False,
                              max_display=min(50, X_sample.shape[1]))
            plt.title(f'{best_model_name} - SHAP Feature Importance (Bar - All Features)', fontsize=11, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"{best_model_name}_shap_bar_all.png",
                        dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()

            print("   SHAP analysis completed with all plots")

        except Exception as e:
            print(f"   SHAP analysis failed: {e}")

    def _plot_interactive_3d(self):
        """绘制交互式3D图"""
        try:
            if self.results:
                best_model_name = list(self.results.keys())[0]
                metrics = self.results[best_model_name]

                fig = go.Figure()
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

        report_path = self.output_dir / "evaluation_summary.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

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
            report_lines.append(f"{i+1:<4} {row['Model']:<15} {row['RMSE']:<12.6f} "
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

    parser = argparse.ArgumentParser(description='Machine Learning Model Evaluation System (ΔTOA)')
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
    print("Machine Learning Model Evaluation System (Target: ΔTOA)")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model Directory: {args.saved_dir}")
    print(f"Data Sampling: {args.sample * 100:.1f}%")
    print(f"Feature Engineering: {'Disabled' if args.no_feature_engineering else 'Enabled'}")
    print(f"SHAP Analysis: {'Disabled' if args.no_shap else 'Enabled'}")
    print("=" * 80)

    try:
        config = ExperimentConfig
        if args.data_dir:
            config.DATA_DIR = Path(args.data_dir)

        evaluator = ModelEvaluator(
            saved_dir=Path(args.saved_dir),
            config=config,
            use_shap=not args.no_shap
        )

        dl = DataLoader(config=config)
        test_data = dl.load_data(sample_fraction=args.sample,
                                 filter_extreme_delta=True,
                                 delta_threshold=0.5)

        X_test, y_test = evaluator.prepare_test_data(
            test_data,
            use_feature_engineering=not args.no_feature_engineering,
            target_column='delta_toa'
        )

        results_df = evaluator.evaluate_models(X_test, y_test)
        evaluator.generate_all_visualizations(X_test, y_test, results_df)
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
                print(f"{i+1:<4} {row['Model']:<15} {row['RMSE']:<12.6f} "
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