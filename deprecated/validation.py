# ==================== validation.py ====================
"""
验证模块
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional
from pathlib import Path
from sklearn.model_selection import KFold
from scipy import stats

from config import ExperimentConfig
from utils import setup_logger


class ModelValidator:
    """模型验证器"""

    def __init__(self, config: ExperimentConfig, logger=None):
        self.config = config
        self.logger = logger or setup_logger('ModelValidator')

    def cross_validate(self, data: pd.DataFrame, band_id: str = 'band3',
                       n_splits: int = 5, model_type: str = 'lut') -> Dict[str, float]:
        """交叉验证"""
        from deprecated.correction_model import ModelFactory

        band_data = data[data['band'] == band_id].copy()

        if len(band_data) < n_splits * 10:
            self.logger.warning(f"数据量不足，减少折数")
            n_splits = min(n_splits, len(band_data) // 10)

        features = ['sza', 'vza', 'aod550', 'rho_true']
        band_data = band_data.dropna(subset=features + ['error_absolute'])

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.config.RANDOM_SEED)

        cv_scores = {
            'mae': [],
            'rmse': [],
            'r2': [],
            'bias': []
        }

        fold = 1
        for train_idx, test_idx in kf.split(band_data):
            self.logger.info(f"交叉验证 - 第 {fold}/{n_splits} 折")

            train_data = band_data.iloc[train_idx]
            test_data = band_data.iloc[test_idx]

            try:
                if model_type == 'lut':
                    model = ModelFactory.create_model(model_type)
                    model.create_lut_from_data(train_data)
                else:
                    X_train = train_data[features].values
                    y_train = train_data['error_absolute'].values
                    model = ModelFactory.create_model(model_type)
                    model.fit(X_train, y_train)

                if model_type == 'lut':
                    X_test = []
                    for _, row in test_data.iterrows():
                        point = [row[feat] for feat in features]
                        X_test.append(point)
                    X_test = np.array(X_test)
                else:
                    X_test = test_data[features].values

                y_test = test_data['error_absolute'].values
                y_pred = model.predict(X_test)

                mae = np.mean(np.abs(y_test - y_pred))
                rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
                ss_res = np.sum((y_test - y_pred) ** 2)
                ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                bias = np.mean(y_test - y_pred)

                cv_scores['mae'].append(mae)
                cv_scores['rmse'].append(rmse)
                cv_scores['r2'].append(r2)
                cv_scores['bias'].append(bias)

                fold += 1

            except Exception as e:
                self.logger.error(f"第 {fold} 折验证失败: {e}")
                continue

        cv_results = {}
        for metric, values in cv_scores.items():
            if values:
                cv_results[f'cv_{metric}_mean'] = np.mean(values)
                cv_results[f'cv_{metric}_std'] = np.std(values)

        self.logger.info(f"交叉验证完成 - 平均RMSE: {cv_results.get('cv_rmse_mean', np.nan):.6f}")

        return cv_results

    def plot_validation_results(self, data: pd.DataFrame, band_id: str = 'band3',
                                model_path: Optional[Path] = None, save_path: Optional[Path] = None):
        """绘制验证结果"""

        band_data = data[data['band'] == band_id].copy()
        features = ['sza', 'vza', 'aod550', 'rho_true']
        band_data = band_data.dropna(subset=features + ['error_absolute'])

        if len(band_data) < 20:
            self.logger.warning("验证数据不足")
            return

        if model_path is None:
            model_files = list(self.config.MODELS_DIR.glob(f"*{band_id}.pkl"))
            if not model_files:
                raise FileNotFoundError(f"找不到 {band_id} 的模型")
            model_path = model_files[0]

        if 'lut' in model_path.stem:
            from deprecated.correction_model import LUTCorrectionModel
            model = LUTCorrectionModel()
            model.load(model_path)
            model_type = 'lut'
        else:
            import joblib
            model_data = joblib.load(model_path)
            if 'formula_type' in model_data:
                from deprecated.correction_model import AnalyticalCorrectionModel
                model = AnalyticalCorrectionModel()
                model.load(model_path)
                model_type = 'analytical'
            else:
                model = model_data
                model_type = 'ml'

        if model_type == 'lut':
            X_test = []
            for _, row in band_data.iterrows():
                point = [row[feat] for feat in features]
                X_test.append(point)
            X_test = np.array(X_test)
        else:
            X_test = band_data[features].values

        y_true = band_data['error_absolute'].values
        y_pred = model.predict(X_test)

        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        bias = np.mean(y_true - y_pred)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 预测 vs 真实值
        ax1 = axes[0, 0]
        ax1.scatter(y_true, y_pred, alpha=0.5, s=10)
        lims = [np.min([ax1.get_xlim(), ax1.get_ylim()]),
                np.max([ax1.get_xlim(), ax1.get_ylim()])]
        ax1.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax1.set_aspect('equal')
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.set_xlabel('True error')
        ax1.set_ylabel('Predicted error')
        ax1.set_title('Predicted vs True')
        ax1.grid(True, alpha=0.3)

        stats_text = f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nBias: {bias:.4f}\nN: {len(y_true)}"
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 残差图
        ax2 = axes[0, 1]
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, s=10)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Predicted error')
        ax2.set_ylabel('Residual')
        ax2.set_title('Residual plot')
        ax2.grid(True, alpha=0.3)

        # 残差直方图
        ax3 = axes[1, 0]
        ax3.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Residual')
        ax3.set_ylabel('Probability density')
        ax3.set_title('Residual distribution')
        ax3.grid(True, alpha=0.3)

        mu, std = stats.norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        p = stats.norm.pdf(x, mu, std)
        ax3.plot(x, p, 'r-', linewidth=2, label=f'N({mu:.4f}, {std:.4f}²)')
        ax3.legend()

        # 按SZA分组的预测性能
        ax4 = axes[1, 1]
        band_data_copy = band_data.copy()
        band_data_copy['pred_error'] = y_pred
        band_data_copy['residual'] = residuals

        sza_bins = np.arange(0, 91, 15)
        band_data_copy['sza_bin'] = pd.cut(band_data_copy['sza'], bins=sza_bins)

        perf_by_sza = band_data_copy.groupby('sza_bin').agg({
            'residual': ['mean', 'std'],
            'error_absolute': 'count'
        })

        x_pos = range(len(perf_by_sza))
        ax4.errorbar(x_pos, perf_by_sza[('residual', 'mean')],
                     yerr=perf_by_sza[('residual', 'std')], fmt='o-', capsize=5)
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax4.set_xlabel('SZA interval (deg)')
        ax4.set_ylabel('Mean residual')
        ax4.set_title('Prediction performance by SZA')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([str(bin_) for bin_ in perf_by_sza.index], rotation=45)
        ax4.grid(True, alpha=0.3)

        ax4_twin = ax4.twinx()
        ax4_twin.bar(x_pos, perf_by_sza[('error_absolute', 'count')],
                     alpha=0.3, color='gray', label='Samples')
        ax4_twin.set_ylabel('Sample count')
        ax4_twin.legend(loc='upper right')

        plt.suptitle(f'Model Validation Results - {band_id} ({model_type})', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"验证结果图已保存: {save_path}")

        plt.show()

        return {
            'mae': mae,
            'rmse': rmse,
            'bias': bias,
            'residual_mean': float(np.mean(residuals)),
            'residual_std': float(np.std(residuals))
        }

    def apply_correction(self, data: pd.DataFrame, band_id: str = 'band3',
                         model_path: Optional[Path] = None) -> pd.DataFrame:
        """应用校正"""

        band_data = data[data['band'] == band_id].copy()

        if len(band_data) == 0:
            self.logger.warning(f"没有 {band_id} 的数据")
            return data

        if model_path is None:
            model_files = list(self.config.MODELS_DIR.glob(f"*{band_id}.pkl"))
            if not model_files:
                raise FileNotFoundError(f"找不到 {band_id} 的模型")
            model_path = model_files[0]

        if 'lut' in model_path.stem:
            from deprecated.correction_model import LUTCorrectionModel
            model = LUTCorrectionModel()
            model.load(model_path)
            model_type = 'lut'
        else:
            import joblib
            model_data = joblib.load(model_path)
            if 'formula_type' in model_data:
                from deprecated.correction_model import AnalyticalCorrectionModel
                model = AnalyticalCorrectionModel()
                model.load(model_path)
                model_type = 'analytical'
            else:
                model = model_data
                model_type = 'ml'

        features = ['sza', 'vza', 'aod550', 'rho_true']

        missing_features = [f for f in features if f not in band_data.columns]
        if missing_features:
            self.logger.warning(f"缺少特征: {missing_features}，使用默认值")
            for feat in missing_features:
                if feat == 'rho_true':
                    band_data[feat] = 0.2
                else:
                    band_data[feat] = 0.0

        if model_type == 'lut':
            X_correct = []
            for _, row in band_data.iterrows():
                point = [row[feat] for feat in features]
                X_correct.append(point)
            X_correct = np.array(X_correct)
        else:
            X_correct = band_data[features].values

        correction = model.predict(X_correct)

        band_data['correction_value'] = correction
        band_data['rho_retrieved_corrected'] = band_data['rho_retrieved'] - correction
        band_data['error_corrected'] = band_data['rho_retrieved_corrected'] - band_data['rho_true']

        original_mae = np.mean(np.abs(band_data['error_absolute']))
        corrected_mae = np.mean(np.abs(band_data['error_corrected']))
        improvement = (original_mae - corrected_mae) / original_mae * 100 if original_mae > 0 else 0

        self.logger.info(f"校正效果 - 原始MAE: {original_mae:.6f}, "
                         f"校正后MAE: {corrected_mae:.6f}, "
                         f"改进: {improvement:.1f}%")

        data_corrected = data.copy()
        for idx in band_data.index:
            data_corrected.loc[idx, 'correction_value'] = band_data.loc[idx, 'correction_value']
            data_corrected.loc[idx, 'rho_retrieved_corrected'] = band_data.loc[idx, 'rho_retrieved_corrected']
            data_corrected.loc[idx, 'error_corrected'] = band_data.loc[idx, 'error_corrected']

        return data_corrected