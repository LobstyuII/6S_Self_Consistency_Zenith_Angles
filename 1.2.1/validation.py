# ==================== validation.py ====================
"""
验证模块
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from scipy import stats

from config import ExperimentConfig
from utils import setup_logger, calculate_statistics


class ModelValidator:
    """模型验证器"""

    def __init__(self, config: ExperimentConfig, logger=None):
        """
        初始化模型验证器

        Parameters:
        -----------
        config : ExperimentConfig
            实验配置
        logger : logging.Logger, optional
            日志记录器
        """
        self.config = config
        self.logger = logger or setup_logger('ModelValidator')

    def cross_validate(self, data: pd.DataFrame, band_id: str = 'band3',
                       n_splits: int = 5, model_type: str = 'lut') -> Dict[str, float]:
        """
        交叉验证

        Parameters:
        -----------
        data : pd.DataFrame
            数据
        band_id : str
            波段ID
        n_splits : int
            K折数
        model_type : str
            模型类型

        Returns:
        --------
        dict
            交叉验证结果
        """
        from correction_model import ModelFactory

        # 筛选波段数据
        band_data = data[data['band'] == band_id].copy()

        if len(band_data) < n_splits * 10:
            self.logger.warning(f"数据量不足，减少折数或增加数据")
            n_splits = min(n_splits, len(band_data) // 10)

        # 准备特征
        features = ['sza', 'vza', 'raa', 'aod550', 'rho_true']
        band_data = band_data.dropna(subset=features + ['error_absolute'])

        # K折交叉验证
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

            # 分割数据
            train_data = band_data.iloc[train_idx]
            test_data = band_data.iloc[test_idx]

            try:
                # 训练模型
                if model_type == 'lut':
                    model = ModelFactory.create_model(model_type)
                    model.create_lut_from_data(train_data)
                else:
                    # 准备训练特征
                    X_train = train_data[features].values
                    y_train = train_data['error_absolute'].values

                    model = ModelFactory.create_model(model_type)
                    model.fit(X_train, y_train)

                # 测试
                if model_type == 'lut':
                    # 为LUT准备测试数据
                    X_test = []
                    for _, row in test_data.iterrows():
                        point = [row[feat] for feat in features]
                        X_test.append(point)
                    X_test = np.array(X_test)
                else:
                    X_test = test_data[features].values

                y_test = test_data['error_absolute'].values
                y_pred = model.predict(X_test)

                # 计算指标
                mae = np.mean(np.abs(y_test - y_pred))
                rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

                # R²
                ss_res = np.sum((y_test - y_pred) ** 2)
                ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

                # 偏差
                bias = np.mean(y_test - y_pred)

                cv_scores['mae'].append(mae)
                cv_scores['rmse'].append(rmse)
                cv_scores['r2'].append(r2)
                cv_scores['bias'].append(bias)

                fold += 1

            except Exception as e:
                self.logger.error(f"第 {fold} 折验证失败: {e}")
                continue

        # 计算平均指标
        cv_results = {}
        for metric, values in cv_scores.items():
            if values:
                cv_results[f'cv_{metric}_mean'] = np.mean(values)
                cv_results[f'cv_{metric}_std'] = np.std(values)

        self.logger.info(f"交叉验证完成 - 平均RMSE: {cv_results.get('cv_rmse_mean', np.nan):.6f}")

        return cv_results

    def validate_on_extreme_conditions(self, data: pd.DataFrame, band_id: str = 'band3',
                                       model_path: Optional[Path] = None) -> Dict[str, float]:
        """
        在极端条件下验证模型

        Parameters:
        -----------
        data : pd.DataFrame
            数据
        band_id : str
            波段ID
        model_path : Path, optional
            模型路径

        Returns:
        --------
        dict
            验证结果
        """
        from correction_model import ModelFactory

        # 筛选波段数据
        band_data = data[data['band'] == band_id].copy()

        # 定义极端条件
        band_data['secz_sza'] = calculate_airmass(band_data['sza'])
        band_data['secz_vza'] = calculate_airmass(band_data['vza'])
        band_data['airmass_product'] = band_data['secz_sza'] * band_data['secz_vza']

        extreme_mask = (
                (band_data['sza'] > 60) |
                (band_data['vza'] > 60) |
                (band_data['airmass_product'] > 4)
        )

        extreme_data = band_data[extreme_mask].copy()
        normal_data = band_data[~extreme_mask].copy()

        if len(extreme_data) < 10:
            self.logger.warning("极端条件数据不足")
            return {}

        # 加载模型
        if model_path is None:
            # 查找默认模型
            model_files = list(self.config.MODELS_DIR.glob(f"*{band_id}.pkl"))
            if not model_files:
                raise FileNotFoundError(f"找不到 {band_id} 的模型")
            model_path = model_files[0]

        # 根据文件扩展名确定模型类型
        if 'lut' in model_path.stem:
            from correction_model import LUTCorrectionModel
            model = LUTCorrectionModel()
            model.load(model_path)
            model_type = 'lut'
        else:
            import joblib
            model_data = joblib.load(model_path)
            if 'formula_type' in model_data:
                from correction_model import AnalyticalCorrectionModel
                model = AnalyticalCorrectionModel()
                model.load(model_path)
                model_type = 'analytical'
            else:
                # 假设是sklearn模型
                model = model_data
                model_type = 'ml'

        # 准备测试数据
        features = ['sza', 'vza', 'raa', 'aod550', 'rho_true']

        # 极端条件验证
        extreme_data_clean = extreme_data.dropna(subset=features + ['error_absolute'])

        if len(extreme_data_clean) == 0:
            self.logger.warning("极端条件没有有效数据")
            return {}

        if model_type == 'lut':
            X_extreme = []
            for _, row in extreme_data_clean.iterrows():
                point = [row[feat] for feat in features]
                X_extreme.append(point)
            X_extreme = np.array(X_extreme)
        else:
            X_extreme = extreme_data_clean[features].values

        y_extreme_true = extreme_data_clean['error_absolute'].values
        y_extreme_pred = model.predict(X_extreme)

        # 正常条件验证
        normal_data_clean = normal_data.dropna(subset=features + ['error_absolute'])

        if len(normal_data_clean) > 0:
            if model_type == 'lut':
                X_normal = []
                for _, row in normal_data_clean.iterrows():
                    point = [row[feat] for feat in features]
                    X_normal.append(point)
                X_normal = np.array(X_normal)
            else:
                X_normal = normal_data_clean[features].values

            y_normal_true = normal_data_clean['error_absolute'].values
            y_normal_pred = model.predict(X_normal)
        else:
            y_normal_true = y_normal_pred = np.array([])

        # 计算指标
        results = {}

        # 极端条件指标
        if len(y_extreme_true) > 0:
            results['extreme_mae'] = float(np.mean(np.abs(y_extreme_true - y_extreme_pred)))
            results['extreme_rmse'] = float(np.sqrt(np.mean((y_extreme_true - y_extreme_pred) ** 2)))
            results['extreme_bias'] = float(np.mean(y_extreme_true - y_extreme_pred))
            results['extreme_n'] = int(len(y_extreme_true))

        # 正常条件指标
        if len(y_normal_true) > 0:
            results['normal_mae'] = float(np.mean(np.abs(y_normal_true - y_normal_pred)))
            results['normal_rmse'] = float(np.sqrt(np.mean((y_normal_true - y_normal_pred) ** 2)))
            results['normal_bias'] = float(np.mean(y_normal_true - y_normal_pred))
            results['normal_n'] = int(len(y_normal_true))

        # 对比
        if 'extreme_rmse' in results and 'normal_rmse' in results:
            results['rmse_ratio'] = results['extreme_rmse'] / results['normal_rmse']

        self.logger.info(f"极端条件验证 - RMSE: {results.get('extreme_rmse', np.nan):.6f}")

        return results

    def plot_validation_results(self, data: pd.DataFrame, band_id: str = 'band3',
                                model_path: Optional[Path] = None, save_path: Optional[Path] = None):
        """
        绘制验证结果

        Parameters:
        -----------
        data : pd.DataFrame
            数据
        band_id : str
            波段ID
        model_path : Path, optional
            模型路径
        save_path : Path, optional
            保存路径
        """
        from correction_model import ModelFactory

        # 筛选数据
        band_data = data[data['band'] == band_id].copy()
        features = ['sza', 'vza', 'raa', 'aod550', 'rho_true']
        band_data = band_data.dropna(subset=features + ['error_absolute'])

        if len(band_data) < 20:
            self.logger.warning("验证数据不足")
            return

        # 加载模型
        if model_path is None:
            model_files = list(self.config.MODELS_DIR.glob(f"*{band_id}.pkl"))
            if not model_files:
                raise FileNotFoundError(f"找不到 {band_id} 的模型")
            model_path = model_files[0]

        # 根据模型类型加载
        if 'lut' in model_path.stem:
            from correction_model import LUTCorrectionModel
            model = LUTCorrectionModel()
            model.load(model_path)
            model_type = 'lut'
        else:
            import joblib
            model_data = joblib.load(model_path)
            if 'formula_type' in model_data:
                from correction_model import AnalyticalCorrectionModel
                model = AnalyticalCorrectionModel()
                model.load(model_path)
                model_type = 'analytical'
            else:
                model = model_data
                model_type = 'ml'

        # 准备测试数据
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

        # 计算指标
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        bias = np.mean(y_true - y_pred)

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 预测 vs 真实值散点图
        ax1 = axes[0, 0]
        ax1.scatter(y_true, y_pred, alpha=0.5, s=10)

        # 添加1:1线
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

        # 添加统计信息
        stats_text = f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nBias: {bias:.4f}\nN: {len(y_true)}"
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 2. 残差图
        ax2 = axes[0, 1]
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, s=10)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Predicted error')
        ax2.set_ylabel('Residual')
        ax2.set_title('Residual plot')
        ax2.grid(True, alpha=0.3)

        # 3. 残差直方图
        ax3 = axes[1, 0]
        ax3.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Residual')
        ax3.set_ylabel('Probability density')
        ax3.set_title('Residual distribution')
        ax3.grid(True, alpha=0.3)

        # 添加正态分布拟合
        from scipy.stats import norm
        mu, std = norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        p = norm.pdf(x, mu, std)
        ax3.plot(x, p, 'r-', linewidth=2, label=f'N({mu:.4f}, {std:.4f}²)')
        ax3.legend()

        # 4. 按SZA分组的预测性能
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

        # 第二y轴显示样本数
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
        """
        应用校正

        Parameters:
        -----------
        data : pd.DataFrame
            需要校正的数据
        band_id : str
            波段ID
        model_path : Path, optional
            模型路径

        Returns:
        --------
        pd.DataFrame
            校正后的数据
        """
        from correction_model import ModelFactory

        # 筛选数据
        band_data = data[data['band'] == band_id].copy()

        if len(band_data) == 0:
            self.logger.warning(f"没有 {band_id} 的数据")
            return data

        # 加载模型
        if model_path is None:
            model_files = list(self.config.MODELS_DIR.glob(f"*{band_id}.pkl"))
            if not model_files:
                raise FileNotFoundError(f"找不到 {band_id} 的模型")
            model_path = model_files[0]

        # 根据模型类型加载
        if 'lut' in model_path.stem:
            from correction_model import LUTCorrectionModel
            model = LUTCorrectionModel()
            model.load(model_path)
            model_type = 'lut'
        else:
            import joblib
            model_data = joblib.load(model_path)
            if 'formula_type' in model_data:
                from correction_model import AnalyticalCorrectionModel
                model = AnalyticalCorrectionModel()
                model.load(model_path)
                model_type = 'analytical'
            else:
                model = model_data
                model_type = 'ml'

        # 准备特征
        features = ['sza', 'vza', 'raa', 'aod550', 'rho_true']

        # 确保所有特征都存在
        missing_features = [f for f in features if f not in band_data.columns]
        if missing_features:
            self.logger.warning(f"缺少特征: {missing_features}，使用默认值")
            for feat in missing_features:
                if feat == 'rho_true':
                    band_data[feat] = 0.2  # 默认地表反射率
                else:
                    band_data[feat] = 0.0

        # 预测校正值
        if model_type == 'lut':
            X_correct = []
            for _, row in band_data.iterrows():
                point = [row[feat] for feat in features]
                X_correct.append(point)
            X_correct = np.array(X_correct)
        else:
            X_correct = band_data[features].values

        correction = model.predict(X_correct)

        # 应用校正
        band_data['correction_value'] = correction
        band_data['rho_retrieved_corrected'] = band_data['rho_retrieved'] - correction
        band_data['error_corrected'] = band_data['rho_retrieved_corrected'] - band_data['rho_true']

        # 计算校正效果
        original_mae = np.mean(np.abs(band_data['error_absolute']))
        corrected_mae = np.mean(np.abs(band_data['error_corrected']))
        improvement = (original_mae - corrected_mae) / original_mae * 100 if original_mae > 0 else 0

        self.logger.info(f"校正效果 - 原始MAE: {original_mae:.6f}, "
                         f"校正后MAE: {corrected_mae:.6f}, "
                         f"改进: {improvement:.1f}%")

        # 更新原始DataFrame
        data_corrected = data.copy()
        for idx in band_data.index:
            data_corrected.loc[idx, 'correction_value'] = band_data.loc[idx, 'correction_value']
            data_corrected.loc[idx, 'rho_retrieved_corrected'] = band_data.loc[idx, 'rho_retrieved_corrected']
            data_corrected.loc[idx, 'error_corrected'] = band_data.loc[idx, 'error_corrected']

        return data_corrected