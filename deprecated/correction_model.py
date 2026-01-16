# ==================== correction_model.py ====================
"""
校正模型模块
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
from scipy import interpolate
from scipy.optimize import curve_fit
import joblib
import json

from config import ExperimentConfig
from utils import setup_logger, calculate_airmass


class CorrectionModel:
    """校正模型基类"""

    def __init__(self, model_type: str = 'lut', logger=None):
        self.model_type = model_type
        self.logger = logger or setup_logger('CorrectionModel')
        self.model = None
        self.metadata = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, filepath: Path):
        raise NotImplementedError

    def load(self, filepath: Path):
        raise NotImplementedError


class LUTCorrectionModel(CorrectionModel):
    """查找表校正模型"""

    def __init__(self, dimensions: List[str] = None, logger=None):
        super().__init__('lut', logger)
        self.dimensions = dimensions or ['sza', 'vza', 'aod550']
        self.grid_points = {}
        self.lut_values = None
        self.interpolator = None

    def create_lut_grid(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """创建LUT网格"""
        grid_points = {}
        for dim in self.dimensions:
            if dim in data.columns:
                unique_vals = np.sort(data[dim].unique())
                grid_points[dim] = unique_vals
                self.logger.info(f"维度 {dim}: {len(unique_vals)} 个点")
        return grid_points

    def create_lut_from_data(self, data: pd.DataFrame, value_col: str = 'error_absolute'):
        """从数据创建LUT"""
        self.grid_points = self.create_lut_grid(data)
        mesh_grids = np.meshgrid(*self.grid_points.values(), indexing='ij')
        lut_shape = tuple(len(vals) for vals in self.grid_points.values())
        self.lut_values = np.full(lut_shape, np.nan)

        self.logger.info("填充LUT...")
        for idx in np.ndindex(lut_shape):
            conditions = []
            for i, (dim_name, dim_vals) in enumerate(self.grid_points.items()):
                target_value = dim_vals[idx[i]]
                conditions.append(data[dim_name] == target_value)

            if conditions:
                mask = conditions[0]
                for cond in conditions[1:]:
                    mask = mask & cond

                if mask.any():
                    self.lut_values[idx] = data.loc[mask, value_col].mean()

        self._create_interpolator()
        fill_rate = np.sum(~np.isnan(self.lut_values)) / self.lut_values.size
        self.logger.info(f"LUT填充率: {fill_rate:.1%}")

        self.metadata = {
            'dimensions': self.dimensions,
            'grid_shape': lut_shape,
            'fill_rate': float(fill_rate),
            'created': pd.Timestamp.now().isoformat()
        }

    def _create_interpolator(self):
        """创建插值器"""
        try:
            points = [vals for vals in self.grid_points.values()]
            valid_mask = ~np.isnan(self.lut_values.ravel())

            if not valid_mask.any():
                raise ValueError("没有有效数据点")

            valid_coords = np.array(np.meshgrid(*points, indexing='ij')).T.reshape(-1, len(points))[valid_mask]
            valid_values = self.lut_values.ravel()[valid_mask]

            self.interpolator = interpolate.LinearNDInterpolator(valid_coords, valid_values, fill_value=0.0)
            self.nn_interpolator = interpolate.NearestNDInterpolator(valid_coords, valid_values)

        except Exception as e:
            self.logger.error(f"创建插值器失败: {e}")
            raise

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.interpolator is None:
            raise ValueError("模型尚未训练")

        predictions = self.interpolator(X)
        nan_mask = np.isnan(predictions)

        if nan_mask.any():
            self.logger.warning(f"{nan_mask.sum()} 个点插值失败，使用最近邻插值")
            predictions[nan_mask] = self.nn_interpolator(X[nan_mask])

        return predictions

    def save(self, filepath: Path):
        save_data = {
            'grid_points': self.grid_points,
            'lut_values': self.lut_values,
            'dimensions': self.dimensions,
            'metadata': self.metadata
        }

        np.savez_compressed(
            filepath,
            **{k: (v if k != 'grid_points' else {key: val for key, val in v.items()})
               for k, v in save_data.items() if k != 'grid_points'}
        )

        grid_points_file = filepath.parent / f"{filepath.stem}_grid.json"
        with open(grid_points_file, 'w') as f:
            json.dump({k: v.tolist() for k, v in self.grid_points.items()}, f)

        self.logger.info(f"LUT模型已保存: {filepath}")

    def load(self, filepath: Path):
        data = np.load(filepath, allow_pickle=True)
        self.lut_values = data['lut_values']
        self.dimensions = data['dimensions'].tolist()
        self.metadata = data['metadata'].item()

        grid_points_file = filepath.parent / f"{filepath.stem}_grid.json"
        with open(grid_points_file, 'r') as f:
            grid_dict = json.load(f)

        self.grid_points = {k: np.array(v) for k, v in grid_dict.items()}
        self._create_interpolator()
        self.logger.info(f"LUT模型已加载: {filepath}")


class AnalyticalCorrectionModel(CorrectionModel):
    """解析校正模型"""

    def __init__(self, formula_type: str = 'secz_linear', logger=None):
        super().__init__('analytical', logger)
        self.formula_type = formula_type
        self.coefficients = None
        self.formula_func = None

    def _secz_linear_formula(self, X: np.ndarray, *coeffs) -> np.ndarray:
        return coeffs[0] * X[:, 0] + coeffs[1] * X[:, 1] + coeffs[2]

    def _secz_poly_formula(self, X: np.ndarray, *coeffs) -> np.ndarray:
        secz_sza = X[:, 0]
        secz_vza = X[:, 1]
        return (coeffs[0] * secz_sza + coeffs[1] * secz_vza +
                coeffs[2] * secz_sza ** 2 + coeffs[3] * secz_vza ** 2 +
                coeffs[4] * secz_sza * secz_vza + coeffs[5])

    def _angle_poly_formula(self, X: np.ndarray, *coeffs) -> np.ndarray:
        sza = X[:, 0]
        vza = X[:, 1]
        sza_rad = np.radians(sza)
        vza_rad = np.radians(vza)

        return (coeffs[0] * sza_rad + coeffs[1] * vza_rad +
                coeffs[2] * sza_rad ** 2 + coeffs[3] * vza_rad ** 2 +
                coeffs[4] * sza_rad * vza_rad + coeffs[5])

    def fit(self, X: np.ndarray, y: np.ndarray):
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        if len(X_valid) < 10:
            self.logger.warning("有效数据点不足，无法训练模型")
            return

        if self.formula_type == 'secz_linear':
            p0 = [0.01, 0.01, 0.0]
            formula = self._secz_linear_formula
        elif self.formula_type == 'secz_poly':
            p0 = [0.01, 0.01, 0.001, 0.001, 0.001, 0.0]
            formula = self._secz_poly_formula
        elif self.formula_type == 'angle_poly':
            p0 = [0.01] * 5 + [0.0]
            formula = self._angle_poly_formula
        else:
            raise ValueError(f"不支持的公式类型: {self.formula_type}")

        try:
            coeffs, _ = curve_fit(formula, X_valid, y_valid, p0=p0, maxfev=5000)
            self.coefficients = coeffs
            self.formula_func = lambda X_pred: formula(X_pred, *coeffs)

            y_pred = self.predict(X_valid)
            ss_res = np.sum((y_valid - y_pred) ** 2)
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

            self.metadata = {
                'formula_type': self.formula_type,
                'coefficients': coeffs.tolist(),
                'r2': float(r2),
                'n_samples': len(X_valid),
                'created': pd.Timestamp.now().isoformat()
            }

            self.logger.info(f"解析模型训练完成 - R² = {r2:.4f}")

        except Exception as e:
            self.logger.error(f"模型训练失败: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.formula_func is None:
            raise ValueError("模型尚未训练")
        return self.formula_func(X)

    def save(self, filepath: Path):
        save_data = {
            'formula_type': self.formula_type,
            'coefficients': self.coefficients,
            'metadata': self.metadata
        }
        joblib.dump(save_data, filepath)
        self.logger.info(f"解析模型已保存: {filepath}")

    def load(self, filepath: Path):
        data = joblib.load(filepath)
        self.formula_type = data['formula_type']
        self.coefficients = data['coefficients']
        self.metadata = data['metadata']

        coeffs = self.coefficients
        if self.formula_type == 'secz_linear':
            self.formula_func = lambda X: self._secz_linear_formula(X, *coeffs)
        elif self.formula_type == 'secz_poly':
            self.formula_func = lambda X: self._secz_poly_formula(X, *coeffs)
        elif self.formula_type == 'angle_poly':
            self.formula_func = lambda X: self._angle_poly_formula(X, *coeffs)

        self.logger.info(f"解析模型已加载: {filepath}")


class ModelFactory:
    """模型工厂"""

    @staticmethod
    def create_model(model_type: str, **kwargs) -> CorrectionModel:
        if model_type == 'lut':
            dimensions = kwargs.get('dimensions', ['sza', 'vza', 'aod550'])
            return LUTCorrectionModel(dimensions=dimensions)
        elif model_type == 'linear':
            return AnalyticalCorrectionModel(formula_type='secz_linear')
        elif model_type == 'polynomial':
            formula_type = kwargs.get('formula_type', 'secz_poly')
            return AnalyticalCorrectionModel(formula_type=formula_type)
        elif model_type == 'ml':
            from sklearn.ensemble import RandomForestRegressor

            class MLCorrectionModel(CorrectionModel):
                def __init__(self, **model_kwargs):
                    super().__init__('ml')
                    self.model = RandomForestRegressor(**model_kwargs)

                def fit(self, X, y):
                    self.model.fit(X, y)
                    self.metadata = {
                        'feature_importance': self.model.feature_importances_.tolist(),
                        'n_estimators': self.model.n_estimators
                    }

                def predict(self, X):
                    return self.model.predict(X)

                def save(self, filepath):
                    joblib.dump(self.model, filepath)

                def load(self, filepath):
                    self.model = joblib.load(filepath)

            return MLCorrectionModel(**kwargs.get('model_kwargs', {}))
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")


class ModelTrainer:
    """模型训练器"""

    def __init__(self, config: ExperimentConfig, logger=None):
        self.config = config
        self.logger = logger or setup_logger('ModelTrainer')
        self.models = {}

    def prepare_training_data(self, data: pd.DataFrame, band_id: str = 'band3',
                              features: List[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        band_data = data[data['band'] == band_id].copy()

        if features is None:
            features = ['sza', 'vza', 'aod550', 'rho_true']

        band_data['secz_sza'] = calculate_airmass(band_data['sza'])
        band_data['secz_vza'] = calculate_airmass(band_data['vza'])

        available_features = [f for f in features if f in band_data.columns]

        valid_mask = (
                band_data['error_absolute'].notna() &
                ~band_data[available_features].isna().any(axis=1)
        )

        band_data_valid = band_data[valid_mask].copy()

        if len(band_data_valid) == 0:
            raise ValueError("没有有效训练数据")

        X = band_data_valid[available_features].values
        y = band_data_valid['error_absolute'].values

        self.logger.info(f"训练数据: {X.shape[0]} 个样本, {X.shape[1]} 个特征")

        return X, y, available_features

    def train_model(self, data: pd.DataFrame, band_id: str = 'band3',
                    model_type: str = 'lut', model_name: str = None, **kwargs):
        if model_name is None:
            model_name = f"{model_type}_{band_id}"

        self.logger.info(f"开始训练模型: {model_name} ({model_type})")

        try:
            X, y, features = self.prepare_training_data(data, band_id, kwargs.get('features'))
            model = ModelFactory.create_model(model_type, **kwargs)

            if model_type == 'lut':
                model.create_lut_from_data(data[data['band'] == band_id])
            else:
                model.fit(X, y)

            y_pred = model.predict(X)
            mae = np.mean(np.abs(y - y_pred))
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))

            self.logger.info(f"模型训练完成 - MAE: {mae:.6f}, RMSE: {rmse:.6f}")

            model_file = self.config.MODELS_DIR / f"{model_name}.pkl"
            model.save(model_file)

            eval_results = {
                'model_name': model_name,
                'model_type': model_type,
                'band_id': band_id,
                'features': features,
                'mae': float(mae),
                'rmse': float(rmse),
                'n_samples': len(y),
                'training_date': pd.Timestamp.now().isoformat()
            }

            eval_file = self.config.MODELS_DIR / f"{model_name}_eval.json"
            with open(eval_file, 'w') as f:
                json.dump(eval_results, f, indent=2)

            self.models[model_name] = model
            return model

        except Exception as e:
            self.logger.error(f"模型训练失败: {e}")
            raise

    def compare_models(self, data: pd.DataFrame, band_id: str = 'band3'):
        model_types = ['lut', 'linear', 'polynomial']
        comparison_results = []

        for model_type in model_types:
            try:
                model_name = f"{model_type}_{band_id}"
                model = self.train_model(data, band_id, model_type, model_name)

                X, y, _ = self.prepare_training_data(data, band_id)
                y_pred = model.predict(X)

                mae = np.mean(np.abs(y - y_pred))
                rmse = np.sqrt(np.mean((y - y_pred) ** 2))
                r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

                comparison_results.append({
                    'model_type': model_type,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'n_features': X.shape[1],
                    'n_samples': len(y)
                })

            except Exception as e:
                self.logger.error(f"模型 {model_type} 比较失败: {e}")

        df_comparison = pd.DataFrame(comparison_results)
        comp_file = self.config.MODELS_DIR / f"model_comparison_{band_id}.csv"
        df_comparison.to_csv(comp_file, index=False)

        self.logger.info(f"模型比较结果已保存: {comp_file}")
        return df_comparison