# ==================== lut_generator.py ====================
"""
最终业务化LUT生成器
基于训练好的ML模型生成非均匀网格LUT
"""
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import warnings

from config import ExperimentConfig
from enhanced_feature_engineering import EnhancedFeatureEngineering
from utils import setup_logger

warnings.filterwarnings('ignore')


class OperationalLUTGenerator:
    """
    业务化LUT生成器
    生成非均匀网格的最终LUT
    """

    def __init__(self, config: ExperimentConfig = None, logger=None):
        self.config = config or ExperimentConfig
        self.logger = logger or setup_logger('OperationalLUTGenerator')
        self.feature_engineer = EnhancedFeatureEngineering()

        # LUT轴配置
        self.lut_config = {
            'sza': {
                'regular': {'range': (0, 70), 'step': 5},  # 0-70°, 步长5°
                'extreme': {'range': (70, 85), 'step': 1}  # 70-85°, 步长1° (加密)
            },
            'vza': {
                'regular': {'range': (0, 70), 'step': 5},  # 0-70°, 步长5°
                'extreme': {'range': (70, 75), 'step': 1}  # 70-75°, 步长1° (加密)
            },
            'raa': {
                'range': (0, 180), 'step': 15  # 0-180°, 步长15°
            },
            'rho_apparent': {
                'range': (0.01, 0.6), 'step': 0.01  # 0.01-0.6, 步长0.01
            },
            'aod550': [0.05, 0.1, 0.2, 0.3, 0.5, 1.0],  # 典型AOD值
            'wavelength': list(self.config.BANDS.values())  # 波段波长
        }

    def generate_lut_axes(self) -> Dict[str, np.ndarray]:
        """
        生成LUT坐标轴（非均匀网格）
        """
        axes = {}

        # 1. SZA轴（非均匀）
        sza_regular = np.arange(
            self.lut_config['sza']['regular']['range'][0],
            self.lut_config['sza']['regular']['range'][1] + 0.1,
            self.lut_config['sza']['regular']['step']
        )

        sza_extreme = np.arange(
            self.lut_config['sza']['extreme']['range'][0],
            self.lut_config['sza']['extreme']['range'][1] + 0.1,
            self.lut_config['sza']['extreme']['step']
        )

        axes['sza'] = np.unique(np.concatenate([sza_regular, sza_extreme]))

        # 2. VZA轴（非均匀）
        vza_regular = np.arange(
            self.lut_config['vza']['regular']['range'][0],
            self.lut_config['vza']['regular']['range'][1] + 0.1,
            self.lut_config['vza']['regular']['step']
        )

        vza_extreme = np.arange(
            self.lut_config['vza']['extreme']['range'][0],
            self.lut_config['vza']['extreme']['range'][1] + 0.1,
            self.lut_config['vza']['extreme']['step']
        )

        axes['vza'] = np.unique(np.concatenate([vza_regular, vza_extreme]))

        # 3. RAA轴
        axes['raa'] = np.arange(
            self.lut_config['raa']['range'][0],
            self.lut_config['raa']['range'][1] + 0.1,
            self.lut_config['raa']['step']
        )

        # 4. 表观反射率轴（作为查找索引）
        axes['rho_apparent'] = np.arange(
            self.lut_config['rho_apparent']['range'][0],
            self.lut_config['rho_apparent']['range'][1] + 0.0001,
            self.lut_config['rho_apparent']['step']
        )

        # 5. AOD轴
        axes['aod550'] = np.array(self.lut_config['aod550'])

        # 6. 波长轴
        wavelengths = [band['wavelength'] for band in self.config.BANDS.values()]
        axes['wavelength'] = np.array(wavelengths)

        # 记录各轴维度
        for axis_name, axis_values in axes.items():
            self.logger.info(
                f"LUT轴 {axis_name}: {len(axis_values)} 个值, 范围: {axis_values.min():.2f}-{axis_values.max():.2f}")

        return axes

    def create_lut_dataframe(self, axes: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        创建LUT数据框（所有参数组合）
        """
        from itertools import product

        # 生成所有组合
        param_combinations = []

        for values in product(
                axes['sza'],
                axes['vza'],
                axes['raa'],
                axes['aod550'],
                axes['wavelength'],
                axes['rho_apparent']
        ):
            sza, vza, raa, aod550, wavelength, rho_apparent = values

            # 计算物理特征
            sza_rad = np.radians(sza)
            vza_rad = np.radians(vza)
            raa_rad = np.radians(raa)

            cos_sza = np.cos(sza_rad)
            cos_vza = np.cos(vza_rad)
            cos_sza_safe = max(cos_sza, 0.001)
            cos_vza_safe = max(cos_vza, 0.001)

            # 计算派生特征
            airmass_sza = 1.0 / cos_sza_safe
            airmass_vza = 1.0 / cos_vza_safe
            total_airmass = airmass_sza + airmass_vza

            # 散射角
            cos_scat = -np.cos(sza_rad) * np.cos(vza_rad) + \
                       np.sin(sza_rad) * np.sin(vza_rad) * np.cos(raa_rad)
            cos_scat = np.clip(cos_scat, -1.0, 1.0)
            scattering_angle = np.degrees(np.arccos(cos_scat))

            # 创建参数组合
            params = {
                'sza': sza,
                'vza': vza,
                'raa': raa,
                'aod550': aod550,
                'wavelength': wavelength,
                'rho_apparent': rho_apparent,  # 这是查找表的输入
                'cos_sza': cos_sza,
                'cos_vza': cos_vza,
                'airmass_sza': airmass_sza,
                'airmass_vza': airmass_vza,
                'total_airmass': total_airmass,
                'scattering_angle': scattering_angle,
                'h2o': 2.0,  # 默认值
                'o3': 0.3,  # 默认值
                'atmos_profile': 'MidlatitudeSummer',
                'aero_profile': 'Continental',
                'is_extreme': (sza > 60) or (vza > 60)
            }

            param_combinations.append(params)

        df = pd.DataFrame(param_combinations)
        self.logger.info(f"LUT参数组合总数: {len(df):,}")

        return df

    def predict_with_ml_model(self, lut_df: pd.DataFrame,
                              ml_model_path: Path) -> pd.DataFrame:
        """
        使用训练好的ML模型预测校正误差
        """
        self.logger.info(f"加载ML模型: {ml_model_path}")

        with open(ml_model_path, 'rb') as f:
            ml_model = pickle.load(f)

        # 准备特征
        features = self.feature_engineer.create_all_features(lut_df)

        # 确保与训练时相同的特征顺序
        expected_features = ml_model.feature_names_in_ if hasattr(ml_model, 'feature_names_in_') else None
        if expected_features is not None:
            # 确保特征顺序一致
            missing_features = set(expected_features) - set(features.columns)
            extra_features = set(features.columns) - set(expected_features)

            if missing_features:
                self.logger.warning(f"缺失特征: {missing_features}")
                for feat in missing_features:
                    features[feat] = 0.0  # 用0填充缺失特征

            # 重新排序特征
            features = features[list(expected_features)]

        # 预测校正误差
        self.logger.info("使用ML模型预测校正误差...")
        predicted_error = ml_model.predict(features)

        # 添加到LUT数据框
        lut_df['ml_correction'] = predicted_error
        lut_df['rho_corrected'] = lut_df['rho_apparent'] - predicted_error

        # 计算统计信息
        self.logger.info(f"校正误差统计:")
        self.logger.info(f"  均值: {predicted_error.mean():.6f}")
        self.logger.info(f"  标准差: {predicted_error.std():.6f}")
        self.logger.info(f"  最小值: {predicted_error.min():.6f}")
        self.logger.info(f"  最大值: {predicted_error.max():.6f}")

        return lut_df

    def create_xarray_dataset(self, lut_df: pd.DataFrame,
                              axes: Dict[str, np.ndarray]) -> xr.Dataset:
        """
        创建xarray Dataset格式的LUT
        """
        self.logger.info("创建xarray Dataset...")

        # 定义维度
        dims = ['wavelength', 'sza', 'vza', 'raa', 'aod550', 'rho_apparent']

        # 初始化数据数组
        shape = (
            len(axes['wavelength']),
            len(axes['sza']),
            len(axes['vza']),
            len(axes['raa']),
            len(axes['aod550']),
            len(axes['rho_apparent'])
        )

        # 创建多维数组
        ml_correction_array = np.full(shape, np.nan, dtype=np.float32)
        rho_corrected_array = np.full(shape, np.nan, dtype=np.float32)

        # 填充数组
        for idx, row in lut_df.iterrows():
            # 找到各个维度的索引
            w_idx = np.where(axes['wavelength'] == row['wavelength'])[0][0]
            sza_idx = np.where(axes['sza'] == row['sza'])[0][0]
            vza_idx = np.where(axes['vza'] == row['vza'])[0][0]
            raa_idx = np.where(axes['raa'] == row['raa'])[0][0]
            aod_idx = np.where(axes['aod550'] == row['aod550'])[0][0]
            rho_idx = np.where(np.isclose(axes['rho_apparent'], row['rho_apparent']))[0][0]

            # 赋值
            ml_correction_array[w_idx, sza_idx, vza_idx, raa_idx, aod_idx, rho_idx] = row['ml_correction']
            rho_corrected_array[w_idx, sza_idx, vza_idx, raa_idx, aod_idx, rho_idx] = row['rho_corrected']

        # 创建Dataset
        ds = xr.Dataset(
            {
                'ml_correction': (dims, ml_correction_array),
                'rho_corrected': (dims, rho_corrected_array)
            },
            coords={
                'wavelength': axes['wavelength'],
                'sza': axes['sza'],
                'vza': axes['vza'],
                'raa': axes['raa'],
                'aod550': axes['aod550'],
                'rho_apparent': axes['rho_apparent']
            },
            attrs={
                'description': '6S几何校正ML LUT',
                'generated_by': 'OperationalLUTGenerator',
                'generation_date': pd.Timestamp.now().isoformat(),
                'model_type': 'ML-based correction',
                'note': 'rho_apparent is input, rho_corrected is output'
            }
        )

        return ds

    def generate_operational_lut(self, ml_model_path: Path,
                                 output_path: Optional[Path] = None) -> xr.Dataset:
        """
        生成业务化LUT的主函数
        """
        self.logger.info("开始生成业务化LUT...")

        # 1. 生成LUT坐标轴
        axes = self.generate_lut_axes()

        # 2. 创建参数组合数据框
        lut_df = self.create_lut_dataframe(axes)

        # 3. 使用ML模型预测校正误差
        lut_df = self.predict_with_ml_model(lut_df, ml_model_path)

        # 4. 创建xarray Dataset
        ds = self.create_xarray_dataset(lut_df, axes)

        # 5. 保存到文件
        if output_path is None:
            output_path = self.config.RESULTS_DIR / "operational_lut.nc"

        # 添加压缩
        encoding = {
            var: {
                'zlib': True,
                'complevel': 4,
                'dtype': 'float32'
            } for var in ds.data_vars
        }

        ds.to_netcdf(output_path, encoding=encoding)
        self.logger.info(f"LUT已保存至: {output_path}")

        # 6. 生成LUT使用说明
        self._generate_lut_documentation(ds, output_path)

        return ds

    def _generate_lut_documentation(self, ds: xr.Dataset, output_path: Path):
        """
        生成LUT使用说明文档
        """
        doc_path = output_path.with_suffix('.txt')

        with open(doc_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("6S几何校正ML LUT 使用说明\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. 文件格式:\n")
            f.write("   - 格式: NetCDF\n")
            f.write("   - 维度: wavelength, sza, vza, raa, aod550, rho_apparent\n")
            f.write("   - 变量: ml_correction, rho_corrected\n\n")

            f.write("2. LUT坐标轴:\n")
            for dim in ds.dims:
                f.write(f"   {dim}: {len(ds[dim])} 个值\n")
                if len(ds[dim]) <= 10:
                    f.write(f"     值: {list(ds[dim].values)}\n")
                else:
                    f.write(f"     范围: {ds[dim].min().item():.2f} - {ds[dim].max().item():.2f}\n")
            f.write("\n")

            f.write("3. 使用方法:\n")
            f.write("   - 输入: 表观反射率 (rho_apparent), 几何参数, 大气参数\n")
            f.write("   - 输出: 校正后的地表反射率 (rho_corrected)\n")
            f.write("   - 公式: rho_corrected = rho_apparent - ml_correction\n\n")

            f.write("4. Python使用示例:\n")
            f.write("   ```python\n")
            f.write("   import xarray as xr\n")
            f.write("   \n")
            f.write("   # 加载LUT\n")
            f.write("   ds = xr.open_dataset('operational_lut.nc')\n")
            f.write("   \n")
            f.write("   # 插值获取校正值\n")
            f.write("   correction = ds['ml_correction'].interp(\n")
            f.write("       wavelength=0.64,\n")
            f.write("       sza=45.0,\n")
            f.write("       vza=30.0,\n")
            f.write("       raa=90.0,\n")
            f.write("       aod550=0.3,\n")
            f.write("       rho_apparent=0.2\n")
            f.write("   )\n")
            f.write("   \n")
            f.write("   rho_corrected = 0.2 - correction.item()\n")
            f.write("   ```\n\n")

            f.write("5. 注意事项:\n")
            f.write("   - 对于边界外的值，使用最近邻或线性插值\n")
            f.write("   - 极端角度 (SZA/VZA > 60°) 区域已加密采样\n")
            f.write("   - LUT基于ML模型生成，适用于Himawari-8/AHI波段\n")

        self.logger.info(f"LUT使用说明已保存至: {doc_path}")

