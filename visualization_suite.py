# ==================== visualization_suite.py ====================
"""
可视化套件（新框架）
目标变量：delta_toa = ρ_TOA^SA - ρ_TOA^PPA
"""
import numpy as np# ==================== visualization_suite.py ====================
"""
可视化套件（新框架）
目标变量：delta_toa = ρ_TOA^SA - ρ_TOA^PPA
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional
from pathlib import Path
import xarray as xr
import warnings
from matplotlib.colors import Normalize, ListedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import argparse
import matplotlib.lines as mlines

warnings.filterwarnings('ignore')

from config import ExperimentConfig
from utils import setup_logger

# 设置专业科研字体
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


class VisualizationSuite:
    """
    可视化套件 - 使用验证网格数据绘制 ΔTOA 相关图形
    """

    def __init__(self, data_dir: Optional[Path] = None,
                 suffix: str = "",
                 logger=None,
                 fixed_rho_true: float = 0.3,
                 fixed_raa: float = 0.0):
        self.logger = logger or setup_logger('VisualizationSuite')
        self.suffix = suffix
        self.fixed_rho_true = fixed_rho_true
        self.fixed_raa = fixed_raa

        if not (0 < self.fixed_rho_true <= 1.0):
            self.logger.warning(f"fixed_rho_true={self.fixed_rho_true} 超出合理范围，重置为0.3")
            self.fixed_rho_true = 0.3

        self.logger.info(f"使用固定参数: rho_true={self.fixed_rho_true}, raa={self.fixed_raa}")

        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = ExperimentConfig.DATA_DIR

        self.logger.info(f"使用数据目录: {self.data_dir}")

        if not self.data_dir.exists():
            self.logger.warning(f"数据目录不存在: {self.data_dir}")

        output_subdir = f"visualization{self.suffix}"
        self.output_dir = ExperimentConfig.MANU_FIGURES_DIR / output_subdir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.validation_data = None

        # 标准大气条件
        self.atmospheric_conditions = {
            'clean': {
                'name': 'Clean (AOD=0.1)',
                'aod550': 0.1,
                'h2o': 2.0,
                'o3': 0.3
            },
            'average': {
                'name': 'Average (AOD=0.3)',
                'aod550': 0.3,
                'h2o': 2.0,
                'o3': 0.3
            },
            'polluted': {
                'name': 'Polluted (AOD=0.5)',
                'aod550': 0.5,
                'h2o': 2.0,
                'o3': 0.3
            }
        }

        self.logger.info(f"初始化 VisualizationSuite:")
        self.logger.info(f"  文件后缀: {self.suffix}")
        self.logger.info(f"  输出目录: {self.output_dir}")

    def _get_validation_filename(self, band_id: str) -> str:
        return f"validation_grid_{band_id}{self.suffix}.nc"

    def _load_validation_grid_data(self) -> pd.DataFrame:
        """加载验证网格数据，重点关注 delta_toa 列，并将值截断至 [-2, 2] 范围内"""
        if self.validation_data is not None:
            return self.validation_data

        all_dfs = []
        self.logger.info("正在加载验证网格数据用于绘图...")
        self.logger.info(f"查找文件模式: validation_grid_*{self.suffix}.nc")
        self.logger.info(f"数据目录: {self.data_dir}")

        if not self.data_dir.exists():
            self.logger.error(f"数据目录不存在: {self.data_dir}")
            return pd.DataFrame()

        for band_id in ExperimentConfig.BANDS.keys():
            filename = self._get_validation_filename(band_id)
            data_file = self.data_dir / filename

            if data_file.exists():
                try:
                    self.logger.info(f"加载文件: {data_file}")
                    ds = xr.open_dataset(data_file)
                    df = ds.to_dataframe().reset_index()
                    ds.close()

                    df['band'] = band_id
                    if 'wavelength' not in df.columns:
                        df['wavelength'] = ExperimentConfig.BANDS[band_id]['wavelength']

                    # 确保 delta_toa 列存在
                    if 'delta_toa' not in df.columns:
                        # 如果存在 rho_toa_sa 和 rho_toa_ppa，则计算
                        if 'rho_toa_sa' in df.columns and 'rho_toa_ppa' in df.columns:
                            df['delta_toa'] = df['rho_toa_sa'] - df['rho_toa_ppa']
                            self.logger.info(f"波段 {band_id}: 从 rho_toa_sa/ppa 计算 delta_toa")
                        else:
                            self.logger.error(f"波段 {band_id} 缺少 delta_toa 且无法计算，跳过")
                            continue

                    # 验证数据合理性
                    self._validate_data_physics(df, band_id)

                    # 将 ΔTOA 截断至 [-2, 2] 范围内（超出部分设为边界值）
                    n_before = len(df)
                    df['delta_toa'] = np.clip(df['delta_toa'], -2.0, 2.0)
                    n_after = len(df)
                    if n_before > n_after:
                        self.logger.warning(f"波段 {band_id}: 截断了 {n_before - n_after} 个超出 [-2,2] 的 ΔTOA 值")

                    self.logger.info(f"  波段 {band_id}: {len(df)} 条记录, "
                                     f"ΔTOA 均值: {df['delta_toa'].mean():.6f}, "
                                     f"范围: [{df['delta_toa'].min():.6f}, {df['delta_toa'].max():.6f}]")

                    all_dfs.append(df)

                except Exception as e:
                    self.logger.error(f"加载波段 {band_id} 失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                self.logger.warning(f"未找到文件: {data_file}")

        if all_dfs:
            self.validation_data = pd.concat(all_dfs, ignore_index=True)
            self.logger.info(f"验证数据总计: {len(self.validation_data)} 条记录")
            self.logger.info(f"全局 ΔTOA 统计: 均值={self.validation_data['delta_toa'].mean():.6f}, "
                             f"标准差={self.validation_data['delta_toa'].std():.6f}")
        else:
            self.logger.error("没有加载到任何验证数据！")
            self.validation_data = pd.DataFrame()

        return self.validation_data

    def _validate_data_physics(self, df: pd.DataFrame, band_id: str):
        """验证数据的物理合理性（略作修改）"""
        # 检查反射率范围
        for col in ['rho_true', 'rho_toa_sa', 'rho_toa_ppa']:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if min_val < 0 or max_val > 1:
                    self.logger.warning(
                        f"  波段 {band_id} 列 {col}: 值范围 [{min_val:.4f}, {max_val:.4f}] 超出合理范围 [0, 1]")

        # 检查角度范围
        for col in ['sza', 'vza']:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if min_val < 0 or max_val > 90:
                    self.logger.warning(
                        f"  波段 {band_id} 列 {col}: 值范围 [{min_val:.1f}, {max_val:.1f}] 超出合理范围 [0, 90]")

        if 'raa' in df.columns:
            min_val = df['raa'].min()
            max_val = df['raa'].max()
            if min_val < 0 or max_val > 180:
                self.logger.warning(
                    f"  波段 {band_id} 列 raa: 值范围 [{min_val:.1f}, {max_val:.1f}] 超出合理范围 [0, 180]")

    def run_all_plots(self):
        """运行所有绘图任务"""
        if self._load_validation_grid_data().empty:
            self.logger.error("验证数据为空，无法绘图")
            return

        self.logger.info(">>> 开始生成物理过程失真图 (Plot 1 - Validation Grid)...")
        self.plot_distortion_process()

        self.logger.info(">>> 开始生成 ΔTOA 分布 Contour 图 (Plot 2 - Validation Grid)...")
        self.plot_delta_toa_contours()

    # =========================================================================
    # Part 1: Weird Line Plots (Distortion Process) - 与旧版类似，但三点的含义需调整
    # =========================================================================
    def plot_distortion_process(self):
        """
        生成过程图：显示 ρ_true, ρ_TOA^SA, ρ_TOA^PPA 的对比
        针对每个 RAA 值分别生成固定SZA和固定VZA图
        """
        data = self._load_validation_grid_data()
        if data.empty:
            self.logger.warning("数据缺失，跳过 Plot 1")
            return

        raa_values = ExperimentConfig.VALIDATION_GRID['raa']
        self.logger.info(f"将为以下 RAA 值生成过程图: {raa_values}")

        fixed_sza_values = [0, 30, 60, 85, 89]
        fixed_vza_values = [0, 30, 60, 85, 89]

        for raa in raa_values:
            self.logger.info(f"正在生成 RAA = {raa}° 的过程图...")
            param_text = f" (ρ={self.fixed_rho_true}, RAA={raa}°)"

            # 固定 SZA 子图
            self._create_fixed_sza_figure(
                data, fixed_sza_values, param_text, raa_value=raa
            )
            # 固定 VZA 子图
            self._create_fixed_vza_figure(
                data, fixed_vza_values, param_text, raa_value=raa
            )

    # ---------- 自定义两段式尺度变换 (新版) ----------
    @staticmethod
    def _forward_segmented_v2(y):
        """将数据坐标 y 映射到轴坐标 [0,1]：两段式 [0,1]占1/3, [1,20]占2/3"""
        y = np.asarray(y)
        b1, b2 = 0.0, 1.0
        b3 = 20.0
        len1 = b2 - b1
        len2 = b3 - b2
        frac1, frac2 = 1/3, 2/3
        pos = np.zeros_like(y, dtype=float)
        mask1 = y <= b2
        pos[mask1] = (y[mask1] - b1) / len1 * frac1
        mask2 = y > b2
        pos[mask2] = frac1 + (y[mask2] - b2) / len2 * frac2
        pos = np.clip(pos, 0, 1)
        return pos

    @staticmethod
    def _inverse_segmented_v2(pos):
        """将轴坐标 [0,1] 映射回数据坐标 y"""
        pos = np.asarray(pos)
        b1, b2 = 0.0, 1.0
        b3 = 20.0
        len1 = b2 - b1
        len2 = b3 - b2
        frac1, frac2 = 1/3, 2/3
        y = np.zeros_like(pos, dtype=float)
        mask1 = pos <= frac1
        y[mask1] = b1 + (pos[mask1] / frac1) * len1
        mask2 = pos > frac1
        y[mask2] = b2 + ((pos[mask2] - frac1) / frac2) * len2
        return y

    # ---------- 固定 SZA 图 ----------
    def _create_fixed_sza_figure(self, data: pd.DataFrame, fixed_sza_values: List,
                                 param_text: str, raa_value: float):
        """创建固定 SZA 图形（新布局：ρ_true 左，SA 与 PPA 右，虚线连接）"""
        vza_levels = [0, 15, 30, 45, 60, 75, 80, 85, 86, 87, 88, 89]
        n_vza = len(vza_levels)
        base_cmap = plt.cm.viridis
        vza_colors = [base_cmap(i / (n_vza - 1)) for i in range(n_vza)]
        vza_cmap = ListedColormap(vza_colors)
        vza_norm = Normalize(vmin=0, vmax=n_vza - 1)

        atm_order = ['clean', 'average', 'polluted']
        fig, axes = plt.subplots(3, 5, figsize=(18, 10),
                                 gridspec_kw={'hspace': 0.35, 'wspace': 0.25})
        if axes.ndim == 1:
            axes = axes.reshape(3, 5)

        # 新坐标定义
        x_true, x_sa, x_ppa = 0.0, 1.0, 1.3

        for row_idx, condition_key in enumerate(atm_order):
            cond_params = self.atmospheric_conditions[condition_key]
            cond_data = data[
                (np.abs(data['aod550'] - cond_params['aod550']) < 0.05) &
                (np.abs(data['h2o'] - cond_params['h2o']) < 0.5) &
                (np.abs(data['rho_true'] - self.fixed_rho_true) < 0.001) &
                (np.abs(data['raa'] - raa_value) < 0.1)
                ]

            for col_idx, sza_val in enumerate(fixed_sza_values):
                ax = axes[row_idx, col_idx]
                sza_data = cond_data[np.abs(cond_data['sza'] - sza_val) < 2.0].copy()

                if sza_data.empty:
                    ax.axis('off')
                    continue

                # 根据 sza_val 是否为极端角度设置 y 轴范围和尺度
                if sza_val >= 85:
                    # 极端角度：使用新版两段式尺度，范围 0-20
                    ax.set_yscale('function', functions=(self._forward_segmented_v2, self._inverse_segmented_v2))
                    ax.set_ylim(0, 20)
                    # 主刻度
                    ax.set_yticks([0.0, 1.0, 20.0])
                    ax.set_yticklabels(['0.0', '1.0', '20.0'])
                    # 次刻度
                    ax.set_yticks([5.0, 10.0, 15.0], minor=True)
                    ax.set_yticklabels(['5', '10', '15'], minor=True)
                    # 添加分界线
                    ax.axhline(y=1, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
                else:
                    # 非极端角度：线性 [0,1]
                    ax.set_ylim(0, 1)
                    ax.set_yticks(np.linspace(0, 1, 6))
                    ax.set_yticklabels([f'{y:.1f}' for y in np.linspace(0, 1, 6)], fontsize=8)
                    ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
                    ax.axhline(y=1, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)

                # 绘制各个 VZA 的数据点及连线
                for vza_val in vza_levels:
                    point_data = sza_data[np.abs(sza_data['vza'] - vza_val) < 2.0]
                    if point_data.empty:
                        continue

                    vals = [
                        point_data['rho_true'].mean(),
                        point_data['rho_toa_sa'].mean(),
                        point_data['rho_toa_ppa'].mean()
                    ]

                    idx = vza_levels.index(vza_val)
                    color = vza_colors[idx]

                    # 根据 VZA 角度设置标记样式
                    if vza_val >= 85:
                        marker = 'D'
                        markersize = 6
                        alpha = 0.9
                    else:
                        marker = 'o'
                        markersize = 5
                        alpha = 0.8

                    # 绘制虚线连接 ρ_true → ρ_SA
                    ax.plot([x_true, x_sa], [vals[0], vals[1]],
                            linestyle='--', linewidth=1.5, color=color, alpha=alpha,
                            marker='', solid_capstyle='round')
                    # 绘制实线连接 ρ_SA → ρ_PPA
                    ax.plot([x_sa, x_ppa], [vals[1], vals[2]],
                            linestyle='-', linewidth=1.5, color=color, alpha=alpha,
                            marker='', solid_capstyle='round')
                    # 绘制三个数据点
                    ax.scatter([x_true, x_sa, x_ppa], vals,
                               marker=marker, s=markersize ** 2, color=color, alpha=alpha,
                               edgecolors='none', zorder=5)

                # 添加垂直虚线分隔左右区域
                ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)

                # 设置 x 轴刻度
                ax.set_xticks([x_true, x_sa, x_ppa])
                ax.set_xticklabels([r'$\rho_{true}$',
                                    r'$\rho_{TOA}^{SA}$',
                                    r'$\rho_{TOA}^{PPA}$'],
                                   fontsize=9)

                # 子图标题
                if sza_val >= 85:
                    ax.set_title(rf'SZA = {sza_val}° (Extreme)', fontsize=11,
                                 fontweight='bold', color='red')
                else:
                    ax.set_title(rf'SZA = {sza_val}°', fontsize=11, fontweight='bold')

                # 行标签
                if col_idx == 0:
                    ax.set_ylabel('Reflectance', fontsize=10)
                    ax.text(-0.4, 0.5, cond_params['name'], transform=ax.transAxes,
                            rotation=90, va='center', fontweight='bold', fontsize=10)

                ax.grid(True, alpha=0.2, linestyle='--')
                ax.tick_params(axis='both', which='major', labelsize=8)

        # 颜色条
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = ScalarMappable(cmap=vza_cmap, norm=vza_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, ticks=np.arange(n_vza))
        cbar.ax.set_yticklabels([str(v) for v in vza_levels])
        cbar.set_label('View Zenith Angle (°)', fontsize=10)
        cbar.ax.tick_params(labelsize=8)

        # 添加分隔线（极端与非极端列之间）
        fig.canvas.draw()  # 确保位置已计算
        for row in range(3):
            ax_left = axes[row, 2]   # 第三列（非极端最后一列）
            ax_right = axes[row, 3]  # 第四列（极端第一列）
            pos_left = ax_left.get_position()
            pos_right = ax_right.get_position()
            x_mid = (pos_left.x1 + pos_right.x0) / 2
            line = mlines.Line2D([x_mid, x_mid], [pos_left.y0, pos_left.y1],
                                 transform=fig.transFigure, color='black', linewidth=1.5, linestyle='-')
            fig.lines.append(line)

        extreme_note = ("Note: Extreme VZA (≥85°) marked with diamonds; dashed: ρ_true→ρ_SA, solid: ρ_SA→ρ_PPA; "
                        "For extreme SZA (≥85°), y-axis uses segmented scaling: [0,1] (1/3 height), [1,20] (2/3 height); "
                        "major ticks at 0.0, 1.0, 20.0; minor ticks at 5, 10, 15.")
        fig.text(0.5, 0.02, extreme_note, ha='center', fontsize=9, style='italic')

        fig.suptitle(
            f'Reflectance Components - Fixed Solar Zenith Angle{param_text}',
            fontsize=13, fontweight='bold', y=0.98)

        output_filename = f"plot1_distortion_fixed_sza_RAA{int(raa_value):03d}{self.suffix}.png"
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Fixed SZA / RAA={raa_value}° 图已保存: {output_path}")

    # ---------- 固定 VZA 图 ----------
    def _create_fixed_vza_figure(self, data: pd.DataFrame, fixed_vza_values: List,
                                 param_text: str, raa_value: float):
        """创建固定 VZA 图形（新布局）"""
        sza_levels = [0, 15, 30, 45, 60, 75, 80, 85, 86, 87, 88, 89]
        n_sza = len(sza_levels)
        base_cmap = plt.cm.plasma
        sza_colors = [base_cmap(i / (n_sza - 1)) for i in range(n_sza)]
        sza_cmap = ListedColormap(sza_colors)
        sza_norm = Normalize(vmin=0, vmax=n_sza - 1)

        atm_order = ['clean', 'average', 'polluted']
        fig, axes = plt.subplots(3, 5, figsize=(18, 10),
                                 gridspec_kw={'hspace': 0.35, 'wspace': 0.25})
        if axes.ndim == 1:
            axes = axes.reshape(3, 5)

        x_true, x_sa, x_ppa = 0.0, 1.0, 1.3

        for row_idx, condition_key in enumerate(atm_order):
            cond_params = self.atmospheric_conditions[condition_key]
            cond_data = data[
                (np.abs(data['aod550'] - cond_params['aod550']) < 0.05) &
                (np.abs(data['h2o'] - cond_params['h2o']) < 0.5) &
                (np.abs(data['rho_true'] - self.fixed_rho_true) < 0.001) &
                (np.abs(data['raa'] - raa_value) < 0.1)
                ]

            for col_idx, vza_val in enumerate(fixed_vza_values):
                ax = axes[row_idx, col_idx]
                vza_data = cond_data[np.abs(cond_data['vza'] - vza_val) < 2.0].copy()

                if vza_data.empty:
                    ax.axis('off')
                    continue

                # 根据 vza_val 是否为极端角度设置 y 轴范围和尺度
                if vza_val >= 85:
                    # 极端角度：使用新版两段式尺度，范围 0-20
                    ax.set_yscale('function', functions=(self._forward_segmented_v2, self._inverse_segmented_v2))
                    ax.set_ylim(0, 20)
                    # 主刻度
                    ax.set_yticks([0.0, 1.0, 20.0])
                    ax.set_yticklabels(['0.0', '1.0', '20.0'])
                    # 次刻度
                    ax.set_yticks([5.0, 10.0, 15.0], minor=True)
                    ax.set_yticklabels(['5', '10', '15'], minor=True)
                    # 添加分界线
                    ax.axhline(y=1, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
                else:
                    # 非极端角度：线性 [0,1]
                    ax.set_ylim(0, 1)
                    ax.set_yticks(np.linspace(0, 1, 6))
                    ax.set_yticklabels([f'{y:.1f}' for y in np.linspace(0, 1, 6)], fontsize=8)
                    ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
                    ax.axhline(y=1, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)

                # 绘制各个 SZA 的数据点及连线
                for sza_val in sza_levels:
                    point_data = vza_data[np.abs(vza_data['sza'] - sza_val) < 2.0]
                    if point_data.empty:
                        continue

                    vals = [
                        point_data['rho_true'].mean(),
                        point_data['rho_toa_sa'].mean(),
                        point_data['rho_toa_ppa'].mean()
                    ]

                    idx = sza_levels.index(sza_val)
                    color = sza_colors[idx]

                    if sza_val >= 85:
                        marker = 's'
                        markersize = 6
                        alpha = 0.9
                    else:
                        marker = 's'
                        markersize = 5
                        alpha = 0.8

                    # 绘制虚线连接 ρ_true → ρ_SA
                    ax.plot([x_true, x_sa], [vals[0], vals[1]],
                            linestyle='--', linewidth=1.5, color=color, alpha=alpha,
                            marker='')
                    # 绘制实线连接 ρ_SA → ρ_PPA
                    ax.plot([x_sa, x_ppa], [vals[1], vals[2]],
                            linestyle='-', linewidth=1.5, color=color, alpha=alpha,
                            marker='')
                    # 绘制三个数据点
                    ax.scatter([x_true, x_sa, x_ppa], vals,
                               marker=marker, s=markersize ** 2, color=color, alpha=alpha,
                               edgecolors='none', zorder=5)

                ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)

                ax.set_xticks([x_true, x_sa, x_ppa])
                ax.set_xticklabels([r'$\rho_{true}$',
                                    r'$\rho_{TOA}^{SA}$',
                                    r'$\rho_{TOA}^{PPA}$'],
                                   fontsize=9)

                if vza_val >= 85:
                    ax.set_title(rf'VZA = {vza_val}° (Extreme)', fontsize=11,
                                 fontweight='bold', color='red')
                else:
                    ax.set_title(rf'VZA = {vza_val}°', fontsize=11, fontweight='bold')

                if col_idx == 0:
                    ax.set_ylabel('Reflectance', fontsize=10)
                    ax.text(-0.4, 0.5, cond_params['name'], transform=ax.transAxes,
                            rotation=90, va='center', fontweight='bold', fontsize=10)

                ax.grid(True, alpha=0.2, linestyle='--')
                ax.tick_params(axis='both', which='major', labelsize=8)

        # 颜色条
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = ScalarMappable(cmap=sza_cmap, norm=sza_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, ticks=np.arange(n_sza))
        cbar.ax.set_yticklabels([str(v) for v in sza_levels])
        cbar.set_label('Solar Zenith Angle (°)', fontsize=10)
        cbar.ax.tick_params(labelsize=8)

        # 添加分隔线（极端与非极端列之间）
        fig.canvas.draw()
        for row in range(3):
            ax_left = axes[row, 2]
            ax_right = axes[row, 3]
            pos_left = ax_left.get_position()
            pos_right = ax_right.get_position()
            x_mid = (pos_left.x1 + pos_right.x0) / 2
            line = mlines.Line2D([x_mid, x_mid], [pos_left.y0, pos_left.y1],
                                 transform=fig.transFigure, color='black', linewidth=1.5, linestyle='-')
            fig.lines.append(line)

        extreme_note = ("Note: Extreme SZA (≥85°) marked with squares; dashed: ρ_true→ρ_SA, solid: ρ_SA→ρ_PPA; "
                        "For extreme VZA (≥85°), y-axis uses segmented scaling: [0,1] (1/3 height), [1,20] (2/3 height); "
                        "major ticks at 0.0, 1.0, 20.0; minor ticks at 5, 10, 15.")
        fig.text(0.5, 0.02, extreme_note, ha='center', fontsize=9, style='italic')

        fig.suptitle(
            f'Reflectance Components - Fixed View Zenith Angle{param_text}',
            fontsize=13, fontweight='bold', y=0.98)

        output_filename = f"plot1_distortion_fixed_vza_RAA{int(raa_value):03d}{self.suffix}.png"
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Fixed VZA / RAA={raa_value}° 图已保存: {output_path}")

    # =========================================================================
    # Part 2: Contour Plots for ΔTOA (modified according to requirements)
    # =========================================================================
    def plot_delta_toa_contours(self):
        """生成 3(AOD) x 6(Band) 的 Contour 图，展示 ΔTOA 随角度的变化"""
        data = self._load_validation_grid_data()
        if data.empty:
            self.logger.warning("数据为空，跳过 Contour 图")
            return

        self._validate_contour_data_quality(data)

        raa_values = ExperimentConfig.VALIDATION_GRID.get('raa', [self.fixed_raa])
        if not isinstance(raa_values, list):
            raa_values = [raa_values]
        if not raa_values:
            raa_values = [self.fixed_raa]

        self.logger.info(f"将为以下 RAA 值生成 ΔTOA Contour 图: {raa_values}")

        for raa in raa_values:
            self._plot_delta_toa_contour_single(data, raa)

    def _plot_delta_toa_contour_single(self, data: pd.DataFrame, raa: float):
        """
        为单个 RAA 绘制 ΔTOA Contour 图。

        关键改动（为了解决“极端角度把色标拉爆，内部变化看不见”的问题）：
        1) 绘图时优先使用未截断的 ΔTOA：若存在 rho_toa_sa/rho_toa_ppa 则现场重算；
        2) 使用以 0 为中心的对称对数归一化 SymLogNorm：
           - |ΔTOA| <= linthresh：线性（保留 10^-3~10^-2 的细微变化）
           - |ΔTOA| >  linthresh：对数压缩（容纳 10^0~10^1 的极端值）
        3) colorbar 明确标注 symlog 与 linthresh，刻度采用对称对数风格。
        """
        from matplotlib.colors import SymLogNorm  # 局部导入，避免你额外改全局 import

        filtered_data = data[
            (np.abs(data['rho_true'] - self.fixed_rho_true) < 0.0001) &
            (np.abs(data['raa'] - raa) < 0.01)
            ].copy()

        if filtered_data.empty:
            self.logger.error(f"错误: 没有找到 rho_true={self.fixed_rho_true}, raa={raa} 的精确匹配数据")
            self.logger.info("尝试使用所有数据，但结果可能不准确...")
            filtered_data = data.copy()

        self.logger.info(f"Contour图 (RAA={raa}) 使用数据: {len(filtered_data)} 条记录")

        # === 优先使用未截断的 ΔTOA（避免被 _load_validation_grid_data 的 clip 掩盖极端值）===
        if ('rho_toa_sa' in filtered_data.columns) and ('rho_toa_ppa' in filtered_data.columns):
            target_col = '_delta_toa_plot'
            filtered_data[target_col] = filtered_data['rho_toa_sa'] - filtered_data['rho_toa_ppa']
            self.logger.info("绘图使用未截断 ΔTOA：由 rho_toa_sa - rho_toa_ppa 现场计算")
        else:
            target_col = 'delta_toa'
            self.logger.warning("未找到 rho_toa_sa/rho_toa_ppa，绘图将使用 delta_toa 列（可能已被截断）")

        atm_order = ['clean', 'average', 'polluted']
        bands = ['band1', 'band2', 'band3', 'band4', 'band5', 'band6']

        # 原始角度网格（可能包含90°）
        sza_grid_full = np.array(ExperimentConfig.VALIDATION_GRID['sza'])
        vza_grid_full = np.array(ExperimentConfig.VALIDATION_GRID['vza'])

        # 仅保留 ≤89° 的角度
        sza_mask = sza_grid_full <= 89
        vza_mask = vza_grid_full <= 89
        sza_grid = sza_grid_full[sza_mask]
        vza_grid = vza_grid_full[vza_mask]

        # 重新建立索引映射（仅用于有效角度）
        sza_to_idx = {val: i for i, val in enumerate(sza_grid)}
        vza_to_idx = {val: i for i, val in enumerate(vza_grid)}

        # 创建网格坐标矩阵（用于绘图）
        X_idx, Y_idx = np.meshgrid(np.arange(len(sza_grid)), np.arange(len(vza_grid)))

        # 记录极端角度位置（用于背景高亮）
        extreme_sza_indices = np.where(sza_grid >= 85)[0]
        extreme_vza_indices = np.where(vza_grid >= 85)[0]

        # === 统计全局范围，用于统一色标（同一张图保持一致，便于比较）===
        all_vals = filtered_data[target_col].to_numpy()
        all_vals = all_vals[np.isfinite(all_vals)]
        if all_vals.size == 0:
            self.logger.error("没有有效的 ΔTOA 值，跳过该 RAA 绘图")
            return

        # 建议用对称范围，便于正负误差对比
        abs_max = float(np.nanmax(np.abs(all_vals)))
        if abs_max == 0:
            abs_max = 1e-12

        # === SymLog 的线性阈值：用于“放大”你关心的内部小变化 ===
        # 经验上取 1e-2 很适合你描述的 10e-3 量级；
        # 但也做了自适应下限，避免数据更小时过度放大噪声。
        positive = np.abs(all_vals[np.abs(all_vals) > 0])
        if positive.size > 0:
            # 取一个稳健尺度：中位数的 0.5 倍 与 1e-2 取更小者，但不低于 1e-4
            robust = float(np.nanmedian(positive)) * 0.5
            linthresh = max(1e-4, min(1e-2, robust))
        else:
            linthresh = 1e-2

        # 如果动态范围不大（例如 abs_max <= 3*linthresh），就用线性 TwoSlope 更直观
        use_symlog = abs_max > 3.0 * linthresh

        self.logger.info(
            f"色标设置: abs_max={abs_max:.3e}, linthresh={linthresh:.3e}, "
            f"mode={'symlog' if use_symlog else 'linear'}"
        )

        if use_symlog:
            norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=-abs_max, vmax=abs_max, base=10)
        else:
            # 小动态范围：线性且 0 为中心
            norm = TwoSlopeNorm(vcenter=0.0, vmin=-abs_max, vmax=abs_max)

        cmap = 'RdBu_r'  # 红正、蓝负、白色在 0 附近

        # === 为 contourf 构造更合理的 levels（线性+对数混合），避免 symlog 下等距 levels 不好看 ===
        def _build_symlog_levels(vmax_abs: float, lt: float, n_lin: int = 9, n_log: int = 12):
            vmax_abs = float(vmax_abs)
            lt = float(lt)
            if vmax_abs <= lt:
                return np.linspace(-vmax_abs, vmax_abs, 21)

            # 线性段：[-lt, lt]
            lin_levels = np.linspace(-lt, lt, n_lin)

            # 对数段：正、负对称
            log_max_exp = np.log10(vmax_abs)
            log_min_exp = np.log10(lt)
            pos = np.logspace(log_min_exp, log_max_exp, n_log)
            neg = -pos[::-1]

            levels = np.unique(np.concatenate([neg, lin_levels, pos]))
            levels.sort()
            return levels

        levels = _build_symlog_levels(abs_max, linthresh) if use_symlog else np.linspace(-abs_max, abs_max, 21)

        fig, axes = plt.subplots(3, 6, figsize=(24, 12), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

        for row_idx, condition_key in enumerate(atm_order):
            cond_params = self.atmospheric_conditions[condition_key]
            cond_data = filtered_data[
                (np.abs(filtered_data['aod550'] - cond_params['aod550']) < 0.001) &
                (np.abs(filtered_data['h2o'] - cond_params['h2o']) < 0.01)
                ]

            for col_idx, band in enumerate(bands):
                ax = axes[row_idx, col_idx]
                band_data = cond_data[cond_data['band'] == band].copy()

                if band_data.empty:
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=10)
                    ax.set_xlabel('SZA (°)', fontsize=9)
                    ax.set_ylabel('VZA (°)', fontsize=9)
                    continue

                # 构建矩阵（仅包含 ≤89° 的角度）
                matrix = np.full((len(vza_grid), len(sza_grid)), np.nan)

                # 将数据点四舍五入到最近的网格点
                band_data['sza_rounded'] = band_data['sza'].apply(
                    lambda x: min(sza_grid, key=lambda g: abs(g - x)) if x <= 89 else np.nan
                )
                band_data['vza_rounded'] = band_data['vza'].apply(
                    lambda x: min(vza_grid, key=lambda g: abs(g - x)) if x <= 89 else np.nan
                )
                band_data.dropna(subset=['sza_rounded', 'vza_rounded'], inplace=True)

                grouped = band_data.groupby(['vza_rounded', 'sza_rounded'])[target_col].median().reset_index()

                for _, row in grouped.iterrows():
                    sza_idx = sza_to_idx[row['sza_rounded']]
                    vza_idx = vza_to_idx[row['vza_rounded']]
                    matrix[vza_idx, sza_idx] = row[target_col]

                coverage = np.sum(~np.isnan(matrix)) / matrix.size * 100
                if coverage < 50:
                    self.logger.warning(f"波段 {band}, 条件 {condition_key}: 数据覆盖率仅 {coverage:.1f}%")

                if np.sum(~np.isnan(matrix)) > 0:
                    contour = ax.contourf(
                        X_idx, Y_idx, matrix,
                        levels=levels, cmap=cmap, norm=norm, extend='both'
                    )

                    # 叠加少量等值线（帮助读者在非线性映射下仍能读出结构）
                    # 只在变化不近似常数时绘制
                    if np.nanstd(matrix) > 0:
                        # 在 |Δ| <= linthresh 附近加密几条线，强调内部变化
                        if use_symlog:
                            fine = np.linspace(-linthresh, linthresh, 7)
                            fine = fine[np.abs(fine) > 0]  # 去掉 0，避免标签/线条重叠
                            ax.contour(X_idx, Y_idx, matrix, levels=fine, colors='k', linewidths=0.4, alpha=0.35)
                        else:
                            mid = np.linspace(-abs_max, abs_max, 6)[1:-1]
                            ax.contour(X_idx, Y_idx, matrix, levels=mid, colors='k', linewidths=0.5, alpha=0.4)

                    # 标记极端角度区域
                    for sza_i in extreme_sza_indices:
                        ax.axvline(x=sza_i, color='yellow', linestyle=':', linewidth=1.5, alpha=0.7)
                    for vza_i in extreme_vza_indices:
                        ax.axhline(y=vza_i, color='yellow', linestyle=':', linewidth=1.5, alpha=0.7)

                    if len(extreme_sza_indices) > 0:
                        ax.axvspan(extreme_sza_indices[0] - 0.5, extreme_sza_indices[-1] + 0.5,
                                   alpha=0.10, color='red')
                    if len(extreme_vza_indices) > 0:
                        ax.axhspan(extreme_vza_indices[0] - 0.5, extreme_vza_indices[-1] + 0.5,
                                   alpha=0.10, color='red')

                    # 数据点位置
                    ax.scatter(
                        [sza_to_idx[v] for v in band_data['sza_rounded']],
                        [vza_to_idx[v] for v in band_data['vza_rounded']],
                        s=10, color='black', alpha=0.25, marker='x'
                    )
                else:
                    ax.text(0.5, 0.5, "No Valid Data", ha='center', va='center', fontsize=10)

                # 标题与标签
                if row_idx == 0:
                    wl = ExperimentConfig.BANDS[band]['wavelength']
                    ax.set_title(f'Band {col_idx + 1} ({wl}µm)', fontweight='bold', fontsize=10)

                if col_idx == 0:
                    ax.text(-0.35, 0.5, cond_params['name'], transform=ax.transAxes,
                            rotation=90, va='center', fontweight='bold', fontsize=10)

                ax.set_xlabel('Solar Zenith Angle (°)', fontsize=9)
                ax.set_ylabel('View Zenith Angle (°)', fontsize=9)

                x_ticks = np.arange(len(sza_grid))
                x_labels = [f'{int(x)}' + ('*' if x >= 85 else '') for x in sza_grid]
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_labels, fontsize=7, rotation=45)

                y_ticks = np.arange(len(vza_grid))
                y_labels = [f'{int(y)}' + ('*' if y >= 85 else '') for y in vza_grid]
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_labels, fontsize=7)

                ax.set_xlim(-0.5, len(sza_grid) - 0.5)
                ax.set_ylim(-0.5, len(vza_grid) - 0.5)
                ax.grid(True, alpha=0.2)

        # === colorbar ===
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        def _build_symlog_ticks(vmax_abs: float, lt: float):
            vmax_abs = float(vmax_abs)
            lt = float(lt)
            if vmax_abs <= lt:
                return np.linspace(-vmax_abs, vmax_abs, 7)

            ticks_pos = []
            e_min = int(np.floor(np.log10(lt)))
            e_max = int(np.ceil(np.log10(vmax_abs)))
            for e in range(e_min, e_max + 1):
                for m in (1, 2, 5):
                    v = m * (10 ** e)
                    if lt <= v <= vmax_abs * 1.0000001:
                        ticks_pos.append(v)
            ticks_pos = np.unique(np.array(ticks_pos, dtype=float))
            ticks = np.concatenate([-ticks_pos[::-1], [0.0], ticks_pos])
            # 确保包含端点（视觉更稳）
            if ticks.size > 0:
                if ticks[0] > -vmax_abs:
                    ticks = np.insert(ticks, 0, -vmax_abs)
                if ticks[-1] < vmax_abs:
                    ticks = np.append(ticks, vmax_abs)
            return ticks

        if use_symlog:
            ticks = _build_symlog_ticks(abs_max, linthresh)
            cbar = fig.colorbar(sm, cax=cbar_ax, ticks=ticks)
            cbar.ax.set_yticklabels([f"{t:.0e}" if t != 0 else "0" for t in ticks])
            cbar.set_label(f'ΔTOA = ρ_TOA^SA - ρ_TOA^PPA (symlog, linthresh={linthresh:.0e})', fontsize=10)
        else:
            ticks = np.linspace(-abs_max, abs_max, 7)
            cbar = fig.colorbar(sm, cax=cbar_ax, ticks=ticks, format='%.2e')
            cbar.set_label('ΔTOA = ρ_TOA^SA - ρ_TOA^PPA (linear, centered at 0)', fontsize=10)

        cbar.ax.tick_params(labelsize=8)

        extreme_note = (
            "* denotes extreme angles (≥85°); yellow dotted lines and red shading highlight extreme angle regions. "
            "Color mapping uses symmetric scaling about 0; when dynamic range is large, symlog is used so both "
            "small interior variations and large extreme-angle excursions remain visible."
        )
        fig.text(0.5, 0.02, extreme_note, ha='center', fontsize=9, style='italic')

        param_text = f" (ρ={self.fixed_rho_true}, RAA={raa}°) - SZA/VZA ≤ 89°"
        fig.suptitle(f'ΔTOA Contour{param_text}', fontsize=14, fontweight='bold', y=0.98)

        output_filename = f"plot2_delta_toa_contour_RAA{int(raa):03d}{self.suffix}.png"
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Plot 2 (RAA={raa}) 已保存: {output_path}")
        self._output_contour_diagnostics(filtered_data, target_col)
        self._output_extreme_angle_statistics(filtered_data, target_col)

    def _validate_contour_data_quality(self, data: pd.DataFrame):
        """验证Contour图数据质量"""
        self.logger.info("验证Contour图数据质量...")
        if 'delta_toa' in data.columns:
            mean_val = data['delta_toa'].mean()
            std_val = data['delta_toa'].std()
            min_val = data['delta_toa'].min()
            max_val = data['delta_toa'].max()
            self.logger.info(f"  ΔTOA 全局统计: 均值={mean_val:.6f}, 标准差={std_val:.6f}, 范围=[{min_val:.6f}, {max_val:.6f}]")
            if max_val > 2.0 or min_val < -2.0:
                self.logger.warning("  ΔTOA 超出 [-2,2] 范围，数据加载时已截断，请留意。")

    def _output_contour_diagnostics(self, data: pd.DataFrame, target_col: str):
        """输出Contour图诊断信息"""
        self.logger.info("=== ΔTOA Contour图诊断信息 ===")
        bands = ['band1', 'band2', 'band3', 'band4', 'band5', 'band6']
        for band in bands:
            band_data = data[data['band'] == band]
            if not band_data.empty and target_col in band_data.columns:
                vals = band_data[target_col]
                self.logger.info(f"波段 {band}:")
                self.logger.info(f"  记录数: {len(band_data)}")
                self.logger.info(f"  均值: {vals.mean():.6f}")
                self.logger.info(f"  标准差: {vals.std():.6f}")
                self.logger.info(f"  范围: [{vals.min():.6f}, {vals.max():.6f}]")
                large = vals[np.abs(vals) > 1.8]  # 接近边界值的统计
                if len(large) > 0:
                    self.logger.warning(f"  发现 {len(large)} 个 |ΔTOA|>1.8 的值（接近截断边界）")

    def _output_extreme_angle_statistics(self, data: pd.DataFrame, target_col: str):
        """输出极端角度的统计信息"""
        self.logger.info("=== 极端角度统计信息 (SZA/VZA ≥ 85°) ===")
        bands = ['band1', 'band2', 'band3', 'band4', 'band5', 'band6']
        for band in bands:
            band_data = data[data['band'] == band]
            if not band_data.empty and target_col in band_data.columns:
                extreme_data = band_data[(band_data['sza'] >= 85) | (band_data['vza'] >= 85)]
                if not extreme_data.empty:
                    vals = extreme_data[target_col]
                    self.logger.info(f"波段 {band} 极端角度:")
                    self.logger.info(f"  记录数: {len(extreme_data)}")
                    self.logger.info(f"  均值: {vals.mean():.6f}")
                    self.logger.info(f"  标准差: {vals.std():.6f}")
                    self.logger.info(f"  范围: [{vals.min():.6f}, {vals.max():.6f}]")
                    for angle_type in ['sza', 'vza']:
                        for threshold in [85, 87, 89]:
                            subset = band_data[band_data[angle_type] >= threshold]
                            if not subset.empty:
                                sub_vals = subset[target_col]
                                self.logger.info(
                                    f"    {angle_type.upper()}≥{threshold}°: {len(subset)} 条记录, "
                                    f"均值={sub_vals.mean():.6f}, 极值=[{sub_vals.min():.6f}, {sub_vals.max():.6f}]")

    # =========================================================================
    # 闭合性误差分析图 - 改为 ΔTOA 分析（可选）
    # =========================================================================
    def plot_delta_toa_analysis(self):
        """绘制 ΔTOA 随角度变化的分析图"""
        data = self._load_validation_grid_data()
        if data.empty:
            self.logger.error("数据为空，无法进行分析")
            return

        filtered_data = data[
            (np.abs(data['rho_true'] - self.fixed_rho_true) < 0.001) &
            (np.abs(data['raa'] - self.fixed_raa) < 0.1)
        ].copy()

        if filtered_data.empty:
            self.logger.warning(f"没有找到 rho_true≈{self.fixed_rho_true}, raa≈{self.fixed_raa} 的数据，使用所有数据")
            filtered_data = data.copy()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        bands = ['band1', 'band2', 'band3', 'band4', 'band5', 'band6']

        for idx, band in enumerate(bands):
            if idx >= len(axes):
                break
            ax = axes[idx]
            band_data = filtered_data[filtered_data['band'] == band]

            if band_data.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax.set_xlabel('SZA (°)', fontsize=9)
                ax.set_ylabel('ΔTOA', fontsize=9)
                ax.set_title(f'{band} - ΔTOA vs SZA', fontsize=10)
                continue

            unique_sza = np.sort(band_data['sza'].unique())
            mean_delta = []
            std_delta = []

            for sza in unique_sza:
                sza_data = band_data[np.abs(band_data['sza'] - sza) < 2.0]
                if len(sza_data) > 0:
                    delta_vals = sza_data['delta_toa']
                    mean_delta.append(delta_vals.mean())
                    std_delta.append(delta_vals.std())
                else:
                    mean_delta.append(np.nan)
                    std_delta.append(np.nan)

            ax.errorbar(unique_sza, mean_delta, yerr=std_delta,
                        marker='o', linestyle='-', capsize=3, linewidth=1.5)

            extreme_mask = unique_sza >= 85
            if np.any(extreme_mask):
                ax.fill_between(unique_sza[extreme_mask],
                                np.array(mean_delta)[extreme_mask] - np.array(std_delta)[extreme_mask],
                                np.array(mean_delta)[extreme_mask] + np.array(std_delta)[extreme_mask],
                                alpha=0.3, color='red', label='Extreme SZA (≥85°)')

            ax.set_xlabel('Solar Zenith Angle (°)', fontsize=9)
            ax.set_ylabel('ΔTOA', fontsize=9)
            ax.set_title(f'{band} - ΔTOA vs SZA (ρ={self.fixed_rho_true})', fontsize=10)
            ax.grid(True, alpha=0.3)

            if np.any(extreme_mask):
                ax.legend(fontsize=8)

            # 添加统计信息
            mean_delta_all = band_data['delta_toa'].mean()
            std_delta_all = band_data['delta_toa'].std()
            ax.text(0.05, 0.95, f'Mean: {mean_delta_all:.6f}\nStd: {std_delta_all:.6f}',
                    transform=ax.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        output_filename = f"delta_toa_analysis{self.suffix}.png"
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"ΔTOA 分析图已保存: {output_path}")


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='Visualization Suite (新框架: ΔTOA)')
    parser.add_argument('--suffix', type=str, default='',
                        help='自定义文件后缀')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='自定义数据目录')
    parser.add_argument('--rho_true', type=float, default=0.3,
                        help='固定地表反射率值 (默认: 0.3)')
    parser.add_argument('--raa', type=float, default=0.0,
                        help='固定相对方位角 (默认: 0.0)')
    parser.add_argument('--run_all', action='store_true',
                        help='运行所有验证网格绘图')
    parser.add_argument('--run_delta_analysis', action='store_true',
                        help='运行 ΔTOA 分析图')

    args = parser.parse_args()

    visualizer = VisualizationSuite(
        data_dir=args.data_dir,
        suffix=args.suffix,
        fixed_rho_true=args.rho_true,
        fixed_raa=args.raa
    )

    if args.run_all:
        visualizer.run_all_plots()

    if args.run_delta_analysis:
        visualizer.plot_delta_toa_analysis()

    if not any([args.run_all, args.run_delta_analysis]):
        visualizer.run_all_plots()


if __name__ == "__main__":
    main()