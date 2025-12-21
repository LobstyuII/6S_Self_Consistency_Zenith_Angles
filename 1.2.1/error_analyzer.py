# ==================== error_analyzer.py ====================
"""
误差分析模块
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from pathlib import Path
from scipy import stats
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from config import ExperimentConfig
from utils import setup_logger, calculate_statistics
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False


class ErrorAnalyzer:
    """误差分析器"""

    def __init__(self, results: Dict[str, pd.DataFrame], logger=None):
        """
        初始化误差分析器

        Parameters:
        -----------
        results : dict
            各波段的模拟结果
        logger : logging.Logger, optional
            日志记录器
        """
        self.results = results
        self.logger = logger or setup_logger('ErrorAnalyzer')

        # 合并所有波段数据
        self.all_data = self._combine_bands()

        # 计算相对误差（如果不存在）
        if 'error_relative' not in self.all_data.columns and 'error_absolute' in self.all_data.columns and 'rho_true' in self.all_data.columns:
            self.all_data['error_relative'] = self.all_data['error_absolute'] / self.all_data['rho_true']
            self.logger.info("计算了相对误差 (error_absolute / rho_true)")

        # 计算统计量
        self.stats = self.calculate_overall_statistics()

    def _combine_bands(self) -> pd.DataFrame:
        """合并所有波段数据"""
        dfs = []
        for band_id, df in self.results.items():
            df_band = df.copy()
            df_band['band'] = band_id

            # 处理特殊情况：波段ID为'all'时，尝试从数据中获取波长
            if band_id == 'all' and 'wavelength' in df_band.columns:
                # 如果已经有波长信息，直接使用
                pass
            elif band_id in ExperimentConfig.BANDS:
                df_band['wavelength'] = ExperimentConfig.BANDS[band_id]['wavelength']
            else:
                # 对于未知波段，使用NaN或默认值
                df_band['wavelength'] = np.nan
                self.logger.warning(f"波段 {band_id} 不在配置中，波长设为NaN")

            dfs.append(df_band)

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()

    def calculate_overall_statistics(self) -> Dict[str, Dict[str, float]]:
        """计算总体统计量"""
        stats_dict = {}

        try:
            # 按波段计算
            for band_id in self.results.keys():
                df_band = self.results[band_id]

                # 检查是否有 error_absolute 列
                if 'error_absolute' in df_band.columns:
                    errors = df_band['error_absolute'].dropna().values

                    if len(errors) > 0:
                        try:
                            stats_dict[band_id] = calculate_statistics(errors, prefix='')
                        except:
                            stats_dict[band_id] = {'error': '计算统计量失败'}

            # 所有波段合并
            all_errors = []
            for df in self.results.values():
                if 'error_absolute' in df.columns:
                    errors = df['error_absolute'].dropna().values
                    all_errors.extend(errors.tolist() if hasattr(errors, 'tolist') else errors)

            all_errors = np.array(all_errors)
            if len(all_errors) > 0:
                try:
                    stats_dict['all'] = calculate_statistics(all_errors, prefix='')
                except:
                    stats_dict['all'] = {'error': '计算总统计量失败'}

        except Exception as e:
            self.logger.error(f"计算统计量失败: {e}")
            stats_dict['error'] = str(e)

        return stats_dict

    def plot_error_distribution(self, save_path: Optional[Path] = None):
        """绘制误差分布图"""
        # 检查是否有数据
        if self.all_data.empty:
            self.logger.warning("没有数据，无法绘制误差分布图")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        # 1. 绝对误差直方图
        if 'error_absolute' in self.all_data.columns:
            errors = self.all_data['error_absolute'].dropna()
            if len(errors) > 0:
                axes[0].hist(errors, bins=50, density=True, alpha=0.7, edgecolor='black')
                axes[0].set_xlabel('Absolute error')
                axes[0].set_ylabel('Probability density')
                axes[0].set_title('Absolute error distribution')
                axes[0].grid(True, alpha=0.3)

                # 添加统计信息
                stats_text = (
                    f"Mean: {errors.mean():.4f}\n"
                    f"Std: {errors.std():.4f}\n"
                    f"RMSE: {np.sqrt(np.mean(errors ** 2)):.4f}\n"
                    f"Samples: {len(errors):,}"
                )
                axes[0].text(0.05, 0.95, stats_text, transform=axes[0].transAxes,
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 2. 相对误差直方图
        if 'error_relative' in self.all_data.columns:
            rel_errors = self.all_data['error_relative'].dropna()
            if len(rel_errors) > 0:
                axes[1].hist(rel_errors, bins=50, density=True, alpha=0.7, edgecolor='black', color='green')
                axes[1].set_xlabel('Relative error')
                axes[1].set_ylabel('Probability density')
                axes[1].set_title('Relative error distribution')
                axes[1].grid(True, alpha=0.3)

                # 添加统计信息
                stats_text = (
                    f"Mean: {rel_errors.mean():.4f}\n"
                    f"Std: {rel_errors.std():.4f}\n"
                    f"Max: {rel_errors.max():.4f}\n"
                    f"Samples: {len(rel_errors):,}"
                )
                axes[1].text(0.05, 0.95, stats_text, transform=axes[1].transAxes,
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        # 3. 误差与SZA的关系
        if 'sza' in self.all_data.columns:
            # 绝对误差与SZA
            valid_data = self.all_data.dropna(subset=['error_absolute', 'sza'])
            if len(valid_data) > 0:
                axes[2].scatter(valid_data['sza'], valid_data['error_absolute'],
                                alpha=0.5, s=1, color='blue', label='Absolute')

            # 相对误差与SZA
            if 'error_relative' in self.all_data.columns:
                valid_data_rel = self.all_data.dropna(subset=['error_relative', 'sza'])
                if len(valid_data_rel) > 0:
                    axes[2].scatter(valid_data_rel['sza'], valid_data_rel['error_relative'],
                                    alpha=0.5, s=1, color='green', label='Relative')

            axes[2].set_xlabel('Solar zenith angle (deg)')
            axes[2].set_ylabel('Error')
            axes[2].set_title('Error vs solar zenith angle')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        # 4. 误差与VZA的关系
        if 'vza' in self.all_data.columns:
            # 绝对误差与VZA
            valid_data = self.all_data.dropna(subset=['error_absolute', 'vza'])
            if len(valid_data) > 0:
                axes[3].scatter(valid_data['vza'], valid_data['error_absolute'],
                                alpha=0.5, s=1, color='blue', label='Absolute')

            # 相对误差与VZA
            if 'error_relative' in self.all_data.columns:
                valid_data_rel = self.all_data.dropna(subset=['error_relative', 'vza'])
                if len(valid_data_rel) > 0:
                    axes[3].scatter(valid_data_rel['vza'], valid_data_rel['error_relative'],
                                    alpha=0.5, s=1, color='green', label='Relative')

            axes[3].set_xlabel('View zenith angle (deg)')
            axes[3].set_ylabel('Error')
            axes[3].set_title('Error vs view zenith angle')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"误差分布图已保存: {save_path}")

    def plot_layered_contours(self,
                              primary_factors: List[str] = ['sza', 'vza'],
                              color_factor: str = 'error_absolute',
                              fixed_conditions: Dict[str, Any] = None,
                              n_subplots: int = 4,
                              save_path: Optional[Path] = None):
        """
        绘制分层Contour图，展示多因素影响

        Parameters:
        -----------
        primary_factors : list
            主坐标轴因素，如['sza', 'vza']
        color_factor : str
            颜色映射因素，如'error_absolute', 'error_relative', 'aod550', 'rho_true'
        fixed_conditions : dict
            固定条件，如{'aod550': [0.1, 0.3, 0.5], 'rho_true': 0.2}
        n_subplots : int
            子图数量（根据可变因素数量）
        """
        if fixed_conditions is None:
            fixed_conditions = {'aod550': 0.3, 'rho_true': 0.2, 'band': 'band3'}

        # 创建子图网格
        fig, axes = plt.subplots(1, n_subplots, figsize=(5 * n_subplots, 4))
        if n_subplots == 1:
            axes = [axes]

        # 获取可变因素列表
        variable_factors = []
        for factor_name, factor_values in fixed_conditions.items():
            if isinstance(factor_values, (list, np.ndarray)) and len(factor_values) > 1:
                variable_factors.append((factor_name, factor_values))

        # 如果可变因素不足，使用固定值创建对比
        if len(variable_factors) < n_subplots:
            # 使用波段作为可变因素
            bands = list(self.results.keys())
            if len(bands) >= n_subplots:
                variable_factors = [('band', bands[:n_subplots])]

        # 绘制每个子图
        for i in range(n_subplots):
            if i < len(variable_factors):
                factor_name, factor_values = variable_factors[i]
                # 选择中间值作为代表
                if isinstance(factor_values, (list, np.ndarray)):
                    mid_value = factor_values[len(factor_values) // 2]
                else:
                    mid_value = factor_values

                # 筛选数据
                subset = self.all_data[self.all_data[factor_name] == mid_value].copy()
                ax = axes[i]

                # 调用内部绘图方法
                self._plot_single_contour(subset, ax, primary_factors, color_factor)
                ax.set_title(f'{factor_name} = {mid_value}', fontsize=10)
            else:
                axes[i].axis('off')

        plt.suptitle(f'Layered Contours: {color_factor} vs {primary_factors[0]} and {primary_factors[1]}',
                     fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"分层Contour图已保存: {save_path}")

    def _plot_single_contour(self, data: pd.DataFrame, ax,
                             primary_factors: List[str],
                             color_factor: str):
        """绘制单个Contour图（内部方法）"""
        if len(data) < 10:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            return

        # 创建网格 - 确保横纵坐标对齐，间隔5度
        x_unique = np.sort(data[primary_factors[0]].unique())
        y_unique = np.sort(data[primary_factors[1]].unique())

        # 如果数据不是5度间隔，重新采样到5度间隔
        if len(x_unique) > 0:
            x_min, x_max = np.floor(x_unique.min() / 5) * 5, np.ceil(x_unique.max() / 5) * 5
            x_grid_vals = np.arange(x_min, x_max + 5, 5)
        else:
            x_grid_vals = np.arange(0, 86, 5)

        if len(y_unique) > 0:
            y_min, y_max = np.floor(y_unique.min() / 5) * 5, np.ceil(y_unique.max() / 5) * 5
            y_grid_vals = np.arange(y_min, y_max + 5, 5)
        else:
            y_grid_vals = np.arange(0, 76, 5)

        # 创建规则网格
        x_grid, y_grid = np.meshgrid(x_grid_vals, y_grid_vals)
        z_grid = np.full_like(x_grid, np.nan)

        # 将数据插值到规则网格上
        from scipy.interpolate import griddata
        points = data[[primary_factors[0], primary_factors[1]]].values
        values = data[color_factor].values

        # 只使用有效数据点
        valid_mask = ~np.isnan(points).any(axis=1) & ~np.isnan(values)
        if np.sum(valid_mask) < 4:  # 至少需要4个点进行插值
            ax.text(0.5, 0.5, 'Insufficient data for interpolation', ha='center', va='center')
            return

        points_valid = points[valid_mask]
        values_valid = values[valid_mask]

        # 插值到规则网格
        z_grid = griddata(points_valid, values_valid, (x_grid, y_grid), method='linear')

        # 绘制等高线
        contour = ax.contourf(x_grid, y_grid, z_grid, levels=20, cmap='RdBu_r')
        ax.contour(x_grid, y_grid, z_grid, levels=10, colors='k', linewidths=0.5, alpha=0.5)

        ax.set_xlabel(f'{primary_factors[0]} (deg)')
        ax.set_ylabel(f'{primary_factors[1]} (deg)')

        # 设置坐标轴刻度为5度间隔
        ax.set_xticks(x_grid_vals)
        ax.set_yticks(y_grid_vals)

        # 设置纵横比相等，确保间隔长度相同
        ax.set_aspect('equal')

        # 确保x和y轴的范围相同（如果数据范围不同）
        x_range = x_grid_vals[-1] - x_grid_vals[0]
        y_range = y_grid_vals[-1] - y_grid_vals[0]
        max_range = max(x_range, y_range)

        if x_range < max_range:
            padding = (max_range - x_range) / 2
            ax.set_xlim(x_grid_vals[0] - padding, x_grid_vals[-1] + padding)
        if y_range < max_range:
            padding = (max_range - y_range) / 2
            ax.set_ylim(y_grid_vals[0] - padding, y_grid_vals[-1] + padding)

        # 添加颜色条
        cbar = plt.colorbar(contour, ax=ax, label=color_factor)
        ax.grid(True, alpha=0.3)

    def plot_3d_error_surface(self,
                              fixed_conditions: Dict[str, float] = None,
                              save_path: Optional[Path] = None):
        """
        绘制3D误差曲面及其2D投影
        """
        # 筛选固定条件下的数据
        if fixed_conditions is None:
            fixed_conditions = {'aod550': 0.3, 'rho_true': 0.2, 'band': 'band3'}

        mask = pd.Series(True, index=self.all_data.index)
        for key, value in fixed_conditions.items():
            if key in self.all_data.columns:
                mask = mask & (self.all_data[key] == value)

        data = self.all_data[mask]

        if len(data) < 10:
            self.logger.warning("数据不足，无法绘制3D曲面")
            return

        # 创建3D图形
        fig = plt.figure(figsize=(15, 5))

        # 子图1：3D曲面 - 绝对误差
        ax1 = fig.add_subplot(131, projection='3d')

        # 三角化曲面
        ax1.plot_trisurf(data['sza'], data['vza'], data['error_absolute'],
                         cmap='viridis', alpha=0.8, edgecolor='none')
        ax1.set_xlabel('SZA (deg)')
        ax1.set_ylabel('VZA (deg)')
        ax1.set_zlabel('Absolute Error')
        ax1.set_title('3D Absolute Error Surface')

        # 子图2：XY投影（Contour）- 绝对误差
        ax2 = fig.add_subplot(132)

        # 使用tricontourf处理不规则网格
        contour = ax2.tricontourf(data['sza'], data['vza'], data['error_absolute'],
                                  levels=20, cmap='RdBu_r')
        plt.colorbar(contour, ax=ax2, label='Absolute Error')
        ax2.set_xlabel('SZA (deg)')
        ax2.set_ylabel('VZA (deg)')
        ax2.set_title('2D Absolute Error Contour')
        ax2.grid(True, alpha=0.3)

        # 子图3：相对误差分布直方图
        ax3 = fig.add_subplot(133)
        if 'error_relative' in data.columns:
            rel_errors = data['error_relative'].dropna()
            if len(rel_errors) > 0:
                ax3.hist(rel_errors, bins=50, density=True,
                         alpha=0.7, edgecolor='black', color='green')
                ax3.set_xlabel('Relative Error')
                ax3.set_ylabel('Density')
                ax3.set_title('Relative Error Distribution')
                ax3.grid(True, alpha=0.3)

                # 添加正态分布拟合
                from scipy.stats import norm
                mu, std = norm.fit(rel_errors)
                x = np.linspace(rel_errors.min(), rel_errors.max(), 100)
                p = norm.pdf(x, mu, std)
                ax3.plot(x, p, 'r-', linewidth=2, label=f'N({mu:.4f}, {std:.4f}²)')
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'No relative error data', ha='center', va='center')
                ax3.set_title('Relative Error Distribution')
        else:
            ax3.text(0.5, 0.5, 'No relative error column', ha='center', va='center')
            ax3.set_title('Relative Error Distribution')

        conditions_str = ', '.join([f'{k}={v}' for k, v in fixed_conditions.items()])
        plt.suptitle(f'3D Error Analysis - Fixed: {conditions_str}', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"3D误差曲面图已保存: {save_path}")

    def create_summary_figure(self,
                              fixed_conditions: Dict[str, float] = None,
                              band_id: str = 'band3',
                              save_path: Optional[Path] = None):
        """
        创建综合摘要图

        修改：删除原来的Error distribution小图，增加相对误差分析
        """
        if fixed_conditions is None:
            fixed_conditions = {'aod550': 0.3, 'rho_true': 0.2}

        # 根据band_id筛选数据
        if band_id == 'all':
            data = self.all_data.copy()
        elif band_id in self.results:
            data = self.results[band_id].copy()
        else:
            self.logger.warning(f"波段 {band_id} 不在结果中")
            return

        # 应用固定条件
        for key, value in fixed_conditions.items():
            if key in data.columns:
                data = data[data[key] == value]

        # 检查是否有误差数据
        if 'error_absolute' not in data.columns or data['error_absolute'].dropna().empty:
            self.logger.warning(f"波段 {band_id} 没有误差数据")
            return

        # 创建2x3的网格布局（6个子图）
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. 绝对误差等高线图
        ax1 = fig.add_subplot(gs[0, 0])

        if len(data) > 10:
            self._plot_single_contour(data, ax1, ['sza', 'vza'], 'error_absolute')
            ax1.set_title('(a) Absolute Error Contour', fontsize=12)
        else:
            ax1.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')

        # 2. 相对误差等高线图
        ax2 = fig.add_subplot(gs[0, 1])

        if len(data) > 10 and 'error_relative' in data.columns:
            self._plot_single_contour(data, ax2, ['sza', 'vza'], 'error_relative')
            ax2.set_title('(b) Relative Error Contour', fontsize=12)
        else:
            ax2.text(0.5, 0.5, 'No relative error data', ha='center', va='center')

        # 3. 3D绝对误差曲面
        ax3 = fig.add_subplot(gs[0, 2], projection='3d')
        if len(data) > 10:
            ax3.plot_trisurf(data['sza'], data['vza'], data['error_absolute'],
                             cmap='viridis', alpha=0.8, edgecolor='none')
            ax3.set_xlabel('SZA')
            ax3.set_ylabel('VZA')
            ax3.set_zlabel('Abs Error')
            ax3.set_title('(c) 3D Absolute Error Surface', fontsize=12)
        else:
            ax3.text(0.5, 0.5, 0.5, 'Insufficient data', ha='center', va='center')

        # 4. SZA与误差关系（绝对和相对）
        ax4 = fig.add_subplot(gs[1, 0])
        if 'sza' in data.columns:
            # 绝对误差与SZA
            valid_data = data.dropna(subset=['error_absolute', 'sza'])
            if len(valid_data) > 0:
                ax4.scatter(valid_data['sza'], valid_data['error_absolute'],
                            alpha=0.5, s=5, color='blue', label='Absolute')

                # 添加趋势线
                x = valid_data['sza'].values
                y = valid_data['error_absolute'].values
                coeffs = np.polyfit(x, y, 1)
                poly = np.poly1d(coeffs)
                x_fit = np.linspace(x.min(), x.max(), 100)
                ax4.plot(x_fit, poly(x_fit), 'b-', linewidth=2, alpha=0.8)

            # 相对误差与SZA
            if 'error_relative' in data.columns:
                valid_data_rel = data.dropna(subset=['error_relative', 'sza'])
                if len(valid_data_rel) > 0:
                    ax4.scatter(valid_data_rel['sza'], valid_data_rel['error_relative'],
                                alpha=0.5, s=5, color='green', label='Relative')

                    # 添加趋势线
                    x_rel = valid_data_rel['sza'].values
                    y_rel = valid_data_rel['error_relative'].values
                    coeffs_rel = np.polyfit(x_rel, y_rel, 1)
                    poly_rel = np.poly1d(coeffs_rel)
                    ax4.plot(x_fit, poly_rel(x_fit), 'g-', linewidth=2, alpha=0.8)

            ax4.set_xlabel('SZA (deg)')
            ax4.set_ylabel('Error')
            ax4.set_title('(d) Error vs SZA', fontsize=12)
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. VZA与误差关系（绝对和相对）
        ax5 = fig.add_subplot(gs[1, 1])
        if 'vza' in data.columns:
            # 绝对误差与VZA
            valid_data = data.dropna(subset=['error_absolute', 'vza'])
            if len(valid_data) > 0:
                ax5.scatter(valid_data['vza'], valid_data['error_absolute'],
                            alpha=0.5, s=5, color='blue', label='Absolute')

                # 添加趋势线
                x = valid_data['vza'].values
                y = valid_data['error_absolute'].values
                coeffs = np.polyfit(x, y, 1)
                poly = np.poly1d(coeffs)
                x_fit = np.linspace(x.min(), x.max(), 100)
                ax5.plot(x_fit, poly(x_fit), 'b-', linewidth=2, alpha=0.8)

            # 相对误差与VZA
            if 'error_relative' in data.columns:
                valid_data_rel = data.dropna(subset=['error_relative', 'vza'])
                if len(valid_data_rel) > 0:
                    ax5.scatter(valid_data_rel['vza'], valid_data_rel['error_relative'],
                                alpha=0.5, s=5, color='green', label='Relative')

                    # 添加趋势线
                    x_rel = valid_data_rel['vza'].values
                    y_rel = valid_data_rel['error_relative'].values
                    coeffs_rel = np.polyfit(x_rel, y_rel, 1)
                    poly_rel = np.poly1d(coeffs_rel)
                    ax5.plot(x_fit, poly_rel(x_fit), 'g-', linewidth=2, alpha=0.8)

            ax5.set_xlabel('VZA (deg)')
            ax5.set_ylabel('Error')
            ax5.set_title('(e) Error vs VZA', fontsize=12)
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 6. 相对误差分布直方图
        ax6 = fig.add_subplot(gs[1, 2])
        if 'error_relative' in data.columns:
            rel_errors = data['error_relative'].dropna()
            if len(rel_errors) > 0:
                ax6.hist(rel_errors, bins=50, density=True,
                         alpha=0.7, edgecolor='black', color='green')
                ax6.set_xlabel('Relative Error')
                ax6.set_ylabel('Density')
                ax6.set_title('(f) Relative Error Distribution', fontsize=12)
                ax6.grid(True, alpha=0.3)

                # 添加统计信息
                stats_text = (
                    f"Mean: {rel_errors.mean():.4f}\n"
                    f"Std: {rel_errors.std():.4f}\n"
                    f"Max: {rel_errors.max():.4f}\n"
                    f"N: {len(rel_errors):,}"
                )
                ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            else:
                ax6.text(0.5, 0.5, 'No relative error data', ha='center', va='center')
                ax6.set_title('(f) Relative Error Distribution', fontsize=12)
        else:
            ax6.text(0.5, 0.5, 'No relative error column', ha='center', va='center')
            ax6.set_title('(f) Relative Error Distribution', fontsize=12)

        conditions_str = ', '.join([f'{k}={v}' for k, v in fixed_conditions.items()])
        plt.suptitle(f'Summary of Geometric Error Analysis - Band: {band_id}\nFixed Conditions: {conditions_str}',
                     fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"综合摘要图已保存: {save_path}")

    def _plot_simple_scatter_with_trend(self, data: pd.DataFrame,
                                        factor: str, ax, error_type='absolute'):
        """绘制简单的散点图与趋势线（内部方法）"""
        if error_type == 'absolute':
            error_col = 'error_absolute'
            color = 'blue'
        else:
            error_col = 'error_relative'
            color = 'green'

        ax.scatter(data[factor], data[error_col],
                   alpha=0.3, s=10, color=color)

        # 添加趋势线
        valid_mask = ~np.isnan(data[factor]) & ~np.isnan(data[error_col])
        if valid_mask.sum() > 2:
            x = data.loc[valid_mask, factor].values
            y = data.loc[valid_mask, error_col].values

            # 线性回归
            coeffs = np.polyfit(x, y, 1)
            poly = np.poly1d(coeffs)
            x_range = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_range, poly(x_range), 'r-', lw=2,
                    label=f'y={coeffs[0]:.4f}x+{coeffs[1]:.4f}')

            # 计算R²
            y_pred = poly(x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel(factor)
        ax.set_ylabel(f'{error_type.capitalize()} Error')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_feature_importance(self, data: pd.DataFrame, ax):
        """绘制特征重要性图（内部方法）"""
        features = ['sza', 'vza', 'aod550', 'rho_true']
        features = [f for f in features if f in data.columns]

        X = data[features].values
        y = data['error_absolute'].values

        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) > 10:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1]

            ax.bar(range(len(features)), importance[indices], align='center')
            ax.set_xticks(range(len(features)))
            ax.set_xticklabels([features[i] for i in indices], rotation=45)
            ax.set_ylabel('Feature Importance')
            ax.set_title('Random Forest Feature Importance')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')