# ==================== plotly_utils.py ====================
"""
交互式可视化工具模块
"""
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path


def create_interactive_contour(data: pd.DataFrame,
                               x_col: str = 'sza',
                               y_col: str = 'vza',
                               z_col: str = 'error_absolute',
                               fixed_params: Dict[str, Any] = None,
                               title: str = None) -> go.Figure:
    """
    创建交互式等高线图（Plotly）

    Parameters:
    -----------
    data : pd.DataFrame
        输入数据
    x_col, y_col, z_col : str
        坐标轴和颜色映射列
    fixed_params : dict
        固定参数条件
    title : str
        图表标题

    Returns:
    --------
    go.Figure
        Plotly图表对象
    """
    # 筛选数据
    plot_data = data.copy()
    if fixed_params:
        for key, value in fixed_params.items():
            if key in plot_data.columns:
                if isinstance(value, (list, np.ndarray)):
                    plot_data = plot_data[plot_data[key].isin(value)]
                else:
                    plot_data = plot_data[plot_data[key] == value]

    if len(plot_data) < 10:
        print("数据不足，无法创建等高线图")
        return None

    # 创建网格数据
    x_unique = np.sort(plot_data[x_col].unique())
    y_unique = np.sort(plot_data[y_col].unique())

    # 插值到规则网格
    x_grid, y_grid = np.meshgrid(x_unique, y_unique)
    z_grid = np.full_like(x_grid, np.nan)

    # 计算平均值
    for i, x_val in enumerate(x_unique):
        for j, y_val in enumerate(y_unique):
            mask = (plot_data[x_col] == x_val) & (plot_data[y_col] == y_val)
            if mask.any():
                z_grid[j, i] = plot_data.loc[mask, z_col].mean()

    # 创建等高线图
    fig = go.Figure(data=go.Contour(
        z=z_grid,
        x=x_unique,
        y=y_unique,
        colorscale='RdBu_r',
        contours=dict(
            showlabels=True,
            labelfont=dict(size=12, color='white')
        ),
        colorbar=dict(
            title=z_col.replace('_', ' ').title(),
            titleside='right'
        ),
        hovertemplate=(
            f"{x_col}: %{{x:.1f}}<br>"
            f"{y_col}: %{{y:.1f}}<br>"
            f"{z_col}: %{{z:.4f}}<br>"
            "<extra></extra>"
        )
    ))

    if title is None:
        title = f'{z_col} vs {x_col} and {y_col}'

    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        template='plotly_white',
        width=800,
        height=600
    )

    return fig


def create_interactive_3d_surface(data: pd.DataFrame,
                                  x_col: str = 'sza',
                                  y_col: str = 'vza',
                                  z_col: str = 'error_absolute',
                                  fixed_params: Dict[str, Any] = None) -> go.Figure:
    """
    创建交互式3D曲面图
    """
    plot_data = data.copy()
    if fixed_params:
        for key, value in fixed_params.items():
            if key in plot_data.columns:
                plot_data = plot_data[plot_data[key] == value]

    if len(plot_data) < 10:
        print("数据不足，无法创建3D曲面图")
        return None

    fig = go.Figure(data=[go.Mesh3d(
        x=plot_data[x_col],
        y=plot_data[y_col],
        z=plot_data[z_col],
        opacity=0.8,
        colorscale='Viridis',
        intensity=plot_data[z_col],
        hovertemplate=(
            f"{x_col}: %{{x:.1f}}<br>"
            f"{y_col}: %{{y:.1f}}<br>"
            f"{z_col}: %{{z:.4f}}<br>"
            "<extra></extra>"
        )
    )])

    fig.update_layout(
        title=f'3D Surface: {z_col} vs {x_col} and {y_col}',
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ),
        template='plotly_white',
        width=800,
        height=600
    )

    return fig


def create_interactive_scatter_matrix(data: pd.DataFrame,
                                      features: List[str] = None,
                                      color_col: str = 'error_absolute',
                                      fixed_params: Dict[str, Any] = None) -> go.Figure:
    """
    创建交互式散点矩阵图
    """
    plot_data = data.copy()
    if fixed_params:
        for key, value in fixed_params.items():
            if key in plot_data.columns:
                plot_data = plot_data[plot_data[key] == value]

    if features is None:
        features = ['sza', 'vza', 'aod550', 'rho_true', 'error_absolute']
        features = [f for f in features if f in plot_data.columns]

    fig = px.scatter_matrix(
        plot_data,
        dimensions=features,
        color=color_col if color_col in plot_data.columns else None,
        title="Scatter Matrix of Key Factors",
        template='plotly_white',
        width=1000,
        height=800
    )

    fig.update_traces(diagonal_visible=False)

    return fig


def save_interactive_plot(fig: go.Figure, save_path: Path):
    """
    保存交互式图表
    """
    if fig is not None:
        fig.write_html(save_path)
        print(f"交互式图表已保存: {save_path}")
    else:
        print("无法保存，图表为空")


# 示例使用函数
def create_all_interactive_plots(data: pd.DataFrame,
                                 output_dir: Path,
                                 band_id: str = 'band3'):
    """
    创建所有交互式图表
    """
    output_dir.mkdir(exist_ok=True)

    # 1. 等高线图
    fig_contour = create_interactive_contour(
        data,
        x_col='sza',
        y_col='vza',
        z_col='error_absolute',
        fixed_params={'band': band_id, 'aod550': 0.3, 'rho_true': 0.2}
    )
    if fig_contour:
        save_interactive_plot(fig_contour, output_dir / f"contour_{band_id}.html")

    # 2. 3D曲面图
    fig_3d = create_interactive_3d_surface(
        data,
        x_col='sza',
        y_col='vza',
        z_col='error_absolute',
        fixed_params={'band': band_id, 'aod550': 0.3, 'rho_true': 0.2}
    )
    if fig_3d:
        save_interactive_plot(fig_3d, output_dir / f"3d_surface_{band_id}.html")

    # 3. 散点矩阵图
    fig_matrix = create_interactive_scatter_matrix(
        data,
        features=['sza', 'vza', 'aod550', 'rho_true', 'error_absolute'],
        color_col='error_absolute',
        fixed_params={'band': band_id}
    )
    if fig_matrix:
        save_interactive_plot(fig_matrix, output_dir / f"scatter_matrix_{band_id}.html")

    return {
        'contour': fig_contour,
        '3d_surface': fig_3d,
        'scatter_matrix': fig_matrix
    }
# ==================== plotly_utils.py 结束 ====================