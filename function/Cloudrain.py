import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt
from scipy.stats import ttest_ind
from scipy.stats import gaussian_kde
import matplotlib.ticker as ticker
from .haxgrid import calculate_optimal_gridsize


def aggregate_data_like_hexmap(df, variables, bounds=(-125, -65, 24, 51), 
                              gridsize=None, mincnt=2):
    """
    使用与蜂窝图相同的聚合方法对数据进行聚合
    
    Parameters:
    -----------
    df : DataFrame
        包含lon, lat和目标变量的数据
    variables : list
        需要聚合的变量列表
    bounds : tuple
        (xmin, xmax, ymin, ymax) 地图边界
    gridsize : int, optional
        网格大小，如果None则自动计算
    mincnt : int
        每个网格最少点数，默认2（与蜂窝图一致）
    
    Returns:
    --------
    aggregated_df : DataFrame
        聚合后的数据
    """
    xmin, xmax, ymin, ymax = bounds
    
    # 1. 使用相同的数据过滤条件
    data = df[
        (df['lon'] >= xmin) & (df['lon'] <= xmax) & 
        (df['lat'] >= ymin) & (df['lat'] <= ymax)
    ].copy()
    
    # 去除所有目标变量中有缺失值的行
    valid_vars = [v for v in variables if v in data.columns]
    data = data[['lon', 'lat'] + valid_vars].dropna()
    
    if len(data) == 0:
        print("No valid data after filtering")
        return pd.DataFrame()
    
    print(f"Original data points: {len(data):,}")
    
    # 2. 使用相同的gridsize计算方法
    if gridsize is None:
        gridsize = calculate_optimal_gridsize(data, bounds)
    
    # 3. 创建六边形网格边界
    def create_hex_grid(data, bounds, gridsize):
        """创建六边形网格并聚合数据"""
        xmin, xmax, ymin, ymax = bounds
        
        # 计算网格参数
        x_range = xmax - xmin
        y_range = ymax - ymin
        
        # 六边形网格的行列数
        nx = int(gridsize)
        ny = int(gridsize * y_range / x_range)  # 保持比例
        
        # 创建网格边界
        x_edges = np.linspace(xmin, xmax, nx + 1)
        y_edges = np.linspace(ymin, ymax, ny + 1)
        
        # 为每个变量创建聚合数据
        aggregated_data = []
        
        for i in range(nx):
            for j in range(ny):
                # 定义网格边界
                x_left, x_right = x_edges[i], x_edges[i + 1]
                y_bottom, y_top = y_edges[j], y_edges[j + 1]
                
                # 找到落在这个网格内的数据点
                mask = (
                    (data['lon'] >= x_left) & (data['lon'] < x_right) &
                    (data['lat'] >= y_bottom) & (data['lat'] < y_top)
                )
                
                grid_data = data[mask]
                
                # 如果该网格内的点数少于mincnt，则跳过
                if len(grid_data) < mincnt:
                    continue
                
                # 计算网格中心点
                center_x = (x_left + x_right) / 2
                center_y = (y_bottom + y_top) / 2
                
                # 为每个变量计算聚合值（使用均值）
                grid_row = {
                    'lon': center_x,
                    'lat': center_y,
                    'point_count': len(grid_data)
                }
                
                for var in valid_vars:
                    if var in grid_data.columns:
                        grid_row[var] = grid_data[var].mean()
                
                aggregated_data.append(grid_row)
        
        return pd.DataFrame(aggregated_data)
    
    # 4. 执行聚合
    aggregated_df = create_hex_grid(data, bounds, gridsize)
    
    print(f"Aggregated data points: {len(aggregated_df):,}")
    print(f"Aggregation ratio: {len(aggregated_df)/len(data):.3f}")
    
    return aggregated_df


def plot_cloudrain_distribution(aggregated_data, vars_primary, var_secondary):

    all_vars = vars_primary + [var_secondary]
    
    # 对每个变量进行99%分位数截断
    for var in all_vars:
        q99 = aggregated_data[var].quantile(0.99)
        aggregated_data[var] = aggregated_data[var].clip(upper=q99)

    # 转换为长格式
    df_long = aggregated_data[all_vars].melt(var_name="Clade", value_name="value")

    # 配色方案
    palette = {
        'final_forest': '#84a354',
        'final_agro': '#b15053', 
        'final_veg': '#cf9a2c',
        'pv_potential_dens': '#2d5016'  
    }

    # 创建双子图布局，减小间距
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), 
                                   gridspec_kw={'width_ratios': [2.5, 1], 'wspace': 0.15})
    # ==== 左图：前三个变量 ====
    sub_primary = df_long[df_long['Clade'].isin(vars_primary)]

    # 优化的布局参数 - 更紧凑的设置
    offset = 0.15
    box_width = 0.1
    violin_width = 0.5
    jitter_width = 0.05

    # 1. 优化Y轴范围
    y_min = sub_primary['value'].min()
    y_max = sub_primary['value'].max()
    y_range = y_max - y_min
    ax1.set_ylim(y_min - 0.05 * y_range, y_max + 0.1 * y_range)

    # 2. 添加坐标轴线
    ax1.spines['left'].set_visible(True)
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['left'].set_linewidth(1.2)
    ax1.spines['bottom'].set_linewidth(1.2)
    ax1.spines['left'].set_color('black')
    ax1.spines['bottom'].set_color('black')

    # 3. 半小提琴图
    pt.half_violinplot(
        x='Clade', y='value', data=sub_primary,
        order=vars_primary,
        palette=[palette[v] for v in vars_primary],
        bw=.2, cut=0, scale="area", 
        width=violin_width,
        inner=None, orient="v", ax=ax1,
        offset=offset  
    )

    # 4. 箱线图
    data_list = [sub_primary[sub_primary['Clade'] == v]['value'] for v in vars_primary]
    positions = np.arange(len(vars_primary)) + offset
    bp = ax1.boxplot(
        data_list,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        medianprops={'linewidth': 2.5, 'color': 'black'},
        whiskerprops={'linewidth': 1.8, 'color': 'black'},
        capprops={'linewidth': 1.8, 'color': 'black'},
        boxprops={'linewidth': 1.2},
        zorder=5
    )
    for patch in bp['boxes']:
        patch.set(facecolor='white', alpha=0.9, edgecolor='black')

    # 5. 修复的密度颜色函数
    def get_density_colors(data, base_color, n_levels=5):
        if len(data) < 2:
            from matplotlib.colors import to_rgba
            return [to_rgba(base_color)] * len(data)
        
        kde = gaussian_kde(data)
        density = kde(data)
        density_percentiles = np.percentile(density, np.linspace(0, 100, n_levels+1))
        
        from matplotlib.colors import to_rgba
        base_rgba = to_rgba(base_color)
        colors = []
        
        for d in density:
            level = np.digitize(d, density_percentiles) - 1
            level = max(0, min(n_levels-1, level))
            alpha = 0.3 + 0.7 * (level / max(1, n_levels-1))
            alpha = max(0.0, min(1.0, alpha))
            color_rgba = (base_rgba[0], base_rgba[1], base_rgba[2], alpha)
            colors.append(color_rgba)
        
        return colors

    # 6. 抖动点图
    for i, var in enumerate(vars_primary):
        vals = sub_primary[sub_primary['Clade'] == var]['value']
        if len(vals) == 0:
            continue
        
        colors = get_density_colors(vals.values, palette[var])
        x_jitter = np.random.normal(i + offset, jitter_width, size=len(vals))
        
        ax1.scatter(
            x_jitter, vals,
            c=colors, s=20,
            edgecolors='white', linewidth=0.2,
            zorder=3
        )
    # 7. 修复X轴设置
    ax1.set_xticks(np.arange(len(vars_primary)) + offset)
    ax1.set_xticklabels([v.replace('final_', '').capitalize() for v in vars_primary],
                        fontsize=12, fontweight='bold')
    
    # 8. 左图美化
    ax1.set_ylabel("Biomass Value (Mg/ha)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("")
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    sns.despine(ax=ax1, trim=True)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax1.tick_params(axis='y', labelsize=11, length=4, width=1)
    ax1.tick_params(axis='x', labelsize=11, length=4, width=1)

    # ==== 右图：pv_potential_dens ====
    sub_secondary = df_long[df_long['Clade']==var_secondary]

    # 同样优化右图的Y轴范围
    pv_min = sub_secondary['value'].min()
    pv_max = sub_secondary['value'].max()
    pv_range = pv_max - pv_min
    ax2.set_ylim(pv_min - 0.05 * pv_range, pv_max + 0.1 * pv_range)

    # 添加坐标轴线
    ax2.spines['left'].set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    ax2.spines['left'].set_linewidth(1.2)
    ax2.spines['bottom'].set_linewidth(1.2)
    ax2.spines['left'].set_color('black')
    ax2.spines['bottom'].set_color('black')

    # 半小提琴图
    pt.half_violinplot(
        x='Clade', y='value', data=sub_secondary,
        order=[var_secondary], palette=[palette[var_secondary]],
        bw=.2, cut=0, scale="area", 
        width=violin_width,
        inner=None, orient="v", ax=ax2, 
        offset=offset
    )

    # 箱线图
    pv_data_list = [sub_secondary['value']]
    bp2 = ax2.boxplot(
        pv_data_list,
        positions=[offset],
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        medianprops={'linewidth': 2.5, 'color': 'black'},
        whiskerprops={'linewidth': 1.8, 'color': 'black'},
        capprops={'linewidth': 1.8, 'color': 'black'},
        boxprops={'linewidth': 1.2},
        zorder=5
    )
    for patch in bp2['boxes']:
        patch.set(facecolor='white', alpha=0.9, edgecolor='black')

    # 抖动点图
    pv_data = sub_secondary['value']
    if len(pv_data) > 0:
        density_colors = get_density_colors(pv_data.values, palette[var_secondary])
        x_positions = np.random.normal(offset, jitter_width, len(pv_data))
        ax2.scatter(x_positions, pv_data, 
                   c=density_colors, s=20,
                   edgecolors='white', linewidth=0.2,
                   zorder=3)

    # 修复右图X轴设置
    ax2.set_xticks([offset])
    ax2.set_xticklabels(['PV Potential\nDensity'], fontsize=12, fontweight='bold')
    
    # 右图美化
    ax2.set_ylabel("PV Density (kWh/m²/year)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("")
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    sns.despine(ax=ax2, trim=True)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax2.tick_params(axis='y', labelsize=11, length=4, width=1)
    ax2.tick_params(axis='x', labelsize=11, length=4, width=1)

    # 整体布局调整
    plt.tight_layout()
    plt.savefig("figure/Figure2_Emission_optimized.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig