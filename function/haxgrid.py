import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, box
import numpy as np
from tqdm import tqdm   
from scipy.spatial import cKDTree, ConvexHull
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon


def calculate_optimal_gridsize(data, bounds, target_hex_per_degree=3):
    """
    动态计算最优六边形网格大小 - 智能增强版
    
    Parameters:
        data: DataFrame, 包含经纬度数据
        bounds: tuple, (xmin, xmax, ymin, ymax) 地图边界
        target_hex_per_degree: float, 目标每度经纬度的六边形数量
    
    Returns:
        optimal_gridsize: int, 最优网格大小
    """
    xmin, xmax, ymin, ymax = bounds
    
    # 1. 计算地理范围
    geo_width = xmax - xmin   # 经度跨度
    geo_height = ymax - ymin  # 纬度跨度
    geo_area = geo_width * geo_height
    
    # 2. 数据密度分析
    data_count = len(data)
    data_density = data_count / geo_area  # 每平方度的数据点数
    
    # 3. 地理形状因子
    shape_ratio = geo_width / geo_height
    if shape_ratio > 2.5:  # 非常宽的地图
        shape_factor = 1.15
    elif shape_ratio > 1.8:  # 较宽的地图
        shape_factor = 1.1
    elif shape_ratio < 0.6:  # 较窄的地图
        shape_factor = 0.9
    else:  # 近似方形
        shape_factor = 1.0
    
    # 4. 基础网格大小计算
    base_gridsize = int(np.sqrt(geo_area) * target_hex_per_degree * shape_factor)
    
    # 5. 根据数据密度调整
    if data_density > 80:      # 超高密度
        density_factor = 1.4
    elif data_density > 50:    # 高密度
        density_factor = 1.25
    elif data_density > 30:    # 中高密度
        density_factor = 1.15
    elif data_density > 15:    # 中等密度  
        density_factor = 1.05
    elif data_density > 5:     # 低密度
        density_factor = 0.9
    else:                      # 极低密度
        density_factor = 0.75
    
    # 6. 根据数据总量调整
    if data_count > 100000:    # 超大数据集
        count_factor = 1.3
    elif data_count > 50000:   # 大数据集
        count_factor = 1.2
    elif data_count > 25000:   # 中大数据集
        count_factor = 1.1
    elif data_count > 10000:   # 中等数据集
        count_factor = 1.0  
    elif data_count > 1000:    # 小数据集
        count_factor = 0.85
    else:                      # 微小数据集
        count_factor = 0.7
    
    # 7. 综合计算最优gridsize
    optimal_gridsize = int(base_gridsize * density_factor * count_factor)
    
    # 8. 稀疏度检测
    def check_sparsity_enhanced(data, bounds, test_gridsize):
        """增强的蜂窝稀疏度检测"""
        xmin, xmax, ymin, ymax = bounds
        
        # 使用更精细的网格进行检测
        grid_size = max(12, int(test_gridsize / 2.5))  # 更精细的网格
        x_edges = np.linspace(xmin, xmax, grid_size + 1)
        y_edges = np.linspace(ymin, ymax, grid_size + 1)
        
        occupied_cells = 0
        cell_densities = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                mask = (
                    (data['lon'] >= x_edges[i]) & (data['lon'] < x_edges[i+1]) &
                    (data['lat'] >= y_edges[j]) & (data['lat'] < y_edges[j+1])
                )
                count = mask.sum()
                if count > 0:
                    occupied_cells += 1
                    cell_densities.append(count)
        
        # 计算稀疏度
        sparsity_ratio = 1 - (occupied_cells / (grid_size * grid_size))
        
        # 计算平均每个蜂窝的数据点数
        avg_points_per_hex = data_count / max(1, occupied_cells)
        
        # 计算密度变异系数（反映数据分布均匀性）
        if cell_densities:
            density_cv = np.std(cell_densities) / (np.mean(cell_densities) + 1e-6)
        else:
            density_cv = 0.0
        
        return sparsity_ratio, avg_points_per_hex, density_cv
    
    # 执行稀疏度检测
    sparsity_ratio, avg_points_per_hex, density_cv = check_sparsity_enhanced(data, bounds, optimal_gridsize)
    
    # 9. 智能稀疏度调整
    if sparsity_ratio > 0.75:  # 极高稀疏度
        sparsity_factor = 0.65
        print(f"  Extremely high sparsity ({sparsity_ratio:.1%} empty hexes), significantly increasing hex size")
    elif sparsity_ratio > 0.6:  # 高稀疏度
        sparsity_factor = 0.75
        print(f"  High sparsity ({sparsity_ratio:.1%} empty hexes), substantially increasing hex size")
    elif sparsity_ratio > 0.45:  # 中等稀疏度
        # 考虑数据密度，如果密度高但稀疏度也高，说明数据极不均匀
        if avg_points_per_hex > 40:  # 高密度不均匀分布
            sparsity_factor = 0.8  # 更保守的调整
            print(f"  Uneven high-density distribution ({sparsity_ratio:.1%} empty hexes, {avg_points_per_hex:.1f} points/hex)")
        else:
            sparsity_factor = 0.85
            print(f"  Medium sparsity ({sparsity_ratio:.1%} empty hexes), moderately increasing hex size")
    elif sparsity_ratio > 0.25:  # 低稀疏度
        sparsity_factor = 0.95
        print(f"  Low sparsity ({sparsity_ratio:.1%} empty hexes), slightly increasing hex size")
    else:  # 密集分布
        sparsity_factor = 1.0
        print(f"  Dense distribution ({sparsity_ratio:.1%} empty hexes), keeping original size")
    
    # 10. 根据平均每蜂窝数据点数调整
    if avg_points_per_hex > 80:  # 极高密度
        hex_density_factor = 1.15  # 增加gridsize，减小蜂窝
        print(f"  Very high hex density ({avg_points_per_hex:.1f} points/hex), decreasing hex size")
    elif avg_points_per_hex > 50:  # 高密度
        hex_density_factor = 1.05  # 轻微增加gridsize
        print(f"  High hex density ({avg_points_per_hex:.1f} points/hex), slightly decreasing hex size")
    elif avg_points_per_hex > 20:  # 适中密度
        hex_density_factor = 1.0
        print(f"  Good hex density ({avg_points_per_hex:.1f} points/hex), keeping size")
    elif avg_points_per_hex > 5:  # 中等密度
        hex_density_factor = 0.9
        print(f"  Medium hex density ({avg_points_per_hex:.1f} points/hex), moderately increasing hex size")
    else:  # 低密度
        hex_density_factor = 0.8
        print(f"  Low hex density ({avg_points_per_hex:.1f} points/hex), increasing hex size")
    
    # 11. 数据分布均匀性调整
    if density_cv > 3.0:  # 分布极不均匀
        uniformity_factor = 0.85
        print(f"  Highly uneven data distribution (CV={density_cv:.2f}), adjusting for clustering")
    elif density_cv > 2.0:  # 分布不均匀
        uniformity_factor = 0.92
        print(f"  Uneven data distribution (CV={density_cv:.2f}), slight adjustment")
    else:
        uniformity_factor = 1.0
        print(f"  Reasonably uniform distribution (CV={density_cv:.2f})")
    
    # 12. 应用所有调整因子
    optimal_gridsize = int(optimal_gridsize * sparsity_factor * hex_density_factor * uniformity_factor)
    
    # 13. 动态上下限
    if data_count > 100000:  # 超大数据集
        max_gridsize = 120
    elif data_count > 50000:  # 大数据集
        max_gridsize = 100
    elif data_count > 25000:  # 中大数据集
        max_gridsize = 85
    else:
        max_gridsize = 70
    
    min_gridsize = max(12, int(np.sqrt(data_count) / 50))  # 动态最小值
    
    optimal_gridsize = max(min_gridsize, min(optimal_gridsize, max_gridsize))
    
    print(f"Grid optimization summary:")
    print(f"  Geographic area: {geo_area:.1f} sq degrees (ratio: {shape_ratio:.2f})")
    print(f"  Data density: {data_density:.1f} points/sq degree")
    print(f"  Data count: {data_count:,} points")
    print(f"  Sparsity ratio: {sparsity_ratio:.1%}")
    print(f"  Avg points/hex: {avg_points_per_hex:.1f}")
    print(f"  Distribution CV: {density_cv:.2f}")
    print(f"  Gridsize range: [{min_gridsize}, {max_gridsize}]")
    print(f"  Final gridsize: {optimal_gridsize}")
    
    return optimal_gridsize


def create_adaptive_hexmap(df, variable='Expectation_net_benefit', 
                          title=None, cmap=None, 
                          bounds=(-125, -65, 24, 51), 
                          helper_shp=None, save_path=None,
                          auto_gridsize=True,
                          manual_gridsize=None):
    """原始的自适应六边形热力图函数"""

    # 数据过滤
    xmin, xmax, ymin, ymax = bounds
    data = df[
        (df['lon'] >= xmin) & (df['lon'] <= xmax) & 
        (df['lat'] >= ymin) & (df['lat'] <= ymax) &
        (df[variable].notna())
    ]
    
    if len(data) == 0:
        print(f"No valid data for {variable}")
        return None
    
    print(f"Data points: {len(data):,}")
    
    # 智能网格大小计算
    if auto_gridsize:
        optimal_gridsize = calculate_optimal_gridsize(data, bounds)
    else:
        optimal_gridsize = manual_gridsize or 30
    
    print(f"Final gridsize: {optimal_gridsize}")
    
    # 配色方案
    if cmap is None:
        from matplotlib.colors import LinearSegmentedColormap
        
        greens_to_reds = [
            '#74c476',  # 浅绿色
            '#41ab5d',  # 中浅绿色
            '#238b45',  # 中绿色
            '#fed976',  # 浅黄色
            '#fd8d3c',  # 橙色
            '#e31a1c',  # 红色
            '#990000'   # 深红色
        ]
        positions = [0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0]
        custom_cmap = LinearSegmentedColormap.from_list("greens_to_reds", 
                                                       list(zip(positions, greens_to_reds)), 
                                                       N=256)
    else:
        custom_cmap = cmap
    
    # 图形设置
    golden_ratio = 1.618
    fig_width = 16
    fig_height = fig_width / golden_ratio
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor('#e6f3ff')
    
    # 底图
    if helper_shp:
        try:
            helper_gdf = gpd.read_file(helper_shp)
            helper_gdf.plot(ax=ax, color='white', 
                          edgecolor='lightgray', linewidth=0.3, alpha=0.9)
        except Exception as e:
            print(f"Helper map error: {e}")

    # 六边形热力图
    hb = ax.hexbin(data['lon'], data['lat'],
                   C=data[variable],
                   gridsize=optimal_gridsize,  
                   cmap=custom_cmap,
                   mincnt=1.8,  
                   alpha=1,  
                   linewidths=0.2, 
                   edgecolors='white',
                   zorder=2
                   )
    
    # 颜色条
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    cbaxes = inset_axes(ax, 
                       width="3%", height="25%",   
                       loc='lower left',
                       bbox_to_anchor=(0.02, 0.05, 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    
    cbar = plt.colorbar(hb, cax=cbaxes, orientation='vertical')
    label_text = "Carbon density\n(Mg/ha)"
    cbar.set_label(label_text, fontsize=10, fontweight='bold',
                  rotation=0, labelpad=20, y=1.05)
    cbar.ax.tick_params(labelsize=8, length=3, width=0.5, pad=2)
    cbar.locator = plt.MaxNLocator(nbins=6)
    cbar.update_ticks()
    
    # 边界和布局
    margin_x = (xmax - xmin) * 0.093
    margin_y = (ymax - ymin) * 0.127
    
    ax.set_xlim(xmin - margin_x, xmax + margin_x)
    ax.set_ylim(ymin - margin_y, ymax + margin_y)
    
    center_lat = (ymin + ymax) / 2
    geo_correction = np.cos(np.radians(center_lat))
    optimal_aspect = golden_ratio * geo_correction
    ax.set_aspect(optimal_aspect)
    
    # 标题和标签
    ax.set_title(title or f'USA {variable.replace("_", " ").title()}', 
                fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel('Longitude', fontsize=13, labelpad=10)
    ax.set_ylabel('Latitude', fontsize=13, labelpad=10)
    
    # 网格和刻度
    ax.tick_params(labelsize=11, pad=5, top=False, right=False)
    ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5, color='gray')
    
    # 经纬度标记
    lon_ticks = np.arange(-120, -60, 20)
    lat_ticks = np.arange(25, 50, 10)
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.set_xticklabels([f'{abs(x)}°W' for x in lon_ticks])
    ax.set_yticklabels([f'{y}°N' for y in lat_ticks])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.2)
        print(f"Saved: {save_path}")
    
    plt.tight_layout(pad=2.0)
    plt.show()
    
    return fig


# ===== 凸包功能 =====

def perform_clustering_analysis(data, n_clusters=None, min_cluster_size=50):
    """
    执行数据聚类分析并返回聚类标签和中心
    
    Parameters:
        data: DataFrame, 包含 lon, lat 列
        n_clusters: int, 聚类数量，如果为None则自动确定
        min_cluster_size: int, 最小聚类大小
        
    Returns:
        cluster_labels: array, 聚类标签
        cluster_centers: array, 聚类中心
        n_clusters: int, 实际聚类数量
    """
    coords = data[['lon', 'lat']].values
    
    if len(coords) < min_cluster_size:
        # 数据量太小，不进行聚类
        return np.zeros(len(coords)), np.mean(coords, axis=0).reshape(1, -1), 1
    
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        best_score = None  # 初始化为None
        
        # 自动确定最优聚类数量
        if n_clusters is None:
            max_clusters = min(8, len(coords)//min_cluster_size)
            if max_clusters < 2:
                return np.zeros(len(coords)), np.mean(coords, axis=0).reshape(1, -1), 1
                
            best_score = -1
            best_k = 2
            
            for k in range(2, max_clusters + 1):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(coords)
                    score = silhouette_score(coords, labels)
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
                except:
                    continue
            
            n_clusters = best_k
        
        # 执行最终聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)
        cluster_centers = kmeans.cluster_centers_
        
        # 如果没有进行自动聚类，计算最终的silhouette score
        if best_score is None:
            try:
                best_score = silhouette_score(coords, cluster_labels)
            except:
                best_score = 0.0
        
        print(f"  Clustering completed: {n_clusters} clusters, silhouette score: {best_score:.3f}")
        
        return cluster_labels, cluster_centers, n_clusters
        
    except ImportError:
        print("  sklearn not available, using simple grid-based clustering")
        # 使用简单的网格聚类作为后备方案
        return simple_grid_clustering(data, n_clusters or 4)


def simple_grid_clustering(data, n_clusters=4):
    """
    简单的网格聚类方法（sklearn的后备方案）
    """
    coords = data[['lon', 'lat']].values
    
    # 计算网格
    grid_size = int(np.sqrt(n_clusters))
    
    lon_min, lon_max = coords[:, 0].min(), coords[:, 0].max()
    lat_min, lat_max = coords[:, 1].min(), coords[:, 1].max()
    
    lon_edges = np.linspace(lon_min, lon_max, grid_size + 1)
    lat_edges = np.linspace(lat_min, lat_max, grid_size + 1)
    
    cluster_labels = np.zeros(len(coords))
    cluster_centers = []
    
    cluster_id = 0
    for i in range(grid_size):
        for j in range(grid_size):
            mask = (
                (coords[:, 0] >= lon_edges[i]) & (coords[:, 0] < lon_edges[i+1]) &
                (coords[:, 1] >= lat_edges[j]) & (coords[:, 1] < lat_edges[j+1])
            )
            if mask.sum() > 0:
                cluster_labels[mask] = cluster_id
                cluster_centers.append(coords[mask].mean(axis=0))
                cluster_id += 1
    
    return cluster_labels, np.array(cluster_centers), cluster_id


def generate_convex_hull(coords, buffer_size):
    """生成标准凸包"""
    try:
        hull = ConvexHull(coords)
        hull_points = coords[hull.vertices]
        
        # 添加缓冲区
        center = coords.mean(axis=0)
        buffered_points = []
        
        for point in hull_points:
            direction = point - center
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                unit_direction = direction / direction_norm
                buffered_point = point + unit_direction * buffer_size
                buffered_points.append(buffered_point)
        
        return Polygon(buffered_points)
    except:
        return None

def generate_convex_hulls(data, cluster_labels, buffer_ratio=0.02):
    """为每个聚类生成凸包"""
    coords = data[['lon', 'lat']].values
    unique_clusters = np.unique(cluster_labels)
    
    hulls = []
    hull_info = []
    
    # 计算缓冲区大小
    lon_range = coords[:, 0].max() - coords[:, 0].min()
    lat_range = coords[:, 1].max() - coords[:, 1].min()
    buffer_size = max(lon_range, lat_range) * buffer_ratio
    
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_coords = coords[mask]
        
        if len(cluster_coords) < 3:
            continue
        
        try:
            hull = generate_convex_hull(cluster_coords, buffer_size)
            
            if hull is not None:
                hulls.append(hull)
                hull_info.append({
                    'cluster_id': cluster_id,
                    'n_points': len(cluster_coords),
                    'center': cluster_coords.mean(axis=0)
                })
                
        except Exception as e:
            print(f"  Warning: Failed to generate hull for cluster {cluster_id}: {e}")
            continue
    
    print(f"  Generated {len(hulls)} convex hulls")
    return hulls, hull_info

def plot_convex_hulls(ax, hulls, hull_info, hull_style='both'):
    """在地图上绘制透明凸包"""
    hull_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
    ]
    
    for i, (hull, info) in enumerate(zip(hulls, hull_info)):
        color = hull_colors[i % len(hull_colors)]
        
        try:
            # 获取凸包坐标
            if hasattr(hull, 'exterior'):
                x_coords, y_coords = hull.exterior.xy
            else:
                continue
            
            # 绘制凸包填充（透明浮层）
            if hull_style in ['fill', 'both']:
                ax.fill(x_coords, y_coords,
                       color=color,
                       alpha=0.25,  # 透明度
                       zorder=8,   # 在热力图之上
                       edgecolor='none')
            
            # 绘制凸包边界
            if hull_style in ['line', 'both']:
                ax.plot(x_coords, y_coords, 
                       color=color, 
                       linewidth=3,
                       alpha=0.9,
                       linestyle='-',
                       zorder=10)  # 最高层级
            
            # 添加聚类标签
            center_x, center_y = info['center']
            ax.annotate(f'C{info["cluster_id"]}', 
                       xy=(center_x, center_y),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=11, fontweight='bold',
                       color='white',
                       bbox=dict(boxstyle='round,pad=0.4', 
                               facecolor=color, 
                               alpha=0.8,
                               edgecolor='white',
                               linewidth=1.5),
                       zorder=15)
            
        except Exception as e:
            print(f"  Warning: Failed to plot hull {i}: {e}")
            continue
    
    print(f"  Plotted {len(hulls)} convex hulls as transparent overlays")

# print("凸包生成函数已加载完成！")


def create_hexmap_with_convex_hulls(df, variable='Expectation_net_benefit', 
                                   title=None, cmap=None, 
                                 bounds=(-125, -65, 24, 51), 
                                   helper_shp=None, save_path=None,
                                   auto_gridsize=True, manual_gridsize=None,
                                   # 凸包相关参数
                                   show_convex_hulls=True,
                                   hull_style='both',
                                   n_clusters=None,
                                   min_cluster_size=100,
                                   buffer_ratio=0.03):
    """
    创建带有透明凸包的六边形热力图
    
    Parameters:
        show_convex_hulls: bool, 是否显示凸包
        hull_style: str, 'line', 'fill', 或 'both'
        n_clusters: int, 聚类数量（None为自动）
        min_cluster_size: int, 最小聚类大小
        buffer_ratio: float, 凸包缓冲区比例
    """
    
    # 数据过滤
    xmin, xmax, ymin, ymax = bounds
    data = df[
        (df['lon'] >= xmin) & (df['lon'] <= xmax) & 
        (df['lat'] >= ymin) & (df['lat'] <= ymax) &
        (df[variable].notna())
    ]
    
    if len(data) == 0:
        print(f"No valid data for {variable}")
        return None
    
    print(f"Data points: {len(data):,}")

    # 智能网格大小计算
    if auto_gridsize:
        optimal_gridsize = calculate_optimal_gridsize(data, bounds)
    else:
        optimal_gridsize = manual_gridsize or 30
    
    print(f"Final gridsize: {optimal_gridsize}")
    
    # 配色方案
    if cmap is None:
        from matplotlib.colors import LinearSegmentedColormap
        
        greens_to_reds = [
            '#74c476',  # 浅绿色
            '#41ab5d',  # 中浅绿色
            '#238b45',  # 中绿色
            '#fed976',  # 浅黄色
            '#fd8d3c',  # 橙色
            '#e31a1c',  # 红色
            '#990000'   # 深红色
        ]
        positions = [0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0]
        custom_cmap = LinearSegmentedColormap.from_list("greens_to_reds", 
                                                       list(zip(positions, greens_to_reds)), 
                                                      N=256)
    else:
        custom_cmap = cmap
    
    # 图形设置
    golden_ratio = 1.618
    fig_width = 16
    fig_height = fig_width / golden_ratio
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor('#e6f3ff')
    
    # 底图
    if helper_shp:
        try:
            helper_gdf = gpd.read_file(helper_shp)
            helper_gdf.plot(ax=ax, color='white', 
                          edgecolor='lightgray', linewidth=0.3, alpha=0.9, zorder=1)
        except Exception as e:
            print(f"Helper map error: {e}")
    
    # 六边形热力图（基础层）
    hb = ax.hexbin(data['lon'], data['lat'],
                   C=data[variable],
                   gridsize=optimal_gridsize,  
                   cmap=custom_cmap,
                   mincnt=1.8,  
                   alpha=1,  
                   linewidths=0.2,
                   edgecolors='white',
                   zorder=5  # 基础热力图层级
                   )
    
    # 聚类分析和凸包生成（透明浮层）
    hulls = []
    hull_info = []
    
    if show_convex_hulls:
        print("Performing clustering analysis...")
        cluster_labels, cluster_centers, actual_n_clusters = perform_clustering_analysis(
            data, n_clusters, min_cluster_size)
        
        print("Generating convex hulls...")
        hulls, hull_info = generate_convex_hulls(
            data, cluster_labels, buffer_ratio)
        
        if hulls:
            plot_convex_hulls(ax, hulls, hull_info, hull_style)
    
    # 颜色条
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    cbaxes = inset_axes(ax, 
                       width="3%", height="25%",   
                       loc='lower left',
                       bbox_to_anchor=(0.02, 0.05, 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    
    cbar = plt.colorbar(hb, cax=cbaxes, orientation='vertical')
    label_text = "Carbon density\n(Mg/ha)"
    cbar.set_label(label_text, fontsize=10, fontweight='bold',
                  rotation=0, labelpad=20, y=1.05)
    cbar.ax.tick_params(labelsize=8, length=3, width=0.5, pad=2)
    cbar.locator = plt.MaxNLocator(nbins=6)
    cbar.update_ticks()
    
    # 边界和布局
    margin_x = (xmax - xmin) * 0.093
    margin_y = (ymax - ymin) * 0.127
    
    ax.set_xlim(xmin - margin_x, xmax + margin_x)
    ax.set_ylim(ymin - margin_y, ymax + margin_y)
    
    center_lat = (ymin + ymax) / 2
    geo_correction = np.cos(np.radians(center_lat))
    optimal_aspect = golden_ratio * geo_correction
    ax.set_aspect(optimal_aspect)
    
    # 标题和标签
    hull_info_text = f" with {len(hulls)} convex hulls" if show_convex_hulls and hulls else ""
    ax.set_title(title or f'USA {variable.replace("_", " ").title()}{hull_info_text}', 
                fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel('Longitude', fontsize=13, labelpad=10)
    ax.set_ylabel('Latitude', fontsize=13, labelpad=10)
    
    # 网格和刻度
    ax.tick_params(labelsize=11, pad=5, top=False, right=False)
    ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5, color='gray')
    
    # 经纬度标记
    lon_ticks = np.arange(-120, -60, 20)
    lat_ticks = np.arange(25, 50, 10)
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.set_xticklabels([f'{abs(x)}°W' for x in lon_ticks])
    ax.set_yticklabels([f'{y}°N' for y in lat_ticks])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.2)
        print(f"Saved: {save_path}")
    
    plt.tight_layout(pad=2.0)
    plt.show()
    
    return fig, (hulls, hull_info) if show_convex_hulls else fig