# 修复内存问题的DEM底图整合函数
import numpy as np
import matplotlib.pyplot as plt
import rioxarray
from shapely.geometry import box
import geopandas as gpd
from pyproj import Transformer
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches

def plot_ccd_optimized_global_style_with_dem_memory_fixed(pixel_optimized_data, us_states_gdf, dem_us, dem_full):
    """
    修复内存问题的CCD优化结果绘制函数，整合DEM底图
    
    Args:
        pixel_optimized_data: 包含优化结果的DataFrame
        us_states_gdf: 美国州界GeoDataFrame
        dem_us: 美国区域DEM数据（xarray DataArray）
        dem_full: 完整DEM数据（xarray DataArray）
    
    Returns:
        fig, ax: 图形对象和轴对象
    """
    # 设置默认字体
    plt.rcParams['font.family'] = 'Arial'
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    usa_bounds_main = {'lon_min': -125, 'lon_max': -65, 'lat_min': 24, 'lat_max': 51}
    
    # 创建边界框并转换投影
    bbox = box(usa_bounds_main['lon_min'], usa_bounds_main['lat_min'],
               usa_bounds_main['lon_max'], usa_bounds_main['lat_max'])
    us_states_bound = us_states_gdf.to_crs(epsg=4326).clip(bbox)
    us_states_albers = us_states_bound.to_crs('ESRI:102003')
    
    # 创建坐标转换器
    transformer = Transformer.from_crs("EPSG:4326", "ESRI:102003", always_xy=True)
    
    # 转换优化数据坐标
    x_proj, y_proj = transformer.transform(pixel_optimized_data['lon'].values, pixel_optimized_data['lat'].values)
    ax.set_facecolor('#f0f8ff')
    
    # 加载辅助底图
    try:
        helper_gdf = gpd.read_file(r'figure\draw_shp\ne_110m_land.shp')
        extended_bbox = box(usa_bounds_main['lon_min'] - 5, usa_bounds_main['lat_min'] - 3,
                            usa_bounds_main['lon_max'] + 5, usa_bounds_main['lat_max'] + 3)
        helper_gdf_proj = helper_gdf.to_crs(epsg=4326).clip(extended_bbox).to_crs('ESRI:102003')
        helper_gdf_proj.plot(ax=ax, color='white', edgecolor='lightgray', linewidth=0.3, alpha=0.9)
    except Exception as e:
        print(f"Helper map error: {e}")
    
    # === 1. 处理外围DEM数据（完整DEM，颜色稍深）===
    print("处理外围DEM数据...")
    if dem_full is not None:
        try:
            # 获取DEM坐标
            if hasattr(dem_full, 'x') and hasattr(dem_full, 'y'):
                dem_lon_full = dem_full.x.values
                dem_lat_full = dem_full.y.values
            elif hasattr(dem_full, 'lon') and hasattr(dem_full, 'lat'):
                dem_lon_full = dem_full.lon.values
                dem_lat_full = dem_full.lat.values
            else:
                print("警告：无法获取DEM坐标信息")
                dem_lon_full = None
                dem_lat_full = None
            
            if dem_lon_full is not None and dem_lat_full is not None:
                # 找到美国范围外的DEM数据
                lon_mask_outside = ~((dem_lon_full >= usa_bounds_main['lon_min']) & 
                                   (dem_lon_full <= usa_bounds_main['lon_max']))
                lat_mask_outside = ~((dem_lat_full >= usa_bounds_main['lat_min']) & 
                                   (dem_lat_full <= usa_bounds_main['lat_max']))
                
                # 获取外围数据的索引
                lon_indices_outside = np.where(lon_mask_outside)[0]
                lat_indices_outside = np.where(lat_mask_outside)[0]
                
                if len(lon_indices_outside) > 0 and len(lat_indices_outside) > 0:
                    # 更激进的降采样外围数据以避免内存问题
                    step_outside = max(1, min(len(lon_indices_outside), len(lat_indices_outside)) // 5000)
                    lon_subset_outside = lon_indices_outside[::step_outside]
                    lat_subset_outside = lat_indices_outside[::step_outside]
                    
                    if len(lon_subset_outside) > 0 and len(lat_subset_outside) > 0:
                        # 获取DEM数据 - 使用更小的数据块
                        dem_data = dem_full.squeeze().values
                        if dem_data.ndim == 3:
                            dem_data = dem_data[0]  # 取第一个时间步
                        
                        # 只选择需要的子集
                        elev_subset_outside = dem_data[np.ix_(lat_subset_outside, lon_subset_outside)]
                        valid_mask_outside = ~np.isnan(elev_subset_outside)
                        valid_elev_outside = elev_subset_outside[valid_mask_outside]
                        
                        if valid_elev_outside.size > 0:
                            # 外围DEM使用稍深的颜色
                            terrain_colors_outside = ListedColormap([
                                "#d4d4d4", "#c0c0c0", "#a8a8a8", "#8f8f8f", 
                                "#767676", "#5d5d5d", "#444444"
                            ])
                            
                            percentiles_outside = np.percentile(valid_elev_outside, [0, 20, 40, 60, 80, 90, 100])
                            norm_outside = BoundaryNorm(percentiles_outside, terrain_colors_outside.N)
                            
                            # 创建网格 - 使用更小的批次
                            lon_grid_outside, lat_grid_outside = np.meshgrid(
                                dem_lon_full[lon_subset_outside], dem_lat_full[lat_subset_outside]
                            )
                            
                            # 分批转换坐标以避免内存问题 - 更小的批次
                            batch_size = 1000
                            x_proj_outside_list = []
                            y_proj_outside_list = []
                            
                            lon_flat = lon_grid_outside.flatten()
                            lat_flat = lat_grid_outside.flatten()
                            
                            for i in range(0, len(lon_flat), batch_size):
                                end_idx = min(i + batch_size, len(lon_flat))
                                batch_lons = lon_flat[i:end_idx]
                                batch_lats = lat_flat[i:end_idx]
                                
                                x_batch, y_batch = transformer.transform(batch_lons, batch_lats)
                                x_proj_outside_list.extend(x_batch)
                                y_proj_outside_list.extend(y_batch)
                            
                            dem_x_proj_outside = np.array(x_proj_outside_list).reshape(lon_grid_outside.shape)
                            dem_y_proj_outside = np.array(y_proj_outside_list).reshape(lat_grid_outside.shape)
                            
                            # 绘制外围DEM（稍深的颜色）
                            ax.pcolormesh(dem_x_proj_outside, dem_y_proj_outside, elev_subset_outside, 
                                         cmap=terrain_colors_outside, norm=norm_outside,
                                         alpha=0.6, zorder=1)
        except Exception as e:
            print(f"外围DEM处理错误: {e}")
    
    # === 2. 处理美国区域DEM数据（dem_us，保持现状颜色）===
    print("处理美国区域DEM数据...")
    if dem_us is not None:
        try:
            # 获取DEM坐标
            if hasattr(dem_us, 'x') and hasattr(dem_us, 'y'):
                dem_lon_us = dem_us.x.values
                dem_lat_us = dem_us.y.values
            elif hasattr(dem_us, 'lon') and hasattr(dem_us, 'lat'):
                dem_lon_us = dem_us.lon.values
                dem_lat_us = dem_us.lat.values
            else:
                print("警告：无法获取DEM坐标信息")
                dem_lon_us = None
                dem_lat_us = None
            
            if dem_lon_us is not None and dem_lat_us is not None:
                # 找到美国范围内的DEM数据
                lon_mask_us = (dem_lon_us >= usa_bounds_main['lon_min']) & (dem_lon_us <= usa_bounds_main['lon_max'])
                lat_mask_us = (dem_lat_us >= usa_bounds_main['lat_min']) & (dem_lat_us <= usa_bounds_main['lat_max'])
                lon_indices_us = np.where(lon_mask_us)[0]
                lat_indices_us = np.where(lat_mask_us)[0]
                
                if len(lon_indices_us) > 0 and len(lat_indices_us) > 0:
                    # 更激进的降采样美国区域数据
                    step_us = max(1, min(len(lon_indices_us), len(lat_indices_us)) // 10000)
                    lon_subset_us = lon_indices_us[::step_us]
                    lat_subset_us = lat_indices_us[::step_us]
                    
                    # 获取DEM数据 - 使用更小的数据块
                    dem_us_data = dem_us.squeeze().values
                    if dem_us_data.ndim == 3:
                        dem_us_data = dem_us_data[0]  # 取第一个时间步
                    
                    # 只选择需要的子集
                    elev_subset_us = dem_us_data[np.ix_(lat_subset_us, lon_subset_us)]
                    valid_mask_us = ~np.isnan(elev_subset_us)
                    valid_elev_us = elev_subset_us[valid_mask_us]
                    
                    if valid_elev_us.size > 0:
                        # 美国区域DEM使用现状颜色（较浅）
                        terrain_colors_us = ListedColormap([
                            "#ffffff", "#ffffff", "#e0f3db", "#a8ddb5",
                            "#e6cfa1", "#c9a86b", "#a97c50"
                        ])
                        
                        percentiles_us = np.percentile(valid_elev_us, [30, 45, 60, 70, 80, 90, 100])
                        norm_us = BoundaryNorm(percentiles_us, terrain_colors_us.N)
                        
                        # 创建网格 - 使用更小的批次
                        lon_grid_us, lat_grid_us = np.meshgrid(
                            dem_lon_us[lon_subset_us], dem_lat_us[lat_subset_us]
                        )
                        
                        # 分批转换坐标 - 更小的批次
                        batch_size = 1000
                        x_proj_us_list = []
                        y_proj_us_list = []
                        
                        lon_flat_us = lon_grid_us.flatten()
                        lat_flat_us = lat_grid_us.flatten()
                        
                        for i in range(0, len(lon_flat_us), batch_size):
                            end_idx = min(i + batch_size, len(lon_flat_us))
                            batch_lons = lon_flat_us[i:end_idx]
                            batch_lats = lat_flat_us[i:end_idx]
                            
                            x_batch, y_batch = transformer.transform(batch_lons, batch_lats)
                            x_proj_us_list.extend(x_batch)
                            y_proj_us_list.extend(y_batch)
                        
                        dem_x_proj_us = np.array(x_proj_us_list).reshape(lon_grid_us.shape)
                        dem_y_proj_us = np.array(y_proj_us_list).reshape(lat_grid_us.shape)
                        
                        # 绘制美国区域DEM（现状颜色）
                        ax.pcolormesh(dem_x_proj_us, dem_y_proj_us, elev_subset_us, 
                                     cmap=terrain_colors_us, norm=norm_us,
                                     alpha=0.5, zorder=2)
        except Exception as e:
            print(f"美国区域DEM处理错误: {e}")
    
    # === 3. 地图边界设置 ===
    xmin, ymin, xmax, ymax = us_states_albers.total_bounds
    mx, my = (xmax - xmin) * 0.08, (ymax - ymin) * 0.08
    ax.set_xlim(xmin - mx, xmax + mx)
    ax.set_ylim(ymin - my, ymax + my)
    us_states_albers.plot(ax=ax, color='none', edgecolor='black', alpha=0.3, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    # === 4. 经纬度网格线 ===
    lon_ticks = np.arange(-120, -60, 10)
    lat_ticks = np.arange(25, 55, 5)
    
    for i, lon in enumerate(lon_ticks):
        lat_line = np.linspace(usa_bounds_main['lat_min'], usa_bounds_main['lat_max'], 100)
        lon_line = np.full_like(lat_line, lon)
        x_line, y_line = transformer.transform(lon_line, lat_line)
        ax.plot(x_line, y_line, color='gray', linewidth=0.3, alpha=0.5, zorder=4)
        
        if lon == -70:
            label_lat = usa_bounds_main['lat_min'] + 2.5
        else:
            label_lat = usa_bounds_main['lat_min'] + 0.5
        x_label, y_label = transformer.transform([lon], [label_lat])
        ax.text(x_label[0], y_label[0], f'{lon}°W', ha='center', va='bottom',
                fontsize=8, color='gray', zorder=6, clip_on=True)
    
    for j, lat in enumerate(lat_ticks):
        lon_line = np.linspace(usa_bounds_main['lon_min'], usa_bounds_main['lon_max'], 100)
        lat_line = np.full_like(lon_line, lat)
        x_line, y_line = transformer.transform(lon_line, lat_line)
        ax.plot(x_line, y_line, color='gray', linewidth=0.3, alpha=0.5, zorder=4)
        
        if lat == 25:
            label_lon = usa_bounds_main['lon_min'] + 2.5
        else:
            label_lon = usa_bounds_main['lon_min'] + 0.5
        x_label, y_label = transformer.transform([label_lon], [lat])
        ax.text(x_label[0], y_label[0], f'{lat}°N', ha='left', va='center',
                fontsize=8, color='gray', zorder=6, clip_on=True)
    
    # === 5. CCD散点图 ===
    if 'ccd_optimized' in pixel_optimized_data.columns:
        ccd_values = pixel_optimized_data['ccd_optimized'].values
    elif 'Expectation_net_benefit' in pixel_optimized_data.columns:
        ccd_values = pixel_optimized_data['Expectation_net_benefit'].values
    else:
        print("警告：未找到CCD优化数据列")
        ccd_values = None
    
    if ccd_values is not None:
        diverging_colors = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8',
                            '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
        ccd_cmap = LinearSegmentedColormap.from_list("ccd_diverging", diverging_colors, N=11)
        ccd_bins = np.percentile(ccd_values, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        ccd_norm = BoundaryNorm(ccd_bins, ncolors=ccd_cmap.N)
        scatter = ax.scatter(x_proj, y_proj, c=ccd_values, cmap=ccd_cmap, norm=ccd_norm,
                             s=2.2, alpha=1.0, edgecolors='none', zorder=5)
        
        # === 6. 颜色条 ===
        cbar_ax = inset_axes(ax, width="33%", height="4%",
                             loc='lower left',
                             bbox_to_anchor=(0.02, 0.02, 1, 1),
                             bbox_transform=ax.transAxes, borderpad=1)
        cbar = plt.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
        cbar.set_ticks(ccd_bins[::2])
        cbar.set_ticklabels([f'{v:.3f}' for v in ccd_bins[::2]])
        cbar.ax.tick_params(labelsize=10) 
        cbar.outline.set_linewidth(0.7)
        
        percentile_labels = [f'{p}%' for p in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100][::2]]
        for i, label in enumerate(percentile_labels):
            cbar_ax.text(i / (len(percentile_labels) - 1), 1.25, label,
                         transform=cbar_ax.transAxes, ha='center', va='bottom',
                         fontsize=10, fontweight='bold')
    
    # === 7. 指北针和比例尺 ===
    def add_nature_north_arrow(ax, size=0.04):
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        dx, dy = x1 - x0, y1 - y0
        desired_size = 0.045
        h = desired_size * dy
        w = h * 0.5
        cx = x0 + 0.965 * dx
        cy = y0 + 0.945 * dy
        coords = [[cx, cy + h / 2], [cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2]]
        ax.add_patch(mpatches.Polygon(coords, closed=True, facecolor="white", edgecolor="black", lw=1.2, zorder=10))
        ax.add_patch(mpatches.Polygon([[cx, cy + h / 2], [cx, cy - h / 2], [cx + w / 2, cy - h / 2]],
                                      closed=True, facecolor="black", edgecolor="black", lw=1.2, zorder=11))
        ax.text(cx, cy - h * 1.0, 'N', ha='center', va='center',
                fontsize=15, fontweight='bold', color='black', zorder=12)
    
    def add_nature_scalebar(ax, length_km=500):
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        dx, dy = x1 - x0, y1 - y0
        bar_x = x0 + 0.80*dx
        bar_y = y0 + 0.03*dy
        bar_height = 0.01*dy
        bar_length = length_km * 1000
        n_segments = 4
        seg_len = bar_length / n_segments
        for i in range(n_segments):
            color = "black" if i % 2 == 0 else "white"
            ax.add_patch(mpatches.Rectangle((bar_x + i*seg_len, bar_y),
                                            seg_len, bar_height,
                                            facecolor=color, edgecolor="black", lw=0.8))
        ax.text(bar_x + bar_length/2, bar_y - 0.008*dy,
                f"{length_km} km", ha="center", va="top",
                fontsize=10, fontweight="bold", color="black")
    
    add_nature_north_arrow(ax, size=0.08)
    add_nature_scalebar(ax, length_km=1000)
    
    plt.tight_layout()
    return fig, ax

