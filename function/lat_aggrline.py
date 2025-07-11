import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_revenue_ratio_by_latitude(df, variable='Revenue_ratio', 
                                 lat_col='lat',
                                 type_col=None,
                                 lat_bin_size=0.5,
                                 figsize=(5,12),
                                 save_path=None):
    """
    Plot revenue ratio aggregated by latitude
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    variable : str, default 'Revenue_ratio'
        Column name of the variable to plot
    lat_col : str, default 'lat'
        Column name containing latitude values
    type_col : str, optional
        Column name for grouping by type
    lat_bin_size : float, default 0.5
        Size of latitude bins in degrees
    figsize : tuple, default (5,12)
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    
    # Create latitude bins with specified interval size
    lat_min = df[lat_col].min()
    lat_max = df[lat_col].max()
    lat_bins = np.arange(lat_min, lat_max + lat_bin_size, lat_bin_size)
    
    df_copy = df.copy()
    df_copy['lat_binned'] = pd.cut(df_copy[lat_col], bins=lat_bins)
    df_copy['lat_center'] = df_copy['lat_binned'].apply(lambda x: x.mid if pd.notna(x) else np.nan)
    
    # Aggregate by binned latitude
    group_cols = ['lat_center']
    if type_col is not None:
        group_cols.append(type_col)
    
    agg = (
        df_copy
        .dropna(subset=[variable, 'lat_center'])
        .groupby(group_cols)[variable]
        .agg(
            mean='mean',
            lowCI=lambda x: np.quantile(x, 0.025),
            highCI=lambda x: np.quantile(x, 0.975),
            count='count'  # Add count
        )
        .reset_index()
    )
    
    # Filter intervals with too few data points
    agg = agg[agg['count'] >= 5]  # At least 5 data points
    
    print(f"Number of latitude intervals after aggregation: {len(agg)}")
    print(f"Average data points per interval: {agg['count'].mean():.1f}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(agg['mean'], agg['lat_center'], linewidth=2.0, color='darkgreen')
    ax.fill_betweenx(agg['lat_center'], agg['lowCI'], agg['highCI'], 
                     alpha=0.3, color='lightgreen')
    
    ax.set_xlabel(variable.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig
