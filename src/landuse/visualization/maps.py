"""
Spatial map visualization
Migrated from: 6.5-6.9 Figure*.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def plot_spatial_map(
    data: xr.DataArray,
    title: str,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot spatial map
    
    Args:
        data: Data to plot
        title: Figure title
        cmap: Colormap
        vmin, vmax: Color limits
        figsize: Figure size
        dpi: Resolution
        save_path: Path to save figure
    
    Returns:
        Figure object
    """
    logger.info(f"Plotting spatial map: {title}")
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot data
    im = data.plot(
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=True,
        cbar_kwargs={'label': data.name or 'Value'}
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved figure: {save_path}")
    
    return fig


def plot_priority_map(
    priority: xr.DataArray,
    synergy: xr.DataArray,
    title: str = "Priority Ranking",
    figsize: Tuple[int, int] = (14, 10),
    dpi: int = 300,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot priority ranking map with synergy overlay
    
    Args:
        priority: Priority ranks
        synergy: 3E-Synergy index
        title: Figure title
        figsize: Figure size
        dpi: Resolution
        save_path: Path to save figure
    
    Returns:
        Figure object
    """
    logger.info("Plotting priority map")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    
    # Priority map
    priority.plot(
        ax=axes[0],
        cmap='RdYlGn_r',
        add_colorbar=True,
        cbar_kwargs={'label': 'Priority Rank (1=highest)'}
    )
    axes[0].set_title('Priority Ranking', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    
    # Synergy map
    synergy.plot(
        ax=axes[1],
        cmap='YlOrRd',
        add_colorbar=True,
        cbar_kwargs={'label': '3E-Synergy Index'}
    )
    axes[1].set_title('3E-Synergy Index', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved figure: {save_path}")
    
    return fig
