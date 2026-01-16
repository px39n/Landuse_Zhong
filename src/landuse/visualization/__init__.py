"""
Visualization module for generating publication-ready figures
"""

from .maps import plot_spatial_map, plot_priority_map
from .figures import generate_all_figures

__all__ = [
    "plot_spatial_map",
    "plot_priority_map",
    "generate_all_figures",
]
