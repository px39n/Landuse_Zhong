"""
Spatial alignment utilities
Migrated from: 2.1 process_csv_for_aligning.ipynb
"""

import numpy as np
import xarray as xr
from typing import Union, Tuple, Optional, Dict, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def align_to_grid(
    data: Union[xr.Dataset, xr.DataArray],
    target_resolution: float = 0.00833333,  # 1/120 degree (~1km)
    target_crs: str = "EPSG:4326",
    method: str = "bilinear"
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Align dataset to target grid
    
    Args:
        data: Input xarray Dataset or DataArray
        target_resolution: Target resolution in degrees
        target_crs: Target coordinate reference system
        method: Resampling method ('bilinear', 'nearest', 'cubic')
    
    Returns:
        Aligned dataset
    """
    logger.info(f"Aligning data to {target_resolution}° resolution using {method}")
    
    # Get current bounds
    lon_min, lon_max = float(data.lon.min()), float(data.lon.max())
    lat_min, lat_max = float(data.lat.min()), float(data.lat.max())
    
    # Create target grid
    target_lon = np.arange(lon_min, lon_max + target_resolution, target_resolution)
    target_lat = np.arange(lat_min, lat_max + target_resolution, target_resolution)
    
    # Interpolate to target grid
    aligned = data.interp(
        lon=target_lon,
        lat=target_lat,
        method=method
    )
    
    logger.info(f"Aligned shape: {aligned.dims}")
    return aligned


def align_datasets(
    datasets: Dict[str, Union[xr.Dataset, xr.DataArray]],
    target_resolution: float = 0.00833333,
    reference_dataset: Optional[str] = None,
    method: str = "bilinear"
) -> Dict[str, Union[xr.Dataset, xr.DataArray]]:
    """
    Align multiple datasets to common grid
    
    Args:
        datasets: Dictionary of named datasets
        target_resolution: Target resolution in degrees
        reference_dataset: Name of dataset to use as reference (optional)
        method: Resampling method
    
    Returns:
        Dictionary of aligned datasets
    """
    logger.info(f"Aligning {len(datasets)} datasets")
    
    # Determine common bounds
    if reference_dataset and reference_dataset in datasets:
        ref = datasets[reference_dataset]
        lon_min, lon_max = float(ref.lon.min()), float(ref.lon.max())
        lat_min, lat_max = float(ref.lat.min()), float(ref.lat.max())
    else:
        # Use intersection of all datasets
        lon_mins, lon_maxs = [], []
        lat_mins, lat_maxs = [], []
        
        for ds in datasets.values():
            lon_mins.append(float(ds.lon.min()))
            lon_maxs.append(float(ds.lon.max()))
            lat_mins.append(float(ds.lat.min()))
            lat_maxs.append(float(ds.lat.max()))
        
        lon_min, lon_max = max(lon_mins), min(lon_maxs)
        lat_min, lat_max = max(lat_mins), min(lat_maxs)
    
    logger.info(f"Common bounds: lon=[{lon_min}, {lon_max}], lat=[{lat_min}, {lat_max}]")
    
    # Create target grid
    target_lon = np.arange(lon_min, lon_max + target_resolution, target_resolution)
    target_lat = np.arange(lat_min, lat_max + target_resolution, target_resolution)
    
    # Align each dataset
    aligned_datasets = {}
    for name, ds in datasets.items():
        logger.info(f"Aligning {name}")
        
        # Clip to common bounds
        ds_clipped = ds.sel(
            lon=slice(lon_min, lon_max),
            lat=slice(lat_min, lat_max)
        )
        
        # Interpolate
        aligned = ds_clipped.interp(
            lon=target_lon,
            lat=target_lat,
            method=method
        )
        
        aligned_datasets[name] = aligned
    
    logger.info("Alignment complete")
    return aligned_datasets


def create_distance_raster(
    reference_points: np.ndarray,
    grid_shape: Tuple[int, int],
    grid_bounds: Tuple[float, float, float, float],
    max_distance: Optional[float] = None
) -> np.ndarray:
    """
    Create distance raster from reference points
    Used for distance-to-feature calculations (towns, grids, roads)
    
    Args:
        reference_points: Array of (lon, lat) coordinates
        grid_shape: Output grid shape (height, width)
        grid_bounds: Grid bounds (west, south, east, north)
        max_distance: Maximum distance to compute (None = no limit)
    
    Returns:
        Distance array in degrees
    """
    from scipy.spatial import cKDTree
    
    height, width = grid_shape
    west, south, east, north = grid_bounds
    
    # Create grid coordinates
    lon = np.linspace(west, east, width)
    lat = np.linspace(south, north, height)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Flatten grid points
    grid_points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    
    # Build KD-tree for reference points
    tree = cKDTree(reference_points)
    
    # Query nearest distances
    distances, _ = tree.query(grid_points)
    
    if max_distance is not None:
        distances = np.minimum(distances, max_distance)
    
    # Reshape to grid
    distance_raster = distances.reshape(height, width)
    
    return distance_raster


def calculate_road_density(
    road_lines: List[np.ndarray],
    grid_shape: Tuple[int, int],
    grid_bounds: Tuple[float, float, float, float],
    cell_size: float
) -> np.ndarray:
    """
    Calculate road density per grid cell
    
    Args:
        road_lines: List of road line geometries as coordinate arrays
        grid_shape: Output grid shape (height, width)
        grid_bounds: Grid bounds (west, south, east, north)
        cell_size: Grid cell size in degrees
    
    Returns:
        Road density array (km/km²)
    """
    height, width = grid_shape
    west, south, east, north = grid_bounds
    
    # Initialize density raster
    density = np.zeros((height, width), dtype=np.float32)
    
    # Calculate cell area (approximate)
    cell_area_km2 = (cell_size * 111) ** 2  # 1 degree ≈ 111 km
    
    for line_coords in road_lines:
        # Iterate through line segments
        for i in range(len(line_coords) - 1):
            lon1, lat1 = line_coords[i]
            lon2, lat2 = line_coords[i + 1]
            
            # Calculate segment length
            dlon = (lon2 - lon1) * 111  # Convert to km
            dlat = (lat2 - lat1) * 111
            length_km = np.sqrt(dlon**2 + dlat**2)
            
            # Get grid indices for segment midpoint
            mid_lon = (lon1 + lon2) / 2
            mid_lat = (lat1 + lat2) / 2
            
            col = int((mid_lon - west) / cell_size)
            row = int((mid_lat - south) / cell_size)
            
            # Add to density if within bounds
            if 0 <= row < height and 0 <= col < width:
                density[row, col] += length_km
    
    # Normalize by cell area
    density /= cell_area_km2
    
    return density
