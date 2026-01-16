"""
Feature extraction utilities
Migrated from: 2.2 process_csv_for_embedding.ipynb
"""

import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract 15-dimensional environmental features for model training
    
    Features:
        Physical (4): land_cover, dem, slope, above_ground_biomass
        Climate (3): near_surface_temperature, wind_speed, shortwave_radiation
        Socioeconomic (8): population_density, gdp_density, distance_to_town,
                          distance_to_grid, road_density_primary/secondary/tertiary
    """
    
    def __init__(self, config: Dict):
        """
        Initialize feature extractor
        
        Args:
            config: Configuration dictionary with feature definitions
        """
        self.config = config
        self.feature_config = config.get("data", {}).get("features", {})
        
        # Build feature list
        self.features = []
        for category in ["physical", "climate", "socioeconomic"]:
            self.features.extend(self.feature_config.get(category, []))
        
        logger.info(f"FeatureExtractor initialized with {len(self.features)} features")
    
    def extract(
        self,
        datasets: Dict[str, xr.Dataset],
        mask: Optional[xr.DataArray] = None
    ) -> xr.Dataset:
        """
        Extract features from aligned datasets
        
        Args:
            datasets: Dictionary of aligned datasets
            mask: Optional mask to filter pixels (e.g., abandonment mask)
        
        Returns:
            Dataset with extracted features
        """
        logger.info("Extracting features")
        
        feature_arrays = {}
        
        for feature_name in self.features:
            if feature_name in datasets:
                feature_arrays[feature_name] = datasets[feature_name]
            else:
                logger.warning(f"Feature {feature_name} not found in datasets")
        
        # Combine into single dataset
        features = xr.Dataset(feature_arrays)
        
        # Apply mask if provided
        if mask is not None:
            features = features.where(mask)
        
        # Handle missing values
        features = self._handle_missing(features)
        
        # Normalize features
        features = self._normalize(features)
        
        logger.info(f"Extracted {len(feature_arrays)} features")
        return features
    
    def _handle_missing(self, features: xr.Dataset) -> xr.Dataset:
        """
        Handle missing values in features
        
        Args:
            features: Feature dataset
        
        Returns:
            Dataset with handled missing values
        """
        for var in features.data_vars:
            # Fill NaN with median
            median_val = float(features[var].median())
            features[var] = features[var].fillna(median_val)
        
        return features
    
    def _normalize(self, features: xr.Dataset) -> xr.Dataset:
        """
        Normalize features to [0, 1] range
        
        Args:
            features: Feature dataset
        
        Returns:
            Normalized dataset
        """
        normalized = features.copy()
        
        for var in features.data_vars:
            vmin = float(features[var].min())
            vmax = float(features[var].max())
            
            if vmax > vmin:
                normalized[var] = (features[var] - vmin) / (vmax - vmin)
            else:
                normalized[var] = features[var] * 0  # All same value -> 0
        
        return normalized
    
    def to_array(
        self,
        features: xr.Dataset,
        flatten: bool = True
    ) -> np.ndarray:
        """
        Convert feature dataset to numpy array
        
        Args:
            features: Feature dataset
            flatten: If True, flatten spatial dimensions
        
        Returns:
            Numpy array of shape (n_samples, n_features) if flatten=True,
            or (height, width, n_features) if flatten=False
        """
        # Stack features
        feature_list = [features[var].values for var in self.features if var in features]
        
        if not feature_list:
            raise ValueError("No features found in dataset")
        
        # Stack along new dimension
        arr = np.stack(feature_list, axis=-1)
        
        if flatten:
            # Flatten spatial dimensions
            height, width, n_features = arr.shape
            arr = arr.reshape(-1, n_features)
            
            # Remove rows with NaN
            arr = arr[~np.isnan(arr).any(axis=1)]
        
        return arr


def extract_features(
    datasets: Dict[str, xr.Dataset],
    feature_list: List[str],
    mask: Optional[xr.DataArray] = None,
    normalize: bool = True
) -> xr.Dataset:
    """
    Convenience function to extract features
    
    Args:
        datasets: Dictionary of aligned datasets
        feature_list: List of feature names to extract
        mask: Optional mask to filter pixels
        normalize: Whether to normalize features
    
    Returns:
        Dataset with extracted features
    """
    feature_arrays = {}
    
    for feature_name in feature_list:
        if feature_name in datasets:
            feature_arrays[feature_name] = datasets[feature_name]
        else:
            logger.warning(f"Feature {feature_name} not found")
    
    features = xr.Dataset(feature_arrays)
    
    if mask is not None:
        features = features.where(mask)
    
    if normalize:
        for var in features.data_vars:
            vmin = float(features[var].min())
            vmax = float(features[var].max())
            if vmax > vmin:
                features[var] = (features[var] - vmin) / (vmax - vmin)
    
    return features
