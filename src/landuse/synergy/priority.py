"""
Priority ranking based on 3E-Synergy Index
"""

import numpy as np
import xarray as xr
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class PriorityRanker:
    """
    Rank pixels by priority for PV deployment
    """
    
    def __init__(self, config: Dict):
        """
        Initialize priority ranker
        
        Args:
            config: Configuration with ranking parameters
        """
        self.config = config
        ranking_config = config.get("synergy", {}).get("ranking", {})
        
        self.attention_kernels = ranking_config.get("attention_kernels", [
            "efficiency",
            "equity",
            "robustness"
        ])
    
    def rank(
        self,
        synergy: xr.DataArray,
        environment: xr.DataArray,
        emission: xr.DataArray,
        economic: xr.DataArray
    ) -> xr.DataArray:
        """
        Rank pixels by priority
        
        Args:
            synergy: 3E-Synergy index
            environment: Environmental suitability
            emission: Emission reduction
            economic: Economic feasibility
        
        Returns:
            Priority ranks (1 = highest priority)
        """
        logger.info("Ranking pixels by priority")
        
        # Flatten and sort by synergy
        synergy_flat = synergy.values.flatten()
        valid_mask = ~np.isnan(synergy_flat)
        
        # Get indices sorted by synergy (descending)
        sorted_indices = np.argsort(-synergy_flat[valid_mask])
        
        # Assign ranks
        ranks = np.full_like(synergy_flat, np.nan)
        ranks[valid_mask] = np.argsort(sorted_indices) + 1
        
        # Reshape to original shape
        priority = xr.DataArray(
            ranks.reshape(synergy.shape),
            coords=synergy.coords,
            dims=synergy.dims
        )
        
        logger.info(f"Ranked {np.sum(valid_mask)} valid pixels")
        
        return priority
    
    def calculate_cumulative_benefits(
        self,
        priority: xr.DataArray,
        environment: xr.DataArray,
        emission: xr.DataArray,
        economic: xr.DataArray,
        n_quantiles: int = 100
    ) -> Dict:
        """
        Calculate cumulative benefits along priority sequence
        
        Args:
            priority: Priority ranks
            environment: Environmental suitability
            emission: Emission reduction
            economic: Economic feasibility
            n_quantiles: Number of quantiles to compute
        
        Returns:
            Dictionary with cumulative benefit curves
        """
        logger.info("Calculating cumulative benefits")
        
        # Flatten arrays
        priority_flat = priority.values.flatten()
        env_flat = environment.values.flatten()
        emission_flat = emission.values.flatten()
        econ_flat = economic.values.flatten()
        
        # Valid mask
        valid_mask = ~np.isnan(priority_flat)
        
        # Sort by priority
        sorted_indices = np.argsort(priority_flat[valid_mask])
        
        env_sorted = env_flat[valid_mask][sorted_indices]
        emission_sorted = emission_flat[valid_mask][sorted_indices]
        econ_sorted = econ_flat[valid_mask][sorted_indices]
        
        # Calculate cumulative sums
        n_total = len(env_sorted)
        quantile_indices = np.linspace(0, n_total, n_quantiles, dtype=int)
        
        cumulative_env = np.zeros(n_quantiles)
        cumulative_emission = np.zeros(n_quantiles)
        cumulative_econ = np.zeros(n_quantiles)
        
        for i, idx in enumerate(quantile_indices):
            cumulative_env[i] = np.mean(env_sorted[:idx]) if idx > 0 else 0
            cumulative_emission[i] = np.sum(emission_sorted[:idx])
            cumulative_econ[i] = np.sum(econ_sorted[:idx])
        
        return {
            "quantiles": np.linspace(0, 1, n_quantiles),
            "cumulative_environment": cumulative_env,
            "cumulative_emission": cumulative_emission,
            "cumulative_economic": cumulative_econ,
            "n_pixels": n_total
        }


def rank_priority(
    synergy: xr.DataArray,
    descending: bool = True
) -> xr.DataArray:
    """
    Simple priority ranking function
    
    Args:
        synergy: 3E-Synergy index
        descending: True for high synergy = high priority
    
    Returns:
        Priority ranks
    """
    synergy_flat = synergy.values.flatten()
    valid_mask = ~np.isnan(synergy_flat)
    
    if descending:
        sorted_indices = np.argsort(-synergy_flat[valid_mask])
    else:
        sorted_indices = np.argsort(synergy_flat[valid_mask])
    
    ranks = np.full_like(synergy_flat, np.nan)
    ranks[valid_mask] = np.argsort(sorted_indices) + 1
    
    return xr.DataArray(
        ranks.reshape(synergy.shape),
        coords=synergy.coords,
        dims=synergy.dims
    )
