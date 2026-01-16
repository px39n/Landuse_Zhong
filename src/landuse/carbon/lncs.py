"""
Land-based Natural Climate Solutions (LNCS) carbon sequestration
"""

import numpy as np
import xarray as xr
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def calculate_lncs_carbon(
    land_use_history: xr.DataArray,
    biomass: xr.DataArray,
    soc: xr.DataArray,
    config: Dict
) -> Dict[str, xr.DataArray]:
    """
    Calculate LNCS carbon sequestration potential
    
    Three strategies:
    1. Afforestation: Above-ground biomass + Below-ground biomass + SOC + litter
    2. Agriculture: Perennial/annual crops, mainly SOC changes
    3. Non-woody Vegetation: Below-ground biomass + SOC potential
    
    Args:
        land_use_history: Historical land use patterns
        biomass: Above-ground biomass data
        soc: Soil organic carbon data
        config: Configuration with LNCS parameters
    
    Returns:
        Dictionary of carbon sequestration by strategy
    """
    logger.info("Calculating LNCS carbon sequestration")
    
    lncs_config = config.get("carbon", {}).get("lncs", {})
    strategies = lncs_config.get("strategies", [
        "afforestation",
        "agriculture",
        "non_woody_vegetation"
    ])
    
    results = {}
    
    # Afforestation
    if "afforestation" in strategies:
        # Above-ground biomass to carbon (typically 0.5 conversion factor)
        agb_carbon = biomass * 0.5
        
        # Below-ground biomass (typically 20-30% of AGB)
        bgb_carbon = agb_carbon * 0.25
        
        # SOC accumulation (simplified)
        soc_carbon = soc * 0.1  # 10% increase over 30 years
        
        # Litter (5% of AGB)
        litter_carbon = agb_carbon * 0.05
        
        results["afforestation"] = agb_carbon + bgb_carbon + soc_carbon + litter_carbon
    
    # Agriculture
    if "agriculture" in strategies:
        # Mainly SOC changes for perennial crops
        results["agriculture"] = soc * 0.15  # 15% increase
    
    # Non-woody vegetation
    if "non_woody_vegetation" in strategies:
        # Grassland/shrubland recovery
        results["non_woody_vegetation"] = biomass * 0.3 + soc * 0.08
    
    logger.info(f"Calculated LNCS for {len(results)} strategies")
    
    return results


def allocate_lncs_strategies(
    land_use_history: xr.DataArray,
    spatial_features: Dict[str, xr.DataArray],
    config: Dict
) -> Dict[str, xr.DataArray]:
    """
    Allocate LNCS strategies based on historical land use preferences
    
    Uses K-d tree + inverse distance weighting (IDW) to assign probabilities
    
    Args:
        land_use_history: Historical land use patterns
        spatial_features: Spatial features (climate, soil, etc.)
        config: Configuration with allocation parameters
    
    Returns:
        Dictionary of strategy allocation probabilities
    """
    allocation_config = config.get("carbon", {}).get("lncs", {}).get("allocation", {})
    
    grid_size = allocation_config.get("grid_size", 10)
    method = allocation_config.get("method", "idw")
    k_neighbors = allocation_config.get("k_neighbors", 5)
    
    logger.info(f"Allocating LNCS strategies using {method} method")
    
    # Simplified allocation (placeholder for full implementation)
    # In full version: 
    # 1. Aggregate to grid_size
    # 2. Analyze historical preferences
    # 3. Use K-d tree for spatial interpolation
    # 4. Apply IDW weighting
    
    strategies = ["afforestation", "agriculture", "non_woody_vegetation"]
    
    # Placeholder: uniform probabilities
    probabilities = {}
    for strategy in strategies:
        probabilities[strategy] = xr.ones_like(land_use_history) / len(strategies)
    
    logger.info("LNCS strategy allocation complete")
    
    return probabilities
