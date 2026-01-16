"""
Photovoltaic emission reduction calculations
Migrated from: 4.1 Emission_reduction_potential.ipynb
"""

import numpy as np
import xarray as xr
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def calculate_pv_emission_reduction(
    solar_radiation: xr.DataArray,
    temperature: xr.DataArray,
    config: Dict
) -> xr.DataArray:
    """
    Calculate PV emission reduction potential
    
    Formula:
        PV_POT = P_R × (I/I_STC) × capacity_density × annual_hours
        P_R = 1 + γ × (T_cell - T_STC)
        Carbon = PV_generation × EF_CM × η_sys × lifetime
    
    Args:
        solar_radiation: Solar radiation data (W/m²)
        temperature: Near-surface temperature data (°C)
        config: Configuration with PV parameters
    
    Returns:
        Total carbon reduction (t CO₂) over lifetime
    """
    pv_config = config.get("carbon", {}).get("pv", {})
    
    # Parameters
    capacity_density = pv_config.get("capacity_density", 0.17)  # kW/m²
    system_efficiency = pv_config.get("system_efficiency", 0.8)
    annual_hours = pv_config.get("annual_hours", 8760)
    lifetime = pv_config.get("lifetime", 30)  # years
    temp_coefficient = pv_config.get("temperature_coefficient", -0.004)  # per °C
    
    # Standard Test Conditions
    I_STC = 1000  # W/m²
    T_STC = 25  # °C
    
    logger.info("Calculating PV emission reduction")
    
    # Temperature correction factor
    P_R = 1 + temp_coefficient * (temperature - T_STC)
    
    # Annual generation potential (kWh/m²/year)
    annual_generation = (
        P_R * (solar_radiation / I_STC) *
        capacity_density * annual_hours * system_efficiency
    )
    
    # Lifetime generation (kWh/m²)
    lifetime_generation = annual_generation * lifetime
    
    # Carbon intensity (kg CO₂/kWh) - US grid average
    # Value from IEA: ~0.4 kg CO₂/kWh
    carbon_intensity = 0.4
    
    # Total carbon reduction (t CO₂/m²)
    carbon_reduction = lifetime_generation * carbon_intensity / 1000
    
    logger.info(
        f"PV emission reduction: "
        f"mean={float(carbon_reduction.mean()):.2f} t CO₂/m²"
    )
    
    return carbon_reduction
