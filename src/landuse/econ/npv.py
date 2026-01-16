"""
Net Present Value (NPV) calculations
Migrated from: 5.1 Economical_feasibility.ipynb
"""

import numpy as np
import xarray as xr
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class NPVCalculator:
    """
    Calculate Net Present Value for PV deployment
    """
    
    def __init__(self, config: Dict):
        """
        Initialize NPV calculator
        
        Args:
            config: Configuration with economic parameters
        """
        self.config = config
        econ_config = config.get("economics", {})
        
        self.baseline_year = econ_config.get("baseline_year", 2020)
        self.target_year = econ_config.get("target_year", 2050)
        self.time_step = econ_config.get("time_step", 10)
        self.discount_rate = econ_config.get("discount_rate", 0.05)
        
        self.scenarios = econ_config.get("scenarios", [])
    
    def calculate(
        self,
        pv_generation: xr.DataArray,
        pv_capacity: xr.DataArray,
        lncs_opportunity_cost: Dict[str, xr.DataArray],
        scenario_prices: Dict[str, xr.DataArray]
    ) -> Dict[str, xr.DataArray]:
        """
        Calculate NPV for each scenario
        
        NPV = Σ[(Revenue - OPEX) / (1+r)^t] - CAPEX - LNCS_cost
        
        Args:
            pv_generation: Annual PV generation (kWh/m²/year)
            pv_capacity: PV capacity (kW/m²)
            lncs_opportunity_cost: LNCS opportunity costs by strategy
            scenario_prices: Electricity prices by scenario
        
        Returns:
            NPV by scenario
        """
        logger.info("Calculating NPV across scenarios")
        
        npv_results = {}
        
        for scenario in self.scenarios:
            scenario_name = scenario["name"]
            logger.info(f"Processing scenario: {scenario_name}")
            
            # Get prices for this scenario
            prices = scenario_prices.get(scenario_name)
            if prices is None:
                logger.warning(f"No prices for scenario {scenario_name}, skipping")
                continue
            
            # Calculate NPV components
            revenue_pv = self._calculate_pv_revenue(pv_generation, prices)
            capex = self._calculate_capex(pv_capacity)
            opex = self._calculate_opex(pv_capacity)
            lncs_cost = self._calculate_lncs_cost(lncs_opportunity_cost)
            
            # Net present value
            npv = revenue_pv - capex - opex - lncs_cost
            
            npv_results[scenario_name] = npv
            
            logger.info(
                f"Scenario {scenario_name}: "
                f"mean NPV = {float(npv.mean()):.2f} k USD/ha"
            )
        
        return npv_results
    
    def _calculate_pv_revenue(
        self,
        generation: xr.DataArray,
        prices: xr.DataArray
    ) -> xr.DataArray:
        """Calculate discounted PV revenue stream"""
        years = range(
            self.baseline_year,
            self.target_year + 1,
            self.time_step
        )
        
        total_revenue = xr.zeros_like(generation)
        
        for i, year in enumerate(years):
            t = i * self.time_step
            discount_factor = 1 / (1 + self.discount_rate) ** t
            
            # Annual revenue discounted
            revenue_t = generation * prices * discount_factor * self.time_step
            total_revenue += revenue_t
        
        return total_revenue
    
    def _calculate_capex(self, capacity: xr.DataArray) -> xr.DataArray:
        """Calculate capital expenditure"""
        # Typical solar CAPEX: $1,000 - $1,500 per kW (2020 dollars)
        capex_per_kw = 1200
        return capacity * capex_per_kw
    
    def _calculate_opex(self, capacity: xr.DataArray) -> xr.DataArray:
        """Calculate operational expenditure (discounted)"""
        # Typical O&M: $15-25 per kW per year
        opex_per_kw_year = 20
        
        years = range(
            self.baseline_year,
            self.target_year + 1,
            self.time_step
        )
        
        total_opex = xr.zeros_like(capacity)
        
        for i, year in enumerate(years):
            t = i * self.time_step
            discount_factor = 1 / (1 + self.discount_rate) ** t
            opex_t = capacity * opex_per_kw_year * discount_factor * self.time_step
            total_opex += opex_t
        
        return total_opex
    
    def _calculate_lncs_cost(
        self,
        lncs_costs: Dict[str, xr.DataArray]
    ) -> xr.DataArray:
        """Calculate LNCS opportunity cost"""
        # Weighted average of LNCS strategies
        # Simplified: equal weights
        
        if not lncs_costs:
            return 0
        
        total_cost = sum(lncs_costs.values()) / len(lncs_costs)
        return total_cost


def calculate_npv(
    pv_generation: xr.DataArray,
    electricity_price: float,
    capex_per_kw: float = 1200,
    opex_per_kw_year: float = 20,
    discount_rate: float = 0.05,
    lifetime: int = 30
) -> xr.DataArray:
    """
    Simplified NPV calculation
    
    Args:
        pv_generation: Annual generation (kWh/m²/year)
        electricity_price: Electricity price ($/kWh)
        capex_per_kw: Capital cost per kW
        opex_per_kw_year: Annual O&M per kW
        discount_rate: Discount rate
        lifetime: Project lifetime (years)
    
    Returns:
        NPV ($/m²)
    """
    # Annual revenue
    annual_revenue = pv_generation * electricity_price
    
    # Annual OPEX
    capacity_kw = 0.17  # kW/m²
    annual_opex = capacity_kw * opex_per_kw_year
    
    # Discounted cash flows
    npv = 0
    for t in range(lifetime):
        discount_factor = 1 / (1 + discount_rate) ** t
        npv += (annual_revenue - annual_opex) * discount_factor
    
    # Subtract CAPEX
    capex = capacity_kw * capex_per_kw
    npv -= capex
    
    return npv
