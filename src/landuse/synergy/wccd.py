"""
Weighted Coupling Coordination Degree (WCCD) calculation
Migrated from: 6.4 3E_synergy_index.ipynb
"""

import numpy as np
import xarray as xr
from typing import Dict, Tuple
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class WCCDCalculator:
    """
    Calculate 3E-Synergy Index using WCCD method
    
    Dimensions:
    - Environment: Deployment probability
    - Emission: Net emission reduction (t CO₂/ha)
    - Economic: Average NPV (k USD/ha)
    
    Formula:
        Coupling degree: C = (∏U_i / (⅓∑U_i)³)^(1/3)
        Coordination degree: T = Σw_i × U_i
        3E-synergy = √(C × T)
    
    Constraint: Σw_i = 1, w_i ≥ 0
    """
    
    def __init__(self, config: Dict):
        """
        Initialize WCCD calculator
        
        Args:
            config: Configuration with synergy parameters
        """
        self.config = config
        synergy_config = config.get("synergy", {})
        
        self.dimensions = synergy_config.get("dimensions", {})
        self.wccd_config = synergy_config.get("wccd", {})
        
        self.method = self.wccd_config.get("method", "SLSQP")
        self.adaptive_weights = self.wccd_config.get("adaptive_weights", True)
    
    def calculate(
        self,
        environment: xr.DataArray,
        emission: xr.DataArray,
        economic: xr.DataArray
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Calculate 3E-Synergy Index
        
        Args:
            environment: Environmental suitability (0-1)
            emission: Net emission reduction (t CO₂/ha)
            economic: Economic NPV (k USD/ha)
        
        Returns:
            Tuple of (synergy_index, optimal_weights)
        """
        logger.info("Calculating 3E-Synergy Index")
        
        # Normalize indicators to [0, 1]
        U_env = self._normalize(environment)
        U_emission = self._normalize(emission)
        U_econ = self._normalize(economic)
        
        if self.adaptive_weights:
            # Optimize weights pixel-by-pixel
            synergy, weights = self._calculate_adaptive(U_env, U_emission, U_econ)
        else:
            # Use fixed equal weights
            weights = np.array([1/3, 1/3, 1/3])
            synergy = self._calculate_wccd(U_env, U_emission, U_econ, weights)
        
        logger.info(f"3E-Synergy calculated: mean={float(synergy.mean()):.3f}")
        
        return synergy, weights
    
    def _normalize(self, data: xr.DataArray) -> xr.DataArray:
        """Normalize to [0, 1]"""
        vmin = float(data.min())
        vmax = float(data.max())
        
        if vmax > vmin:
            normalized = (data - vmin) / (vmax - vmin)
        else:
            normalized = xr.zeros_like(data)
        
        return normalized
    
    def _calculate_wccd(
        self,
        U1: xr.DataArray,
        U2: xr.DataArray,
        U3: xr.DataArray,
        weights: np.ndarray
    ) -> xr.DataArray:
        """
        Calculate WCCD with given weights
        
        Args:
            U1, U2, U3: Normalized indicators
            weights: Weights (must sum to 1)
        
        Returns:
            WCCD synergy index
        """
        w1, w2, w3 = weights
        
        # Coupling degree
        product = U1 * U2 * U3
        sum_avg = (U1 + U2 + U3) / 3
        C = (product / (sum_avg ** 3)) ** (1/3)
        
        # Handle division by zero
        C = xr.where(sum_avg > 1e-6, C, 0)
        
        # Coordination degree
        T = w1 * U1 + w2 * U2 + w3 * U3
        
        # Synergy index
        synergy = np.sqrt(C * T)
        
        return synergy
    
    def _calculate_adaptive(
        self,
        U1: xr.DataArray,
        U2: xr.DataArray,
        U3: xr.DataArray
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Calculate WCCD with adaptive (optimized) weights per pixel
        
        Returns:
            Tuple of (synergy_index, optimal_weights)
        """
        logger.info("Optimizing weights adaptively")
        
        # Convert to numpy arrays
        u1 = U1.values.flatten()
        u2 = U2.values.flatten()
        u3 = U3.values.flatten()
        
        n_pixels = len(u1)
        synergy_values = np.zeros(n_pixels)
        weight_values = np.zeros((n_pixels, 3))
        
        # Optimize for each pixel (can be parallelized)
        for i in range(n_pixels):
            if i % 10000 == 0:
                logger.info(f"Optimizing pixel {i}/{n_pixels}")
            
            # Objective: maximize WCCD
            def objective(w):
                C = ((u1[i] * u2[i] * u3[i]) / 
                     (((u1[i] + u2[i] + u3[i]) / 3) ** 3 + 1e-9)) ** (1/3)
                T = w[0] * u1[i] + w[1] * u2[i] + w[2] * u3[i]
                return -(C * T) ** 0.5  # Negative for minimization
            
            # Constraints
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1), (0, 1), (0, 1)]
            
            # Initial guess
            w0 = np.array([1/3, 1/3, 1/3])
            
            # Optimize
            result = minimize(
                objective,
                w0,
                method=self.method,
                bounds=bounds,
                constraints=constraints
            )
            
            weight_values[i] = result.x
            synergy_values[i] = -result.fun
        
        # Reshape back to original shape
        synergy = xr.DataArray(
            synergy_values.reshape(U1.shape),
            coords=U1.coords,
            dims=U1.dims
        )
        
        weights = xr.DataArray(
            weight_values.reshape((*U1.shape, 3)),
            coords={**U1.coords, 'dimension': ['environment', 'emission', 'economic']},
            dims=[*U1.dims, 'dimension']
        )
        
        return synergy, weights


def calculate_3e_synergy(
    environment: xr.DataArray,
    emission: xr.DataArray,
    economic: xr.DataArray,
    adaptive_weights: bool = True
) -> xr.DataArray:
    """
    Convenience function to calculate 3E-Synergy
    
    Args:
        environment: Environmental suitability
        emission: Emission reduction potential
        economic: Economic feasibility
        adaptive_weights: Whether to optimize weights per pixel
    
    Returns:
        3E-Synergy index
    """
    config = {
        'synergy': {
            'wccd': {
                'method': 'SLSQP',
                'adaptive_weights': adaptive_weights
            }
        }
    }
    
    calculator = WCCDCalculator(config)
    synergy, _ = calculator.calculate(environment, emission, economic)
    
    return synergy
