"""
Ranking optimization with efficiency kernels
"""

import numpy as np
from typing import Callable, Dict
import logging

logger = logging.getLogger(__name__)


class EfficiencyKernel:
    """
    Efficiency kernel functions for ranking optimization
    
    Kernel functions define preference over ranking positions:
    - Decreasing: Prefer high values at front (efficiency-focused)
    - Uniform: Equal weight to all positions
    - Increasing: Prefer distribution across all positions (equity-focused)
    """
    
    @staticmethod
    def decreasing(u: np.ndarray) -> np.ndarray:
        """
        Decreasing kernel: w(u) = 1 - u
        
        Prioritizes early positions (high-value items first)
        """
        return 1 - u
    
    @staticmethod
    def uniform(u: np.ndarray) -> np.ndarray:
        """
        Uniform kernel: w(u) = 1
        
        Equal weight to all positions
        """
        return np.ones_like(u)
    
    @staticmethod
    def increasing(u: np.ndarray) -> np.ndarray:
        """
        Increasing kernel: w(u) = u
        
        Prioritizes later positions (equity-focused)
        """
        return u
    
    @staticmethod
    def exponential(u: np.ndarray, alpha: float = 2.0) -> np.ndarray:
        """
        Exponential kernel: w(u) = exp(-alpha * u)
        
        Strongly prioritizes early positions
        """
        return np.exp(-alpha * u)
    
    @staticmethod
    def power(u: np.ndarray, p: float = 2.0) -> np.ndarray:
        """
        Power kernel: w(u) = (1 - u)^p
        
        Adjustable priority curve
        """
        return (1 - u) ** p


def optimize_ranking(
    values: np.ndarray,
    areas: np.ndarray,
    kernel_type: str = "decreasing",
    kernel_params: Dict = None
) -> Tuple[np.ndarray, float]:
    """
    Optimize ranking with efficiency kernel
    
    Args:
        values: Benefit values per pixel
        areas: Pixel areas
        kernel_type: Type of efficiency kernel
        kernel_params: Additional kernel parameters
    
    Returns:
        Tuple of (optimal_ranking, efficiency_score)
    """
    if kernel_params is None:
        kernel_params = {}
    
    n = len(values)
    total_values = values * areas
    
    # Create kernel weights
    u = np.arange(1, n + 1) / n
    
    if kernel_type == "decreasing":
        weights = EfficiencyKernel.decreasing(u)
    elif kernel_type == "uniform":
        weights = EfficiencyKernel.uniform(u)
    elif kernel_type == "increasing":
        weights = EfficiencyKernel.increasing(u)
    elif kernel_type == "exponential":
        alpha = kernel_params.get("alpha", 2.0)
        weights = EfficiencyKernel.exponential(u, alpha)
    elif kernel_type == "power":
        p = kernel_params.get("p", 2.0)
        weights = EfficiencyKernel.power(u, p)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # For decreasing kernels, sort by value (descending)
    if kernel_type in ["decreasing", "exponential", "power"]:
        ranking = np.argsort(-total_values)
    # For increasing kernels, sort by value (ascending)
    elif kernel_type == "increasing":
        ranking = np.argsort(total_values)
    # For uniform, any ranking gives same result, use value-based
    else:
        ranking = np.argsort(-total_values)
    
    # Calculate efficiency
    ranked_values = total_values[ranking]
    efficiency = np.sum(ranked_values * weights)
    
    return ranking, efficiency
