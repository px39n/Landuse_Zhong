"""
Economic feasibility analysis
NPV calculation across policy scenarios
"""

from .npv import calculate_npv, NPVCalculator
from .scenarios import load_ar6_scenarios

__all__ = [
    "calculate_npv",
    "NPVCalculator",
    "load_ar6_scenarios",
]
