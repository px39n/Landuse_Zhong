"""
Carbon emission reduction assessment
PV vs LNCS (Land-based Natural Climate Solutions)
"""

from .pv_emission import calculate_pv_emission_reduction
from .lncs import calculate_lncs_carbon, allocate_lncs_strategies

__all__ = [
    "calculate_pv_emission_reduction",
    "calculate_lncs_carbon",
    "allocate_lncs_strategies",
]
