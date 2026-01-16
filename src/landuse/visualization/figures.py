"""
Generate all publication figures
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def generate_all_figures(
    results: Dict,
    output_dir: str,
    dpi: int = 300
) -> None:
    """
    Generate all publication figures
    
    Figures:
    - Figure 1: Environmental suitability
    - Figure 2: Policy matrix + Priority map
    - Figure 3: Carbon comparison (PV vs LNCS)
    - Figure 4: Cumulative benefit curves
    
    Args:
        results: Dictionary with all result datasets
        output_dir: Output directory
        dpi: Figure resolution
    """
    logger.info("Generating publication figures")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Environmental Suitability
    logger.info("Generating Figure 1: Environmental Suitability")
    generate_figure1(results, output_path, dpi)
    
    # Figure 2: Policy Matrix & Priority
    logger.info("Generating Figure 2: Policy Matrix & Priority")
    generate_figure2(results, output_path, dpi)
    
    # Figure 3: Carbon Comparison
    logger.info("Generating Figure 3: Carbon Comparison")
    generate_figure3(results, output_path, dpi)
    
    # Figure 4: Cumulative Benefits
    logger.info("Generating Figure 4: Cumulative Benefits")
    generate_figure4(results, output_path, dpi)
    
    logger.info(f"All figures saved to {output_path}")


def generate_figure1(results: Dict, output_path: Path, dpi: int) -> None:
    """Figure 1: Environmental suitability spatial distribution"""
    from .maps import plot_spatial_map
    
    env_suitability = results.get("environment")
    if env_suitability is None:
        logger.warning("Environmental suitability data not found, skipping Figure 1")
        return
    
    fig = plot_spatial_map(
        env_suitability,
        title="Environmental Suitability for PV Deployment",
        cmap="YlGnBu",
        dpi=dpi,
        save_path=str(output_path / "Figure1_environmental_suitability.pdf")
    )
    plt.close(fig)


def generate_figure2(results: Dict, output_path: Path, dpi: int) -> None:
    """Figure 2: Policy scenario matrix and priority map"""
    from .maps import plot_priority_map
    
    priority = results.get("priority")
    synergy = results.get("synergy")
    
    if priority is None or synergy is None:
        logger.warning("Priority/synergy data not found, skipping Figure 2")
        return
    
    fig = plot_priority_map(
        priority,
        synergy,
        title="Priority Ranking for PV Deployment",
        dpi=dpi,
        save_path=str(output_path / "Figure2_priority_map.pdf")
    )
    plt.close(fig)


def generate_figure3(results: Dict, output_path: Path, dpi: int) -> None:
    """Figure 3: Carbon emission reduction comparison"""
    pv_carbon = results.get("pv_carbon")
    lncs_carbon = results.get("lncs_carbon")
    
    if pv_carbon is None or lncs_carbon is None:
        logger.warning("Carbon data not found, skipping Figure 3")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    
    # Bar plot comparing PV vs LNCS
    categories = ['PV', 'LNCS', 'Net Reduction']
    values = [
        float(pv_carbon.sum()),
        float(lncs_carbon.sum()),
        float(pv_carbon.sum() - lncs_carbon.sum())
    ]
    
    bars = ax.bar(categories, values, color=['#2E86AB', '#A23B72', '#F18F01'])
    
    ax.set_ylabel('Carbon Reduction (Gt CO₂)', fontsize=12)
    ax.set_title('Carbon Emission Reduction Potential (2020-2050)', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / "Figure3_carbon_comparison.pdf", 
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def generate_figure4(results: Dict, output_path: Path, dpi: int) -> None:
    """Figure 4: Cumulative benefit curves"""
    cumulative = results.get("cumulative_benefits")
    
    if cumulative is None:
        logger.warning("Cumulative benefits data not found, skipping Figure 4")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=dpi)
    
    quantiles = cumulative["quantiles"]
    
    # Environment
    axes[0].plot(quantiles * 100, cumulative["cumulative_environment"], 
                 linewidth=2, color='#2E86AB')
    axes[0].set_xlabel('Deployment Fraction (%)', fontsize=11)
    axes[0].set_ylabel('Avg Suitability', fontsize=11)
    axes[0].set_title('Environmental Benefit', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Emission
    axes[1].plot(quantiles * 100, cumulative["cumulative_emission"], 
                 linewidth=2, color='#A23B72')
    axes[1].set_xlabel('Deployment Fraction (%)', fontsize=11)
    axes[1].set_ylabel('Cumulative Emission Reduction (Gt CO₂)', fontsize=11)
    axes[1].set_title('Emission Reduction', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    # Economic
    axes[2].plot(quantiles * 100, cumulative["cumulative_economic"], 
                 linewidth=2, color='#F18F01')
    axes[2].set_xlabel('Deployment Fraction (%)', fontsize=11)
    axes[2].set_ylabel('Cumulative NPV (Billion USD)', fontsize=11)
    axes[2].set_title('Economic Benefit', fontsize=12, fontweight='bold')
    axes[2].grid(alpha=0.3)
    
    plt.suptitle('Cumulative Benefits Along Priority Sequence', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / "Figure4_cumulative_benefits.pdf", 
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)
