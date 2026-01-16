"""
Stage 8: 3E-Synergy Index Calculation
Migrated from: 6.4 3E_synergy_index.ipynb

Calculates Weighted Coupling Coordination Degree (WCCD)
"""

import sys
from pathlib import Path
import yaml
import logging
import argparse
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from landuse.data import DataManifest, DataCatalog
from landuse.synergy import WCCDCalculator, PriorityRanker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def calculate_synergy(config: dict) -> None:
    """
    Calculate 3E-Synergy Index
    
    Process:
    1. Load three dimensions:
       - Environment: Deployment probability (Stage 5)
       - Emission: Net emission reduction (Stage 6)
       - Economic: Average NPV (Stage 7)
    2. Calculate WCCD with adaptive weights
    3. Rank pixels by priority
    4. Calculate cumulative benefit curves
    5. Export results
    """
    logger.info("=== Stage 8: 3E-Synergy Index ===")
    
    manifest = DataManifest("manifest.json")
    manifest.load()
    
    catalog = DataCatalog(config)
    
    # TODO: Load input data
    # This is a placeholder for the full implementation
    
    logger.info("Loading 3E dimensions...")
    
    # Placeholder: Load from previous stages
    # environment = xr.open_dataarray(catalog.get_path("results", "env_probability.nc"))
    # emission = xr.open_dataarray(catalog.get_path("results", "net_emission.nc"))
    # economic = xr.open_dataarray(catalog.get_path("results", "npv_mean.nc"))
    
    # Placeholder data
    import numpy as np
    
    lon = np.arange(-125, -66, 0.1)
    lat = np.arange(24, 50, 0.1)
    
    environment = xr.DataArray(
        np.random.rand(len(lat), len(lon)),
        coords=[("lat", lat), ("lon", lon)]
    )
    emission = xr.DataArray(
        np.random.rand(len(lat), len(lon)) * 5000,  # t CO2/ha
        coords=[("lat", lat), ("lon", lon)]
    )
    economic = xr.DataArray(
        np.random.randn(len(lat), len(lon)) * 1000,  # k USD/ha
        coords=[("lat", lat), ("lon", lon)]
    )
    
    logger.info("Calculating WCCD...")
    
    # Calculate 3E-Synergy
    wccd_calculator = WCCDCalculator(config)
    synergy, weights = wccd_calculator.calculate(
        environment,
        emission,
        economic
    )
    
    logger.info(f"3E-Synergy: mean={float(synergy.mean()):.3f}, std={float(synergy.std()):.3f}")
    
    # Rank priorities
    logger.info("Ranking priorities...")
    
    ranker = PriorityRanker(config)
    priority = ranker.rank(synergy, environment, emission, economic)
    
    # Calculate cumulative benefits
    cumulative = ranker.calculate_cumulative_benefits(
        priority,
        environment,
        emission,
        economic,
        n_quantiles=100
    )
    
    logger.info(f"Ranked {cumulative['n_pixels']} pixels")
    
    # Save results
    output_paths = {}
    
    synergy_path = catalog.get_path("results", "3e_synergy.nc")
    synergy.to_netcdf(synergy_path)
    output_paths["synergy"] = synergy_path
    logger.info(f"Saved: {synergy_path}")
    
    weights_path = catalog.get_path("results", "3e_weights.nc")
    weights.to_netcdf(weights_path)
    output_paths["weights"] = weights_path
    logger.info(f"Saved: {weights_path}")
    
    priority_path = catalog.get_path("results", "priority_ranks.nc")
    priority.to_netcdf(priority_path)
    output_paths["priority"] = priority_path
    logger.info(f"Saved: {priority_path}")
    
    # Save cumulative benefits
    import json
    cumulative_path = catalog.get_path("results", "cumulative_benefits.json")
    with open(cumulative_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        cumulative_json = {
            k: v.tolist() if hasattr(v, 'tolist') else v
            for k, v in cumulative.items()
        }
        json.dump(cumulative_json, f, indent=2)
    output_paths["cumulative"] = cumulative_path
    logger.info(f"Saved: {cumulative_path}")
    
    # Register outputs
    for name, path in output_paths.items():
        manifest.register_artifact(
            stage="stage8_synergy",
            name=name,
            path=path,
            artifact_type="netcdf" if path.endswith(".nc") else "json"
        )
    
    manifest.save()
    logger.info("=== Stage 8 Complete ===")


def main():
    parser = argparse.ArgumentParser(description="Stage 8: 3E-Synergy Index")
    parser.add_argument("--config", default="configs/global.yaml", help="Config file path")
    args = parser.parse_args()
    
    config = load_config(args.config)
    calculate_synergy(config)


if __name__ == "__main__":
    main()
