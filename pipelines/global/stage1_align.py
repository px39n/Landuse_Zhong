"""
Stage 1: Spatial Alignment
Migrated from: 2.1 process_csv_for_aligning.ipynb

Aligns all datasets to common 1/120° grid
"""

import sys
from pathlib import Path
import yaml
import logging
import argparse
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from landuse.data import DataManifest, DataCatalog
from landuse.indicators import align_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def align_data(config: dict) -> None:
    """
    Align all datasets to common grid
    
    Process:
    1. Load abandonment mask, PV sites, environmental features
    2. Align to target resolution (1/120° ~ 1km)
    3. Export aligned datasets
    """
    logger.info("=== Stage 1: Spatial Alignment ===")
    
    manifest = DataManifest("manifest.json")
    manifest.load()
    
    catalog = DataCatalog(config)
    
    # Target resolution
    target_resolution = config["data"]["resolution"]
    logger.info(f"Target resolution: {target_resolution}° (~1km)")
    
    # TODO: Load datasets
    # This is a placeholder for the full implementation
    # Full implementation would:
    # 1. Load abandonment mask
    # 2. Load PV site locations
    # 3. Load all 15 environmental features
    # 4. Align using indicators.align_datasets()
    
    datasets = {}
    
    # Example: Load from catalog
    # abandonment_path = catalog.get_path("abandonment", "abandonment_mask.nc")
    # datasets["abandonment"] = xr.open_dataset(abandonment_path)
    
    logger.info(f"Loaded {len(datasets)} datasets")
    
    # Align datasets
    if datasets:
        aligned = align_datasets(
            datasets,
            target_resolution=target_resolution,
            method="bilinear"
        )
        
        # Save aligned datasets
        for name, ds in aligned.items():
            output_path = catalog.get_path("aligned", f"{name}_aligned.nc")
            ds.to_netcdf(output_path)
            
            manifest.register_artifact(
                stage="stage1_align",
                name=f"{name}_aligned",
                path=output_path,
                artifact_type="netcdf"
            )
            
            logger.info(f"Saved: {output_path}")
    
    manifest.save()
    logger.info("=== Stage 1 Complete ===")


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Spatial Alignment")
    parser.add_argument("--config", default="configs/global.yaml", help="Config file path")
    args = parser.parse_args()
    
    config = load_config(args.config)
    align_data(config)


if __name__ == "__main__":
    main()
