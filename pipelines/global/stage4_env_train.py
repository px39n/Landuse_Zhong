"""
Stage 4: Environmental Suitability Training
Migrated from: 3.0 pre-training.ipynb

Two-stage machine learning:
1. GMM for environmental structure identification
2. Transformer-ResNet for binary classification
"""

import sys
from pathlib import Path
import yaml
import logging
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from landuse.data import DataManifest, DataCatalog
from landuse.env_model import (
    GMMTrainer,
    TransformerResNetClassifier,
    NegativeSampler
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_environmental_model(config: dict) -> None:
    """
    Train environmental suitability model
    
    Process:
    1. Load positive samples (PV sites)
    2. Train GMM to identify environmental clusters
    3. Generate negative samples from low-density regions
    4. Train Transformer-ResNet classifier
    5. Evaluate and save models
    """
    logger.info("=== Stage 4: Environmental Suitability Training ===")
    
    manifest = DataManifest("manifest.json")
    manifest.load()
    
    catalog = DataCatalog(config)
    
    # TODO: Load training data
    # This is a placeholder for the full implementation
    # Full implementation would:
    # 1. Load PV site features (positive samples)
    # 2. Load abandonment candidate features
    
    logger.info("Loading training data...")
    
    # Placeholder data
    # In full version, load from:
    # - stage2_embed output: PV features
    # - stage3_predprep output: Candidate features
    
    X_positive = np.random.randn(1000, 15)  # Placeholder
    X_candidates = np.random.randn(100000, 15)  # Placeholder
    
    # ===== Step 1: Train GMM =====
    logger.info("Step 1: Training GMM...")
    
    gmm_trainer = GMMTrainer(config)
    gmm_pipeline = gmm_trainer.train(
        X_positive,
        search_components=False
    )
    
    # Calibrate GMM
    gmm_trainer.calibrate(X_positive[:200])  # Use subset for calibration
    
    # Save GMM
    gmm_path = catalog.get_path("models", "gmm_pipeline.pkl")
    gmm_trainer.save(gmm_path)
    logger.info(f"GMM saved: {gmm_path}")
    
    # ===== Step 2: Negative Sampling =====
    logger.info("Step 2: Generating negative samples...")
    
    # Score candidates with GMM
    log_density = gmm_trainer.score_samples(X_candidates, method='log_density')
    
    # Sample negatives from low-density regions
    df_candidates = pd.DataFrame(X_candidates, columns=[f"f{i}" for i in range(15)])
    
    sampler = NegativeSampler(config)
    df_negative = sampler.sample(
        df_candidates,
        log_density,
        n_positive=len(X_positive)
    )
    
    X_negative = df_negative.values
    logger.info(f"Generated {len(X_negative)} negative samples")
    
    # ===== Step 3: Train Transformer-ResNet =====
    logger.info("Step 3: Training Transformer-ResNet classifier...")
    
    # Combine positive and negative samples
    X_train = np.vstack([X_positive, X_negative])
    y_train = np.hstack([
        np.ones(len(X_positive)),
        np.zeros(len(X_negative))
    ])
    
    # Split train/val/test
    split_config = config["model"]["split"]
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train,
        test_size=split_config["test"],
        random_state=split_config["random_state"]
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=split_config["val"] / (1 - split_config["test"]),
        random_state=split_config["random_state"]
    )
    
    logger.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # Build and train model
    classifier = TransformerResNetClassifier(config)
    classifier.build()
    
    history = classifier.train(
        X_train, y_train,
        X_val, y_val,
        epochs=config["model"]["transformer_resnet"]["epochs"],
        batch_size=config["model"]["transformer_resnet"]["batch_size"]
    )
    
    # Evaluate
    metrics = classifier.evaluate(X_test, y_test)
    logger.info(f"Test metrics: {metrics}")
    
    # Save model
    model_path = catalog.get_path("models", "transformer_resnet.h5")
    classifier.save(model_path)
    logger.info(f"Model saved: {model_path}")
    
    # Register in manifest
    manifest.register_artifact(
        stage="stage4_env_train",
        name="gmm_pipeline",
        path=gmm_path,
        artifact_type="model",
        metadata={
            "n_components": gmm_trainer.n_components,
            "n_positive": len(X_positive)
        }
    )
    
    manifest.register_artifact(
        stage="stage4_env_train",
        name="transformer_resnet",
        path=model_path,
        artifact_type="model",
        metadata=metrics
    )
    
    manifest.save()
    logger.info("=== Stage 4 Complete ===")


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Environmental Training")
    parser.add_argument("--config", default="configs/global.yaml", help="Config file path")
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_environmental_model(config)


if __name__ == "__main__":
    main()
