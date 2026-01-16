"""
Negative sampling strategies for binary classification
Migrated from: function/negative_sampling.py
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class NegativeSampler:
    """
    Generate negative samples using GMM log-density scores
    """
    
    def __init__(self, config: Dict):
        """
        Initialize negative sampler
        
        Args:
            config: Configuration dictionary with sampling parameters
        """
        self.config = config
        sampling_config = config.get("model", {}).get("negative_sampling", {})
        
        self.ratio = sampling_config.get("ratio", 1.0)
        self.method = sampling_config.get("method", "stratified")
        self.density_threshold = sampling_config.get("density_threshold", 5)
        self.random_state = sampling_config.get("random_state", 42)
    
    def sample(
        self,
        candidates: pd.DataFrame,
        log_density: np.ndarray,
        n_positive: int
    ) -> pd.DataFrame:
        """
        Sample negative samples from candidates
        
        Args:
            candidates: Candidate pixel features
            log_density: GMM log-density scores for candidates
            n_positive: Number of positive samples (to determine negative count)
        
        Returns:
            Sampled negative samples
        """
        n_negative = int(n_positive * self.ratio)
        
        logger.info(f"Sampling {n_negative} negative samples (ratio={self.ratio})")
        
        if self.method == "stratified":
            return self._stratified_sample(candidates, log_density, n_negative)
        elif self.method == "threshold":
            return self._threshold_sample(candidates, log_density, n_negative)
        elif self.method == "random":
            return self._random_sample(candidates, n_negative)
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")
    
    def _stratified_sample(
        self,
        candidates: pd.DataFrame,
        log_density: np.ndarray,
        n_samples: int
    ) -> pd.DataFrame:
        """
        Stratified sampling based on log-density percentiles
        
        Focuses on low-density regions (below threshold percentile)
        """
        threshold = np.percentile(log_density, self.density_threshold)
        
        # Filter low-density candidates
        low_density_mask = log_density < threshold
        low_density_candidates = candidates[low_density_mask].copy()
        
        if len(low_density_candidates) < n_samples:
            logger.warning(
                f"Not enough low-density candidates ({len(low_density_candidates)}). "
                f"Using all and sampling additional from medium density."
            )
            
            # Use all low-density + sample from medium density
            n_additional = n_samples - len(low_density_candidates)
            medium_density_mask = ~low_density_mask
            medium_candidates = candidates[medium_density_mask]
            
            additional = medium_candidates.sample(
                n=n_additional,
                random_state=self.random_state
            )
            
            return pd.concat([low_density_candidates, additional], ignore_index=True)
        
        # Sample from low-density region
        sampled = low_density_candidates.sample(
            n=n_samples,
            random_state=self.random_state
        )
        
        logger.info(
            f"Sampled {n_samples} negative samples from "
            f"{len(low_density_candidates)} low-density candidates "
            f"(threshold={threshold:.2f})"
        )
        
        return sampled
    
    def _threshold_sample(
        self,
        candidates: pd.DataFrame,
        log_density: np.ndarray,
        n_samples: int
    ) -> pd.DataFrame:
        """
        Sample from below threshold only (no fallback)
        """
        threshold = np.percentile(log_density, self.density_threshold)
        low_density_mask = log_density < threshold
        low_density_candidates = candidates[low_density_mask]
        
        if len(low_density_candidates) < n_samples:
            raise ValueError(
                f"Not enough candidates below threshold: "
                f"{len(low_density_candidates)} < {n_samples}"
            )
        
        return low_density_candidates.sample(
            n=n_samples,
            random_state=self.random_state
        )
    
    def _random_sample(
        self,
        candidates: pd.DataFrame,
        n_samples: int
    ) -> pd.DataFrame:
        """
        Simple random sampling (baseline)
        """
        return candidates.sample(n=n_samples, random_state=self.random_state)


def generate_negative_samples(
    candidates: pd.DataFrame,
    log_density: np.ndarray,
    n_positive: int,
    ratio: float = 1.0,
    density_percentile: float = 5.0,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Convenience function to generate negative samples
    
    Args:
        candidates: Candidate pixel features
        log_density: GMM log-density scores
        n_positive: Number of positive samples
        ratio: Negative to positive ratio
        density_percentile: Percentile threshold for low-density
        random_state: Random seed
    
    Returns:
        Sampled negative samples
    """
    n_negative = int(n_positive * ratio)
    threshold = np.percentile(log_density, density_percentile)
    
    low_density_mask = log_density < threshold
    low_density_candidates = candidates[low_density_mask]
    
    if len(low_density_candidates) < n_negative:
        logger.warning(
            f"Insufficient low-density candidates. "
            f"Using {len(low_density_candidates)} instead of {n_negative}"
        )
        return low_density_candidates
    
    return low_density_candidates.sample(
        n=n_negative,
        random_state=random_state
    )
