"""
Gaussian Mixture Model (GMM) for environmental structure identification
Migrated from: function/gmm_training.py
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import logging

logger = logging.getLogger(__name__)


class GMMTrainer:
    """
    GMM trainer for environmental clustering
    Identifies environmental structure in PV site features
    """
    
    def __init__(self, config: Dict):
        """
        Initialize GMM trainer
        
        Args:
            config: Configuration dictionary with GMM parameters
        """
        self.config = config
        self.gmm_config = config.get("model", {}).get("gmm", {})
        
        self.n_components = self.gmm_config.get("n_components", 23)
        self.covariance_type = self.gmm_config.get("covariance_type", "full")
        self.n_init = self.gmm_config.get("n_init", 10)
        self.random_state = self.gmm_config.get("random_state", 42)
        
        self.pipeline = None
        self.calibration_stats = None
    
    def build_pipeline(self) -> Pipeline:
        """
        Build preprocessing + GMM pipeline
        
        Returns:
            Sklearn Pipeline
        """
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('gmm', GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_init=self.n_init,
                random_state=self.random_state
            ))
        ])
        
        return pipeline
    
    def train(
        self,
        X: np.ndarray,
        search_components: bool = False,
        component_range: Optional[List[int]] = None
    ) -> Pipeline:
        """
        Train GMM
        
        Args:
            X: Training features (n_samples, n_features)
            search_components: Whether to search for optimal n_components
            component_range: Range of components to search
        
        Returns:
            Trained pipeline
        """
        logger.info(f"Training GMM with n_components={self.n_components}")
        
        if not search_components:
            # Train with fixed n_components
            self.pipeline = self.build_pipeline()
            self.pipeline.fit(X)
            
            # Calculate BIC for reporting
            X_transformed = self.pipeline[:-1].transform(X)
            bic = self.pipeline[-1].bic(X_transformed)
            logger.info(f"GMM trained. BIC: {bic:.2f}")
        
        else:
            # Grid search for optimal n_components
            if component_range is None:
                component_range = list(range(15, 35, 2))
            
            logger.info(f"Searching n_components in range: {component_range}")
            
            param_grid = {
                'gmm__n_components': component_range,
                'gmm__covariance_type': ['full', 'diag'],
                'gmm__n_init': [10, 20]
            }
            
            base_pipeline = self.build_pipeline()
            
            # Use BIC as scoring
            def bic_scorer(estimator, X):
                X_transformed = estimator[:-1].transform(X)
                return -estimator[-1].bic(X_transformed)
            
            grid_search = GridSearchCV(
                base_pipeline,
                param_grid,
                scoring=bic_scorer,
                cv=5,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X)
            
            self.pipeline = grid_search.best_estimator_
            self.n_components = grid_search.best_params_['gmm__n_components']
            
            logger.info(f"Best n_components: {self.n_components}")
            logger.info(f"Best score: {grid_search.best_score_:.2f}")
        
        return self.pipeline
    
    def score_samples(
        self,
        X: np.ndarray,
        method: str = 'log_density'
    ) -> np.ndarray:
        """
        Score samples using GMM
        
        Args:
            X: Input features
            method: Scoring method ('log_density', 'density', 'probability')
        
        Returns:
            Scores for each sample
        """
        if self.pipeline is None:
            raise ValueError("GMM must be trained first")
        
        X_transformed = self.pipeline[:-1].transform(X)
        gmm = self.pipeline[-1]
        
        log_density = gmm.score_samples(X_transformed)
        
        if method == 'log_density':
            return log_density
        elif method == 'density':
            return np.exp(log_density)
        elif method == 'probability':
            # Normalize to [0, 1] using sigmoid
            if self.calibration_stats is None:
                logger.warning("No calibration stats, using current batch statistics")
                mu = log_density.mean()
            else:
                mu = self.calibration_stats['mu']
            
            return 1.0 / (1.0 + np.exp(-(log_density - mu)))
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calibrate(self, X_calibration: np.ndarray) -> None:
        """
        Calibrate scoring using independent calibration set
        
        Args:
            X_calibration: Calibration features
        """
        log_density = self.score_samples(X_calibration, method='log_density')
        
        self.calibration_stats = {
            'mu': float(np.mean(log_density)),
            'std': float(np.std(log_density)),
            'min': float(np.min(log_density)),
            'max': float(np.max(log_density)),
            'median': float(np.median(log_density)),
            'q05': float(np.quantile(log_density, 0.05)),
            'q95': float(np.quantile(log_density, 0.95))
        }
        
        logger.info(f"Calibration stats: mu={self.calibration_stats['mu']:.2f}, "
                   f"std={self.calibration_stats['std']:.2f}")
    
    def save(self, path: str) -> None:
        """Save trained pipeline"""
        if self.pipeline is None:
            raise ValueError("No trained pipeline to save")
        
        save_dict = {
            'pipeline': self.pipeline,
            'calibration_stats': self.calibration_stats,
            'config': self.gmm_config
        }
        
        joblib.dump(save_dict, path)
        logger.info(f"GMM pipeline saved to {path}")
    
    def load(self, path: str) -> None:
        """Load trained pipeline"""
        save_dict = joblib.load(path)
        
        self.pipeline = save_dict['pipeline']
        self.calibration_stats = save_dict.get('calibration_stats')
        
        # Update n_components from loaded model
        self.n_components = self.pipeline[-1].n_components
        
        logger.info(f"GMM pipeline loaded from {path}")


# Convenience functions

def train_gmm(
    X: np.ndarray,
    n_components: int = 23,
    covariance_type: str = 'full',
    search: bool = False
) -> Pipeline:
    """
    Convenience function to train GMM
    
    Args:
        X: Training features
        n_components: Number of mixture components
        covariance_type: Covariance type
        search: Whether to search for optimal parameters
    
    Returns:
        Trained pipeline
    """
    config = {
        'model': {
            'gmm': {
                'n_components': n_components,
                'covariance_type': covariance_type,
                'n_init': 10,
                'random_state': 42
            }
        }
    }
    
    trainer = GMMTrainer(config)
    return trainer.train(X, search_components=search)


def score_samples(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """
    Score samples using trained GMM pipeline
    
    Args:
        pipeline: Trained GMM pipeline
        X: Input features
    
    Returns:
        Log-density scores
    """
    X_transformed = pipeline[:-1].transform(X)
    return pipeline[-1].score_samples(X_transformed)
