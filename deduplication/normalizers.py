"""
Z-score normalization for feature vectors.
Computes and applies normalization parameters across all Petri nets.
"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np


@dataclass
class FeatureStatistics:
    """Statistics for a single feature across all nets."""
    mean: float
    std: float
    min: float
    max: float
    feature_name: str


class ZScoreFeatureNormalizer:
    """
    Computes and applies z-score normalization for feature vectors.
    
    Must be fitted on ALL nets before use in deduplication.
    Stores mean and std for each feature to enable consistent normalization.
    """
    
    def __init__(self, feature_names: List[str]):
        """
        Initialize normalizer.
        
        Args:
            feature_names: Ordered list of feature names
        """
        self.feature_names = feature_names
        self.statistics: Dict[str, FeatureStatistics] = {}
    
    def fit(self, all_feature_vectors: np.ndarray):
        """
        Compute normalization parameters from all feature vectors.
        
        Args:
            all_feature_vectors: Array of shape [n_nets, n_features]
        """
        for i, fname in enumerate(self.feature_names):
            values = all_feature_vectors[:, i]
            self.statistics[fname] = FeatureStatistics(
                mean=np.mean(values),
                std=np.std(values),
                min=np.min(values),
                max=np.max(values),
                feature_name=fname
            )
    
    def normalize(self, feature_vector: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization to a single feature vector.
        
        Args:
            feature_vector: Array of shape [n_features]
            
        Returns:
            Normalized vector where each feature has mean=0, std=1
            (relative to the distribution of all nets used in fit)
        """
        normalized = np.zeros_like(feature_vector, dtype=float)
        for i, fname in enumerate(self.feature_names):
            stat = self.statistics[fname]
            if stat.std > 0:
                normalized[i] = (feature_vector[i] - stat.mean) / stat.std
            else:
                normalized[i] = 0.0
        return normalized