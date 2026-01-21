"""
Feature extraction for Petri nets and traces for alignment heuristic recommendation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Union
import numpy as np


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for feature extraction from Petri nets and traces.

    Provides common interface for extracting features as numpy arrays or dicts,
    with automatic conversion between the two representations.

    Includes optional caching mechanism to avoid re-extracting features
    from the same Petri net multiple times.
    """

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Returns ordered list of feature names."""
        pass

    @abstractmethod
    def _extract_features_internal(self, *args, **kwargs) -> Dict[str, float]:
        """Internal feature extraction method. Must return a flat dict."""
        pass

    def extract(
        self,
        *args,
        return_as_dict: bool = False,
        **kwargs,
    ) -> Union[np.ndarray, Dict[str, float]]:
        """
        Extract features from input.

        Args:
            return_as_dict: If True, return dict. Otherwise return numpy array.

        Returns:
            Feature vector as numpy array or dict.
        """

        # Extract features
        feature_dict = self._extract_features_internal(*args, **kwargs)
        assert set(feature_dict.keys()) == set(self.feature_names), (
            f"Extracted features do not match expected feature names. "
            f"Expected: {self.feature_names}, but got: {feature_dict.keys()}"
        )

        if return_as_dict:
            return feature_dict
        return self.dict_to_vector(feature_dict)

    def extract_batched(
        self,
        *args,
        return_as_dict: bool = False,
        **kwargs,
    ) -> list[Dict[str, float]] | list[np.ndarray]:
        """
        Extract features from batch of inputs.

        Args:
            return_as_dict: If True, return dict. Otherwise return numpy array.

        Returns:
            Feature vector as numpy array or dict.
        """
        feats = self._extract_features_batch(*args, **kwargs)
        if return_as_dict:
            return feats
        return [self.dict_to_vector(feat) for feat in feats]

    def dict_to_vector(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy array using feature_names order."""
        return np.nan_to_num(
            np.array([feature_dict[k] for k in self.feature_names])
        )

    def vector_to_dict(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """Convert feature vector to dict using feature_names order."""
        return dict(zip(self.feature_names, feature_vector))
