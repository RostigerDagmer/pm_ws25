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

    def __init__(self, use_cache: bool = True):
        """
        Initialize feature extractor.

        Args:
            use_cache: Whether to cache extracted features
        """
        self.use_cache = use_cache
        self._feature_cache = {} if use_cache else None

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Returns ordered list of feature names."""
        pass

    @abstractmethod
    def _extract_features_internal(self, *args, **kwargs) -> Dict[str, float]:
        """Internal feature extraction method. Must return a flat dict."""
        pass

    @abstractmethod
    def _compute_cache_key(self, *args, **kwargs):
        """
        Compute cache key from arguments.

        Subclasses must implement this to define how to generate unique
        cache keys for their specific inputs.

        Returns:
            Hashable cache key (typically int from id(), or tuple of ids)
        """
        raise NotImplementedError(
            "Subclasses must implement _compute_cache_key"
        )

    def extract(
        self,
        *args,
        return_as_dict: bool = False,
        use_cache: bool = None,
        **kwargs,
    ) -> Union[np.ndarray, Dict[str, float]]:
        """
        Extract features from input.

        Args:
            return_as_dict: If True, return dict. Otherwise return numpy array.
            use_cache: Override instance cache setting. If None, uses self.use_cache.

        Returns:
            Feature vector as numpy array or dict.
        """
        should_cache = self.use_cache if use_cache is None else use_cache

        # Check cache if enabled
        if should_cache:
            cache_key = self._compute_cache_key(*args, **kwargs)
            if cache_key is not None and cache_key in self._feature_cache:
                cached_dict = self._feature_cache[cache_key]
                if return_as_dict:
                    return cached_dict
                return self.dict_to_vector(cached_dict)

        # Extract features
        feature_dict = self._extract_features_internal(*args, **kwargs)
        assert set(feature_dict.keys()) == set(self.feature_names), (
            f"Extracted features do not match expected feature names. "
            f"Expected: {self.feature_names}, but got: {feature_dict.keys()}"
        )

        # Cache if enabled
        if should_cache:
            cache_key = self._compute_cache_key(*args, **kwargs)
            if cache_key is not None:
                self._feature_cache[cache_key] = feature_dict

        if return_as_dict:
            return feature_dict
        return self.dict_to_vector(feature_dict)

    def extract_batched(
        self,
        *args,
        return_as_dict: bool = False,
        use_cache: bool = None,
        **kwargs,
    ) -> list[Dict[str, float]] | list[np.ndarray]:
        """Extract features for a batch of traces."""
        # TODO: caching
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
