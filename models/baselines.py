"""
Baseline classifiers for comparison.
"""

from typing import Dict, Any
import numpy as np
from collections import Counter

from models.base import ClassificationModel


class SingleBestSolver(ClassificationModel):
    """
    Baseline that always predicts the globally fastest heuristic.

    Computes which heuristic is fastest across all combinations in the
    training set and always returns that heuristic.
    """

    def _default_hyperparameters(self) -> Dict[str, Any]:
        return {}

    def _train_classifier(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> str:
        """Find most common label (globally fastest heuristic)."""
        counter = Counter(y_train)
        most_common_class = counter.most_common(1)[0][0]

        # Store the string label directly
        return self.label_encoder.inverse_transform([most_common_class])[
            0
        ]  # This will set self.model to that label

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return uniform probability for the single best solver."""
        n_samples = X.shape[0]
        n_classes = len(self.label_encoder.classes_)

        # Find index of the best solver
        best_class_idx = list(self.label_encoder.classes_).index(self.model)

        # Create probability matrix (one-hot for best class)
        proba = np.zeros((n_samples, n_classes))
        proba[:, best_class_idx] = 1.0
        return proba

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Return feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        return {name: 0.0 for name in self.feature_extractor.feature_names}


class RandomClassifier(ClassificationModel):
    """
    Baseline that randomly selects a heuristic based on training label distribution.

    Uses the observed frequency of each heuristic as the probability
    distribution for random selection.
    """

    def _default_hyperparameters(self) -> Dict[str, Any]:
        return {'random_state': 42}

    def _train_classifier(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Dict[int, float]:
        """Compute label frequency distribution."""
        counter = Counter(y_train)
        total = len(y_train)

        # Store distribution as {class_idx: probability}
        distribution = {cls: count / total for cls, count in counter.items()}
        return distribution

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return training label distribution for all samples."""
        n_samples = X.shape[0]
        n_classes = len(self.label_encoder.classes_)

        # Create probability matrix based on training distribution
        proba = np.zeros((n_samples, n_classes))
        for cls_idx, prob in self.model.items():
            proba[:, cls_idx] = prob

        return proba

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Return feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        return {name: 0.0 for name in self.feature_extractor.feature_names}
