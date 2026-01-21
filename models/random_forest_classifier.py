"""
Random Forest classifier for alignment heuristic recommendation.
"""

from typing import Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRFClassifier
import logging

from models.base import ClassificationModel


class RandomForestClassifier(ClassificationModel):
    """
    Random Forest-based classifier for predicting best alignment heuristic.

    Uses scikit-learn's Random Forest implementation to predict which
    heuristic will be fastest for a given (model, trace) pair.
    """

    def _default_hyperparameters(self) -> Dict[str, Any]:
        """Default Random Forest hyperparameters."""
        return {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 0,
        }

    def _train_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> SklearnRFClassifier:
        """Train Random Forest classifier."""
        model = SklearnRFClassifier(**self.hyperparameters)
        model.fit(X_train, y_train)

        logging.info(f"Random Forest training complete.")
        return model

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Return feature importance from Random Forest model.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        importance_scores = self.model.feature_importances_
        feature_names = self.feature_extractor.feature_names

        return dict(zip(feature_names, importance_scores))
