"""
Gradient Boosting classifier for alignment heuristic recommendation.
"""

from typing import Dict, Any
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBClassifier
import logging

from models.base import ClassificationModel


class GradientBoostingClassifier(ClassificationModel):
    """
    Gradient Boosting-based classifier for predicting best alignment heuristic.

    Uses scikit-learn's Gradient Boosting implementation to predict which
    heuristic will be fastest for a given (model, trace) pair.
    """

    def _default_hyperparameters(self) -> Dict[str, Any]:
        """Default Gradient Boosting hyperparameters."""
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'max_features': 'sqrt',
            'random_state': 42,
            'verbose': 0,
        }

    def _train_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> SklearnGBClassifier:
        """Train Gradient Boosting classifier."""
        model = SklearnGBClassifier(**self.hyperparameters)
        model.fit(X_train, y_train)

        logging.info(f"Gradient Boosting training complete.")
        return model

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Return feature importance from Gradient Boosting model.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        importance_scores = self.model.feature_importances_
        feature_names = self.feature_extractor.feature_names

        return dict(zip(feature_names, importance_scores))
