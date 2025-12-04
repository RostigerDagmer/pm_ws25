"""
XGBoost-based classifier for alignment heuristic recommendation.
"""

from typing import Dict, Any
import numpy as np
from xgboost import XGBClassifier
import logging

from models.base import ClassificationModel


class XGBoostClassifier(ClassificationModel):
    """
    XGBoost-based classifier for predicting best alignment heuristic.

    Uses gradient boosting with tree-based models to predict which
    heuristic will be fastest for a given (model, trace) pair.
    """

    def _default_hyperparameters(self) -> Dict[str, Any]:
        """Default XGBoost hyperparameters."""
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'n_jobs': -1,
        }

    def _train_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> XGBClassifier:
        """Train XGBoost classifier."""
        model = XGBClassifier(**self.hyperparameters)
        model.fit(X_train, y_train)

        logging.info(f"XGBoost training complete.")
        return model

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Return feature importance from XGBoost model.

        Returns:
            Dictionary mapping feature names to importance scores (gain-based)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        importance_scores = self.model.feature_importances_
        feature_names = self.feature_extractor.feature_names

        return dict(zip(feature_names, importance_scores))
