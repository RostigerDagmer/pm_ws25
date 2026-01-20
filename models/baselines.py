"""
Baseline classifiers for comparison.
"""

from typing import Dict, Any
import time
import numpy as np
from collections import Counter
from models.base import ClassificationModel, PredictionResult
from dataloaders.net import ProcessModelDataset
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import Trace


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
        self.most_common_class = counter.most_common(1)[0][0]

        # Store the string label directly
        return self.label_encoder.inverse_transform([self.most_common_class])[
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

    def predict_heuristic(
        self, model: PetriNet, im: Marking, fm: Marking, trace: Trace
    ) -> PredictionResult:
        return PredictionResult(
            predicted_heuristic=self.label_encoder.inverse_transform(
                [self.most_common_class]
            )[0],
            confidence=1.0,
            feature_extraction_time=0.0,
            classification_time=0.0,
        )

    def predict_batched(
        self, model: ProcessModelDataset.ItemType, traces: list[Trace]
    ) -> list[PredictionResult]:
        return [
            PredictionResult(
                predicted_heuristic=self.label_encoder.inverse_transform(
                    [self.most_common_class]
                )[0],
                confidence=1.0,
                feature_extraction_time=0.0,
                classification_time=0.0,
            )
            for _ in range(len(traces))
        ]


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
    ) -> Dict[str, Any]:
        """Compute label frequency distribution and initialize RNG."""
        counter = Counter(y_train)
        total = len(y_train)

        # Store distribution as {class_idx: probability}
        distribution = {cls: count / total for cls, count in counter.items()}

        rng = np.random.RandomState(self.hyperparameters['random_state'])

        return {'distribution': distribution, 'rng': rng}

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Sample randomly from the training distribution for each sample."""
        n_samples = X.shape[0]
        n_classes = len(self.label_encoder.classes_)

        rng = self.model['rng']

        classes = list(self.model['distribution'].keys())
        probs = [self.model['distribution'][cls] for cls in classes]

        # Instead of assigning probabilities, we sample one class per instance
        #   because predict_heuristics always picks the argmax.
        proba = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            sampled_class = rng.choice(classes, p=probs)
            proba[i, sampled_class] = 1.0

        return proba

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Return feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        return {name: 0.0 for name in self.feature_extractor.feature_names}

    def predict_heuristic(
        self, model: PetriNet, im: Marking, fm: Marking, trace: Trace
    ) -> PredictionResult:
        t_clf_start = time.perf_counter()
        X = np.zeros(1)
        proba = self._predict_proba(X)[0]
        predicted_class = np.argmax(proba)
        confidence = proba[predicted_class]
        predicted_heuristic = self.label_encoder.inverse_transform(
            [predicted_class]
        )[0]
        t_clf_end = time.perf_counter()
        return PredictionResult(
            predicted_heuristic=self.most_common_class,
            confidence=1.0,
            feature_extraction_time=0.0,
            classification_time=0.0,
        )

    def predict_batched(
        self, model: ProcessModelDataset.ItemType, traces: list[Trace]
    ) -> list[PredictionResult]:

        t_clf_start = time.perf_counter()
        X = np.zeros(len(traces))
        proba = self._predict_proba(X)
        predicted_classes = np.argmax(proba, axis=1)
        confidence = proba[:, predicted_classes]
        predicted_heuristics = self.label_encoder.inverse_transform(
            predicted_classes
        )
        t_clf_end = time.perf_counter()

        classification_time = t_clf_end - t_clf_start

        return [
            PredictionResult(
                predicted_heuristic=h,
                confidence=conf,
                feature_extraction_time=0.0,
                classification_time=classification_time / len(traces),
            )
            for h, conf in zip(predicted_heuristics, confidence)
        ]
