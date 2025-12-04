"""
Base abstract class for alignment heuristic classifiers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import pickle
import hashlib
import json
import time
import logging
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from dataloaders.runs import RunDataset
from features.extractors import BaseFeatureExtractor
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import Trace
from pm4py.objects.petri_net.utils.petri_utils import construct_trace_net

from models.utils import normalize_datasets, validate_aligner_consistency, iter_combined_datasets


@dataclass
class PredictionResult:
    predicted_heuristic: str
    confidence: Optional[float] = None  # Optional confidence score
    total_prediction_time: float = 0.0
    feature_extraction_time: float = 0.0
    classification_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'predicted_heuristic': self.predicted_heuristic,
            'confidence': self.confidence,
            'total_prediction_time': self.total_prediction_time,
            'feature_extraction_time': self.feature_extraction_time,
            'classification_time': self.classification_time,
        }


class ClassificationModel(ABC):
    """
    Abstract base class for alignment heuristic classification models.
    Handles model training, caching, prediction, and feature importance analysis.
    Supports training on multiple RunDatasets for combining different logs.
    """

    def __init__(
        self,
        run_datasets: Union[RunDataset, List[RunDataset]],
        feature_extractor: BaseFeatureExtractor,
        cache_dir: Optional[Path] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        force_retrain: bool = False,
    ):
        """
        Args:
            run_datasets: Single RunDataset or list of RunDatasets for training
            feature_extractor: Feature extractor for model/trace pairs
            cache_dir: Directory for model cache (defaults to first dataset's base_path / .cache_models)
            hyperparameters: Model-specific hyperparameters
            force_retrain: Force retraining even if cached model exists
        """
        # Normalize to list and validate
        self.run_datasets = normalize_datasets(run_datasets)
        validate_aligner_consistency(self.run_datasets)

        self.feature_extractor = feature_extractor
        self.hyperparameters = hyperparameters or self._default_hyperparameters()

        # Setup cache directory
        if cache_dir is None:
            self.cache_dir = Path("cache/models")
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.label_encoder = None  # Maps string labels to integers
        self.is_trained = False

        cache_key = self._compute_cache_key()
        self.cache_file = self.cache_dir / f"{self.__class__.__name__}_{cache_key}.pkl"

        if force_retrain or not self._load_from_cache():
            logging.info(f"Training new {self.__class__.__name__}...")
            self._train()
            self._save_to_cache()
        else:
            logging.info(f"Loaded {self.__class__.__name__} from cache.")

    @abstractmethod
    def _default_hyperparameters(self) -> Dict[str, Any]:
        """Return default hyperparameters for this model type."""
        pass

    @abstractmethod
    def _train_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Any:
        """
        Train the classifier model.

        Args:
            X_train: Feature matrix [n_samples, n_features]
            y_train: Label vector [n_samples] (integer encoded)

        Returns:
            Trained model instance
        """
        pass

    @abstractmethod
    def _predict_proba(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix [n_samples, n_features]

        Returns:
            Probability matrix [n_samples, n_classes]
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Return feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        return {name: 0.0 for name in self.feature_extractor.feature_names}

    def _compute_cache_key(self) -> str:
        """Compute cache key from datasets, feature extractor, and hyperparameters."""
        key_data = {
            'dataset_hashes': sorted([ds.hash() for ds in self.run_datasets]),
            'feature_extractor_names': self.feature_extractor.feature_names,
            'hyperparameters': self.hyperparameters,
            'model_class': self.__class__.__name__
        }
        return hashlib.sha1(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()[:16]

    def _load_from_cache(self) -> bool:
        """Load model from cache. Returns True if successful."""
        if not self.cache_file.exists():
            return False

        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            self.model = cache_data['model']
            self.label_encoder = cache_data['label_encoder']
            self.is_trained = True
            return True
        except Exception as e:
            logging.warning(f"Failed to load model from cache: {e}")
            return False

    def _save_to_cache(self):
        """Save model to cache."""
        cache_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Saved model to {self.cache_file}")

    def _train(self):
        """
        Extract features and train classifier on RunDatasets.
        Uses iter_combined_datasets() to get the fastest heuristic per combination.
        """
        X_list = []
        y_list = []

        logging.info("Extracting features and identifying fastest heuristics...")
        for model, trace, results_dict in tqdm(
            iter_combined_datasets(self.run_datasets),
            desc="Training data preparation"
        ):
            # Extract features
            trace_net, trace_im, trace_fm = construct_trace_net(trace)
            features = self.feature_extractor.extract(
                model.pm, model.im, model.fm,
                trace_net, trace_im, trace_fm,
                return_as_dict=False
            )

            # Find fastest heuristic
            best_algo = None
            best_time = float('inf')

            for algo_name, (algo, item, perf_list) in results_dict.items():
                # Compute mean time from perf counters
                durations = [p.duration for p in perf_list if p.duration is not None and p.duration != float('inf')]
                if durations:
                    mean_time = np.mean(durations)
                    if mean_time < best_time:
                        best_time = mean_time
                        best_algo = algo_name

            if best_algo is not None:
                X_list.append(features)
                y_list.append(best_algo)

        X_train = np.array(X_list)
        y_train_str = np.array(y_list)

        if len(X_train) == 0:
            raise ValueError("No valid training samples found. Check your RunDatasets.")

        # Encode labels to integers
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(y_train_str)

        logging.info(f"Training on {len(X_train)} samples with {X_train.shape[1]} features")
        logging.info(f"Label distribution: {dict(zip(*np.unique(y_train_str, return_counts=True)))}")

        # Train classifier
        self.model = self._train_classifier(X_train, y_train)
        self.is_trained = True

    def predict_heuristic(
        self,
        model: PetriNet,
        im: Marking,
        fm: Marking,
        trace: Trace
    ) -> PredictionResult:
        """ Predict best heuristic for a single model/trace pair."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call _train() first.")

        t_start = time.perf_counter()

        # Extract features
        t_fe_start = time.perf_counter()
        trace_net, trace_im, trace_fm = construct_trace_net(trace)
        features = self.feature_extractor.extract(
            model, im, fm,
            trace_net, trace_im, trace_fm,
            return_as_dict=False
        )
        t_fe_end = time.perf_counter()
        feature_extraction_time = t_fe_end - t_fe_start

        # Predict
        t_clf_start = time.perf_counter()
        X = features.reshape(1, -1)
        proba = self._predict_proba(X)[0]
        predicted_class = np.argmax(proba)
        confidence = proba[predicted_class]
        predicted_heuristic = self.label_encoder.inverse_transform([predicted_class])[0]
        t_clf_end = time.perf_counter()
        classification_time = t_clf_end - t_clf_start

        t_end = time.perf_counter()
        total_time = t_end - t_start

        return PredictionResult(
            predicted_heuristic=predicted_heuristic,
            confidence=float(confidence),
            total_prediction_time=total_time,
            feature_extraction_time=feature_extraction_time,
            classification_time=classification_time
        )
