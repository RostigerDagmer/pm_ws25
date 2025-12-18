"""
Base abstract class for alignment heuristic classifiers.
"""

from dataloaders.net import ProcessModelDataset
from dataloaders.labels import LabelDataset
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import pickle
import hashlib
import json
import time
import logging
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass

from features.extractors import BaseFeatureExtractor
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import Trace
from pm4py.objects.petri_net.utils.petri_utils import construct_trace_net


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
        dataset: Optional[LabelDataset] = None,
        feature_extractor: Optional[BaseFeatureExtractor] = None,
        tables: Optional[List[pd.DataFrame]] = None,
        cache_dir: Optional[Path] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        force_retrain: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            dataset: Optional LabelDataset for training
            feature_extractor: Feature extractor for model/trace pairs
            table: Optional compressed table of ids + feature_vector for training
            cache_dir: Directory for model cache (defaults to first dataset's base_path / .cache_models)
            hyperparameters: Model-specific hyperparameters
            force_retrain: Force retraining even if cached model exists
        """

        self.tables = tables
        self.dataset = dataset

        if tables is None:
            if dataset is None:
                raise ValueError(
                    "At least one of dataset or tables must be provided."
                )
        else:
            self.dataset = None

        self.feature_extractor = feature_extractor
        self.hyperparameters = (
            hyperparameters or self._default_hyperparameters()
        )

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
        self.cache_file = (
            self.cache_dir / f"{self.__class__.__name__}_{cache_key}.pkl"
        )
        self.device = device or torch.device("cpu")

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
        self, X_train: np.ndarray, y_train: np.ndarray
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
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
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
            'dataset_hashes': (self.dataset.hash() if self.dataset else None,),
            'feature_extractor_names': self.feature_extractor.feature_names,
            'hyperparameters': self.hyperparameters,
            'model_class': self.__class__.__name__,
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

    def _get_xy_dataset(self):
        logging.info(
            "Extracting features and identifying fastest heuristics..."
        )
        X_list, y_list = [], []
        for item in tqdm(
            self.dataset,
            desc="Training data preparation",
        ):
            # Extract features
            trace_net, trace_im, trace_fm = construct_trace_net(item.trace)
            features = self.feature_extractor.extract(
                item.model.pm,
                item.model.im,
                item.model.fm,
                trace_net,
                trace_im,
                trace_fm,
                return_as_dict=False,
            )

            X_list.append(features)
            y_list.append(item.algo)
        return X_list, y_list

    def _get_xy_table(self):
        X_list, y_list = [], []
        for table in self.tables:
            parsed_features = (
                table['feature_vector']
                .apply(
                    lambda x: np.fromstring(
                        x.strip('[]').replace('\n', ' '), sep=' '
                    )
                )
                .tolist()
            )
            X_list.extend(parsed_features)
            y_list.extend(table['aligner'].tolist())
        return X_list, y_list

    def _train(self):
        """
        Extract features and train classifier on RunDatasets.
        Uses iter_combined_datasets() to get the fastest heuristic per combination.
        """

        if self.tables is not None:
            X_list, y_list = self._get_xy_table()
        else:
            X_list, y_list = self._get_xy_dataset()

        X_train = np.array(X_list)
        y_train_str = np.array(y_list)

        if len(X_train) == 0:
            raise ValueError(
                "No valid training samples found. Check your RunDatasets."
            )

        # Encode labels to integers
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(y_train_str)

        logging.info(
            f"Training on {len(X_train)} samples with {X_train.shape[1]} features"
        )
        logging.info(
            f"Label distribution: {dict(zip(*np.unique(y_train_str, return_counts=True)))}"
        )

        # Train classifier
        self.model = self._train_classifier(X_train, y_train)
        self.is_trained = True

    def predict_heuristic(
        self, model: PetriNet, im: Marking, fm: Marking, trace: Trace
    ) -> PredictionResult:
        """Predict best heuristic for a single model/trace pair."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call _train() first.")

        t_start = time.perf_counter()

        # Extract features
        t_fe_start = time.perf_counter()
        trace_net, trace_im, trace_fm = construct_trace_net(trace)
        features = self.feature_extractor.extract(
            model,
            im,
            fm,
            trace_net,
            trace_im,
            trace_fm,
            return_as_dict=False,
            use_cache=False,  # Disable cache to measure actual extraction time
        )
        t_fe_end = time.perf_counter()
        feature_extraction_time = t_fe_end - t_fe_start

        # Predict
        t_clf_start = time.perf_counter()
        X = features.reshape(1, -1)
        proba = self._predict_proba(X)[0]
        predicted_class = np.argmax(proba)
        confidence = proba[predicted_class]
        predicted_heuristic = self.label_encoder.inverse_transform(
            [predicted_class]
        )[0]
        t_clf_end = time.perf_counter()
        classification_time = t_clf_end - t_clf_start

        t_end = time.perf_counter()
        total_time = t_end - t_start

        return PredictionResult(
            predicted_heuristic=predicted_heuristic,
            confidence=float(confidence),
            total_prediction_time=total_time,
            feature_extraction_time=feature_extraction_time,
            classification_time=classification_time,
        )

    def predict_batched(
        self, model: ProcessModelDataset.ItemType, traces: list[Trace]
    ) -> list[PredictionResult]:

        t_fe_start = time.perf_counter()
        trace_nets = [construct_trace_net(trace) for trace in traces]
        features = self.feature_extractor.extract_batched(
            model.pm,
            model.im,
            model.fm,
            trace_nets,
            return_as_dict=False,
            use_cache=False,
        )
        t_fe_end = time.perf_counter()
        feature_extraction_time = t_fe_end - t_fe_start

        predictions = []
        for feature in features:
            t_clf_start = time.perf_counter()
            X = feature.reshape(1, -1)
            proba = self._predict_proba(X)[0]
            predicted_class = np.argmax(proba)
            confidence = proba[predicted_class]
            predicted_heuristic = self.label_encoder.inverse_transform(
                [predicted_class]
            )[0]
            t_clf_end = time.perf_counter()
            classification_time = t_clf_end - t_clf_start
            predictions.append(
                PredictionResult(
                    predicted_heuristic=predicted_heuristic,
                    confidence=float(confidence),
                    total_prediction_time=t_clf_end - t_clf_start,
                    feature_extraction_time=feature_extraction_time,
                    classification_time=classification_time,
                )
            )

        return predictions
