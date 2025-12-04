"""
Evaluation framework for alignment heuristic recommenders.
"""

from typing import Dict, Any, List, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from models.base import ClassificationModel
from dataloaders.runs import RunDataset
from models.utils import normalize_datasets, validate_aligner_consistency, iter_combined_datasets


@dataclass
class EvaluationMetrics:
    # Classification metrics
    accuracy: float
    precision_per_class: Dict[str, float]
    recall_per_class: Dict[str, float]
    f1_per_class: Dict[str, float]
    confusion_matrix: np.ndarray
    class_labels: List[str]

    # Time performance metrics
    mean_actual_time: float  # Mean time of predicted heuristic
    mean_optimal_time: float  # Mean time of fastest heuristic
    performance_ratio: float  # actual / optimal (lower is better, 1.0 is perfect)
    time_savings_vs_worst: float  # Savings compared to always picking worst

    # Per-heuristic timing statistics
    heuristic_timings: Dict[str, Dict[str, float]]

    # Feature importance
    feature_importance: Dict[str, float]

    # Prediction timing
    mean_prediction_time: float
    mean_feature_extraction_time: float
    mean_classification_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'accuracy': self.accuracy,
            'precision_per_class': self.precision_per_class,
            'recall_per_class': self.recall_per_class,
            'f1_per_class': self.f1_per_class,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'class_labels': self.class_labels,
            'mean_actual_time': self.mean_actual_time,
            'mean_optimal_time': self.mean_optimal_time,
            'performance_ratio': self.performance_ratio,
            'time_savings_vs_worst': self.time_savings_vs_worst,
            'heuristic_timings': self.heuristic_timings,
            'feature_importance': self.feature_importance,
            'mean_prediction_time': self.mean_prediction_time,
            'mean_feature_extraction_time': self.mean_feature_extraction_time,
            'mean_classification_time': self.mean_classification_time,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 80,
            "EVALUATION SUMMARY",
            "=" * 80,
            f"Classification Accuracy: {self.accuracy:.2%}",
            f"Performance Ratio (actual/optimal): {self.performance_ratio:.3f}",
            f"Mean Actual Time: {self.mean_actual_time:.4f}s",
            f"Mean Optimal Time: {self.mean_optimal_time:.4f}s",
            f"Time Savings vs Worst: {self.time_savings_vs_worst:.2%}",
            "",
            "Per-Class Metrics:",
        ]

        for label in self.class_labels:
            lines.append(
                f"  {label}: "
                f"P={self.precision_per_class.get(label, 0):.2%} "
                f"R={self.recall_per_class.get(label, 0):.2%} "
                f"F1={self.f1_per_class.get(label, 0):.2%}"
            )

        lines.extend([
            "",
            "Prediction Timing:",
            f"  Mean Total: {self.mean_prediction_time:.6f}s",
            f"  Mean Feature Extraction: {self.mean_feature_extraction_time:.6f}s",
            f"  Mean Classification: {self.mean_classification_time:.6f}s",
            "",
            "Top 5 Important Features:",
        ])

        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for fname, importance in sorted_features:
            lines.append(f"  {fname}: {importance:.4f}")

        lines.append("=" * 80)
        return "\n".join(lines)


class RecommenderEvaluator:
    """ Evaluator for alignment heuristic recommenders."""

    def __init__(
        self,
        classifier: ClassificationModel,
        run_datasets: Union[RunDataset, List[RunDataset]]
    ):
        self.classifier = classifier
        self.run_datasets = normalize_datasets(run_datasets)
        validate_aligner_consistency(self.run_datasets)

    def evaluate(self) -> EvaluationMetrics:
        """ Evaluate classifier on run_datasets. """
        logging.info("Starting evaluation...")

        # Collect predictions and ground truth
        y_true = []
        y_pred = []
        actual_times = []
        optimal_times = []
        worst_times = []
        prediction_times = []
        feature_extraction_times = []
        classification_times = []

        # Track per-heuristic performance
        heuristic_runs = defaultdict(list)

        for model, trace, results_dict in tqdm(
            iter_combined_datasets(self.run_datasets),
            desc="Evaluating"
        ):
            prediction = self.classifier.predict_heuristic(
                model.pm, model.im, model.fm, trace
            )
            y_pred.append(prediction.predicted_heuristic)

            # Track timing
            prediction_times.append(prediction.total_prediction_time)
            feature_extraction_times.append(prediction.feature_extraction_time)
            classification_times.append(prediction.classification_time)

            # Find fastest heuristic (ground truth)
            best_algo = None
            best_time = float('inf')
            worst_time = 0.0
            heuristic_times = {}

            for algo_name, (algo, item, perf_list) in results_dict.items():
                durations = [p.duration for p in perf_list if p.duration is not None and p.duration != float('inf')]
                if durations:
                    mean_time = np.mean(durations)
                    heuristic_times[algo_name] = mean_time
                    heuristic_runs[algo_name].append(mean_time)

                    if mean_time < best_time:
                        best_time = mean_time
                        best_algo = algo_name
                    if mean_time > worst_time:
                        worst_time = mean_time

            y_true.append(best_algo)
            optimal_times.append(best_time)
            worst_times.append(worst_time)

            # Get actual time for predicted heuristic
            actual_time = heuristic_times[prediction.predicted_heuristic]
            actual_times.append(actual_time)

        accuracy = accuracy_score(y_true, y_pred)
        
        unique_labels = sorted(set(y_true) | set(y_pred))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=unique_labels, average=None, zero_division=0
        )

        # Build per-class dictionaries
        precision_dict = dict(zip(unique_labels, precision))
        recall_dict = dict(zip(unique_labels, recall))
        f1_dict = dict(zip(unique_labels, f1))

        conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_labels)

        # Compute time performance metrics
        mean_actual = np.mean(actual_times)
        mean_optimal = np.mean(optimal_times)
        mean_worst = np.mean(worst_times)

        performance_ratio = mean_actual / mean_optimal if mean_optimal > 0 else float('inf')
        time_savings = (mean_worst - mean_actual) / (mean_worst - mean_optimal) if (mean_worst - mean_optimal) > 0 else 0.0

        # Compute per-heuristic timing statistics
        heuristic_timings = {}
        for heuristic, times in heuristic_runs.items():
            heuristic_timings[heuristic] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'median': np.median(times),
                'min': np.min(times),
                'max': np.max(times),
                'count': len(times)
            }

        # Get feature importance
        feature_importance = self.classifier.get_feature_importance()

        metrics = EvaluationMetrics(
            accuracy=accuracy,
            precision_per_class=precision_dict,
            recall_per_class=recall_dict,
            f1_per_class=f1_dict,
            confusion_matrix=conf_matrix,
            class_labels=unique_labels,
            mean_actual_time=mean_actual,
            mean_optimal_time=mean_optimal,
            performance_ratio=performance_ratio,
            time_savings_vs_worst=time_savings,
            heuristic_timings=heuristic_timings,
            feature_importance=feature_importance,
            mean_prediction_time=np.mean(prediction_times),
            mean_feature_extraction_time=np.mean(feature_extraction_times),
            mean_classification_time=np.mean(classification_times),
        )

        logging.info("Evaluation complete!")
        logging.info(f"\n{metrics.summary()}")

        return metrics

    def compare_with_baselines(
        self,
        baselines: List[ClassificationModel]
    ) -> pd.DataFrame:
        """ Compare classifier with baseline models."""
        results = []

        main_metrics = self.evaluate()
        results.append({
            'model': self.classifier.__class__.__name__,
            'accuracy': main_metrics.accuracy,
            'performance_ratio': main_metrics.performance_ratio,
            'mean_actual_time': main_metrics.mean_actual_time,
            'mean_prediction_time': main_metrics.mean_prediction_time,
        })

        # Evaluate baselines
        for baseline in baselines:
            logging.info(f"\nEvaluating baseline: {baseline.__class__.__name__}")
            evaluator = RecommenderEvaluator(baseline, self.run_datasets)
            baseline_metrics = evaluator.evaluate()

            results.append({
                'model': baseline.__class__.__name__,
                'accuracy': baseline_metrics.accuracy,
                'performance_ratio': baseline_metrics.performance_ratio,
                'mean_actual_time': baseline_metrics.mean_actual_time,
                'mean_prediction_time': baseline_metrics.mean_prediction_time,
            })

        df = pd.DataFrame(results)

        # Add relative performance columns
        df['relative_to_best'] = df['performance_ratio'] / df['performance_ratio'].min()

        return df
