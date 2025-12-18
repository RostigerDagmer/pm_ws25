"""
Evaluation framework for alignment heuristic recommenders.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import json
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from models.base import ClassificationModel
from dataloaders.labels import LabelDataset


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
    mean_alignment_time_only: (
        float  # Alignment time only (predicted heuristic)
    )
    mean_alignment_time_with_prediction: (
        float  # Total: feature extraction + classification + alignment
    )
    mean_optimal_alignment_time: float  # Fastest heuristic alignment time
    performance_ratio_alignment_only: float  # alignment_only / optimal
    performance_ratio_with_prediction: float  # with_prediction / optimal
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
            'mean_alignment_time_only': self.mean_alignment_time_only,
            'mean_alignment_time_with_prediction': self.mean_alignment_time_with_prediction,
            'mean_optimal_alignment_time': self.mean_optimal_alignment_time,
            'performance_ratio_alignment_only': self.performance_ratio_alignment_only,
            'performance_ratio_with_prediction': self.performance_ratio_with_prediction,
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
            "",
            "Alignment Time Performance:",
            f"  Alignment Only (predicted heuristic): {self.mean_alignment_time_only:.4f}s",
            f"  With Prediction Overhead: {self.mean_alignment_time_with_prediction:.4f}s",
            f"  Optimal (fastest heuristic): {self.mean_optimal_alignment_time:.4f}s",
            f"  Performance Ratio (alignment only): {self.performance_ratio_alignment_only:.3f}x",
            f"  Performance Ratio (with prediction): {self.performance_ratio_with_prediction:.3f}x",
            f"  Time Savings vs Worst: {self.time_savings_vs_worst:.2%}",
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

        lines.extend(
            [
                "",
                "Confusion Matrix:",
                "(Rows: Actual/True labels, Columns: Predicted labels)",
                "",
            ]
        )

        # Format confusion matrix
        lines.append(self._format_confusion_matrix())

        lines.extend(
            [
                "",
                "Prediction Timing:",
                f"  Mean Total: {self.mean_prediction_time * 1000:.3f}ms",
                f"  Mean Feature Extraction: {self.mean_feature_extraction_time * 1000:.3f}ms",
                f"  Mean Classification: {self.mean_classification_time * 1000:.3f}ms",
                "",
                "Top 5 Important Features:",
            ]
        )

        if self.feature_importance is not None:
            sorted_features = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            for fname, importance in sorted_features:
                lines.append(f"  {fname}: {importance:.4f}")

        lines.append("=" * 80)
        return "\n".join(lines)

    def _format_confusion_matrix(self) -> str:
        """Format confusion matrix as a readable table."""

        def clean_and_abbreviate(label: str, max_len: int = 25) -> str:
            cleaned = label.replace("VERSION_", "")
            if len(cleaned) <= max_len:
                return cleaned
            return cleaned[:max_len]

        labels = [clean_and_abbreviate(label) for label in self.class_labels]

        # Calculate column widths
        max_label_width = max(len(label) for label in labels)
        col_width = max(max_label_width, 8)

        # Header row
        header = " " * max_label_width + "  "
        header += "  ".join(f"{label:>{col_width}}" for label in labels)
        header += "  " + f"{'TOTAL':>{col_width}}"

        lines = [header]

        # Data rows
        for i, row_label in enumerate(labels):
            row_sum = self.confusion_matrix[i].sum()
            row = f"{row_label:<{max_label_width}}  "
            row += "  ".join(
                f"{self.confusion_matrix[i, j]:>{col_width}}"
                for j in range(len(labels))
            )
            row += "  " + f"{row_sum:>{col_width}}"
            lines.append(row)

        # Total row
        col_sums = self.confusion_matrix.sum(axis=0)
        total_sum = self.confusion_matrix.sum()
        total_row = f"{'TOTAL':<{max_label_width}}  "
        total_row += "  ".join(
            f"{col_sums[j]:>{col_width}}" for j in range(len(labels))
        )
        total_row += "  " + f"{total_sum:>{col_width}}"
        lines.append(total_row)

        return "\n".join(lines)


class RecommenderEvaluator:
    """Evaluator for alignment heuristic recommenders."""

    def __init__(
        self,
        classifier: ClassificationModel,
        dataset: LabelDataset,
    ):
        self.classifier = classifier
        self.dataset = dataset

    def evaluate_batched(self) -> EvaluationMetrics:
        """Evaluate classifier on run_datasets."""
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

        for runs in tqdm(self.dataset.iter_by_model(), desc="Evaluating"):
            dataset_ids, runs = zip(*runs)
            model = runs[0].model.deserialize()
            traces = [run.trace for run in runs]

            predictions = self.classifier.predict_batched(model, traces)
            y_pred.extend(
                [prediction.predicted_heuristic for prediction in predictions]
            )

            # Track timing
            prediction_times.extend(
                [
                    prediction.total_prediction_time
                    for prediction in predictions
                ]
            )
            feature_extraction_times.extend(
                [
                    prediction.feature_extraction_time
                    for prediction in predictions
                ]
            )
            classification_times.extend(
                [prediction.classification_time for prediction in predictions]
            )

            # Find fastest heuristic (ground truth)

            for dataset_id, gt, prediction in zip(
                dataset_ids, runs, predictions
            ):
                # print(f"dataset_id: {dataset_id}")
                # print(f"gt: {gt.comb_id}")
                # print(f"prediction: {prediction}")

                # print(f"{self.dataset.df[(self.dataset.df['dataset_id'] == dataset_id) & (self.dataset.df['comb_id'] == gt.comb_id)]}")

                all_alignments = {
                    item.algo: item.perf
                    for item in self.dataset.get_combination_results(
                        dataset_id, gt.comb_id
                    )
                }

                durations = {
                    algo: np.mean(
                        [
                            (
                                p["duration"]
                                if p["duration"] != float('inf')
                                else 20.0
                            )
                            for p in perf
                        ]
                    )
                    for algo, perf in all_alignments.items()  # TODO: read timeout from ds
                }

                # print(f"durations: {durations}")

                best_time = min(durations.values())
                worst_time = max(durations.values())
                best_algo = gt.algo

                y_true.append(best_algo)
                optimal_times.append(best_time)
                worst_times.append(worst_time)

                # Get actual time for predicted heuristic
                actual_time = durations[prediction.predicted_heuristic]
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
        mean_alignment_only = np.mean(actual_times)
        mean_prediction = np.mean(prediction_times)
        mean_alignment_with_pred = mean_alignment_only + mean_prediction
        mean_optimal = np.mean(optimal_times)
        mean_worst = np.mean(worst_times)

        performance_ratio_alignment_only = (
            mean_alignment_only / mean_optimal
            if mean_optimal > 0
            else float('inf')
        )
        performance_ratio_with_pred = (
            mean_alignment_with_pred / mean_optimal
            if mean_optimal > 0
            else float('inf')
        )
        time_savings = (
            (mean_worst - mean_alignment_only) / (mean_worst - mean_optimal)
            if (mean_worst - mean_optimal) > 0
            else 0.0
        )

        # Compute per-heuristic timing statistics
        heuristic_timings = {}
        for heuristic, times in heuristic_runs.items():
            heuristic_timings[heuristic] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'median': np.median(times),
                'min': np.min(times),
                'max': np.max(times),
                'count': len(times),
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
            mean_alignment_time_only=mean_alignment_only,
            mean_alignment_time_with_prediction=mean_alignment_with_pred,
            mean_optimal_alignment_time=mean_optimal,
            performance_ratio_alignment_only=performance_ratio_alignment_only,
            performance_ratio_with_prediction=performance_ratio_with_pred,
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

    def evaluate(self) -> EvaluationMetrics:
        """Evaluate classifier on run_datasets."""
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

        for dataset_id, run in tqdm(self.dataset, desc="Evaluating"):
            model = run.model.deserialize()
            trace = run.trace
            prediction = self.classifier.predict_heuristic(
                model.pm, model.im, model.fm, trace
            )
            y_pred.append(prediction.predicted_heuristic)

            # Track timing
            prediction_times.append(prediction.total_prediction_time)
            feature_extraction_times.append(prediction.feature_extraction_time)
            classification_times.append(prediction.classification_time)

            # Find fastest heuristic (ground truth)
            all_alignments = {
                item.algo: item.perf
                for item in self.dataset.get_combination_results(
                    dataset_id, run.comb_id
                )
            }

            durations = {
                algo: np.mean(
                    [
                        (
                            p["duration"]
                            if p["duration"] != float('inf')
                            else 20.0
                        )
                        for p in perf
                    ]
                )
                for algo, perf in all_alignments.items()  # TODO: read timeout from ds
            }

            best_time = min(durations.values())
            worst_time = max(durations.values())
            best_algo = min(durations, key=durations.get)

            y_true.append(best_algo)
            optimal_times.append(best_time)
            worst_times.append(worst_time)

            # Get actual time for predicted heuristic
            actual_time = durations[prediction.predicted_heuristic]
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
        mean_alignment_only = np.mean(actual_times)
        mean_prediction = np.mean(prediction_times)
        mean_alignment_with_pred = mean_alignment_only + mean_prediction
        mean_optimal = np.mean(optimal_times)
        mean_worst = np.mean(worst_times)

        performance_ratio_alignment_only = (
            mean_alignment_only / mean_optimal
            if mean_optimal > 0
            else float('inf')
        )
        performance_ratio_with_pred = (
            mean_alignment_with_pred / mean_optimal
            if mean_optimal > 0
            else float('inf')
        )
        time_savings = (
            (mean_worst - mean_alignment_only) / (mean_worst - mean_optimal)
            if (mean_worst - mean_optimal) > 0
            else 0.0
        )

        # Compute per-heuristic timing statistics
        heuristic_timings = {}
        for heuristic, times in heuristic_runs.items():
            heuristic_timings[heuristic] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'median': np.median(times),
                'min': np.min(times),
                'max': np.max(times),
                'count': len(times),
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
            mean_alignment_time_only=mean_alignment_only,
            mean_alignment_time_with_prediction=mean_alignment_with_pred,
            mean_optimal_alignment_time=mean_optimal,
            performance_ratio_alignment_only=performance_ratio_alignment_only,
            performance_ratio_with_prediction=performance_ratio_with_pred,
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
        self, baselines: List[ClassificationModel]
    ) -> pd.DataFrame:
        """Compare classifier with baseline models."""
        results = []

        main_metrics = self.evaluate_batched()
        results.append(
            {
                'model': self.classifier.__class__.__name__,
                'accuracy': main_metrics.accuracy,
                'performance_ratio_alignment_only': main_metrics.performance_ratio_alignment_only,
                'performance_ratio_with_prediction': main_metrics.performance_ratio_with_prediction,
                'mean_alignment_time_only': main_metrics.mean_alignment_time_only,
                'mean_alignment_time_with_prediction': main_metrics.mean_alignment_time_with_prediction,
                'mean_prediction_time': main_metrics.mean_prediction_time,
            }
        )

        # Evaluate baselines
        for baseline in baselines:
            logging.info(
                f"\nEvaluating baseline: {baseline.__class__.__name__}"
            )
            evaluator = RecommenderEvaluator(baseline, self.dataset)
            baseline_metrics = evaluator.evaluate_batched()

            results.append(
                {
                    'model': baseline.__class__.__name__,
                    'accuracy': baseline_metrics.accuracy,
                    'performance_ratio_alignment_only': baseline_metrics.performance_ratio_alignment_only,
                    'performance_ratio_with_prediction': baseline_metrics.performance_ratio_with_prediction,
                    'mean_alignment_time_only': baseline_metrics.mean_alignment_time_only,
                    'mean_alignment_time_with_prediction': baseline_metrics.mean_alignment_time_with_prediction,
                    'mean_prediction_time': baseline_metrics.mean_prediction_time,
                }
            )

        df = pd.DataFrame(results)

        # Add relative performance columns
        df['relative_to_best_alignment_only'] = (
            df['performance_ratio_alignment_only']
            / df['performance_ratio_alignment_only'].min()
        )
        df['relative_to_best_with_prediction'] = (
            df['performance_ratio_with_prediction']
            / df['performance_ratio_with_prediction'].min()
        )

        return df

    @staticmethod
    def save_results(
        metrics: EvaluationMetrics,
        comparison_df: pd.DataFrame,
        output_dir: Path,
        train_count: int,
        test_count: int,
    ):
        """Save evaluation results to output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics as JSON
        with open(output_dir / "metrics.json", 'w') as f:
            metrics_dict = RecommenderEvaluator._convert_to_serializable(
                metrics.to_dict()
            )
            json.dump(metrics_dict, f, indent=2)

        # Save summary as txt
        with open(output_dir / "summary.txt", 'w') as f:
            f.write("ML CLASSIFIER EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Train datasets: {train_count}\n")
            f.write(f"Test datasets: {test_count}\n\n")
            f.write("=" * 80 + "\n\n")
            f.write(metrics.summary())
            f.write("\n\n" + "=" * 80 + "\n\n")
            f.write("Baseline Comparison:\n")
            f.write(comparison_df.to_string())

        logging.info(f"Results saved to: {output_dir}")

    @staticmethod
    def _convert_to_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {
                key: RecommenderEvaluator._convert_to_serializable(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [
                RecommenderEvaluator._convert_to_serializable(item)
                for item in obj
            ]
        return obj
