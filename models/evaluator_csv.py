"""
Evaluation framework for alignment heuristic recommenders.

Uses combination-based metrics where ground truth can be multiple near-optimal
heuristics (within a tolerance threshold of the best).
"""

from typing import Dict, Any, List, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import json
from collections import defaultdict, Counter
import pandas as pd
import time

from models.base import ClassificationModel
from dataloaders.runs import RunDataset
from models.utils import (
    normalize_datasets,
    validate_aligner_consistency,
    iter_combined_datasets,
)


# Default tolerance thresholds for near-optimal heuristics
TOLERANCE_THRESHOLDS = [0.0, 0.10, 0.20]

HEURISTIC_ALIASES = {
    'VERSION_DIJKSTRA_NO_HEURISTICS': 'Dijkstra',
    'VERSION_STATE_EQUATION_A_STAR_ILP': 'A*-ILP',
    'VERSION_STATE_EQUATION_A_STAR': 'A*',
    'VERSION_REQUIRED_MODEL_MOVE': 'RequiredModelMove',
    'VERSION_REMAINING_TRACE': 'RemainingActivities',
}

def get_heuristic_alias(full_name: str) -> str:
    """Get short alias for heuristic name."""
    return HEURISTIC_ALIASES.get(full_name, full_name)


def heuristics_are_similar(
    time_a: float, time_b: float, relative_threshold: float
) -> bool:
    """ Determine if two heuristic execution times are similar using percentage comparison. """
    if relative_threshold == 0.0:
        return time_a == time_b

    # Handle zero times
    min_time = min(time_a, time_b)
    if min_time == 0:
        return time_a == time_b

    relative_diff = abs(time_a - time_b) / min_time
    return relative_diff <= relative_threshold


def get_near_optimal_heuristics(
    heuristic_times: Dict[str, float], threshold: float
) -> Tuple[str, ...]:
    """
    Get all heuristics within threshold of the best.

    Args:
        heuristic_times: Dict mapping heuristic name to execution time
        threshold: Relative threshold for near-optimal (e.g., 0.10 for 10%)

    Returns:
        Tuple of heuristic names that are near-optimal (sorted for consistency)
    """
    if not heuristic_times:
        return ()

    best_time = min(heuristic_times.values())
    near_optimal = []

    for heuristic, time in heuristic_times.items():
        if heuristics_are_similar(time, best_time, threshold):
            near_optimal.append(heuristic)

    return tuple(sorted(near_optimal))

def parse_feature_vector(s):
    s_clean = str(s).strip('[]').replace('\n', ' ')
    values = [float(x) for x in s_clean.split() if x]
    return np.array(values)


@dataclass
class HeuristicSpecificMetrics:
    """Binary classification metrics for a specific heuristic (One-vs-Rest)."""

    heuristic_name: str
    true_positives: int  # Predicted H AND H is in near-optimal set
    false_positives: int  # Predicted H AND H is NOT in near-optimal set
    false_negatives: int  # Predicted NOT H AND H is in near-optimal set
    true_negatives: int  # Predicted NOT H AND H is NOT in near-optimal set
    precision: float  # TP / (TP + FP) - When we predict H, how often is it correct?
    recall: float  # TP / (TP + FN) - When H is optimal, how often do we predict it?
    accuracy: float  # (TP + TN) / (TP + FP + FN + TN) - Overall correctness
    f1_score: float  # 2 * (precision * recall) / (precision + recall)
    total_optimal_samples: int  # TP + FN - Samples where this heuristic is optimal


@dataclass
class CombinationMetrics:
    """Metrics for a specific combination of near-optimal heuristics."""

    combination: Tuple[str, ...]  # e.g., ("A*", "Dijkstra")
    support: int  # How often this combination is the ground truth
    correct_predictions: int  # How often was prediction in this combination
    recall: float  # correct_predictions / support - When this combo is optimal, how often do we predict a member?


@dataclass
class ToleranceLevelMetrics:
    """Metrics for a specific tolerance level (e.g., 0%, 10%, 20%)."""

    threshold: float
    combination_metrics: Dict[Tuple[str, ...], CombinationMetrics]
    per_heuristic_metrics: Dict[str, HeuristicSpecificMetrics]  # Binary metrics per heuristic
    overall_accuracy: float  # Prediction is in near-optimal set
    macro_accuracy: float  # Average accuracy across all combinations
    total_samples: int
    prediction_counts: Dict[str, int]  # How often each heuristic was predicted


@dataclass
class EvaluationMetrics:
    """Complete evaluation metrics across multiple tolerance levels."""

    # Metrics per tolerance level (0%, 10%, 20%)
    tolerance_metrics: Dict[float, ToleranceLevelMetrics]

    # Time performance metrics
    mean_alignment_time_only: float  # Alignment time only (predicted heuristic)
    mean_alignment_time_with_prediction: float  # Total: feature + classification + alignment
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
        tolerance_dict = {}
        for threshold, level_metrics in self.tolerance_metrics.items():
            combo_dict = {}
            for combo, metrics in level_metrics.combination_metrics.items():
                # Use aliases for keys
                combo_aliases = [get_heuristic_alias(h) for h in combo]
                combo_key = " + ".join(combo_aliases)
                combo_dict[combo_key] = {
                    'support': metrics.support,
                    'correct_predictions': metrics.correct_predictions,
                    'recall': metrics.recall,
                }
            # Convert prediction counts to use aliases
            prediction_counts_aliased = {
                get_heuristic_alias(h): count
                for h, count in level_metrics.prediction_counts.items()
            }

            # Convert per-heuristic metrics to use aliases
            per_heuristic_dict = {}
            for heuristic, metrics in level_metrics.per_heuristic_metrics.items():
                alias = get_heuristic_alias(heuristic)
                per_heuristic_dict[alias] = {
                    'true_positives': metrics.true_positives,
                    'false_positives': metrics.false_positives,
                    'false_negatives': metrics.false_negatives,
                    'true_negatives': metrics.true_negatives,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'accuracy': metrics.accuracy,
                    'f1_score': metrics.f1_score,
                    'total_optimal_samples': metrics.total_optimal_samples,
                }

            tolerance_dict[f"{threshold:.0%}"] = {
                'threshold': threshold,
                'overall_accuracy': level_metrics.overall_accuracy,
                'macro_accuracy': level_metrics.macro_accuracy,
                'total_samples': level_metrics.total_samples,
                'prediction_counts': prediction_counts_aliased,
                'combination_metrics': combo_dict,
                'per_heuristic_metrics': per_heuristic_dict,
            }

        return {
            'tolerance_metrics': tolerance_dict,
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
            "",
            "Alignment Time Performance:",
            f"  Alignment Only (predicted heuristic): {self.mean_alignment_time_only:.4f}s",
            f"  With Prediction Overhead: {self.mean_alignment_time_with_prediction:.4f}s",
            f"  Optimal (fastest heuristic): {self.mean_optimal_alignment_time:.4f}s",
            f"  Performance Ratio (alignment only): {self.performance_ratio_alignment_only:.3f}x",
            f"  Performance Ratio (with prediction): {self.performance_ratio_with_prediction:.3f}x",
            f"  Time Savings vs Worst: {self.time_savings_vs_worst:.2%}",
        ]

        # Add metrics for each tolerance level
        for threshold in sorted(self.tolerance_metrics.keys()):
            level = self.tolerance_metrics[threshold]
            lines.extend(self._format_tolerance_level(level))

        # Prediction timing
        lines.extend(
            [
                "",
                "-" * 80,
                "Prediction Timing:",
                f"  Mean Total: {self.mean_prediction_time * 1000:.3f}ms",
                f"  Mean Feature Extraction: {self.mean_feature_extraction_time * 1000:.3f}ms",
                f"  Mean Classification: {self.mean_classification_time * 1000:.3f}ms",
                "",
                "Top 5 Important Features:",
            ]
        )

        sorted_features = sorted(
            self.feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:5]
        for fname, importance in sorted_features:
            lines.append(f"  {fname}: {importance:.4f}")

        lines.append("=" * 80)
        return "\n".join(lines)

    def _format_tolerance_level(self, level: ToleranceLevelMetrics) -> List[str]:
        """Format metrics for one tolerance level."""
        lines = [
            "",
            "=" * 80,
            f"TOLERANCE LEVEL: {level.threshold:.0%}",
            "=" * 80,
            "",
            "Overall Metrics:",
            f"  Accuracy:       {level.overall_accuracy:>6.2%}",
            f"  Macro Accuracy: {level.macro_accuracy:>6.2%}",
            f"  Total Samples:  {level.total_samples:>6}",
            "",
        ]

        # Add prediction distribution
        lines.extend([
            "Prediction Distribution:",
            "",
            f"  {'Heuristic':<30} {'Count':>8} {'Percentage':>12}",
            "  " + "-" * 52,
        ])

        # Sort by count (most predicted first)
        sorted_preds = sorted(
            level.prediction_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for heuristic, count in sorted_preds:
            alias = get_heuristic_alias(heuristic)
            percentage = 100.0 * count / level.total_samples if level.total_samples > 0 else 0.0
            lines.append(f"  {alias:<30} {count:>8} {percentage:>11.2f}%")

        lines.extend([
            "",
            "Per-Combination Ground Truth Metrics:",
            "",
        ])

        # Calculate max combination length for dynamic column width
        max_combo_len = 0
        combo_strs = []
        for combo, _ in sorted(
            level.combination_metrics.items(),
            key=lambda x: x[1].support,
            reverse=True,
        ):
            combo_aliases = [get_heuristic_alias(h) for h in combo]
            combo_str = " + ".join(combo_aliases)
            combo_strs.append(combo_str)
            max_combo_len = max(max_combo_len, len(combo_str))

        # Set column width with minimum of 30 and maximum of 60
        combo_width = min(max(max_combo_len, 30), 60)

        # Header for combination table
        lines.append(
            f"  {'Combination':<{combo_width}} {'Support':>8} {'Correct':>8} {'Recall':>10}"
        )
        lines.append("  " + "-" * (combo_width + 28))

        # Sort by support (most frequent first)
        sorted_combos = sorted(
            level.combination_metrics.items(),
            key=lambda x: x[1].support,
            reverse=True,
        )

        for (combo, metrics), combo_str in zip(sorted_combos, combo_strs):
            lines.append(
                f"  {combo_str:<{combo_width}} {metrics.support:>8} "
                f"{metrics.correct_predictions:>8} {metrics.recall:>10.2%}"
            )

        return lines


class RecommenderEvaluator:
    """Evaluator for alignment heuristic recommenders."""

    def __init__(
        self,
        classifier: ClassificationModel,
        run_datasets: Union[RunDataset, List[RunDataset]] = None,
        tables: List = None,
        tolerance_thresholds: List[float] = None,
    ):
        if run_datasets is None and tables is None:
            raise ValueError("Either run_datasets or tables must be provided")
        if run_datasets is not None and tables is not None:
            raise ValueError("Cannot provide both run_datasets and tables")

        self.classifier = classifier
        self.run_datasets = normalize_datasets(run_datasets) if run_datasets is not None else None
        self.tables = tables
        self.tolerance_thresholds = tolerance_thresholds or TOLERANCE_THRESHOLDS

        if self.run_datasets is not None:
            validate_aligner_consistency(self.run_datasets)

    def evaluate(self) -> Union[EvaluationMetrics, Dict[str, EvaluationMetrics]]:
        """
        Evaluate classifier on run_datasets or tables with combination-based metrics.

        Returns:
            - If tables provided: Dict[str, EvaluationMetrics] with per-dataset metrics + 'overall'
            - If run_datasets provided: Single EvaluationMetrics object
        """
        logging.info("Starting evaluation...")

        # Use table-based evaluation if tables are provided
        if self.tables is not None:
            return self._evaluate_from_tables()

        # Otherwise use run_datasets evaluation
        # Collect predictions and timing data
        predictions = []
        all_heuristic_times = []  # List of dicts: {heuristic: time}
        actual_times = []
        optimal_times = []
        worst_times = []
        prediction_times = []
        feature_extraction_times = []
        classification_times = []

        # Track per-heuristic performance
        heuristic_runs = defaultdict(list)

        for model, trace, results_dict in tqdm(
            iter_combined_datasets(self.run_datasets), desc="Evaluating"
        ):
            prediction = self.classifier.predict_heuristic(
                model.pm, model.im, model.fm, trace
            )
            predictions.append(prediction.predicted_heuristic)

            # Track timing
            prediction_times.append(prediction.combined_prediction_time)
            feature_extraction_times.append(prediction.feature_extraction_time)
            classification_times.append(prediction.classification_time)

            # Collect all heuristic times for this sample
            best_time = float('inf')
            worst_time = 0.0
            heuristic_times = {}

            for algo_name, (_, _, perf_list) in results_dict.items():
                durations = [
                    p.duration for p in perf_list if p.duration is not None
                ]
                durations = [
                    20.0 if dur == float('inf') else dur for dur in durations
                ]  # set to timeout value
                if durations:
                    mean_time = np.mean(durations)
                    heuristic_times[algo_name] = mean_time
                    heuristic_runs[algo_name].append(mean_time)

                    if mean_time < best_time:
                        best_time = mean_time
                    if mean_time > worst_time:
                        worst_time = mean_time

            all_heuristic_times.append(heuristic_times)
            optimal_times.append(best_time)
            worst_times.append(worst_time)

            # Get actual time for predicted heuristic
            actual_time = heuristic_times[prediction.predicted_heuristic]
            actual_times.append(actual_time)

        # Calculate metrics for each tolerance level
        tolerance_metrics = {}
        for threshold in self.tolerance_thresholds:
            tolerance_metrics[threshold] = self._calculate_tolerance_level_metrics(
                predictions, all_heuristic_times, threshold
            )

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

        # Compute per-heuristic timing statistics (excluding inf values)
        heuristic_timings = {}
        for heuristic, times in heuristic_runs.items():
            times_arr = np.array(times)
            times_finite = times_arr[np.isfinite(times_arr)]
            heuristic_timings[heuristic] = {
                'mean': np.mean(times_finite) if len(times_finite) > 0 else 0.0,
                'std': np.std(times_finite) if len(times_finite) > 0 else 0.0,
                'median': np.median(times_finite) if len(times_finite) > 0 else 0.0,
                'min': np.min(times_finite) if len(times_finite) > 0 else 0.0,
                'max': np.max(times_finite) if len(times_finite) > 0 else 0.0,
                'count': len(times),
                'count_finite': len(times_finite),
                'count_inf': len(times) - len(times_finite),
            }

        # Get feature importance
        feature_importance = self.classifier.get_feature_importance()

        metrics = EvaluationMetrics(
            tolerance_metrics=tolerance_metrics,
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

    def _evaluate_single_table(
        self, table: pd.DataFrame, dataset_name: str, show_progress: bool = False
    ) -> EvaluationMetrics:
        """
        Evaluate classifier on a single table and return full EvaluationMetrics.

        Args:
            table: DataFrame with feature_vector and aligner columns
            dataset_name: Name of the dataset for logging
            show_progress: If True, show tqdm progress bar

        Returns:
            EvaluationMetrics with complete metrics for this table
        """
        table = table.copy()
        table['features'] = table['feature_vector'].apply(parse_feature_vector)

        # Group by combination_id
        grouped = table.groupby('combination_id')

        predictions = []
        all_heuristic_times = []
        actual_times = []
        optimal_times = []
        worst_times = []
        prediction_times = []
        heuristic_runs = defaultdict(list)

        # Wrap iterator with tqdm if progress requested
        iterator = tqdm(grouped, desc=f"Evaluating {dataset_name}") if show_progress else grouped

        for combination_id, group in iterator:
            # Get feature vector
            features = np.array(group.iloc[0]['features']).reshape(1, -1)

            # Predict and measure time
            start_time = time.perf_counter()
            proba = self.classifier._predict_proba(features)[0]
            predicted_class = np.argmax(proba)
            predicted_heuristic = self.classifier.label_encoder.inverse_transform([predicted_class])[0]
            prediction_time = time.perf_counter() - start_time

            predictions.append(predicted_heuristic)
            prediction_times.append(prediction_time)

            # Build heuristic_times dict
            heuristic_times = {}
            for _, row in group.iterrows():
                heuristic_times[row['aligner']] = row['time_total_mean']
                heuristic_runs[row['aligner']].append(row['time_total_mean'])

            all_heuristic_times.append(heuristic_times)

            # Find best and worst times
            best_time = min(heuristic_times.values())
            worst_time = max(heuristic_times.values())
            optimal_times.append(best_time)
            worst_times.append(worst_time)

            # Get predicted heuristic's time
            if predicted_heuristic in heuristic_times:
                actual_times.append(heuristic_times[predicted_heuristic])
            else:
                actual_times.append(worst_time)

        # Calculate metrics for each tolerance level
        tolerance_metrics = {}
        for threshold in self.tolerance_thresholds:
            tolerance_metrics[threshold] = self._calculate_tolerance_level_metrics(
                predictions, all_heuristic_times, threshold
            )

        # Filter out infinite values before computing statistics
        actual_times_arr = np.array(actual_times)
        optimal_times_arr = np.array(optimal_times)
        worst_times_arr = np.array(worst_times)

        # Log inf value statistics (only for overall)
        if dataset_name == 'overall':
            n_inf_actual = np.sum(~np.isfinite(actual_times_arr))
            n_inf_optimal = np.sum(~np.isfinite(optimal_times_arr))
            n_inf_worst = np.sum(~np.isfinite(worst_times_arr))
            if n_inf_actual > 0 or n_inf_optimal > 0 or n_inf_worst > 0:
                logging.warning(
                    f"Found infinite timing values: "
                    f"{n_inf_actual} in actual, {n_inf_optimal} in optimal, {n_inf_worst} in worst. "
                    f"These will be excluded from mean calculations."
                )

        # Compute time performance metrics (excluding inf values)
        actual_times_finite = actual_times_arr[np.isfinite(actual_times_arr)]
        optimal_times_finite = optimal_times_arr[np.isfinite(optimal_times_arr)]
        worst_times_finite = worst_times_arr[np.isfinite(worst_times_arr)]

        mean_alignment_only = np.mean(actual_times_finite) if len(actual_times_finite) > 0 else 0.0
        mean_prediction = np.mean(prediction_times) if len(prediction_times) > 0 else 0.0
        mean_alignment_with_pred = mean_alignment_only + mean_prediction
        mean_optimal = np.mean(optimal_times_finite) if len(optimal_times_finite) > 0 else 0.0
        mean_worst = np.mean(worst_times_finite) if len(worst_times_finite) > 0 else 0.0

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

        # Compute per-heuristic timing statistics (excluding inf values)
        heuristic_timings = {}
        for heuristic, times in heuristic_runs.items():
            times_arr = np.array(times)
            times_finite = times_arr[np.isfinite(times_arr)]
            heuristic_timings[heuristic] = {
                'mean': np.mean(times_finite) if len(times_finite) > 0 else 0.0,
                'std': np.std(times_finite) if len(times_finite) > 0 else 0.0,
                'median': np.median(times_finite) if len(times_finite) > 0 else 0.0,
                'min': np.min(times_finite) if len(times_finite) > 0 else 0.0,
                'max': np.max(times_finite) if len(times_finite) > 0 else 0.0,
                'count': len(times),
                'count_finite': len(times_finite),
                'count_inf': len(times) - len(times_finite),
            }

        # Get feature importance
        feature_importance = self.classifier.get_feature_importance()

        return EvaluationMetrics(
            tolerance_metrics=tolerance_metrics,
            mean_alignment_time_only=mean_alignment_only,
            mean_alignment_time_with_prediction=mean_alignment_with_pred,
            mean_optimal_alignment_time=mean_optimal,
            performance_ratio_alignment_only=performance_ratio_alignment_only,
            performance_ratio_with_prediction=performance_ratio_with_pred,
            time_savings_vs_worst=time_savings,
            heuristic_timings=heuristic_timings,
            feature_importance=feature_importance,
            mean_prediction_time=mean_prediction,
            mean_feature_extraction_time=0.0,  # Features already extracted in CSV
            mean_classification_time=mean_prediction,  # Same as prediction time in CSV mode
        )

    def _evaluate_from_tables(self) -> Dict[str, EvaluationMetrics]:
        """
        Evaluate classifier using pre-computed CSV tables.

        Returns:
            Dict mapping dataset_name to EvaluationMetrics.
            Includes 'overall' key for combined metrics across all datasets.
        """
        logging.info(f"Evaluating using {len(self.tables)} CSV tables...")

        # Evaluate each dataset individually
        all_metrics = {}
        for table in self.tables:
            dataset_name = table.attrs.get('dataset_name', 'unknown')
            logging.info(f"  Evaluating dataset: {dataset_name}")
            dataset_metrics = self._evaluate_single_table(table, dataset_name)
            all_metrics[dataset_name] = dataset_metrics

        # Combine all tables for overall evaluation
        combined_df = pd.concat(self.tables, ignore_index=True)
        combined_df.attrs['dataset_name'] = 'overall'
        logging.info(f"\nEvaluating overall (combined): {len(combined_df)} total rows")

        overall_metrics = self._evaluate_single_table(combined_df, 'overall', show_progress=True)
        all_metrics['overall'] = overall_metrics

        logging.info("Table-based evaluation complete!")
        logging.info(f"\n{overall_metrics.summary()}")

        return all_metrics

    def _calculate_per_heuristic_metrics_binary(
        self,
        predictions: List[str],
        all_heuristic_times: List[Dict[str, float]],
        threshold: float,
    ) -> Dict[str, HeuristicSpecificMetrics]:
        """
        Calculate binary classification metrics per heuristic (One-vs-Rest).

        For each heuristic H:
        - TP: Predicted H AND H is in near-optimal set
        - FP: Predicted H AND H is NOT in near-optimal set
        - FN: Predicted NOT H AND H is in near-optimal set
        - TN: Predicted NOT H AND H is NOT in near-optimal set

        Args:
            predictions: List of predicted heuristic names
            all_heuristic_times: List of dicts mapping heuristic name to time
            threshold: Tolerance threshold (e.g., 0.10 for 10%)

        Returns:
            Dict mapping heuristic name to HeuristicSpecificMetrics
        """
        heuristic_stats = defaultdict(lambda: {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'tn': 0
        })

        all_heuristics = set(all_heuristic_times[0].keys()) if all_heuristic_times else set()

        for pred, heuristic_times in zip(predictions, all_heuristic_times):
            # Get near-optimal set for this sample
            near_optimal = get_near_optimal_heuristics(heuristic_times, threshold)

            # For each heuristic, determine TP/FP/FN/TN
            for heuristic in all_heuristics:
                predicted_h = (pred == heuristic)
                optimal_h = (heuristic in near_optimal)

                if predicted_h and optimal_h:
                    heuristic_stats[heuristic]['tp'] += 1
                elif predicted_h and not optimal_h:
                    heuristic_stats[heuristic]['fp'] += 1
                elif not predicted_h and optimal_h:
                    heuristic_stats[heuristic]['fn'] += 1
                else:  # not predicted_h and not optimal_h
                    heuristic_stats[heuristic]['tn'] += 1

        # Calculate metrics
        per_heuristic_metrics = {}
        for heuristic, stats in heuristic_stats.items():
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']
            tn = stats['tn']

            # Precision: When we predict H, how often is it correct?
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            # Recall: When H is optimal, how often do we predict it?
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # Accuracy: Overall correctness for this heuristic
            accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

            # F1 Score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            per_heuristic_metrics[heuristic] = HeuristicSpecificMetrics(
                heuristic_name=heuristic,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                true_negatives=tn,
                precision=precision,
                recall=recall,
                accuracy=accuracy,
                f1_score=f1,
                total_optimal_samples=tp + fn,
            )

        return per_heuristic_metrics

    def _calculate_tolerance_level_metrics(
        self,
        predictions: List[str],
        all_heuristic_times: List[Dict[str, float]],
        threshold: float,
    ) -> ToleranceLevelMetrics:
        """
        Calculate metrics for a specific tolerance level.

        Args:
            predictions: List of predicted heuristic names
            all_heuristic_times: List of dicts mapping heuristic name to time
            threshold: Tolerance threshold (e.g., 0.10 for 10%)

        Returns:
            ToleranceLevelMetrics for this threshold
        """
        # Get ground truth combinations for each sample
        ground_truth_combos = []
        for heuristic_times in all_heuristic_times:
            combo = get_near_optimal_heuristics(heuristic_times, threshold)
            ground_truth_combos.append(combo)

        combo_support = Counter(ground_truth_combos)

        # Count predictions per heuristic
        prediction_counts = Counter(predictions)

        # Count correct predictions per combination
        combo_correct = defaultdict(int)
        overall_correct = 0

        for pred, gt_combo in zip(predictions, ground_truth_combos):
            if pred in gt_combo:
                overall_correct += 1
                combo_correct[gt_combo] += 1

        # Build CombinationMetrics for each combination
        combination_metrics = {}
        all_recalls = []

        for combo, support in combo_support.items():
            correct = combo_correct[combo]
            recall = correct / support if support > 0 else 0.0

            combination_metrics[combo] = CombinationMetrics(
                combination=combo,
                support=support,
                correct_predictions=correct,
                recall=recall,
            )

            all_recalls.append(recall)

        # Calculate overall metrics
        total_samples = len(predictions)
        overall_accuracy = overall_correct / total_samples if total_samples > 0 else 0.0
        macro_accuracy = np.mean(all_recalls) if all_recalls else 0.0

        # Calculate per-heuristic binary metrics
        per_heuristic_metrics = self._calculate_per_heuristic_metrics_binary(
            predictions, all_heuristic_times, threshold
        )

        return ToleranceLevelMetrics(
            threshold=threshold,
            combination_metrics=combination_metrics,
            per_heuristic_metrics=per_heuristic_metrics,
            overall_accuracy=overall_accuracy,
            macro_accuracy=macro_accuracy,
            total_samples=total_samples,
            prediction_counts=dict(prediction_counts),
        )

    def compare_with_baselines(
        self, baselines: List[ClassificationModel]
    ) -> pd.DataFrame:
        """Compare classifier with baseline models across all tolerance levels."""
        results = []

        main_result = self.evaluate()
        # Handle both return types: Dict or single EvaluationMetrics
        main_metrics = main_result['overall'] if isinstance(main_result, dict) else main_result

        result_row = {
            'model': self.classifier.__class__.__name__,
            'performance_ratio_alignment_only': main_metrics.performance_ratio_alignment_only,
            'performance_ratio_with_prediction': main_metrics.performance_ratio_with_prediction,
            'mean_alignment_time_only': main_metrics.mean_alignment_time_only,
            'mean_alignment_time_with_prediction': main_metrics.mean_alignment_time_with_prediction,
            'mean_prediction_time': main_metrics.mean_prediction_time,
        }
        # Add accuracy for each tolerance level
        for threshold, level_metrics in main_metrics.tolerance_metrics.items():
            result_row[f'accuracy_{threshold:.0%}'] = level_metrics.overall_accuracy
            result_row[f'macro_accuracy_{threshold:.0%}'] = level_metrics.macro_accuracy
        results.append(result_row)

        # Evaluate baselines
        for baseline in baselines:
            logging.info(
                f"\nEvaluating baseline: {baseline.__class__.__name__}"
            )
            evaluator = RecommenderEvaluator(
                baseline,
                run_datasets=self.run_datasets,
                tables=self.tables,
                tolerance_thresholds=self.tolerance_thresholds
            )
            baseline_result = evaluator.evaluate()
            # Handle both return types
            baseline_metrics = baseline_result['overall'] if isinstance(baseline_result, dict) else baseline_result

            result_row = {
                'model': baseline.__class__.__name__,
                'performance_ratio_alignment_only': baseline_metrics.performance_ratio_alignment_only,
                'performance_ratio_with_prediction': baseline_metrics.performance_ratio_with_prediction,
                'mean_alignment_time_only': baseline_metrics.mean_alignment_time_only,
                'mean_alignment_time_with_prediction': baseline_metrics.mean_alignment_time_with_prediction,
                'mean_prediction_time': baseline_metrics.mean_prediction_time,
            }
            # Add accuracy for each tolerance level
            for threshold, level_metrics in baseline_metrics.tolerance_metrics.items():
                result_row[f'accuracy_{threshold:.0%}'] = level_metrics.overall_accuracy
                result_row[f'macro_accuracy_{threshold:.0%}'] = level_metrics.macro_accuracy
            results.append(result_row)

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
