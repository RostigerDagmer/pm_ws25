"""
Evaluation framework for alignment heuristic recommenders.

Uses combination-based metrics where ground truth can be multiple near-optimal
heuristics (within a tolerance threshold of the best).
"""

from typing import Dict, Any, List, Union, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import json
from collections import defaultdict, Counter

from models.base import ClassificationModel
from dataloaders.labels import LabelDataset


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
    """Determine if two heuristic execution times are similar using percentage comparison."""
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
    recall: float  # correct_predictions / support


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
class DatasetEvaluationData:
    """Accumulates evaluation data for a single dataset during evaluation."""

    predictions: List[str] = field(default_factory=list)
    all_heuristic_times: List[Dict[str, float]] = field(default_factory=list)
    actual_times: List[float] = field(default_factory=list)
    optimal_times: List[float] = field(default_factory=list)
    worst_times: List[float] = field(default_factory=list)
    prediction_times: List[float] = field(default_factory=list)
    feature_extraction_times: List[float] = field(default_factory=list)
    classification_times: List[float] = field(default_factory=list)
    heuristic_runs: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )


@dataclass
class EvaluationMetrics:
    """Complete evaluation metrics across multiple tolerance levels."""

    # Metrics per tolerance level (0%, 10%, 20%)
    tolerance_metrics: Dict[float, ToleranceLevelMetrics]

    # Time performance metrics
    mean_alignment_time_only: (
        float  # Alignment time only (predicted heuristic)
    )
    mean_alignment_time_with_prediction: (
        float  # Total: feature + classification + alignment
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

    def _format_tolerance_level(
        self, level: ToleranceLevelMetrics
    ) -> List[str]:
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
        lines.extend(
            [
                "Prediction Distribution:",
                "",
                f"  {'Heuristic':<30} {'Count':>8} {'Percentage':>12}",
                "  " + "-" * 52,
            ]
        )

        # Sort by count (most predicted first)
        sorted_preds = sorted(
            level.prediction_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for heuristic, count in sorted_preds:
            alias = get_heuristic_alias(heuristic)
            percentage = (
                100.0 * count / level.total_samples
                if level.total_samples > 0
                else 0.0
            )
            lines.append(f"  {alias:<30} {count:>8} {percentage:>11.2f}%")

        lines.extend(
            [
                "",
                "Per-Combination Ground Truth Metrics:",
                "",
            ]
        )

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
        dataset: LabelDataset,
        tolerance_thresholds: Optional[List[float]] = None,
    ):
        self.classifier = classifier
        self.tolerance_thresholds = (
            tolerance_thresholds or TOLERANCE_THRESHOLDS
        )
        self.dataset = dataset

    def evaluate(self, batched: bool = True, print_summary: bool = True) -> Dict[str, EvaluationMetrics]:
        """
        Evaluate classifier on run_datasets.

        Returns:
            Dict mapping dataset_id to EvaluationMetrics.
            Includes 'overall' key for combined metrics across all datasets.
        """
        logging.info("Starting evaluation...")

        # Collect data per dataset (including 'overall' as special key)
        dataset_data: Dict[str, DatasetEvaluationData] = defaultdict(
            DatasetEvaluationData
        )

        for runs in tqdm(self.dataset.iter_by_model(), desc="Evaluating"):
            dataset_ids, runs = zip(*runs)
            model = runs[0].model.deserialize()
            traces = [run.trace for run in runs]

            predictions = []
            if batched:
                predictions = self.classifier.predict_batched(model, traces)
            else:
                for trace in traces:
                    prediction = self.classifier.predict_heuristic(model.pm, model.im, model.fm, trace)
                    predictions.append(prediction)

            # Process each prediction
            for dataset_id, gt, prediction in zip(dataset_ids, runs, predictions):
                # Get all heuristic times
                all_alignments = {
                    item.algo: item.perf
                    for item in self.dataset.get_combination_results(
                        dataset_id, gt.comb_id
                    )
                }

                durations = {
                    algo: np.mean([
                        (p["duration"] if p["duration"] != float('inf') else 20.0)
                        for p in perf
                    ])
                    for algo, perf in all_alignments.items()
                }

                best_time = min(durations.values())
                worst_time = max(durations.values())
                actual_time = durations[prediction.predicted_heuristic]

                # Store in both dataset-specific and overall
                for key in [dataset_id, 'overall']:
                    data = dataset_data[key]
                    data.predictions.append(prediction.predicted_heuristic)
                    data.all_heuristic_times.append(durations)
                    data.actual_times.append(actual_time)
                    data.optimal_times.append(best_time)
                    data.worst_times.append(worst_time)
                    data.prediction_times.append(prediction.combined_prediction_time)
                    data.feature_extraction_times.append(prediction.feature_extraction_time)
                    data.classification_times.append(prediction.classification_time)

                    for algo, time in durations.items():
                        data.heuristic_runs[algo].append(time)

        # Build metrics for each dataset
        all_metrics = {}

        logging.info(f"\nEvaluating {len(dataset_data) - 1} datasets individually...")
        for dataset_id, data in dataset_data.items():  # Includes 'overall'
            logging.info(f"  Dataset: {dataset_id[:8]} ({len(data.predictions)} samples)")
            all_metrics[dataset_id] = self._compute_metrics_from_data(data)

        logging.info("Evaluation complete!")
        if print_summary:
            logging.info(f"\n{all_metrics['overall'].summary()}")
    
        return all_metrics

    def _compute_metrics_from_data(
        self, data: DatasetEvaluationData
    ) -> EvaluationMetrics:
        """Helper to compute EvaluationMetrics from collected data."""
        # Calculate tolerance metrics
        tolerance_metrics = {}
        for threshold in self.tolerance_thresholds:
            tolerance_metrics[threshold] = self._calculate_tolerance_level_metrics(
                data.predictions, data.all_heuristic_times, threshold
            )

        # Time performance metrics
        mean_alignment_only = np.mean(data.actual_times)
        mean_prediction = np.mean(data.prediction_times)
        mean_alignment_with_pred = mean_alignment_only + mean_prediction
        mean_optimal = np.mean(data.optimal_times)
        mean_worst = np.mean(data.worst_times)

        performance_ratio_alignment_only = (
            mean_alignment_only / mean_optimal if mean_optimal > 0 else float('inf')
        )
        performance_ratio_with_pred = (
            mean_alignment_with_pred / mean_optimal if mean_optimal > 0 else float('inf')
        )
        time_savings = (
            (mean_worst - mean_alignment_only) / (mean_worst - mean_optimal)
            if (mean_worst - mean_optimal) > 0 else 0.0
        )

        # Per-heuristic timing statistics
        heuristic_timings = {}
        for heuristic, times in data.heuristic_runs.items():
            heuristic_timings[heuristic] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'median': np.median(times),
                'min': np.min(times),
                'max': np.max(times),
                'count': len(times),
            }

        # Feature importance (same for all datasets)
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
            mean_prediction_time=np.mean(data.prediction_times),
            mean_feature_extraction_time=np.mean(data.feature_extraction_times),
            mean_classification_time=np.mean(data.classification_times),
        )

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
        overall_accuracy = (
            overall_correct / total_samples if total_samples > 0 else 0.0
        )
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
        self,
        baselines: List[ClassificationModel],
        main_result: Dict[str, EvaluationMetrics] = None,
    ) -> pd.DataFrame:
        """
        Compare classifier with baseline models across all tolerance levels.

        Args:
            baselines: List of baseline classifiers to compare against
            main_result: Optional pre-computed main classifier results.
                        If None, will call evaluate_batched()

        Returns:
            DataFrame with comparison metrics
        """
        results = []

        # Extract 'overall' metrics
        main_metrics = main_result['overall']

        result_row = {
            'model': self.classifier.__class__.__name__,
            'performance_ratio_alignment_only': main_metrics.performance_ratio_alignment_only,
            'performance_ratio_with_prediction': main_metrics.performance_ratio_with_prediction,
            'mean_alignment_time_only': main_metrics.mean_alignment_time_only,
            'mean_alignment_time_with_prediction': main_metrics.mean_alignment_time_with_prediction,
            'mean_prediction_time': main_metrics.mean_prediction_time,
            'mean_feature_extraction_time': main_metrics.mean_feature_extraction_time,
            'mean_classification_time': main_metrics.mean_classification_time,
        }
        # Add accuracy for each tolerance level
        for threshold, level_metrics in main_metrics.tolerance_metrics.items():
            result_row[f'accuracy_{threshold:.0%}'] = (
                level_metrics.overall_accuracy
            )
            result_row[f'macro_accuracy_{threshold:.0%}'] = (
                level_metrics.macro_accuracy
            )
        results.append(result_row)

        # Evaluate baselines
        for baseline in baselines:
            logging.info(
                f"\nEvaluating baseline: {baseline.__class__.__name__}"
            )
            evaluator = RecommenderEvaluator(
                baseline, self.dataset, self.tolerance_thresholds
            )
            baseline_result = evaluator.evaluate()
            # Extract 'overall' metrics
            baseline_metrics = baseline_result['overall']

            result_row = {
                'model': baseline.__class__.__name__,
                'performance_ratio_alignment_only': baseline_metrics.performance_ratio_alignment_only,
                'performance_ratio_with_prediction': baseline_metrics.performance_ratio_with_prediction,
                'mean_alignment_time_only': baseline_metrics.mean_alignment_time_only,
                'mean_alignment_time_with_prediction': baseline_metrics.mean_alignment_time_with_prediction,
                'mean_prediction_time': baseline_metrics.mean_prediction_time,
                'mean_feature_extraction_time': baseline_metrics.mean_feature_extraction_time,
                'mean_classification_time': baseline_metrics.mean_classification_time,
            }
            # Add accuracy for each tolerance level
            for (
                threshold,
                level_metrics,
            ) in baseline_metrics.tolerance_metrics.items():
                result_row[f'accuracy_{threshold:.0%}'] = (
                    level_metrics.overall_accuracy
                )
                result_row[f'macro_accuracy_{threshold:.0%}'] = (
                    level_metrics.macro_accuracy
                )
            results.append(result_row)

        df = pd.DataFrame(results)

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
