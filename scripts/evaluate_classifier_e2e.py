"""
Evaluation script for ML classification across all XES datasets.

This script:
1. Iterates over XES datasets (hardcoded train/test split)
2. Creates ProcessModelDataset with models for each dataset
3. Applies deduplication with UniqueProcessModelDataset
4. Creates RunDataset with multiple alignment heuristics
5. Trains XGBoostClassifier on train datasets
6. Evaluates on test datasets and compares with baselines
"""

import json
import logging
from pathlib import Path
import numpy as np

from dataloaders.xes_log import XESEventLogDataset
from dataloaders.net import ProcessModelDataset, VariantRandomDistributionSampler
from dataloaders.unique_net import UniqueProcessModelDataset
from dataloaders.runs import RunDataset, AlignerSpec, SimplePerturbedTraceSampler
from deduplication.deduplicator import DeduplicationConfig
from features.extractors import CompositeFeatureExtractor
from models import (
    XGBoostClassifier,
    SingleBestSolver,
    RandomClassifier,
    RecommenderEvaluator
)
from pm4py.discovery import discover_petri_net_inductive
from util.distributions import ExponentialSpec, NormalSpec

logging.basicConfig(level=logging.INFO)

OUTPUT_DIR = Path("outputs") / "evaluate_classifier"

TRAIN_DATASETS = {
    'd9769f3d-0ab0-4fb8-803b-0d1120ffcf54': ['Hospital_log.xes'],
    # '63a8435a-077d-4ece-97cd-2c76d394d99c': ['BPIC15_2.xes'],
    # 'ed445cdd-27d5-4d77-a1f7-59fe7360cfbe': ['BPIC15_3.xes'],
    # '679b11cf-47cd-459e-a6de-9ca614e25985': ['BPIC15_4.xes'],
    # '3301445f-95e8-4ff0-98a4-901f1f204972': ['BPI%20Challenge%202018.xes'],
    # '3926db30-f712-4394-aebc-75976070e91f': ['BPI_Challenge_2012.xes'],
    # 'a6f651a7-5ce0-4bc6-8be1-a7747effa1cc': ['RequestForPayment.xes'],
    # '6af6d5f0-f44c-49be-aac8-8eaa5fe4f6fd': ['Hospital%20Billing%20-%20Event%20Log.xes'],
    # '33632f3c-5c48-40cf-8d8f-2db57f5a6ce7': ['Sepsis%20Cases%20-%20Event%20Log.xes'],
    # 'd06aff4b-79f0-45e6-8ec8-e19730c248f1': ['BPI_Challenge_2019.xes'],
    # '3537c19d-6c64-4b1d-815d-915ab0e479da': ['BPI_Challenge_2013_open_problems.xes'],
    # '500573e6-accc-4b0c-9576-aa5468b10cee': ['BPI_Challenge_2013_incidents.xes'],
    # '91fd1fa8-4df4-4b1a-9a3f-0116c412378f': ['InternationalDeclarations.xes'],
    # 'fb84cf2d-166f-4de2-87be-62ee317077e5': ['PrepaidTravelCost.xes'],
    # '12683249': ['Road_Traffic_Fine_Management_Process.xes'],
}

TEST_DATASETS = {
    'a0addfda-2044-4541-a450-fdcc9fe16d17': ['BPIC15_1.xes'],
    # 'b32c6fe5-f212-4286-9774-58dd53511cf8': ['BPIC15_5.xes'],
    # '5f3067df-f10b-45da-b98b-86ae4c7a310b': ['BPI%20Challenge%202017.xes'],
    # 'db35afac-2133-40f3-a565-2dc77a9329a3': ['PermitLog.xes'],
    # '6a0a26d2-82d0-4018-b1cd-89afb0e8627f': ['DomesticDeclarations.xes'],
    # 'c2c3b154-ab26-4b31-a0e8-8f2350ddac11': ['BPI_Challenge_2013_closed_problems.xes'],
}


def create_run_dataset(dataset_uuid: str, filename: str, dedup_config: DeduplicationConfig):
    """Create RunDataset for a single XES file."""
    logging.info(f"\nProcessing: {filename}")

    dataset_path = Path("data") / dataset_uuid / filename

    try:
        log_dataset = XESEventLogDataset(str(dataset_path), attribute="concept:name")

        pm_dataset = ProcessModelDataset(
            log_dataset=log_dataset,
            discovery_methods={"inductive": discover_petri_net_inductive},
            param_grid={
                "noise_threshold": [0.0, 0.1, 0.2, 0.3, 0.4],
                "disable_fallthroughs": [True],
            },
            sampler_specs={
                "variant_random": VariantRandomDistributionSampler(
                    seed=1,
                    n_subsets=1000,
                    max_len_subset=100,
                    min_len_subset=10,
                    len_distribution=ExponentialSpec(rate=1.0 / 100.0),
                    freq_distribution=NormalSpec(mean=10.0, std=5.0),
                    reconstruct_frequency=True,
                )
            },
            cached=True,
            max_models=100,
        )

        unique_dataset = UniqueProcessModelDataset(
            base_dataset=pm_dataset,
            dedup_config=dedup_config,
            force_recompute=False,
        )

        sampler = SimplePerturbedTraceSampler(
            seed=42,
            ds=unique_dataset,
            slice=range(0, 10)
        )

        run_base_path = Path("cache") / "run_datasets" / filename.replace('.xes', '')
        run_base_path.mkdir(parents=True, exist_ok=True)

        run_dataset = RunDataset(
            base_path=run_base_path,
            process_model_dataset=unique_dataset,
            aligners=AlignerSpec.A_STAR.value,
            trace_sampler=sampler,
            n_runs=1,
        )

        logging.info(f"Created RunDataset with {len(run_dataset)} items")
        return run_dataset

    except Exception as e:
        logging.error(f"ERROR processing {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def save_results(metrics, comparison_df, train_count: int, test_count: int):
    """Save evaluation results to output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "metrics.json", 'w') as f:
        metrics_dict = convert_to_serializable(metrics.to_dict())
        json.dump(metrics_dict, f, indent=2)

    comparison_df.to_csv(OUTPUT_DIR / "baseline_comparison.csv", index=False)

    with open(OUTPUT_DIR / "summary.txt", 'w') as f:
        f.write("ML CLASSIFIER EVALUATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Train datasets: {train_count}\n")
        f.write(f"Test datasets: {test_count}\n\n")
        f.write("="*80 + "\n\n")
        f.write(metrics.summary())
        f.write("\n\n" + "="*80 + "\n\n")
        f.write("Baseline Comparison:\n")
        f.write(comparison_df.to_string())

    logging.info(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    dedup_config = DeduplicationConfig()

    # Create train RunDatasets
    logging.info(f"\nCreating {sum(len(f) for f in TRAIN_DATASETS.values())} train RunDatasets...")
    train_run_datasets = []
    for dataset_uuid, files in TRAIN_DATASETS.items():
        for filename in files:
            run_dataset = create_run_dataset(dataset_uuid, filename, dedup_config)
            if run_dataset is not None:
                train_run_datasets.append(run_dataset)

    # Create test RunDatasets
    logging.info(f"\nCreating {sum(len(f) for f in TEST_DATASETS.values())} test RunDatasets...")
    test_run_datasets = []
    for dataset_uuid, files in TEST_DATASETS.items():
        for filename in files:
            run_dataset = create_run_dataset(dataset_uuid, filename, dedup_config)
            if run_dataset is not None:
                test_run_datasets.append(run_dataset)

    logging.info("\nCreating feature extractor...")
    feature_extractor = CompositeFeatureExtractor(use_cache=True)

    logging.info("\nTraining XGBoostClassifier...")
    classifier = XGBoostClassifier(
        run_datasets=train_run_datasets,
        feature_extractor=feature_extractor,
        cache_dir=Path("cache") / "models",
        force_retrain=False,
    )

    # Train baselines
    logging.info("\nTraining baseline classifiers...")
    single_best = SingleBestSolver(
        run_datasets=train_run_datasets,
        feature_extractor=feature_extractor,
        cache_dir=Path("cache") / "models",
    )

    random_clf = RandomClassifier(
        run_datasets=train_run_datasets,
        feature_extractor=feature_extractor,
        cache_dir=Path("cache") / "models",
    )

    # Evaluate
    logging.info("\nEvaluating on test datasets...")
    evaluator = RecommenderEvaluator(
        classifier=classifier,
        run_datasets=test_run_datasets
    )

    metrics = evaluator.evaluate()

    # Compare with baselines
    logging.info("\nComparing with baselines...")
    comparison_df = evaluator.compare_with_baselines([single_best, random_clf])
    logging.info("\n" + comparison_df.to_string())
    
    save_results(metrics, comparison_df, len(train_run_datasets), len(test_run_datasets))
    