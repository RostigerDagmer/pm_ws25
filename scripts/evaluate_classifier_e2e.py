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

from typing import Optional
from experiments.simulation.dataset import get_synthetic_dataset
from util.rng import RNG
import yaml
from configs.schema import PipelineConfig
import json
import logging
from pathlib import Path
import numpy as np
import os

from dataloaders.runs import RunDataset
from features import CompositeFeatureExtractor
from models import (
    XGBoostClassifier,
    SingleBestSolver,
    RandomClassifier,
)
from models.evaluator_csv import RecommenderEvaluator
from scripts.generate_dataset import build_pipeline
import pandas as pd

logging.basicConfig(level=logging.INFO)

SEED = 1
OUTPUT_DIR = Path("outputs") / "evaluate_classifier"

# Legacy code: These dictionaries are no longer used.
# The script now automatically discovers all datasets using find_existing_tables()
# which searches cache/.runs, data/runs, and cache/.runs_synthetic for CSV files.

# TRAIN_DATASETS = {
#     'd9769f3d-0ab0-4fb8-803b-0d1120ffcf54': ['Hospital_log.xes'],
#     '63a8435a-077d-4ece-97cd-2c76d394d99c': ['BPIC15_2.xes'],
#     'ed445cdd-27d5-4d77-a1f7-59fe7360cfbe': ['BPIC15_3.xes'],
#     '679b11cf-47cd-459e-a6de-9ca614e25985': ['BPIC15_4.xes'],
#     '3301445f-95e8-4ff0-98a4-901f1f204972': ['BPI%20Challenge%202018.xes'],
#     '3926db30-f712-4394-aebc-75976070e91f': ['BPI_Challenge_2012.xes'],
#     'a6f651a7-5ce0-4bc6-8be1-a7747effa1cc': ['RequestForPayment.xes'],
#     '6af6d5f0-f44c-49be-aac8-8eaa5fe4f6fd': [
#         'Hospital%20Billing%20-%20Event%20Log.xes'
#     ],
#     '33632f3c-5c48-40cf-8d8f-2db57f5a6ce7': [
#         'Sepsis%20Cases%20-%20Event%20Log.xes'
#     ],
#     'd06aff4b-79f0-45e6-8ec8-e19730c248f1': ['BPI_Challenge_2019.xes'],
#     '3537c19d-6c64-4b1d-815d-915ab0e479da': [
#         'BPI_Challenge_2013_open_problems.xes'
#     ],
#     '500573e6-accc-4b0c-9576-aa5468b10cee': [
#         'BPI_Challenge_2013_incidents.xes'
#     ],
#     '91fd1fa8-4df4-4b1a-9a3f-0116c412378f': ['InternationalDeclarations.xes'],
#     'fb84cf2d-166f-4de2-87be-62ee317077e5': ['PrepaidTravelCost.xes'],
#     '12683249': ['Road_Traffic_Fine_Management_Process.xes'],
#     'a0addfda-2044-4541-a450-fdcc9fe16d17': ['BPIC15_1.xes'],
# }

# TEST_DATASETS = {
#     'b32c6fe5-f212-4286-9774-58dd53511cf8': ['BPIC15_5.xes'],
#     '5f3067df-f10b-45da-b98b-86ae4c7a310b': ['BPI%20Challenge%202017.xes'],
#     'db35afac-2133-40f3-a565-2dc77a9329a3': ['PermitLog.xes'],
#     '6a0a26d2-82d0-4018-b1cd-89afb0e8627f': ['DomesticDeclarations.xes'],
#     'c2c3b154-ab26-4b31-a0e8-8f2350ddac11': [
#         'BPI_Challenge_2013_closed_problems.xes'
#     ],
# }


def find_existing_tables(
    root: Path,
):
    """
    Load tables for training and evaluation.

    - train: Uses .train.csv (labels only, one best aligner per combination)
    - test: Uses .runs.csv filtered by .test.csv combination_ids
      (complete data with all aligners per combination for proper evaluation)
    - eval: Uses .eval.csv (labels only)
    """
    train_tables = []
    test_tables = []
    eval_tables = []

    # Load train tables (labels only)
    for table_path in root.glob("**/*.train.csv"):
        train_tables.append(pd.read_csv(table_path))

    # Load test tables: runs.csv filtered by test.csv combination_ids
    test_label_paths = list(root.glob("**/*.test.csv"))
    for test_label_path in test_label_paths:
        # Get corresponding runs.csv file
        runs_path = test_label_path.parent / test_label_path.name.replace('.test.csv', '.runs.csv')
        if not runs_path.exists():
            logging.warning(f"No runs.csv found for {test_label_path}, using test.csv directly")
            test_tables.append(pd.read_csv(test_label_path))
            continue

        # Load test combination_ids
        test_labels = pd.read_csv(test_label_path)
        test_combination_ids = set(test_labels['combination_id'].unique())

        # Load full runs data and filter by test combination_ids
        runs_df = pd.read_csv(runs_path)
        test_filtered = runs_df[runs_df['combination_id'].isin(test_combination_ids)]

        logging.info(f"Test set {test_label_path.name}: {len(test_combination_ids)} combinations, "
                     f"{len(test_filtered)} total rows (all aligners)")
        test_tables.append(test_filtered)

    # Load eval tables (labels only)
    for table_path in root.glob("**/*.eval.csv"):
        eval_tables.append(pd.read_csv(table_path))

    return train_tables, test_tables, eval_tables


def get_natural_dataset(
    log_path: str,
    config: str,
    base_path: Optional[str] = None,
    skip_init: bool = False,
) -> RunDataset:
    RNG.initialize(SEED)
    cfg_dict = yaml.safe_load(open(config))
    cfg = PipelineConfig.model_validate(cfg_dict)
    cfg.log_path = log_path
    cfg.alignment.cache_path = base_path

    cfg.seed = SEED
    # Use SLURM_CPUS_PER_TASK if available, otherwise default to 16
    cfg.alignment.workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 16))

    # Skip config <-> cache check
    # (This is unsafe if process models are being referenced in the cache that are not included in the current config)
    return build_pipeline(cfg, skip_init=skip_init)


if __name__ == "__main__":

    config_path = "configs/default.yaml"
    cache_path = "cache/.runs"

    # Create train RunDatasets
    # This is too heavy for now
    # logging.info(
    #     f"\nCreating {sum(len(f) for f in TRAIN_DATASETS.values())} train RunDatasets..."
    # )
    # train_run_datasets = []
    # for dataset_uuid, files in TRAIN_DATASETS.items():
    #     for filename in files:
    #         run_dataset = get_natural_dataset(
    #             str(Path("data") / dataset_uuid / filename),
    #             config_path,
    #             cache_path,
    #         )
    #         if run_dataset is not None:
    #             train_run_datasets.append(run_dataset)
    #
    # train_run_datasets.append(
    #     get_synthetic_dataset(Path(cache_path), seed=SEED, count=50)
    # )

    # Can merge individually extracted tables with precomputed features
    logging.info("\nLoading pre-computed CSV tables...")
    # Search in cache/.runs, data/runs, and data/runs_synthetic for CSV files
    train_tables, test_tables, eval_tables = [], [], []
    for search_path in [Path(cache_path), Path("data/runs"), Path("cache/.runs_synthetic")]:
        if search_path.exists():
            logging.info(f"  Searching in: {search_path}")
            t_train, t_test, t_eval = find_existing_tables(search_path)
            train_tables.extend(t_train)
            test_tables.extend(t_test)
            eval_tables.extend(t_eval)
            logging.info(f"    Found: {len(t_train)} train, {len(t_test)} test, {len(t_eval)} eval tables")

    logging.info("\nCreating feature extractor...")
    feature_extractor = CompositeFeatureExtractor(use_cache=True)

    logging.info("\nTraining XGBoostClassifier...")
    classifier = XGBoostClassifier(
        tables=train_tables,
        feature_extractor=feature_extractor,
        cache_dir=Path("cache") / "models",
        force_retrain=True,
    )

    # Train baselines
    logging.info("\nTraining baseline classifiers...")
    single_best = SingleBestSolver(
        tables=train_tables,
        feature_extractor=feature_extractor,
        cache_dir=Path("cache") / "models",
        force_retrain=True,
    )

    random_clf = RandomClassifier(
        tables=train_tables,
        feature_extractor=feature_extractor,
        cache_dir=Path("cache") / "models",
        force_retrain=True,
    )

    # Test data now uses pre-computed CSV files (loaded in test_tables above)
    # Old approach (loading from XES files) is commented out:
    # logging.info("\nLoading test RunDatasets (using cached alignment runs)...")
    # test_run_datasets = []
    # for dataset_uuid, files in TEST_DATASETS.items():
    #     for filename in files:
    #         logging.info(f"Loading: {filename}")
    #         run_dataset = get_natural_dataset(
    #             str(Path("data") / dataset_uuid / filename),
    #             config_path,
    #             cache_path,
    #             use_cache=True,  # Use cached alignment runs for speed
    #         )
    #         if run_dataset is not None:
    #             test_run_datasets.append(run_dataset)
    #             logging.info(f"  ✓ Loaded {len(run_dataset)} runs from cache")
    #
    # test_run_datasets.append(
    #     get_synthetic_dataset(Path(cache_path), seed=SEED + 1, count=4)
    # )

    # Evaluate using test tables (CSV files)
    logging.info(f"\nEvaluating on {len(test_tables)} test datasets...")
    evaluator = RecommenderEvaluator(
        classifier=classifier, tables=test_tables
    )

    metrics = evaluator.evaluate()

    # Compare with baselines
    logging.info("\nComparing with baselines...")
    comparison_df = evaluator.compare_with_baselines([single_best, random_clf])
    logging.info("\n" + comparison_df.to_string())

    RecommenderEvaluator.save_results(
        metrics=metrics,
        comparison_df=comparison_df,
        output_dir=OUTPUT_DIR,
        train_count=len(train_tables),
        test_count=len(test_tables),
    )
