"""Hyperparameter optimization for XGBoostClassifier using Optuna."""

import json
import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import optuna
import pandas as pd

from dataloaders.labels import LabelDataset
from dataloaders.util import get_natural_dataset, find_existing_tables
from features import CompositeFeatureExtractor
from models import XGBoostClassifier, RecommenderEvaluator
from util.rng import RNG

from dataloaders.util import create_tables
from scripts.create_labels import DF_SCHEMA, format_row

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

SEED = 42

# =============================================================================
# Dataset Configuration (60/20/20 Split by Samples)
# =============================================================================

# ~62.000 Samples (61%)
TRAIN_DATASETS = {
    'd9769f3d-0ab0-4fb8-803b-0d1120ffcf54': ['Hospital_log.xes'],
    '63a8435a-077d-4ece-97cd-2c76d394d99c': ['BPIC15_2.xes'],
    'ed445cdd-27d5-4d77-a1f7-59fe7360cfbe': ['BPIC15_3.xes'],
    '679b11cf-47cd-459e-a6de-9ca614e25985': ['BPIC15_4.xes'],
    'd06aff4b-79f0-45e6-8ec8-e19730c248f1': ['BPI_Challenge_2019.xes'],
    '3926db30-f712-4394-aebc-75976070e91f': ['BPI_Challenge_2012.xes'],
    'db35afac-2133-40f3-a565-2dc77a9329a3': ['PermitLog.xes'],
    'fb84cf2d-166f-4de2-87be-62ee317077e5': ['PrepaidTravelCost.xes'],
    '33632f3c-5c48-40cf-8d8f-2db57f5a6ce7': [
        'Sepsis%20Cases%20-%20Event%20Log.xes'
    ],
}

# ~22.000 Samples (21%)
VAL_DATASETS = {
    'a0addfda-2044-4541-a450-fdcc9fe16d17': ['BPIC15_1.xes'],
    '3301445f-95e8-4ff0-98a4-901f1f204972': ['BPI%20Challenge%202018.xes'],
    '91fd1fa8-4df4-4b1a-9a3f-0116c412378f': ['InternationalDeclarations.xes'],
    '3537c19d-6c64-4b1d-815d-915ab0e479da': [
        'BPI_Challenge_2013_open_problems.xes'
    ],
}

# ~18.000 Samples (18%) -> not used in this script
TEST_DATASETS = {
    '6af6d5f0-f44c-49be-aac8-8eaa5fe4f6fd': [
        'Hospital%20Billing%20-%20Event%20Log.xes'
    ],
    'b32c6fe5-f212-4286-9774-58dd53511cf8': ['BPIC15_5.xes'],
    '5f3067df-f10b-45da-b98b-86ae4c7a310b': ['BPI%20Challenge%202017.xes'],
    '6a0a26d2-82d0-4018-b1cd-89afb0e8627f': ['DomesticDeclarations.xes'],
    'c2c3b154-ab26-4b31-a0e8-8f2350ddac11': [
        'BPI_Challenge_2013_closed_problems.xes'
    ],
    'a6f651a7-5ce0-4bc6-8be1-a7747effa1cc': ['RequestForPayment.xes'],
    '12683249': ['Road_Traffic_Fine_Management_Process.xes'],
    '500573e6-accc-4b0c-9576-aa5468b10cee': [
        'BPI_Challenge_2013_incidents.xes'
    ],
}


def load_all_train_tables(cache_path: Path) -> list[pd.DataFrame]:
    """Load ALL .train.csv tables from cache path."""
    train_tables, _, _ = find_existing_tables(cache_path)
    logging.info(f"Loaded {len(train_tables)} train tables")
    return train_tables


def load_rundatasets(dataset_config: dict, config_path: str, cache_path: str):
    """Load RunDatasets for a given dataset configuration."""
    run_datasets = []
    for dataset_uuid, files in dataset_config.items():
        for filename in files:
            logging.info(f"Loading: {filename}")
            run_dataset = get_natural_dataset(
                str(Path("data") / dataset_uuid / filename),
                config_path,
                cache_path,
                seed=SEED,
                num_workers=8,
            )
            if run_dataset is not None:
                run_datasets.append(run_dataset)
    return run_datasets


def create_objective(
    train_tables: list[pd.DataFrame],
    val_dataset: LabelDataset,
    feature_extractor: CompositeFeatureExtractor,
    cache_dir: Path,
):
    """Create Optuna objective function that minimizes performance ratio."""

    def objective(trial: optuna.Trial) -> float:
        hyperparameters = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float(
                'learning_rate', 0.01, 0.3, log=True
            ),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float(
                'colsample_bytree', 0.6, 1.0
            ),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float(
                'reg_lambda', 1e-8, 1.0, log=True
            ),
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'random_state': SEED,
            'n_jobs': -1,
        }

        classifier = XGBoostClassifier(
            tables=train_tables,
            feature_extractor=feature_extractor,
            cache_dir=cache_dir / f"trial_{trial.number}",
            hyperparameters=hyperparameters,
            force_retrain=True,
        )

        evaluator = RecommenderEvaluator(
            classifier=classifier,
            dataset=val_dataset,
        )
        metrics = evaluator.evaluate(batched=True, print_summary=False)
        performance_ratio = metrics['overall'].performance_ratio_alignment_only

        logging.info(
            f"Trial {trial.number}: perf_ratio={performance_ratio:.4f}"
        )
        return performance_ratio

    return objective


def run_optimization(
    n_trials: int,
    use_existing_tables: bool,
    config_path: str = "configs/default.yaml",
    cache_path: str = "cache/.runs",
    output_dir: Path = None,
):
    """Run hyperparameter optimization."""
    RNG.initialize(SEED)

    if output_dir is None:
        output_dir = Path("outputs") / f"hyperopt_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    if use_existing_tables:
        train_tables = load_all_train_tables(Path(cache_path))
    else:
        train_rundatasets = load_rundatasets(
            TRAIN_DATASETS, config_path, cache_path
        )
        fe = CompositeFeatureExtractor()
        train_tables = []
        for run_dataset in train_rundatasets:
            t_train, _, _ = create_tables(
                run_dataset,
                train_ratio=1.0,
                test_ratio=0.0,
                schema=DF_SCHEMA,
                fe=fe,
                fmt_row=format_row,
                force_recompute=False,
            )
            train_tables.append(t_train)

    total_samples = sum(len(t) for t in train_tables)
    logging.info(f"Training samples: {total_samples}")

    # Load validation dataset
    val_rundatasets = load_rundatasets(VAL_DATASETS, config_path, cache_path)
    val_dataset = LabelDataset(val_rundatasets)
    logging.info(f"Validation samples: {len(val_dataset)}")

    feature_extractor = CompositeFeatureExtractor(use_cache=True)

    study = optuna.create_study(
        direction="minimize",
        study_name="xgboost_hyperopt",
        storage=f"sqlite:///{output_dir / 'optuna_study.db'}",
        load_if_exists=True,
    )

    objective = create_objective(
        train_tables=train_tables,
        val_dataset=val_dataset,
        feature_extractor=feature_extractor,
        cache_dir=output_dir / "models",
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Save results
    logging.info(f"Best trial: {study.best_trial.number}")
    logging.info(f"Best perf_ratio: {study.best_value:.4f}")
    logging.info(f"Best params: {study.best_params}")

    best_params = {
        'best_trial': study.best_trial.number,
        'best_performance_ratio': study.best_value,
        'best_hyperparameters': study.best_params,
        'n_trials': n_trials,
    }

    with open(output_dir / "best_hyperparameters.json", 'w') as f:
        json.dump(best_params, f, indent=2)

    return study


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=5)
    # Existing tables might contain data leakage -> only use for testing
    parser.add_argument("--use-existing-tables", action="store_true")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--cache-path", type=str, default="cache/.runs")
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    run_optimization(
        n_trials=args.n_trials,
        use_existing_tables=args.use_existing_tables,
        config_path=args.config,
        cache_path=args.cache_path,
        output_dir=output_dir,
    )
