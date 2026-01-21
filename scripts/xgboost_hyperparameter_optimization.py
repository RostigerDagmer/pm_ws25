"""Hyperparameter optimization for XGBoostClassifier using Optuna."""

import json
import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import optuna
import pandas as pd

from dataloaders.labels import TableLabelDataset
from dataloaders.util import find_existing_tables
from features import CompositeFeatureExtractor
from models import XGBoostClassifier, RecommenderEvaluator
from util.rng import RNG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

SEED = 42

def load_tables(
    cache_path: Path,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    """Load train, eval, and runs tables from cache path."""
    train_tables, _, eval_tables, runs_tables = find_existing_tables(
        cache_path, include_runs=True
    )
    logging.info(
        f"Loaded {len(train_tables)} train, {len(eval_tables)} eval, "
        f"{len(runs_tables)} runs tables"
    )
    return train_tables, eval_tables, runs_tables


def create_objective(
    train_tables: list[pd.DataFrame],
    val_dataset: TableLabelDataset,
    feature_extractor: CompositeFeatureExtractor,
    cache_dir: Path,
):
    """Create Optuna objective function that minimizes performance ratio."""

    def objective(trial: optuna.Trial) -> float:
        # Search ranges refined based on best trial 21 (perf_ratio=1.072)
        hyperparameters = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
            'max_depth': trial.suggest_int('max_depth', 7, 14),
            'learning_rate': trial.suggest_float(
                'learning_rate', 0.1, 0.5, log=True
            ),
            'min_child_weight': trial.suggest_int('min_child_weight', 2, 8),
            'subsample': trial.suggest_float('subsample', 0.65, 0.9),
            'colsample_bytree': trial.suggest_float(
                'colsample_bytree', 0.8, 1.0
            ),
            'gamma': trial.suggest_float('gamma', 0.05, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 0.5, log=True),
            'reg_lambda': trial.suggest_float(
                'reg_lambda', 1e-4, 0.1, log=True
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
    cache_path: str = "cache/.runs",
    output_dir: Path = None,
):
    """Run hyperparameter optimization using cached tables."""
    RNG.initialize(SEED)

    if output_dir is None:
        output_dir = Path("outputs") / f"hyperopt_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading cached tables...")
    train_tables, eval_tables, runs_tables = load_tables(Path(cache_path))
    val_dataset = TableLabelDataset(
        test_tables=eval_tables,
        runs_tables=runs_tables,
    )

    total_samples = sum(len(t) for t in train_tables)
    logging.info(f"Training samples: {total_samples}")
    logging.info(f"Validation samples: {len(val_dataset)}")

    feature_extractor = CompositeFeatureExtractor()

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--cache-path", type=str, default=f"cache/.runs")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    run_optimization(
        n_trials=args.n_trials,
        cache_path=args.cache_path,
        output_dir=output_dir,
    )
