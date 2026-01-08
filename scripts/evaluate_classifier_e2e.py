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

from dataloaders.labels import LabelDataset
from scripts.create_labels import DF_SCHEMA, format_row
import torch
from models.spectral_model import SpectralModel
from argparse import ArgumentParser
from util.rng import RNG
import logging
from pathlib import Path

from features import CompositeFeatureExtractor
from models import (
    XGBoostClassifier,
    SingleBestSolver,
    RandomClassifier,
    RecommenderEvaluator,
)
from dataloaders.util import (
    get_natural_dataset,
    get_synthetic_dataset,
    create_tables,
    find_existing_tables,
)
from datetime import datetime

logging.basicConfig(level=logging.INFO)

SEED = 1

OUTPUT_DIR = (
    Path("outputs") / f"evaluate_classifier_{datetime.now():%Y%m%d_%H%M%S}"
)

TRAIN_DATASETS = {
    'd9769f3d-0ab0-4fb8-803b-0d1120ffcf54': ['Hospital_log.xes'],
    '63a8435a-077d-4ece-97cd-2c76d394d99c': ['BPIC15_2.xes'],
    'ed445cdd-27d5-4d77-a1f7-59fe7360cfbe': ['BPIC15_3.xes'],
    '679b11cf-47cd-459e-a6de-9ca614e25985': ['BPIC15_4.xes'],
    '3301445f-95e8-4ff0-98a4-901f1f204972': ['BPI%20Challenge%202018.xes'],
    '3926db30-f712-4394-aebc-75976070e91f': ['BPI_Challenge_2012.xes'],
    '6af6d5f0-f44c-49be-aac8-8eaa5fe4f6fd': [
        'Hospital%20Billing%20-%20Event%20Log.xes'
    ],
    'd06aff4b-79f0-45e6-8ec8-e19730c248f1': ['BPI_Challenge_2019.xes'],
    '3537c19d-6c64-4b1d-815d-915ab0e479da': [
        'BPI_Challenge_2013_open_problems.xes'
    ],
    # 'a0addfda-2044-4541-a450-fdcc9fe16d17': ['BPIC15_1.xes'],
    # 'a6f651a7-5ce0-4bc6-8be1-a7747effa1cc': ['RequestForPayment.xes'],
    # '500573e6-accc-4b0c-9576-aa5468b10cee': [
    #     'BPI_Challenge_2013_incidents.xes'
    # ],
    '91fd1fa8-4df4-4b1a-9a3f-0116c412378f': ['InternationalDeclarations.xes'],
    'fb84cf2d-166f-4de2-87be-62ee317077e5': ['PrepaidTravelCost.xes'],
    # '12683249': ['Road_Traffic_Fine_Management_Process.xes'],
    # '33632f3c-5c48-40cf-8d8f-2db57f5a6ce7': [
    #     'Sepsis%20Cases%20-%20Event%20Log.xes'
    # ],
}

TEST_DATASETS = {
    'b32c6fe5-f212-4286-9774-58dd53511cf8': ['BPIC15_5.xes'],
    '5f3067df-f10b-45da-b98b-86ae4c7a310b': ['BPI%20Challenge%202017.xes'],
    'db35afac-2133-40f3-a565-2dc77a9329a3': ['PermitLog.xes'],
    '6a0a26d2-82d0-4018-b1cd-89afb0e8627f': ['DomesticDeclarations.xes'],
    'c2c3b154-ab26-4b31-a0e8-8f2350ddac11': [
        'BPI_Challenge_2013_closed_problems.xes'
    ],
}


if __name__ == "__main__":

    RNG.initialize(SEED)

    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--use-tables",
        action="store_true",
        help="Use existing feature tables to shortcut feature extraction",
    )
    args = arg_parser.parse_args()

    config_path = "configs/default.yaml"
    cache_path = "cache/.runs"

    # Create train RunDatasets
    logging.info(
        f"\nCreating {sum(len(f) for f in TRAIN_DATASETS.values())} train RunDatasets..."
    )

    train_run_datasets = None
    train_tables, test_tables, eval_tables = None, None, None

    if args.use_tables:
        train_tables, test_tables, eval_tables = find_existing_tables(
            Path(cache_path)
        )
    else:
        train_run_datasets = []
        for dataset_uuid, files in TRAIN_DATASETS.items():
            for filename in files:
                print(f"Loading: {filename}")
                run_dataset = get_natural_dataset(
                    str(Path("data") / dataset_uuid / filename),
                    config_path,
                    cache_path,
                    seed=SEED,
                    num_workers=16,
                )
                if run_dataset is not None:
                    train_run_datasets.append(run_dataset)

        train_run_datasets.append(
            get_synthetic_dataset(
                Path(cache_path),
                seed=SEED,
                num_models=200,
                num_traces=32,
                min_depth=2,
                max_depth=3,
            )
        )
        train_tables, test_tables, eval_tables = [], [], []
        for run_dataset in train_run_datasets:
            t_train, t_test, t_eval = create_tables(
                run_dataset,
                train_ratio=0.7,
                test_ratio=0.2,
                schema=DF_SCHEMA,
                fe=CompositeFeatureExtractor(),
                fmt_row=format_row,
                force_recompute=True,
            )
            del run_dataset
            train_tables.append(t_train)
            test_tables.append(t_test)
            eval_tables.append(t_eval)

    # Relying on existing tables can skip single threaded feature extraction

    logging.info("\nCreating feature extractor...")
    feature_extractor = CompositeFeatureExtractor(use_cache=True)

    logging.info("\nTraining XGBoostClassifier...")
    classifier = XGBoostClassifier(
        tables=train_tables,
        feature_extractor=feature_extractor,
        cache_dir=Path("cache") / "models",
        force_retrain=True,
    )

    has_transformer_model = Path("transformer_model.pth").exists()

    if has_transformer_model:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        transformer_model = SpectralModel(
            d_model=128,
            d_trace=128,
            hidden_dim=256,
            mlp_hidden_dim=512,
            n_classes=6,
            num_heads=4,
            n_layers=2,
            n_self_attn=2,
            dropout=0.1,
            pretraining=False,
        ).to(device)

        transformer_model.load_state_dict(torch.load("transformer_model.pth"))
        transformer_model.eval()

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

    logging.info("\nLoading test RunDatasets (using cached alignment runs)...")
    test_run_datasets = []
    for dataset_uuid, files in TEST_DATASETS.items():
        for filename in files:
            logging.info(f"Loading: {filename}")
            run_dataset = get_natural_dataset(
                str(Path("data") / dataset_uuid / filename),
                config_path,
                cache_path,
                seed=SEED,
                num_workers=16,
            )
            test_run_datasets.append(run_dataset)

    test_run_datasets.append(
        get_synthetic_dataset(
            Path(cache_path),
            seed=SEED + 1,
            num_models=100,
            num_traces=32,
            min_depth=2,
            max_depth=3,
        )
    )

    test_dataset = LabelDataset(test_run_datasets)

    # Evaluate
    logging.info(f"\nEvaluating on test datasets [{len(test_dataset)}]")
    evaluator = RecommenderEvaluator(
        classifier=classifier, dataset=test_dataset
    )

    metrics = evaluator.evaluate_batched()

    # Compare with baselines
    logging.info("\nComparing with baselines...")
    comparison_df = evaluator.compare_with_baselines(
        [single_best, random_clf]
        + ([transformer_model] if has_transformer_model else [])
    )
    logging.info("\n" + comparison_df.to_string())

    RecommenderEvaluator.save_results(
        metrics=overall_metrics,
        comparison_df=comparison_df,
        output_dir=OUTPUT_DIR,
        train_count=len(train_tables),
        test_count=len(test_tables),
    )

    # Generate HTML report
    logging.info("\nGenerating HTML report...")

    report_gen = EvaluationReportGenerator(metrics_dict=all_metrics)
    report_gen.to_html(OUTPUT_DIR / "evaluation_report.html")
    logging.info(f"HTML report available at: {OUTPUT_DIR / 'evaluation_report.html'}")
