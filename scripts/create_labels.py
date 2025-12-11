import argparse
import logging
import os
import yaml
import random
from torch.utils.data import DataLoader, Dataset
from configs.schema import PipelineConfig
from dataloaders.runs import RunDataset, PerfCounter
from scripts.generate_dataset import build_pipeline
from pm4py.objects.petri_net.utils.petri_utils import construct_trace_net
import numpy as np
import pandas as pd
from tqdm import tqdm

from features.extractors import CompositeFeatureExtractor
from util.rng import RNG

DF_SCHEMA = [
    "item_id",
    "combination_id",
    "model_id",
    "trace_id",
    "aligner",
    "feature_vector",
    "time_total_mean",
    "time_total_std",
    "time_total_median",
    "time_search_mean",
    "time_lp_mean",
]


def get_stats(stats: list[PerfCounter]) -> dict[str, float]:
    durations = [s.duration for s in stats if s.duration is not None]
    ms = [s.extract_metrics() for s in stats]
    search_times = [s["search_time"] for s in ms]
    lp_times = [s["lp_time"] for s in ms]

    def compute_metrics(data: list[float]) -> dict[str, float]:
        return {
            "mean": float(np.mean(data)) if data else 0.0,
            "std": float(np.std(data)) if data else 0.0,
            "median": float(np.median(data)) if data else 0.0,
        }

    return {
        "mean_total": compute_metrics(durations)["mean"],
        "std_total": compute_metrics(durations)["std"],
        "median_total": compute_metrics(durations)["median"],
        "mean_search": compute_metrics(search_times)["mean"],
        "std_search": compute_metrics(search_times)["std"],
        "median_search": compute_metrics(search_times)["median"],
        "mean_lp": compute_metrics(lp_times)["mean"],
        "std_lp": compute_metrics(lp_times)["std"],
        "median_lp": compute_metrics(lp_times)["median"],
    }


def format_row(
    run: RunDataset.ItemType, feature_vector: np.typing.NDArray[np.float32]
) -> pd.Series:
    stats = get_stats(run.perf)

    return pd.Series(
        {
            "item_id": run.item_id,
            "combination_id": run.comb_id,
            "model_id": run.model.hash(),
            "trace_id": RunDataset._hash_trace(run.trace),
            "aligner": run.algo,
            "feature_vector": feature_vector,
            # metric for decision making
            "time_total_mean": stats.get("mean_total"),
            "time_total_std": stats.get("std_total"),
            "time_total_median": stats.get("median_total"),
            # breakdown metrics
            "time_search_mean": stats.get("mean_search"),
            "time_lp_mean": stats.get("mean_lp"),
        }
    )


def collate(batch: list[RunDataset.SerializedItemType]) -> pd.DataFrame:
    df_local = pd.DataFrame(columns=DF_SCHEMA)
    fe = CompositeFeatureExtractor()
    for run in batch:
        run = run.deserialize()
        model, trace, item, perf, algo = (
            run.model,
            run.trace,
            run.item,
            run.perf,
            run.algo,
        )
        trace_net, trace_im, trace_fm = construct_trace_net(trace)
        fv = fe.extract(
            model.pm, model.im, model.fm, trace_net, trace_im, trace_fm
        )
        row = format_row(run, fv)
        # supress future warning
        df_local = (
            pd.DataFrame([row])
            if df_local.empty
            else pd.concat([df_local, pd.DataFrame([row])], ignore_index=True)
        )

    return df_local


def split_dataframes(
    labels: pd.DataFrame,
    train_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_ids = labels["combination_id"].unique()
    random.shuffle(unique_ids)

    n = len(unique_ids)
    n_train = int(n * train_ratio)
    n_test = int(n * test_ratio)

    train_ids = set(unique_ids[:n_train])
    test_ids = set(unique_ids[n_train : n_train + n_test])
    eval_ids = set(unique_ids[n_train + n_test :])  # residual into eval

    train_df = labels[labels["combination_id"].isin(train_ids)]
    test_df = labels[labels["combination_id"].isin(test_ids)]
    eval_df = labels[labels["combination_id"].isin(eval_ids)]

    print(
        f"SPLIT SIZES → train:{len(train_df)}  test:{len(test_df)}  eval:{len(eval_df)}"
    )
    return train_df, test_df, eval_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--test", type=float, default=0.2)
    parser.add_argument("--eval", type=float, default=0.1)
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of alignment runs even if cached data exists",
    )

    args = parser.parse_args()
    # load config
    cfg_dict = yaml.safe_load(open(args.config))
    cfg = PipelineConfig.model_validate(cfg_dict)

    # override shallow keys safely
    if args.runs:
        cfg.alignment.runs = args.runs
    if args.workers:
        cfg.alignment.workers = args.workers
    if args.seed:
        cfg.seed = args.seed

    cfg.log_path = args.path
    RNG.initialize(cfg.seed)

    logging.info(cfg)

    # Use cached data unless --force-recompute is specified
    skip_init = not args.force_recompute
    if skip_init:
        logging.info(
            "Cache mode enabled: Will use cached alignment runs if available"
        )
    else:
        logging.info(
            "Force recompute mode: Will regenerate all alignment runs"
        )

    run_dataset = build_pipeline(cfg, skip_init=skip_init)

    # Check if output files already exist (for resuming interrupted jobs)
    base = run_dataset.save_path.with_suffix('')
    train_csv = f"{base}.train.csv"
    test_csv = f"{base}.test.csv"
    eval_csv = f"{base}.eval.csv"

    if not args.force_recompute and all(
        os.path.exists(f) for f in [train_csv, test_csv, eval_csv]
    ):
        logging.info(
            f"Output files already exist at {base}.*.csv - skipping feature extraction"
        )
        logging.info("Use --force-recompute to regenerate")
        exit(0)

    df = pd.DataFrame(columns=DF_SCHEMA)

    dataloader = DataLoader(
        run_dataset.serialized,
        batch_size=512,
        shuffle=False,
        num_workers=4,  # cfg.alignment.workers if cfg.alignment.workers > 0 else os.cpu_count(),
        persistent_workers=True,
        collate_fn=collate,
    )

    for df_batch in tqdm(dataloader, desc="Extracting features from runs"):
        df = pd.concat(
            [df, df_batch],
            ignore_index=True,
        )

    best_indices = df.groupby("combination_id")["time_total_mean"].idxmin()
    labels = df.loc[best_indices]

    # print(f"labels.head(): {labels.head()}")
    print("\nBest Aligner Labels (Head):")
    print(labels[["aligner", "time_total_mean", "time_total_std"]].head())
    print("Summary statistics (minimum time across aligners):")
    print(labels["time_total_mean"].describe())
    print("Distribution of aligners chosen:")
    print(labels["aligner"].value_counts())

    total_ratio = args.train + args.test + args.eval
    print(
        f"Split ratios → train:{args.train}  test:{args.test}  eval:{args.eval}  total:{total_ratio}"
    )
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0 (got {total_ratio})")

    base = run_dataset.save_path.with_suffix('')
    train_df, test_df, eval_df = split_dataframes(
        labels, args.train, args.test
    )
    train_df.to_csv(f"{base}.train.csv", index=False)
    test_df.to_csv(f"{base}.test.csv", index=False)
    eval_df.to_csv(f"{base}.eval.csv", index=False)

    df.to_csv(f"{base}.runs.csv", index=False)
    labels.to_csv(f"{base}.labels.csv", index=False)
