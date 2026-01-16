import argparse
import logging
import yaml
from configs.schema import PipelineConfig
from dataloaders.runs import RunDataset, PerfCounter
import numpy as np
import pandas as pd

from features import CompositeFeatureExtractor
from util.rng import RNG
from dataloaders.util import create_tables, build_pipeline

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
    parser.add_argument("--force-recompute", action="store_true")

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

    run_dataset = build_pipeline(cfg)

    _, _, _ = create_tables(
        run_dataset,
        args.train,
        args.test,
        DF_SCHEMA,
        CompositeFeatureExtractor(),
        format_row,
        force_recompute=args.force_recompute,
    )
