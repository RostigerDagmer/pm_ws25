#!/usr/bin/env python3
"""
Generate training labels from synthetic process models.

OUTPUT: Same CSV format as create_labels.py - can be combined with real data!
  - <hash>.train.csv
  - <hash>.test.csv
  - <hash>.eval.csv
  - <hash>.runs.csv
  - <hash>.labels.csv
"""
import argparse
import logging
import os
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from configs.schema import PipelineConfig
from dataloaders.runs import RunDataset, SyntheticTraceSampler
from dataloaders.synthetic import SyntheticProcessModelDataset
from scripts.create_labels import collate, split_dataframes, DF_SCHEMA
from util.rng import RNG
from util.distributions import (
    CategoricalSpec,
    PoissonSpec,
    BernoulliDepthLinearSpec,
)
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def build_synthetic_pipeline(
    cfg: PipelineConfig,
    param_grid: list[tuple[dict, int]],
    n_traces: int = 100,
    max_trace_length: int = 50,
    skip_init: bool = False,
) -> RunDataset:
    """Build RunDataset from synthetic models"""
    logging.info("Creating synthetic process model dataset...")

    synthetic_dataset = SyntheticProcessModelDataset(
        param_grid=param_grid,
        cached=True,
        num_workers=cfg.alignment.workers or os.cpu_count() or 1,
    )

    logging.info(f"✓ Generated {len(synthetic_dataset)} synthetic models")

    trace_sampler = SyntheticTraceSampler(
        seed=RNG.get_seed(),
        ds=synthetic_dataset,
        slice=range(n_traces),  # Number of traces per model
        steps=max_trace_length,  # Maximum trace length
        batch_size=128,  # Batch size for simulation
    )

    base_path = cfg.alignment.cache_path or Path("data/runs_synthetic")

    return RunDataset(
        base_path=base_path,
        process_model_dataset=synthetic_dataset,
        aligners=cfg.alignment.resolve(),
        trace_sampler=trace_sampler,
        n_runs=cfg.alignment.runs,
        n_workers=cfg.alignment.workers,
        write_batch_size=cfg.alignment.write_batch_size,
        skip_init=skip_init,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training labels from synthetic models (compatible with real data)"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--test", type=float, default=0.2)
    parser.add_argument("--eval", type=float, default=0.1)
    parser.add_argument("--n-models", type=int, default=100)
    parser.add_argument("--n-traces", type=int, default=100)
    parser.add_argument("--max-trace-length", type=int, default=50)
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--force-recompute", action="store_true")

    args = parser.parse_args()

    print("=" * 60)
    print("SYNTHETIC DATA GENERATION")
    print("=" * 60)

    # Load config
    cfg_dict = yaml.safe_load(open(args.config))
    cfg = PipelineConfig.model_validate(cfg_dict)

    if args.runs:
        cfg.alignment.runs = args.runs
    if args.workers:
        cfg.alignment.workers = args.workers
    if args.seed:
        cfg.seed = args.seed

    RNG.initialize(cfg.seed)

    print(f"\nConfig: {args.n_models} models, {args.n_traces} traces/model")
    print(f"Depth: {args.min_depth}-{args.max_depth}, Workers: {cfg.alignment.workers}\n")

    # 3 model configurations
    models_per_config = args.n_models // 3
    param_grid = [
        ({
            "dist_params": {
                "op": CategoricalSpec([0.3, 0.3, 0.3, 0.1]),
                "seq_len": PoissonSpec(4),
                "p_stop": BernoulliDepthLinearSpec(base=0.2, slope=0.1),
                "width": PoissonSpec(3),
            },
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
        }, models_per_config),
        ({
            "dist_params": {
                "op": CategoricalSpec([0.2, 0.2, 0.2, 0.4]),
                "seq_len": PoissonSpec(4),
                "p_stop": BernoulliDepthLinearSpec(base=0.2, slope=0.1),
                "width": PoissonSpec(3),
            },
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
        }, models_per_config),
        ({
            "dist_params": {
                "op": CategoricalSpec([0.2, 0.4, 0.2, 0.2]),
                "seq_len": PoissonSpec(4),
                "p_stop": BernoulliDepthLinearSpec(base=0.2, slope=0.1),
                "width": PoissonSpec(3),
            },
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
        }, args.n_models - 2 * models_per_config),
    ]

    skip_init = not args.force_recompute
    run_dataset = build_synthetic_pipeline(
        cfg, param_grid, args.n_traces, args.max_trace_length, skip_init
    )

    # Check cache
    base = run_dataset.save_path.with_suffix('')
    if not args.force_recompute and all(
        Path(f"{base}.{split}.csv").exists() for split in ["train", "test", "eval"]
    ):
        logging.info(f"✓ Output exists: {base}.*.csv (use --force-recompute to regenerate)")
        exit(0)

    # Extract features
    logging.info("Extracting features...")
    df = pd.DataFrame(columns=DF_SCHEMA)

    dataloader = DataLoader(
        run_dataset.serialized,
        batch_size=512,
        shuffle=False,
        num_workers=cfg.alignment.workers,  # Use all available workers for faster feature extraction
        persistent_workers=True,
        collate_fn=collate,
    )

    for df_batch in tqdm(dataloader, desc="Features"):
        df = pd.concat([df, df_batch], ignore_index=True)

    # Find best aligners
    best_indices = df.groupby("combination_id")["time_total_mean"].idxmin()
    labels = df.loc[best_indices]

    print(f"\n✓ {len(df)} runs, {len(labels)} best aligners")
    print("\nAligner distribution:")
    print(labels["aligner"].value_counts())

    # Split and save
    if not np.isclose(args.train + args.test + args.eval, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    train_df, test_df, eval_df = split_dataframes(labels, args.train, args.test)

    # Ensure directory exists
    Path(base).parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(f"{base}.train.csv", index=False)
    test_df.to_csv(f"{base}.test.csv", index=False)
    eval_df.to_csv(f"{base}.eval.csv", index=False)
    df.to_csv(f"{base}.runs.csv", index=False)
    labels.to_csv(f"{base}.labels.csv", index=False)

    print(f"\n✓ Saved to {base}.*.csv")
    print(f"  - train: {len(train_df)} samples")
    print(f"  - test: {len(test_df)} samples")
    print(f"  - eval: {len(eval_df)} samples")
    print(f"\n💡 Add this path to evaluate_classifier_e2e.py to combine with real data!")
