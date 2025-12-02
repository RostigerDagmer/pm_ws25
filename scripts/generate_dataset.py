import argparse
import yaml
from pathlib import Path
from configs.schema import PipelineConfig
from dataloaders.net import ProcessModelDataset
from dataloaders.unique_net import UniqueProcessModelDataset
from dataloaders.util import build_dataset
from dataloaders.runs import RunDataset
from util.rng import RNG
import logging

logging.basicConfig(level=logging.INFO)


def build_pipeline(cfg: PipelineConfig) -> RunDataset:
    log_dataset = build_dataset(cfg.log_path)

    pm_dataset = ProcessModelDataset(
        log_dataset=log_dataset,
        discovery_methods=cfg.discovery.resolve(),
        param_grid=cfg.discovery.params,
        sampler_specs={
            sampler.name: sampler.build() for sampler in cfg.discovery.samplers
        },
        cached=True,
        num_workers=cfg.discovery.workers or cfg.alignment.workers,
    )

    if cfg.deduplication:
        pm_dataset = UniqueProcessModelDataset(
            base_dataset=pm_dataset,
            dedup_config=cfg.deduplication.config,
            force_recompute=cfg.deduplication.force_recompute,
        )

    return RunDataset(
        base_path=Path("data/runs"),
        process_model_dataset=pm_dataset,
        aligners=cfg.alignment.resolve(),
        trace_sampler=cfg.alignment.sampler.build(ds=pm_dataset),
        n_runs=cfg.alignment.runs,
        n_workers=cfg.alignment.workers,
        write_batch_size=cfg.alignment.write_batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--seed", type=int)

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
    rng = RNG()
    rng.initialize(cfg.seed)

    logging.info(cfg)

    dataset = build_pipeline(cfg, rng)
