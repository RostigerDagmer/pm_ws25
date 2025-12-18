import argparse
import yaml
from configs.schema import PipelineConfig
from util.rng import RNG
import logging
from dataloaders.util import build_pipeline

logging.basicConfig(level=logging.INFO)

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

    dataset = build_pipeline(cfg)
