from util.rng import RNG
from dataloaders.synthetic import SyntheticProcessModelDataset
from dataloaders.runs import RunDataset, AlignerSpec, SyntheticTraceSampler
from pathlib import Path


def get_synthetic_dataset(
    cache_path: Path, seed: int = 1, count: int = 20, device: str = "cpu"
) -> RunDataset:
    RNG.initialize(seed)

    N_RUNS = 5  # set number of runs here

    from util.distributions import (
        CategoricalSpec,
        PoissonSpec,
        BernoulliDepthLinearSpec,
    )

    MAX_DEPTH = 2
    MIN_DEPTH = 1

    synthetic_dataset = SyntheticProcessModelDataset(
        param_grid=[
            (
                {  # and dominant
                    "dist_params": {
                        "op": CategoricalSpec([0.1, 0.6, 0.2, 0.1]),
                        "seq_len": PoissonSpec(4),
                        "p_stop": BernoulliDepthLinearSpec(
                            base=0.2, slope=0.1
                        ),
                        "width": PoissonSpec(3),
                    },
                    "min_depth": MIN_DEPTH,
                    "max_depth": MAX_DEPTH,
                },
                count,  # Number of models per config
            ),
            (
                {  # xor dominant
                    "dist_params": {
                        "op": CategoricalSpec([0.6, 0.1, 0.2, 0.1]),
                        "seq_len": PoissonSpec(4),
                        "p_stop": BernoulliDepthLinearSpec(
                            base=0.2, slope=0.1
                        ),
                        "width": PoissonSpec(3),
                    },
                    "min_depth": MIN_DEPTH,
                    "max_depth": MAX_DEPTH,
                },
                count,  # Number of models per config
            ),
            (
                {  # shallow and wide xor dominant
                    "dist_params": {
                        "op": CategoricalSpec([0.6, 0.1, 0.2, 0.1]),
                        "seq_len": PoissonSpec(1),
                        "p_stop": BernoulliDepthLinearSpec(
                            base=0.5, slope=0.3
                        ),
                        "width": PoissonSpec(10),
                    },
                    "min_depth": MIN_DEPTH,
                    "max_depth": MAX_DEPTH,
                },
                count,  # Number of models per config
            ),
            (
                {  # shallow and wide xor / loop dominant
                    "dist_params": {
                        "op": CategoricalSpec([0.4, 0.1, 0.4, 0.1]),
                        "seq_len": PoissonSpec(1),
                        "p_stop": BernoulliDepthLinearSpec(
                            base=0.5, slope=0.3
                        ),
                        "width": PoissonSpec(10),
                    },
                    "min_depth": MIN_DEPTH,
                    "max_depth": MAX_DEPTH,
                },
                count,  # Number of models per config
            ),
            (
                {  # loop dominant
                    "dist_params": {
                        "op": CategoricalSpec([0.15, 0.15, 0.6, 0.1]),
                        "seq_len": PoissonSpec(4),
                        "p_stop": BernoulliDepthLinearSpec(
                            base=0.2, slope=0.1
                        ),
                        "width": PoissonSpec(3),
                    },
                    "min_depth": MIN_DEPTH,
                    "max_depth": MAX_DEPTH,
                },
                count,  # Number of models per config
            ),
        ],
    )
    trace_sampler = SyntheticTraceSampler(
        ds=synthetic_dataset,
        seed=RNG.get_seed(),
        batch_size=128,
        slice=range(0, 8),
        steps=40,
        device=device,
    )

    return RunDataset(
        cache_path,
        synthetic_dataset,
        AlignerSpec.A_STAR.value,
        trace_sampler,
        n_runs=N_RUNS,
        n_workers=20,
    )
