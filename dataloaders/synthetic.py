import os
from typing import Any, Iterator
from torch.utils.data import Dataset
from dataloaders.base import BaseEventLogDataset
from dataloaders.serializable import (
    Serializable,
    Deserializable,
    WithSerializedView,
)
from experiments.simulation import models, simulate
from experiments.simulation.structured_net import StructuredNet, TensorNet
from pm4py.objects.log.obj import Trace, EventLog
from util.distributions import DistParam
from dataclasses import asdict, dataclass
import hashlib
import json
import torch
from util.rng import RNG


class SyntheticEventLogDataset(BaseEventLogDataset):
    def __init__(
        self,
        model: models.StructuredNet,
        n_traces: int = 256,
        max_trace_length: int = 50,
        batch_size: int = 256,
        **kwargs,
    ):
        """
        Args:
            n_traces (int): Number of traces to simulate.
            model (StructuredNet): Process model to simulate from.
            max_trace_length (int): Maximum length of each trace.
            **kwargs: Passed to BaseEventLogDataset.
        """
        super().__init__(source_path=None, **kwargs)
        self.model = model
        self.n_traces = n_traces
        self.max_trace_length = max_trace_length
        self.N = model.to_tensor()
        self.batch_size = batch_size
        log = []
        for _ in range(
            (self.n_traces + self.batch_size - 1) // self.batch_size
        ):
            batch = simulate.simulate_batch(
                (self.N.pre, self.N.post),
                self.N.M0,
                self.N.Mf,
                self.N.labels,
                steps=self.max_trace_length,
                batch_size=self.batch_size,
                compact=True,
            )
            log.extend(simulate.apply_labels(batch, self.N.labels))

        self.log = EventLog(log)

    def _load_log(self, source_path, **kwargs):
        return None

    def __getitem__(self, idx: int) -> Trace:
        return self.log[idx]


@dataclass
class ItemType(Serializable["SerializedItemType"]):
    stnet: StructuredNet  # wraps .net, .im, .fm
    params: dict[str, Any | DistParam]  # whatever you used to generate it

    # unify with discovered models
    @property
    def pm(self):
        return self.stnet.net

    @property
    def im(self):
        return self.stnet.im

    @property
    def fm(self):
        return self.stnet.fm

    def hash(self) -> str:
        return hashlib.sha1(
            json.dumps(self.params, sort_keys=True, default=asdict).encode()
        ).hexdigest()

    def serialize(self) -> "SerializedItemType":
        return SerializedItemType(self.params, self.stnet.to_tensor())


@dataclass
class SerializedItemType(Deserializable[ItemType]):
    params: dict
    net: TensorNet

    def hash(self) -> str:
        return hashlib.sha1(
            json.dumps(self.params, sort_keys=True, default=asdict).encode()
        ).hexdigest()

    def deserialize(self) -> ItemType:
        stnet = StructuredNet.from_tensor(self.net)
        return ItemType(
            stnet=stnet,
            params=self.params,
        )


class SyntheticProcessModelDataset(
    Dataset[ItemType], WithSerializedView[ItemType, SerializedItemType]
):

    def __init__(
        self,
        param_grid: list[tuple[dict, int]],
        max_models=None,
        cached=False,
        cache_dir=None,
        num_workers=0,
    ):
        self.param_grid = param_grid
        self.configurations = [
            {**p, "index": i}
            for params, reps in param_grid
            for i, p in enumerate([params] * reps)
        ]
        self.max_models = max_models
        self.cached = cached
        self.cache_dir = cache_dir
        self.log = None
        self.num_workers = (
            num_workers if num_workers > 0 else os.cpu_count() or 1
        )

    def __len__(self):
        return len(self.configurations)

    def __iter__(self) -> Iterator[ItemType]:
        for i in range(len(self)):
            yield self[i]

    def _get_serialized(self, idx: int) -> SerializedItemType:
        return self[idx].serialize()

    def __getitem__(self, idx: int) -> ItemType:
        cfg = {
            k: v for k, v in self.configurations[idx].items() if k != 'index'
        }
        # combine seed from rng and index
        config_hash = hashlib.sha1(
            json.dumps(
                self.configurations[idx], sort_keys=True, default=asdict
            ).encode()
        ).hexdigest()
        sample_seed = (RNG.get_seed() + int(config_hash, 16)) % 2**32
        model = models.sample_net(
            generator=torch.Generator().manual_seed(sample_seed), **cfg
        )
        return ItemType(model, self.configurations[idx])

    def hash(self) -> str:
        # global hash of all params
        return hashlib.sha1(
            json.dumps(
                self.configurations, sort_keys=True, default=asdict
            ).encode()
        ).hexdigest()


if __name__ == "__main__":

    from util.distributions import (
        CategoricalSpec,
        PoissonSpec,
        BernoulliDepthLinearSpec,
    )

    RNG.initialize(42)

    dist_params = {
        "op": CategoricalSpec([0.3, 0.3, 0.3, 0.1]),
        "seq_len": PoissonSpec(4),
        "p_stop": BernoulliDepthLinearSpec(base=0.2, slope=0.1),
        "width": PoissonSpec(4),
    }

    stnet = models.sample_net(dist_params, generator=RNG.torch_generator())
    dataset = SyntheticEventLogDataset(
        model=stnet,
        n_traces=10,
        max_trace_length=20,
    )
    for trace in dataset:
        print(trace)

    model_dataset = SyntheticProcessModelDataset(
        param_grid=[
            (
                {
                    "dist_params": {
                        "op": CategoricalSpec([0.3, 0.3, 0.3, 0.1]),
                        "seq_len": PoissonSpec(4),
                        "p_stop": BernoulliDepthLinearSpec(
                            base=0.2, slope=0.1
                        ),
                        "width": PoissonSpec(4),
                    },
                    "depth": 4,
                },
                3,
            ),
            (
                {
                    "dist_params": {
                        "op": CategoricalSpec([0.1, 0.5, 0.3, 0.1]),
                        "seq_len": PoissonSpec(4),
                        "p_stop": BernoulliDepthLinearSpec(
                            base=0.2, slope=0.1
                        ),
                        "width": PoissonSpec(4),
                    },
                    "depth": 4,
                },
                3,
            ),
            (
                {
                    "dist_params": {
                        "op": CategoricalSpec([0.1, 0.3, 0.5, 0.1]),
                        "seq_len": PoissonSpec(4),
                        "p_stop": BernoulliDepthLinearSpec(
                            base=0.2, slope=0.1
                        ),
                        "width": PoissonSpec(4),
                    },
                    "depth": 4,
                },
                3,
            ),
            (
                {
                    "dist_params": {
                        "op": CategoricalSpec([0.1, 0.3, 0.3, 0.3]),
                        "seq_len": PoissonSpec(4),
                        "p_stop": BernoulliDepthLinearSpec(
                            base=0.2, slope=0.1
                        ),
                        "width": PoissonSpec(4),
                    },
                    "depth": 4,
                },
                3,
            ),
        ],
    )

    for item in model_dataset:
        print(item.hash())
        print(item.stnet)
