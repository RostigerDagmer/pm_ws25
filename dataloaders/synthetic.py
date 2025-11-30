from typing import Any, Generator, Optional, Union
from dataloaders.base import BaseEventLogDataset
from dataloaders.net import ProcessModelDataset
from experiments.simulation import models, driver, simulate
from experiments.simulation.structured_net import StructuredNet
from pm4py.objects.log.obj import EventLog, Trace
from experiments.simulation.models import DistParam
from dataclasses import dataclass
import hashlib
import json
import torch


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
                (self.N['pre'], self.N['post']),
                self.N['M0'],
                self.N['Mf'],
                self.N['labels'],
                steps=self.max_trace_length,
                batch_size=self.batch_size,
                compact=True,
            )
            simulate.apply_labels(batch, self.N['labels'])
            log.extend(batch)

        self.log = log

    def _load_log(self, source_path, **kwargs):
        return None

    def __getitem__(self, idx: int) -> Trace:
        return self.log[idx]


class EmptyEventLogDataset(BaseEventLogDataset):
    def __init__(
        self,
        **kwargs,
    ):
        """
        Args:
            **kwargs: Passed to BaseEventLogDataset.
        """
        super().__init__(source_path=None, **kwargs)

    def _load_log(self, source_path, **kwargs):
        return simulate.EventLog()

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> simulate.Trace:
        return None


class SyntheticProcessModelDataset(ProcessModelDataset):
    @dataclass
    class ItemType:
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
            serializable_params = {
                **self.params,
                "dist_params": {
                    k: (v.__dict__ if hasattr(v, "__dict__") else v)
                    for k, v in self.params.get("dist_params", {}).items()
                },
            }
            return hashlib.sha1(
                json.dumps(serializable_params, sort_keys=True).encode()
            ).hexdigest()

    class SerializedView:
        @dataclass
        class ItemType:
            params: dict  # enough to re-sample or reconstruct

            def hash(self) -> str:
                serializable_params = {
                    **self.params,
                    "dist_params": {
                        k: (v.__dict__ if hasattr(v, "__dict__") else v)
                        for k, v in self.params.get("dist_params", {}).items()
                    },
                }
                return hashlib.sha1(
                    json.dumps(serializable_params, sort_keys=True).encode()
                ).hexdigest()

            def deserialize(self) -> "SyntheticProcessModelDataset.ItemType":
                # inspect function arguments of sample_net
                args = models.sample_net.__code__.co_varnames
                stnet = models.sample_net(
                    **{k: v for k, v in self.params.items() if k in args}
                )  # deterministic
                return SyntheticProcessModelDataset.ItemType(
                    stnet=stnet,
                    params=self.params,
                )

        def __init__(self, ds: "ProcessModelDataset"):
            self.ds = ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx: int) -> ItemType:
            return self.ds.__get_serialized__(idx)

        def __iter__(self) -> Generator[ItemType, None, None]:
            for i in range(len(self.ds)):
                yield self.ds.__get_serialized__(i)

    def __init__(self, param_grid: list[tuple[dict, int]]):
        super().__init__(
            log_dataset=EmptyEventLogDataset(),
            discovery_methods={},
            param_grid={},
            sampler_specs={},
            max_models=None,
            cached=False,
            cache_dir=None,
            num_workers=None,
        )
        self.configurations = [
            {**p, "index": i}
            for params, reps in param_grid
            for i, p in enumerate([params] * reps)
        ]
        self.serialized: list[
            SyntheticProcessModelDataset.SerializedView.ItemType
        ] = [
            SyntheticProcessModelDataset.SerializedView.ItemType(p)
            for p in self.configurations
        ]

    def __get_serialized__(
        self, idx: int
    ) -> ProcessModelDataset.SerializedView.ItemType:
        return self.serialized[idx]

    def hash(self) -> str:
        # global hash of all params
        return hashlib.sha1(
            json.dumps(self.configurations, sort_keys=True).encode()
        ).hexdigest()


if __name__ == "__main__":

    from experiments.simulation.models import (
        CategoricalSpec,
        PoissonSpec,
        BernoulliDepthLinearSpec,
    )

    dist_params = {
        "op": CategoricalSpec([0.3, 0.3, 0.3, 0.1]),
        "seq_len": PoissonSpec(4),
        "p_stop": BernoulliDepthLinearSpec(base=0.2, slope=0.1),
    }
    stnet = models.sample_net(dist_params)
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
                    },
                    "depth": 4,
                },
                3,
            ),
        ]
    )

    for item in model_dataset:
        print(item.hash())
        print(item.stnet)
