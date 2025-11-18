from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
import logging
import marshal
import multiprocessing
from pathlib import Path
import pickle
import time
from torch.utils.data import Dataset
from typing import Any, Iterator, Optional, Union
import os
import hashlib
import json
from abc import ABC, abstractmethod
from itertools import product

from tqdm import tqdm
from experiments.simulation.noise import inject_noise_trace
from features.extractors import CompositeFeatureExtractor
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.log.obj import EventLog, Trace
import cProfile
import pstats
from dataclasses import dataclass

from dataloaders.net import ProcessModelDataset
from pm4py.algo.conformance.alignments.petri_net.algorithm import (
    Variants,
    apply,
)
from pm4py.objects.petri_net.utils.petri_utils import construct_trace_net
from pm4py.util import typing

SEED = 42


class Aligner(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(
        self, pm: PetriNet, im: Marking, fm: Marking, trace: Trace
    ) -> Union[typing.AlignmentResult, typing.ListAlignments]:
        pass

    @abstractmethod
    def hash(self) -> str:
        return self.name


class PerfCounter:
    def __init__(self):
        self._profiler = cProfile.Profile()
        self.duration: float | None = None
        self.stats: pstats.Stats | None = None

    def __enter__(self):
        self._t_start = time.perf_counter()
        self._profiler.enable()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._profiler.disable()
        self._t_end = time.perf_counter()
        self.duration = self._t_end - self._t_start
        self.stats = pstats.Stats(self._profiler).strip_dirs()
        return False

    def dict(self) -> dict[str, Any]:
        return {
            "duration": self.duration,
            "stats": marshal.dumps(self.stats.stats),
        }

    @staticmethod
    def from_dict(d) -> "PerfCounter":
        if isinstance(d, PerfCounter):
            logging.warning(
                "PerfCounter.from_dict called on PerfCounter instance"
            )
            return d
        duration = d["duration"]
        stats = pstats.Stats()
        stats.stats = marshal.loads(d["stats"])
        pc = PerfCounter()
        pc.duration = duration
        pc.stats = stats
        return pc


class TraceSampler(ABC):

    def __init__(
        self,
        ds: ProcessModelDataset,
        seed: Optional[int] = None,
        slice: Optional[range] = None,
    ):
        self.source_path = ds.cache_dir
        self.log = ds.log
        self.range = slice if slice is not None else range(len(self.log))
        self.seed = seed

    def __len__(self):
        return len(self.range)

    @abstractmethod
    def __getitem__(self, index: int) -> Trace:
        # here you would modify traces from the original log or similar
        raise NotImplementedError

    def __iter__(self) -> Iterator[Trace]:
        for i in self.range:
            yield self.__getitem__(i)

    def hash(self) -> str:
        base = {
            "name": str(self.__class__),
            "ds": str(self.source_path),
            "range": (self.range.start, self.range.stop, self.range.step),
        }
        return hashlib.sha1(
            json.dumps(base, sort_keys=True).encode()
        ).hexdigest()


class RunDataset(Dataset):

    @dataclass
    class ItemType:
        item_id: str
        model: ProcessModelDataset.ItemType
        trace: Trace
        item: Union[typing.AlignmentResult, typing.ListAlignments]
        perf: PerfCounter
        algo: str
        comb_id: str  # an identifier for the combination of model, trace (to group by aligner)

    def __init__(
        self,
        base_path: Path,
        process_model_dataset: ProcessModelDataset,
        aligners: Sequence[Aligner],
        trace_sampler: TraceSampler.__class__,
        multiprocessing: bool = True,
        slice: Optional[range] = None,
    ):
        self.base_path = base_path
        self.pm_dataset = process_model_dataset
        self.trace_sampler = trace_sampler(
            self.pm_dataset, seed=SEED, slice=slice
        )
        self.aligners = aligners
        self.items: dict[str, "RunDataset.ItemType"] = {}
        self.index: list[str] = []
        if multiprocessing:
            self._init_cache_mp()
        else:
            self._init_cache()

    def _tryload(self):
        path = self.save_path()
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.items = pickle.load(f)
                print(f"tryload::self.items: {len(self.items)}")
                self.index = list(self.items.keys())
                print(f"tryload::self.index: {len(self.index)}")

    def save_path(self):
        return self.base_path / Path(f"{self._hash_config()}.pkl")

    @staticmethod
    def _process_item(
        hash: str,
        model: ProcessModelDataset.SerializedView.ItemType,
        trace: Trace,
        aligner: Aligner | str,
    ) -> "RunDataset.ItemType":

        deser_model = model.deserialize()
        if isinstance(aligner, str):
            # reconstruct aligner from name (in the mp case)
            aligner = next(
                a for a in AlignerSpec.ALL.value if a.name == aligner
            )
        with PerfCounter() as pc:
            item = aligner(
                deser_model.pm, deser_model.im, deser_model.fm, trace
            )
        stats = pc.dict()
        return RunDataset.ItemType(
            hash,
            deser_model,
            trace,
            item,
            stats,
            aligner.name,
            RunDataset._hash_comb(trace, deser_model),
        )

    def _init_cache(self):
        self._tryload()
        for trace in tqdm(self.trace_sampler, total=len(self.trace_sampler)):
            for model in tqdm(
                self.pm_dataset.serialized, total=len(self.pm_dataset)
            ):
                for aligner in self.aligners:
                    h = self._hash_item(
                        model, trace, aligner
                    )  # should be deterministic in the result since aligner should be deterministic in the result
                    if h in self.items:
                        continue
                    self.items[h] = RunDataset._process_item(
                        h, model, trace, aligner
                    )

        self.index = list(self.items.keys())
        os.makedirs(self.base_path, exist_ok=True)
        with open(self.save_path(), "wb") as f:
            pickle.dump(self.items, f)

    def _init_cache_mp(self):
        self._tryload()
        total = (
            len(self.trace_sampler)
            * len(self.pm_dataset.serialized)
            * len(self.aligners)
        )

        num_workers = multiprocessing.cpu_count()
        logging.info(
            f"Populating run dataset cache using {num_workers} workers..."
        )

        existing_items = self.items
        new_items = {}
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = set()
            preparing = tqdm(total=total, desc="Preparing")
            aligned = tqdm(total=0, desc="Aligned")  # dynamic total

            for m, t, a in product(
                self.pm_dataset.serialized, self.trace_sampler, self.aligners
            ):
                item_id = RunDataset._hash_item(m, t, a)
                if item_id in existing_items:
                    preparing.update(1)
                    continue

                fut = pool.submit(
                    RunDataset._process_item,
                    item_id,
                    m,
                    t,
                    a.name,
                )
                futures.add(fut)
                aligned.total += 1  # update total dynamically
                preparing.update(1)

                # **consume any futures that are ready immediately**
                done = {f for f in futures if f.done()}
                for d in done:
                    item = d.result()
                    new_items[item.item_id] = item
                    futures.remove(d)
                    aligned.update(1)

            # after producer loop: drain remaining futures
            for fut in as_completed(futures):
                item: RunDataset.ItemType = fut.result()
                new_items[item.item_id] = item
                aligned.update(1)

        # finalize
        self.items.update(new_items)
        self.index = list(self.items.keys())

        path = self.save_path()
        os.makedirs(path.parent, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.items, f)

    def _hash_config(self):
        base: dict[str, str | list[str]] = {
            "model_ds_hash": self.pm_dataset.hash(),
            "trace_sampler_hash": self.trace_sampler.hash(),
            "aligner_hash": [a.hash() for a in self.aligners],
        }
        return hashlib.sha1(
            json.dumps(base, sort_keys=True).encode()
        ).hexdigest()

    @staticmethod
    def _hash_trace(trace: Trace) -> str:
        return hashlib.sha1(
            json.dumps(
                [str(event) for event in trace], sort_keys=True
            ).encode()
        ).hexdigest()

    @staticmethod
    def _hash_item(
        model: ProcessModelDataset.SerializedView.ItemType,
        trace: Trace,
        aligner: Aligner,
    ) -> str:
        item: dict[str, str | int] = {
            "model_hash": model.hash(),
            "trace_hash": RunDataset._hash_trace(trace),
            "aligner_hash": aligner.hash(),
        }
        return hashlib.sha1(
            json.dumps(item, sort_keys=True).encode()
        ).hexdigest()

    @staticmethod
    def _hash_comb(trace: Trace, model: ProcessModelDataset.ItemType) -> str:
        item: dict[str, str | int] = {
            "model_hash": model.hash(),
            "trace_hash": RunDataset._hash_trace(trace),
        }
        return hashlib.sha1(
            json.dumps(item, sort_keys=True).encode()
        ).hexdigest()

    def __getitem__(self, index: int) -> "RunDataset.ItemType":
        key = self.index[index]
        # deserialize perf
        item = self.items[key]
        item.perf = PerfCounter.from_dict(item.perf)
        return item


class PM4pyAligner(Aligner):
    def __init__(self, variant: Variants):
        super().__init__(name=str(variant))
        self.variant = variant

    def __call__(self, pm: PetriNet, im: Marking, fm: Marking, trace: Trace):
        return apply(
            EventLog([trace]),
            pm,
            im,
            fm,
            None,
            variant=self.variant,
        )

    def hash(self) -> str:
        base = {"variant": str(self.variant.value)}
        return hashlib.sha1(
            json.dumps(base, sort_keys=True).encode()
        ).hexdigest()


class AlignerSpec(Enum):
    ALL = list(map(lambda v: PM4pyAligner(v), Variants))
    A_STAR = [
        PM4pyAligner(Variants.VERSION_STATE_EQUATION_A_STAR),
        PM4pyAligner(Variants.VERSION_DIJKSTRA_LESS_MEMORY),
    ]


class SimplePerturbedTraceSampler(TraceSampler):
    def __init__(
        self,
        ds: ProcessModelDataset,
        seed: Optional[int] = None,
        slice: Optional[range] = None,
    ):
        super().__init__(ds, seed, slice)

    def __getitem__(self, index: int) -> Trace:
        trace: Trace = self.log[index]
        return inject_noise_trace(trace=trace, seed=self.seed)


if __name__ == "__main__":
    import torch
    from dataloaders.net import VariantRandomDistributionSampler
    from dataloaders.csv_log import CSVEventLogDataset
    from dataloaders.xes_log import XESEventLogDataset
    from pm4py.discovery import discover_petri_net_inductive
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    PLOT = False

    path = "data/63a8435a-077d-4ece-97cd-2c76d394d99c/BPIC15_2.xes"

    log_dataset = XESEventLogDataset(path)

    # subset length distribution: defines the distribution of lengths across samples
    len_distribution = torch.distributions.Exponential(
        torch.tensor([1.0 / 100.0])
    )
    mean, std = 10.0, 5.0
    freq_distribution = torch.distributions.Normal(
        mean, std
    )  # defines the reordering of traces by defining the sampling distribution over index(index)

    if PLOT:
        # plot length distribution
        lengths = torch.arange(0, 500)
        probs = torch.exp(-len_distribution.rate * lengths)
        plt.plot(lengths.numpy(), probs.numpy())
        plt.title("Exponential Length Distribution (λ=1/100)")
        plt.xlabel("Trace Length")
        plt.ylabel("Probability Density")
        plt.grid()
        plt.show()

        # plot frequency distribution
        x = torch.linspace(-10, 30, 100)
        coeff = 1.0 / (std * torch.sqrt(torch.tensor(2.0 * 3.141592653589793)))
        exponent = -0.5 * ((x - mean) / std) ** 2
        probs = coeff * torch.exp(exponent)
        plt.plot(x.numpy(), probs.numpy())
        plt.title("Normal Frequency Distribution (μ=10, σ=5)")
        plt.xlabel("Trace Frequency")
        plt.ylabel("Probability Density")
        plt.grid()
        plt.show()

    pm_dataset = ProcessModelDataset(
        log_dataset=log_dataset,
        discovery_methods={"inductive": discover_petri_net_inductive},
        param_grid={
            "noise_threshold": [0.0, 0.1, 0.2, 0.3],
            "disable_fallthroughs": [True],
        },
        sampler_specs={
            "variant3": VariantRandomDistributionSampler(
                n_subsets=50,  # number of subsets: defines how often the log is sampled... basically
                max_len_subset=150,
                min_len_subset=20,  # max_length_subset: limits the possible length of each sample (what is fed to the discovery algorithm)
                len_distribution=len_distribution,
                freq_distribution=freq_distribution,
            )
        },
        cached=True,
    )

    run_dataset = RunDataset(
        Path('./data/runs'),
        pm_dataset,
        AlignerSpec.A_STAR.value,
        SimplePerturbedTraceSampler,
        multiprocessing=True,
        slice=range(0, 25),  # <- for testing
    )

    fe = CompositeFeatureExtractor()

    def format_row(
        run: RunDataset.ItemType, feature_vector: np.typing.NDArray[np.float32]
    ) -> pd.Series:
        return pd.Series(
            {
                "item_id": run.item_id,
                "combination_id": run.comb_id,
                "model_id": run.model.hash(),
                "trace_id": RunDataset._hash_trace(run.trace),
                "aligner": run.algo,
                "feature_vector": feature_vector,
                "time": run.perf.duration,  # <- probably shouldn't be total time but only search time
            }
        )

    df = pd.DataFrame(
        columns=[
            "item_id",
            "combination_id",
            "model_id",
            "trace_id",
            "aligner",
            "feature_vector",
            "time",
        ]
    )

    for run in run_dataset:
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
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    print(df.head())

    # group by combination_id and choose the minimum time across aligners
    labels = df.loc[df.groupby("combination_id")["time"].idxmin()]
    print(f"labels.head(): {labels.head()}")
    print("Summary statistics (minimum time across aligners):")
    print(labels["time"].describe())
    print("Distribution of aligners chosen:")
    print(labels["aligner"].value_counts())

    # write to csv
    labels.to_csv("./test_labels.csv", index=False)
