from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
import logging
import multiprocessing
from pathlib import Path
import pickle
import time
from torch.utils.data import Dataset
from typing import Any, Iterator, Union
import os
import hashlib
import json
from abc import ABC, abstractmethod
from itertools import product

from tqdm import tqdm
from dataloaders.base import make_feature_fn
from dataloaders.csv_log import CSVEventLogDataset
from experiments.simulation.noise import inject_noise_trace
from features.extractors import CompositeFeatureExtractor
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.log.obj import EventLog, Trace

from dataloaders.net import (
    DISCOVERY_METHODS,
    PARAM_GRID,
    SAMPLER_SPECS,
    ProcessModelDataset,
)
from pm4py.algo.conformance.alignments.petri_net.algorithm import (
    Variants,
    apply,
)
from pm4py.objects.petri_net.utils.petri_utils import construct_trace_net
from pm4py.util import typing


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
        pass


class PerfCounter:
    def __init__(self):
        self.time = time.perf_counter()
        # other perf counters?

    def __sub__(self, other: "PerfCounter"):
        ret = PerfCounter()
        ret.time = self.time - other.time
        return ret

    def dict(self) -> dict[str, float]:
        return {"time": self.time}


class TraceSampler(ABC):

    def __init__(self, ds: ProcessModelDataset):
        self.source_path = ds.cache_dir
        self.log = ds.log

    def __len__(self):
        return len(self.log)

    @abstractmethod
    def __getitem__(self, index: int) -> Trace:
        # here you would modify traces from the original log or similar
        raise NotImplementedError

    def __iter__(self) -> Iterator[Trace]:
        for i in range(len(self)):
            yield self.__getitem__(i)

    def hash(self) -> str:
        base = {
            "name": str(self.__class__),
            "ds": str(self.source_path),
        }
        return hashlib.sha1(
            json.dumps(base, sort_keys=True).encode()
        ).hexdigest()


class RunDataset(Dataset):

    item_type = tuple[
        ProcessModelDataset.ItemType,
        Trace,
        Union[typing.AlignmentResult, typing.ListAlignments],
        dict[str, Any],
        str,
    ]

    def __init__(
        self,
        base_path: Path,
        process_model_dataset: ProcessModelDataset,
        aligners: Sequence[Aligner],
        trace_sampler: TraceSampler.__class__,
        multiprocessing: bool = True,
    ):
        self.base_path = base_path
        self.pm_dataset = process_model_dataset
        self.trace_sampler = trace_sampler(self.pm_dataset)
        self.aligners = aligners
        self.items: dict[str, "RunDataset.item_type"] = {}
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
                self.index = list(self.items.keys())

    def save_path(self):
        return self.base_path / Path(f"{self._hash_config()}.pkl")

    @staticmethod
    def _process_item(
        model: "ProcessModelDataset.ItemType",
        trace: Trace,
        aligner: Aligner,
    ) -> "RunDataset.item_type":
        start = PerfCounter()
        item = aligner(model.pm, model.im, model.fm, trace)
        end = PerfCounter()
        return (
            model,
            trace,
            item,
            (end - start).dict(),
            aligner.name,
        )

    def _init_cache(self):
        self._tryload()
        for trace in tqdm(self.trace_sampler, total=len(self.trace_sampler)):
            for model in tqdm(self.pm_dataset, total=len(self.pm_dataset)):
                for aligner in self.aligners:
                    h = self._hash_item(
                        model, trace, aligner
                    )  # should be deterministic in the result since aligner should be deterministic in the result
                    if h in self.items:
                        continue
                    self.items[h] = RunDataset._process_item(
                        model, trace, aligner
                    )

        os.makedirs(self.base_path, exist_ok=True)
        with open(self.save_path(), "wb") as f:
            pickle.dump(self.items, f)

    def _init_cache_mp(self):
        self._tryload()

        total = (
            len(self.trace_sampler) * len(self.pm_dataset) * len(self.aligners)
        )
        num_workers = min(multiprocessing.cpu_count(), 64)
        logging.info(
            f"Populating run dataset cache using {num_workers} workers..."
        )

        existing_items = self.items
        new_items = {}

        with ProcessPoolExecutor(max_workers=num_workers) as pool:

            results = pool.map(
                lambda args: RunDataset._process_item(*args),
                [
                    (t, m, a)
                    for t, m, a in product(
                        self.trace_sampler, self.pm_dataset, self.aligners
                    )
                    if RunDataset._hash_item(t, m, a) not in existing_items
                ],
            )

            for model, trace, aligner, perf, item, algo in tqdm(
                results, total=total, desc="Aligned"
            ):
                h = self._hash_item(model, trace, aligner)
                new_items[h] = (model, trace, item, perf, algo)

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
    def _hash_item(
        model: ProcessModelDataset.ItemType,
        trace: Trace,
        aligner: Aligner,
    ) -> str:
        item: dict[str, str | int] = {
            "model_hash": model.hash(),
            "trace_hash": trace.__hash__(),  # <- this hashes statically
            "aligner_hash": aligner.hash(),
        }
        return hashlib.sha1(
            json.dumps(item, sort_keys=True).encode()
        ).hexdigest()

    def __getitem__(self, index: int) -> "RunDataset.item_type":
        key = self.index[index]
        return self.items[key]


class AstarAligner(Aligner):
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
    ALL = list(map(lambda v: AstarAligner(v), Variants))


class SimplePerturbedTraceSampler(TraceSampler):
    def __init__(self, ds: ProcessModelDataset):
        super().__init__(ds)

    def __getitem__(self, index: int) -> Trace:
        trace: Trace = self.log[index]
        return inject_noise_trace(trace=trace)


if __name__ == "__main__":

    path = "data/c3f3ba2d-e81e-4274-87c7-882fa1dbab0d/BPI2016_Werkmap_Messages.csv"
    log_dataset = CSVEventLogDataset(
        path,
        case_id_col="CustomerID",
        timestamp_col="EventDateTime",
        activity_col="HandlingChannelID",
        sep=";",
        feature_fn=make_feature_fn,
    )

    pm_dataset = ProcessModelDataset(
        log_dataset=log_dataset,
        discovery_methods=DISCOVERY_METHODS.GURANTEED_SOUND,
        param_grid=PARAM_GRID.STANDARD,
        sampler_specs=SAMPLER_SPECS.STANDARD,
        cached=True,
    )

    run_dataset = RunDataset(
        Path('./data/runs'),
        pm_dataset,
        AlignerSpec.ALL.value,
        SimplePerturbedTraceSampler,
        multiprocessing=False,
    )

    fe = CompositeFeatureExtractor()

    for run in run_dataset:
        print(run)
        model, trace, item, perf, algo = run
        trace_net, trace_im, trace_fm = construct_trace_net(trace)
        fv = fe.extract(
            model.pm, model.im, model.fm, trace_net, trace_im, trace_fm
        )
        print(fv)
        break
