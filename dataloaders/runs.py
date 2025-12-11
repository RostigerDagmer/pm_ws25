from dataloaders.util import _normalize_log_input
from dataloaders.net import deserialize_sampler
from concurrent.futures._base import Future
from concurrent.futures._base import as_completed
from concurrent.futures import TimeoutError
import pebble
from collections.abc import Sequence

from enum import Enum
from io import BufferedWriter
import logging
import marshal
import multiprocessing
from pathlib import Path
import pickle
import time
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import os
import hashlib
import json
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from dataloaders.synthetic import SyntheticProcessModelDataset
from experiments.simulation.noise import inject_noise_trace
from experiments.simulation.simulate import simulate_batch, apply_labels
from features.extractors import CompositeFeatureExtractor
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.log.obj import EventLog, Trace
import cProfile
import pstats
from dataclasses import dataclass, asdict

from dataloaders.net import ProcessModelDataset
from dataloaders.serializable import (
    Deserializable,
    Serializable,
    WithSerializedView,
    Hashable,
)
from dataloaders.unique_net import UniqueProcessModelDataset
from pm4py.algo.conformance.alignments.petri_net.algorithm import (
    Variants,
    apply,
)
from pm4py.util import typing
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from util.rng import RNG
import random
import gc


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


class PM4pyAligner(Aligner):
    def __init__(self, variant_name: str):
        super().__init__(name=variant_name)
        self.variant_name = variant_name

    @property
    def variant(self) -> Variants:
        return Variants[self.variant_name]

    def __call__(self, pm: PetriNet, im: Marking, fm: Marking, trace: Trace):
        return apply(
            EventLog([trace]),
            pm,
            im,
            fm,
            None,
            variant=self.variant,  # Enum member; its .value is the module
        )

    def hash(self) -> str:
        base = {"variant": f"{self.variant_name}"}
        return hashlib.sha1(
            json.dumps(base, sort_keys=True).encode()
        ).hexdigest()


class AlignerSpec(Enum):
    ALL = list(map(lambda v: PM4pyAligner(v.name), Variants))
    A_STAR = [
        PM4pyAligner("VERSION_DIJKSTRA_NO_HEURISTICS"),
        PM4pyAligner("VERSION_REMAINING_TRACE"),
        PM4pyAligner("VERSION_REQUIRED_ACTIVITIES"),
        PM4pyAligner("VERSION_STATE_EQUATION_A_STAR"),
        PM4pyAligner("VERSION_STATE_EQUATION_A_STAR_ILP"),
        # PM4pyAligner(Variants.VERSION_INCREMENTAL_A_STAR),
    ]


class PerfCounter(Serializable[dict[str, Any]]):
    def __init__(self):
        self._profiler = cProfile.Profile()
        self.duration: float | None = None
        self.stats: pstats.Stats | None = None

        # Specific metrics extracted from profile
        self.search_time: float = 0.0
        self.lp_time: float = 0.0

    def __enter__(self):
        self._t_start = time.perf_counter()
        self._profiler.enable()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._profiler.disable()
        self._t_end = time.perf_counter()
        self.duration = self._t_end - self._t_start
        self.stats = pstats.Stats(self._profiler).strip_dirs()
        self.extract_metrics()
        return False

    def extract_metrics(self) -> Optional[dict[str, float]]:
        """
        Parses the pstats to find cumulative time for specific functions
        (__search and LP solvers).
        """
        if not self.stats:
            return

        self.search_time = 0.0
        self.lp_time = 0.0

        # ps.stats is a dict: (filename, line, funcname) -> (cc, nc, tt, ct, callers)
        for func, (cc, nc, tt, ct, callers) in self.stats.stats.items():
            fname = func[2]

            if "__search" in fname or "synchr" in fname:
                self.search_time += ct
            elif "cvxopt.glpk.lp" in fname or "cvxopt.glpk.ilp" in fname:
                self.lp_time += ct
        return {
            "search_time": self.search_time,
            "lp_time": self.lp_time,
        }

    @staticmethod
    def inf():
        duration = float("inf")
        stats = pstats.Stats()
        search_time = float("inf")
        lp_time = float("inf")
        # fill in stats
        stats.stats = {
            "__search": (0, 0, search_time, 0, {}),
            "cvxopt.glpk.lp": (0, 0, lp_time, 0, {}),
        }
        ret = PerfCounter()
        ret.duration = duration
        ret.stats = stats
        ret.search_time = search_time
        ret.lp_time = lp_time
        return ret

    def _dict(self) -> dict[str, Any]:
        return {
            "duration": self.duration,
            "stats": marshal.dumps(self.stats.stats) if self.stats else None,
        }

    def serialize(self) -> dict[str, Any]:
        return self._dict()

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "PerfCounter":
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
        seed: int,
        ds: Union[
            ProcessModelDataset,
            UniqueProcessModelDataset,
            SyntheticProcessModelDataset,
        ],
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
            "seed": self.seed,
        }
        return hashlib.sha1(
            json.dumps(base, sort_keys=True).encode()
        ).hexdigest()

    def iter_for_model(self, model: Any) -> Iterator[Trace]:
        # ignore by default
        return self.__iter__()


@dataclass
class NoiseParams:
    p_insert: float = 0.1
    p_delete: float = 0.1
    p_swap: float = 0.05

    def hash(self):
        base = asdict(self)
        return hashlib.sha1(
            json.dumps(base, sort_keys=True).encode()
        ).hexdigest()


class SimplePerturbedTraceSampler(TraceSampler):
    def __init__(
        self,
        seed: int,
        ds: Union[ProcessModelDataset, UniqueProcessModelDataset],
        noise_params: NoiseParams | dict[str, float] = NoiseParams(),
        slice: Optional[range] = None,
    ):
        super().__init__(seed=seed, ds=ds, slice=slice)
        if isinstance(noise_params, dict):
            noise_params = NoiseParams(**noise_params)
        self.noise_params = noise_params

    def hash(self) -> str:
        base = {
            "name": str(self.__class__),
            "ds": str(self.source_path),
            "seed": self.seed,
            "noise": self.noise_params.hash(),
        }
        return hashlib.sha1(
            json.dumps(base, sort_keys=True).encode()
        ).hexdigest()

    def __getitem__(self, index: int) -> Trace:
        trace: Trace = self.log[index]
        return inject_noise_trace(
            trace=trace,
            p_insert=self.noise_params.p_insert,
            p_delete=self.noise_params.p_delete,
            p_swap=self.noise_params.p_swap,
            seed=self.seed,
        )


class SubsetAwarePerturbedTraceSampler(TraceSampler):
    def __init__(
        self,
        seed: int,
        ds: Union[ProcessModelDataset, UniqueProcessModelDataset],
        noise_params: NoiseParams | dict[str, float] = NoiseParams(),
        slice: Optional[range] = None,
        pick_random: bool = False,
    ):
        super().__init__(seed=seed, ds=ds, slice=slice)
        if isinstance(noise_params, dict):
            noise_params = NoiseParams(**noise_params)
        self.noise_params = noise_params
        self.pick_random = pick_random

    def __getitem__(self, index: int) -> Trace:
        trace: Trace = self.log[index]
        return inject_noise_trace(
            trace=trace,
            p_insert=self.noise_params.p_insert,
            p_delete=self.noise_params.p_delete,
            p_swap=self.noise_params.p_swap,
            seed=self.seed,
        )

    def hash(self) -> str:
        base = {
            "name": str(self.__class__),
            "ds": str(self.source_path),
            "seed": self.seed,
            "noise": self.noise_params.hash(),
        }
        return hashlib.sha1(
            json.dumps(base, sort_keys=True).encode()
        ).hexdigest()

    def iter_for_model(
        self, model: ProcessModelDataset.SerializedItemType
    ) -> Iterator[Trace]:
        org_sampler = deserialize_sampler(model.sampler)
        subset = org_sampler.sample_once(self.log, model.subset_idx)
        for i in self.range:
            if i >= len(subset):
                break
            if self.pick_random:
                trace = random.choice(subset)
            else:
                trace = subset[i]
            yield inject_noise_trace(
                trace=trace,
                p_insert=self.noise_params.p_insert,
                p_delete=self.noise_params.p_delete,
                p_swap=self.noise_params.p_swap,
                seed=self.seed,
            )


class SyntheticTraceSampler(TraceSampler):
    def __init__(
        self,
        seed: int,
        ds: SyntheticProcessModelDataset,
        slice: Optional[range] = None,
        batch_size: int = 128,
        steps: int = 100,
        noise: NoiseParams | dict[str, float] = NoiseParams(),
        device: str = "cpu",
    ):
        if slice is None:
            slice = range(batch_size)
        super().__init__(seed=seed, ds=ds, slice=slice)
        self.pm_dataset = ds
        self.batch_size = batch_size
        self.steps = steps
        if isinstance(noise, dict):
            noise = NoiseParams(**noise)
        self.noise = noise
        self.device = device
        self._traces_per_model = (
            slice.stop - slice.start if slice is not None else batch_size
        )

    def __len__(self) -> int:
        return self._traces_per_model

    def __getitem__(self, index: int) -> Trace:
        raise RuntimeError(
            "SyntheticTraceSampler must be used via iter_for_model(model)"
        )

    def __iter__(self):
        raise RuntimeError(
            "SyntheticTraceSampler must be used via iter_for_model(model)"
        )

    def hash(self) -> str:
        base = {
            "seed": self.seed,
            "batch_size": self.batch_size,
            "steps": self.steps,
            "noise": self.noise.hash(),
        }
        return hashlib.sha1(
            json.dumps(base, sort_keys=True).encode()
        ).hexdigest()

    def iter_for_model(
        self, model: SyntheticProcessModelDataset.SerializedItemType
    ) -> Iterator[Trace]:
        N = model.net

        pre = N.pre.to(self.device)
        post = N.post.to(self.device)
        M0 = N.M0.to(self.device)
        Mf = N.Mf.to(self.device)
        labels = N.labels  # list[str]

        # for determinism: derive a per-model seed
        model_seed = (int(model.hash(), 16) ^ self.seed) % 0xFFFFFFFFFFFF

        # generate traces in batches but expose as individual Trace objects
        remaining = self._traces_per_model
        trace_index = 0
        generator = torch.Generator(device=self.device).manual_seed(model_seed)
        while remaining > 0:
            bsz = min(self.batch_size, remaining)
            seq_batch = simulate_batch(
                (pre, post),
                M0,
                Mf,
                labels,
                steps=self.steps,
                batch_size=bsz,
                compact=True,
                generator=generator,
            )  # shape [B, steps'] with integer label ids

            log = apply_labels(seq_batch, labels)
            for trace in log:
                trace_hash = RunDataset._hash_trace(trace)
                trace_seed = (int(trace_hash, 16) ^ model_seed) % 2**64
                noisy_trace = inject_noise_trace(
                    trace,
                    p_insert=self.noise.p_insert,
                    p_delete=self.noise.p_delete,
                    p_swap=self.noise.p_swap,
                    labels=labels,
                    activity_key="concept:name",
                    seed=trace_seed,  # deterministic per-trace
                )
                yield noisy_trace
                trace_index += 1
                remaining -= 1
                if remaining <= 0:
                    break


@dataclass
class ItemType(Serializable["SerializedItemType"]):
    item_id: str
    model: ProcessModelDataset.ItemType | SyntheticProcessModelDataset.ItemType
    trace: Trace
    item: Union[typing.AlignmentResult, typing.ListAlignments]
    perf: list[PerfCounter]
    algo: str
    comb_id: str  # an identifier for the combination of model, trace (to group by aligner)

    def serialize(self) -> "SerializedItemType":
        return SerializedItemType(
            self.item_id,
            self.model.serialize(),
            self.trace,
            self.item,
            [p.serialize() for p in self.perf],
            self.algo,
            self.comb_id,
        )


@dataclass
class SerializedItemType(Deserializable[ItemType]):
    item_id: str
    model: (
        ProcessModelDataset.SerializedItemType
        | SyntheticProcessModelDataset.SerializedItemType
        | str
    )  # either a serialized model or a key into the process model dataset
    trace: Trace
    item: Union[typing.AlignmentResult, typing.ListAlignments]
    perf: list[dict[str, Any]]
    algo: str
    comb_id: str  # an identifier for the combination of model, trace (to group by aligner)

    def deserialize(self) -> ItemType:
        if isinstance(self.model, str):
            raise ValueError("Cannot deserialize model from key")
        return ItemType(
            self.item_id,
            self.model.deserialize(),
            self.trace,
            self.item,
            [PerfCounter.from_dict(p) for p in self.perf],
            self.algo,
            self.comb_id,
        )


class RunDataset(
    Dataset[ItemType], WithSerializedView[ItemType, SerializedItemType]
):

    def __init__(
        self,
        base_path: Path,
        process_model_dataset: Union[
            ProcessModelDataset,
            UniqueProcessModelDataset,
            SyntheticProcessModelDataset,
        ],
        aligners: Sequence[Aligner],
        trace_sampler: TraceSampler,
        n_runs: int = 1,  # Number of runs per trace/model pair
        n_workers: int = 0,
        write_batch_size: int = 100,
        timeout: float = 20.0,
        skip_init: bool = False,
    ):
        self.base_path = base_path
        self.pm_dataset = process_model_dataset
        self.save_path = (
            self.base_path / Path(f"{(self.pm_dataset.log_uuid)}.pkl")
            if isinstance(
                self.pm_dataset,
                (ProcessModelDataset, UniqueProcessModelDataset),
            )
            else self.base_path / Path("synthetic.pkl")
        )
        self.trace_sampler = trace_sampler
        self.aligners = aligners
        self.n_runs = n_runs
        self.items: dict[str, "RunDataset.SerializedItemType"] = {}
        self.combinations: dict[str, list[str]] = (
            {}
        )  # pivot table from combination_id -> item_ids
        self.index: list[str] = []
        self.n_workers = n_workers
        self.write_batch_size = write_batch_size
        self.timeout = timeout
        if skip_init:
            self._tryload()
        else:
            self._init_cache_mp()

    @property
    def log_uuid(self) -> str:
        return self.pm_dataset.log_uuid

    def __iter__(self) -> Iterator[ItemType]:
        for i in range(len(self)):
            yield self[i]

    @staticmethod
    def _flush_batch(
        f: BufferedWriter, batch: dict[str, "RunDataset.SerializedItemType"]
    ):
        pickle.dump(batch, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())

    def _tryload(self):
        path = self.save_path
        if not os.path.exists(path):
            return
        self.items = {}
        self.combinations = {}

        with open(path, "rb") as f:
            while True:
                try:
                    chunk = pickle.load(f)  # each chunk = dict of N items
                    self.items.update(chunk)
                    for item in chunk.values():
                        self.combinations.setdefault(item.comb_id, []).append(
                            item.item_id
                        )
                except EOFError:
                    break

        self.index = list(self.items.keys())

    @staticmethod
    def _stuff_timeout_item(
        hash: str,
        model: (
            ProcessModelDataset.SerializedItemType
            | SyntheticProcessModelDataset.SerializedItemType
        ),
        trace: Trace,
        aligner: Aligner,
        n_runs: int = 1,
    ) -> SerializedItemType:
        # Create infinite stats
        stats = [PerfCounter.inf().serialize() for _ in range(n_runs)]
        deser_model = model.deserialize()

        return SerializedItemType(
            hash,
            model,
            trace,
            None,
            stats,
            aligner.name,
            RunDataset._hash_comb(trace, deser_model),
        )

    @staticmethod
    def _process_item(
        hash: str,
        model: (
            ProcessModelDataset.SerializedItemType
            | SyntheticProcessModelDataset.SerializedItemType
        ),
        trace: Trace,
        aligner: Aligner,
        n_runs: int = 1,
    ) -> SerializedItemType:

        deser_model = model.deserialize()
        file_path = f"./dbg/{model.hash()}.svg"
        if logging.getLogger().level == logging.DEBUG and not os.path.exists(
            file_path
        ):
            gviz = pn_visualizer.apply(
                deser_model.pm,
                deser_model.im,
                deser_model.fm,
                parameters={"format": "svg"},
            )
            pn_visualizer.save(gviz, file_path)
        # MULTI-RUN BENCHMARK LOGIC
        stats: list[dict[str, Any]] = []
        last_item = None

        # Loop n times to collect statistics
        for _ in range(n_runs):
            with PerfCounter() as pc:
                item = aligner(
                    deser_model.pm,
                    deser_model.im,
                    deser_model.fm,
                    trace,
                )

            stats.append(pc.serialize())
            logging.debug(
                f"duration [{aligner.name}]: {pc.duration}" + f" - {file_path}"
                if logging.getLogger().level == logging.DEBUG
                else ""
            )
            last_item = item

        # Calculate aggregates
        return SerializedItemType(
            hash,
            model.hash(),
            trace,
            last_item,
            stats,
            aligner.name,
            RunDataset._hash_comb(trace, deser_model),
        )

    def __len__(self):
        return len(self.index)

    def _init_cache_mp(self):
        self._tryload()
        logging.info("Initializing run dataset cache (multiprocessing)...")
        logging.info("Current number of items: %d", len(self.items))
        total = (
            len(self.trace_sampler)
            * len(self.pm_dataset.serialized)
            * len(self.aligners)
        )

        num_workers = (
            self.n_workers
            if self.n_workers > 0
            else multiprocessing.cpu_count()
        )
        logging.info(
            f"Populating run dataset cache using {num_workers} workers..."
        )

        existing_items = self.items
        new_items: dict[str, RunDataset.SerializedItemType] = {}
        new_combinations: dict[str, list[str]] = {}
        batch: dict[str, RunDataset.SerializedItemType] = {}
        os.makedirs(self.save_path.parent, exist_ok=True)

        with (
            pebble.ProcessPool(max_workers=num_workers) as pool,
            open(self.save_path, "ab") as f,
        ):
            futures: dict[Future[RunDataset.SerializedItemType], tuple] = {}
            scheduled = tqdm(total=total, desc="Scheduled")
            aligned = tqdm(total=0, desc="Aligned")  # dynamic total
            written = tqdm(total=0, desc="Written")  # dynamic total
            seen_hashes = set()
            irrelevant = set(self.items.keys())

            for m in self.pm_dataset.serialized:
                for t in self.trace_sampler.iter_for_model(m):
                    for a in self.aligners:
                        item_id = RunDataset._hash_item(m, t, a)
                        try:
                            irrelevant.remove(item_id)
                        except KeyError:
                            pass
                        if item_id in existing_items or item_id in seen_hashes:
                            scheduled.total -= 1
                            continue
                        seen_hashes.add(item_id)
                        fut = pool.schedule(
                            RunDataset._process_item,
                            args=(
                                item_id,
                                m,
                                t,
                                a,
                                self.n_runs,
                            ),
                            timeout=self.timeout,
                        )
                        futures[fut] = (item_id, m, t, a)
                        aligned.total += 1  # update total dynamically
                        written.total += 1  # update total dynamically
                        scheduled.update(1)

            # drain futures
            for fut in as_completed(futures):
                try:
                    item = fut.result()
                except TimeoutError:
                    # Handle timeout by stuffing
                    i_id, mod, tr, alg = futures[fut]
                    item = RunDataset._stuff_timeout_item(
                        i_id, mod, tr, alg, self.n_runs
                    )
                except Exception as e:
                    logging.error(f"Error processing item: {e}")
                    continue

                new_items[item.item_id] = item
                batch[item.item_id] = item
                new_combinations.setdefault(item.comb_id, []).append(
                    item.item_id
                )
                aligned.update(1)

                if len(batch) >= self.write_batch_size:
                    RunDataset._flush_batch(f, batch)
                    written.update(len(batch))
                    batch.clear()

            # final flush
            if batch:
                RunDataset._flush_batch(f, batch)
                written.update(len(batch))
        # finalize
        self.items = {
            k: v for k, v in self.items.items() if k not in irrelevant
        }
        self.combinations = {
            k: [v for v in vals if v not in irrelevant]
            for k, vals in self.combinations.items()
        }
        self.combinations = {
            k: v for k, v in self.combinations.items() if v
        }  # filter empty
        self.items.update(new_items)
        self.combinations.update(new_combinations)
        self.index = list(self.items.keys())
        gc.collect()

    @staticmethod
    def _hash_trace(trace: Trace) -> str:
        return hashlib.sha1(
            json.dumps(
                [str(event) for event in trace], sort_keys=True
            ).encode()
        ).hexdigest()

    @staticmethod
    def _hash_item(
        model: Hashable,
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
    def _hash_comb(trace: Trace, model: Hashable) -> str:
        item: dict[str, str | int] = {
            "model_hash": model.hash(),
            "trace_hash": RunDataset._hash_trace(trace),
        }
        return hashlib.sha1(
            json.dumps(item, sort_keys=True).encode()
        ).hexdigest()

    def _get_serialized(
        self, idx: int | str
    ) -> "RunDataset.SerializedItemType":
        if isinstance(idx, str):
            key = idx
        else:
            key = self.index[idx]
        item = self.items[key]
        # replace model index with model object
        if isinstance(item.model, (str, int)):
            item.model = self.pm_dataset.serialized[item.model]
        return item

    def __getitem__(self, index: int | str) -> "RunDataset.ItemType":
        return self._get_serialized(index).deserialize()

    def iter_by_combination(
        self,
    ) -> Iterator[
        Tuple[
            Union[
                "ProcessModelDataset.ItemType",
                "SyntheticProcessModelDataset.ItemType",
            ],
            Trace,
            Dict[
                str,
                Tuple[
                    str,
                    Union[typing.AlignmentResult, typing.ListAlignments],
                    List[PerfCounter],
                ],
            ],
        ]
    ]:
        """
        Iterate over RunDataset grouped by combination_id.

        Yields for each (Model, Trace) combination:
            - model: ProcessModelDataset.ItemType
            - trace: Trace object
            - results_dict: Dict[algo_name -> (algo, item, perf)]
        """

        for comb_id in self.combinations:
            items = [self[item_id] for item_id in self.combinations[comb_id]]
            model = items[0].model
            trace = items[0].trace
            results_dict = {
                item.algo: (item.algo, item.item, item.perf) for item in items
            }
            yield (model, trace, results_dict)


def get_stats(stats: list[PerfCounter]) -> dict[str, float]:
    durations = [s.duration for s in stats if s.duration is not None]
    ms = [s.extract_metrics() for s in stats]
    search_times = [s["search_time"] for s in ms]
    lp_times = [s["lp_time"] for s in ms]

    def compute_metrics(data: list[float]) -> dict[str, float]:
        # replace inf values with large positive values
        data = [1000.0 if x == np.inf else x for x in data]
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


if __name__ == "__main__":
    from dataloaders.net import VariantRandomDistributionSampler
    from pm4py.objects.petri_net.utils.petri_utils import construct_trace_net
    import matplotlib.pyplot as plt
    import pandas as pd

    RNG.initialize(3)
    logging.basicConfig(level=logging.INFO)

    # CONFIGURATION
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
                20,  # Number of models per config
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
                20,  # Number of models per config
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
                20,  # Number of models per config
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
                20,  # Number of models per config
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
                20,  # Number of models per config
            ),
        ],
    )
    trace_sampler = SyntheticTraceSampler(
        ds=synthetic_dataset,
        seed=RNG.get_seed(),
        batch_size=128,
        slice=range(0, 8),
        steps=40,
    )

    run_dataset = RunDataset(
        Path("cache/.runs"),
        synthetic_dataset,
        AlignerSpec.A_STAR.value,
        trace_sampler,
        n_runs=N_RUNS,
        n_workers=20,
    )

    fe = CompositeFeatureExtractor()

    def format_row(
        run: RunDataset.ItemType,  # feature_vector: np.typing.NDArray[np.float32]
    ) -> pd.Series:
        stats = get_stats(run.perf)

        return pd.Series(
            {
                "item_id": run.item_id,
                "combination_id": run.comb_id,
                "model_id": run.model.hash(),
                "trace_id": RunDataset._hash_trace(run.trace),
                "aligner": run.algo,
                # "feature_vector": feature_vector,
                # metric for decision making
                "time_total_mean": stats.get("mean_total"),
                "time_total_std": stats.get("std_total"),
                "time_total_median": stats.get("median_total"),
                # breakdown metrics
                "time_search_mean": stats.get("mean_search"),
                "time_lp_mean": stats.get("mean_lp"),
            }
        )

    df = pd.DataFrame(
        columns=[
            "item_id",
            "combination_id",
            "model_id",
            "trace_id",
            "aligner",
            # "feature_vector",
            "time_total_mean",
            "time_total_std",
            "time_total_median",
            "time_search_mean",
            "time_lp_mean",
        ]
    )

    for run in tqdm(run_dataset, desc="Extracting features from runs"):
        model, trace, item, perf, algo = (
            run.model,
            run.trace,
            run.item,
            run.perf,
            run.algo,
        )
        # trace_net, trace_im, trace_fm = construct_trace_net(trace)
        # fv = fe.extract(
        #     model.pm, model.im, model.fm, trace_net, trace_im, trace_fm
        # )
        row = format_row(run)  # , fv)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    print(df.head())

    # group by combination_id and choose the minimum time across aligners
    best_indices = df.groupby("combination_id")["time_total_mean"].idxmin()
    labels = df.loc[best_indices]

    # print(f"labels.head(): {labels.head()}")
    print("\nBest Aligner Labels (Head):")
    print(labels[["aligner", "time_total_mean", "time_total_std"]].head())
    print("Summary statistics (minimum time across aligners):")
    print(labels["time_total_mean"].describe())
    print("Distribution of aligners chosen:")
    print(labels["aligner"].value_counts())
