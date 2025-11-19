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
from sklearn.ensemble import GradientBoostingClassifier

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
        PM4pyAligner(Variants.VERSION_STATE_EQUATION_A_STAR_ILP),
        PM4pyAligner(Variants.VERSION_INCREMENTAL_A_STAR),
    ]



class PerfCounter:
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

    def extract_metrics(self):
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

    def dict(self) -> dict[str, Any]:
        return {
            "duration": self.duration,
            "search_time": self.search_time,
            "lp_time": self.lp_time,
            # "stats": marshal.dumps(self.stats.stats) if self.stats else None
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

    @dataclass
    class SerializedItemType:
        item_id: str
        model: ProcessModelDataset.SerializedView.ItemType
        trace: Trace
        item: Union[typing.AlignmentResult, typing.ListAlignments]
        perf: dict[str, Any]
        algo: str
        comb_id: str  # an identifier for the combination of model, trace (to group by aligner)

        def deserialize(self) -> "RunDataset.ItemType":
            return RunDataset.ItemType(
                self.item_id,
                self.model.deserialize(),
                self.trace,
                self.item,
                PerfCounter.from_dict(self.perf),
                self.algo,
                self.comb_id,
            )

    def __init__(
        self,
        base_path: Path,
        process_model_dataset: ProcessModelDataset,
        aligners: Sequence[Aligner],
        trace_sampler: TraceSampler.__class__,
        n_runs: int = 1,  # Number of runs per trace/model pair
        multiprocessing: bool = True,
        slice: Optional[range] = None,
    ):
        self.base_path = base_path
        self.pm_dataset = process_model_dataset
        self.trace_sampler = trace_sampler(
            self.pm_dataset, seed=SEED, slice=slice
        )
        self.aligners = aligners
        self.n_runs = n_runs
        self.items: dict[str, "RunDataset.SerializedItemType"] = {}
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
        hash: str,
        model: ProcessModelDataset.SerializedView.ItemType,
        trace: Trace,
        aligner: Aligner | str,
        n_runs: int = 1,
    ) -> "RunDataset.SerializedItemType":

        deser_model = model.deserialize()
        if isinstance(aligner, str):
            # reconstruct aligner from name (in the mp case)
            aligner = next(
                a for a in AlignerSpec.ALL.value if a.name == aligner
            )

            # MULTI-RUN BENCHMARK LOGIC
            durations = []
            search_times = []
            lp_times = []
            last_item = None

            # Loop n times to collect statistics
            for _ in range(n_runs):
                with PerfCounter() as pc:
                    item = aligner(
                        deser_model.pm, deser_model.im, deser_model.fm, trace
                    )

                durations.append(pc.duration)
                search_times.append(pc.search_time)
                lp_times.append(pc.lp_time)
                last_item = item

            # Calculate aggregates
            stats = {
                "mean_total": float(np.mean(durations)),
                "std_total": float(np.std(durations)),
                "median_total": float(np.median(durations)),
                "min_total": float(np.min(durations)),

                "mean_search": float(np.mean(search_times)),
                "std_search": float(np.std(search_times)),

                "mean_lp": float(np.mean(lp_times)),
                "std_lp": float(np.std(lp_times)),

                "n_runs": n_runs
            }

            return RunDataset.SerializedItemType(
                hash,
                model,
                trace,
                last_item,
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
                        h, model, trace, aligner, self.n_runs
                    )

        self.index = list(self.items.keys())
        os.makedirs(self.base_path, exist_ok=True)
        with open(self.save_path(), "wb") as f:
            pickle.dump(self.items, f)

    def __len__(self):
        return len(self.index)

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
                    self.n_runs,  # pass n_runs to worker
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
        base: dict[str, str | list[str] | int] = {
            "model_ds_hash": self.pm_dataset.hash(),
            "trace_sampler_hash": self.trace_sampler.hash(),
            "aligner_hash": [a.hash() for a in self.aligners],
            "n_runs": self.n_runs  # Hash must include n_runs to invalidate cache if changed
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
        # deserialize
        item = self.items[key].deserialize()
        return item


if __name__ == "__main__":
    import torch
    from dataloaders.net import VariantRandomDistributionSampler
    from dataloaders.csv_log import CSVEventLogDataset
    from dataloaders.xes_log import XESEventLogDataset
    from pm4py.discovery import discover_petri_net_inductive
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # CONFIGURATION
    PLOT = False
    N_RUNS = 5  # set number of runs here
    path = "../data/63a8435a-077d-4ece-97cd-2c76d394d99c/BPIC15_2.xes"

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
        n_runs=N_RUNS,
        multiprocessing=True,
        slice=range(0, 10),  # <- for testing
    )

    fe = CompositeFeatureExtractor()

    def format_row(
        run: RunDataset.ItemType, feature_vector: np.typing.NDArray[np.float32]
    ) -> pd.Series:
        stats = run.perf
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


    df = pd.DataFrame(
        columns=[
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
            "time_lp_mean"
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
        trace_net, trace_im, trace_fm = construct_trace_net(trace)
        fv = fe.extract(
            model.pm, model.im, model.fm, trace_net, trace_im, trace_fm
        )
        row = format_row(run, fv)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    print(df.head())

    # group by combination_id and choose the minimum time across aligners
    best_indices = df.groupby("combination_id")["time_total_mean"].idxmin()
    labels = df.loc[best_indices]

    #print(f"labels.head(): {labels.head()}")
    print("\nBest Aligner Labels (Head):")
    print(labels[["aligner", "time_total_mean", "time_total_std"]].head())
    print("Summary statistics (minimum time across aligners):")
    print(labels["time_total_mean"].describe())
    print("Distribution of aligners chosen:")
    print(labels["aligner"].value_counts())

    # write to csv
    labels.to_csv("./test_labels.csv", index=False)
    df.to_csv("./all_runs_aggregated.csv", index=False)
    print("\nSaved aggregated results to './test_labels_aggregated.csv' and './all_runs_aggregated.csv'")

    # GradientBoostingClassifier training
    X = np.vstack(labels["feature_vector"].to_numpy())
    y = labels["aligner"].to_numpy()

    clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=SEED,
    )
    clf.fit(X, y)
    print("Classifier trained.")
    print(f"Feature importances: {clf.feature_importances_}")

    # save model
    with open("./aligner_time_predictor.pkl", "wb") as f:
        pickle.dump(clf, f)
