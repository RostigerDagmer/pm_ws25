from dataclasses import dataclass
from typing import Callable, Optional, Any, Union
import torch
from torch.utils.data import Dataset
from dataloaders.base import BaseEventLogDataset
from dataloaders.util import _normalize_log_input
from pm4py.discovery import (
    discover_petri_net_alpha,
    discover_petri_net_alpha_plus,
    discover_petri_net_heuristics,
    discover_petri_net_ilp,
    discover_petri_net_inductive,
)
from itertools import product
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import hashlib
import json
import inspect
import random
import os
import pandas as pd
from tqdm import tqdm
from enum import Enum
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from collections import defaultdict, Counter

from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import Marking, PetriNet


logging.getLogger(None)
logging.basicConfig(level=logging.INFO)


class TraceSubset(Sequence):
    """Lightweight, list-like wrapper that carries index metadata."""

    def __init__(self, data, indices=None):
        self._data = list(data)
        self.indices = indices  # may be None

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        return str(self.indices)


class Sampler(ABC):
    def __init__(self, n_subsets=1, name=None, seed=None):
        self.n_subsets = n_subsets
        self.name = name or self.__class__.__name__
        self.seed = seed
        torch.manual_seed(self.seed or 42)
        self._cache = {}  # (id(log)) -> dict of computed features

    @abstractmethod
    def sample_once(self, log, subset_idx: int):
        raise NotImplementedError

    def __call__(self, log):
        subsets = []
        for i in range(self.n_subsets):
            if self.seed is not None:
                # derive deterministic 32-bit seed from (sampler_name, seed, subset_idx)
                key = f"{self.name}:{self.seed}:{i}".encode()
                seed_int = int(hashlib.sha1(key).hexdigest(), 16) % (2**32)
                random.seed(seed_int)
            subsets.append(self.sample_once(log, i))
        return subsets

    def get_cached(self, log, key, compute_fn):
        """Cache helper: compute once per-log per-key."""
        lid = id(log)
        if lid not in self._cache:
            self._cache[lid] = {}
        cache = self._cache[lid]
        if key not in cache:
            cache[key] = compute_fn(log)
        return cache[key]


class FullSampler(Sampler):
    def sample_once(self, log, subset_idx):
        # keep indices=None for full log
        return TraceSubset(log, indices=None)


class RandomSubsetSampler(Sampler):
    def __init__(self, frac=0.5, **kwargs):
        super().__init__(**kwargs)
        if not (0.0 < frac <= 1.0):
            raise ValueError("frac must be in (0, 1]")
        self.frac = frac

    def sample_once(self, log, subset_idx):
        n = len(log)
        k = max(1, int(round(self.frac * n)))
        idx = random.sample(range(n), k=k)
        return TraceSubset((log[i] for i in idx), indices=idx)


class LengthStratifiedSampler(Sampler):
    def __init__(self, n_bins=3, **kwargs):
        super().__init__(n_subsets=n_bins, **kwargs)

    def sample_once(self, log, subset_idx):
        import torch

        lengths = self.get_cached(
            log,
            "lengths",
            lambda l: torch.tensor([len(t) for t in l], dtype=torch.float32),
        )
        quantiles = self.get_cached(
            log,
            f"length_bins_{self.n_subsets}",
            lambda l: torch.quantile(
                lengths, torch.linspace(0, 1, self.n_subsets + 1)
            ),
        )

        lo, hi = quantiles[subset_idx], quantiles[subset_idx + 1]
        mask = (lengths >= lo) & (lengths <= hi)
        idx = torch.nonzero(mask).view(-1).tolist()
        subset = [log[i] for i in idx]
        return TraceSubset(subset, indices=idx)


class TemporalDriftSampler(Sampler):
    def __init__(self, n_bins=3, **kwargs):
        super().__init__(n_subsets=n_bins, **kwargs)

    def sample_once(self, log, subset_idx):

        def compute_medians(l):
            medians = []
            for trace in l:
                ts = [
                    e.get("time:timestamp")
                    for e in trace
                    if "time:timestamp" in e
                ]
                if not ts:
                    medians.append(float("nan"))
                    continue
                tvals = pd.to_datetime(ts).view("int64")
                medians.append(
                    float(
                        torch.median(torch.tensor(tvals, dtype=torch.float64))
                    )
                )
            return torch.tensor(medians, dtype=torch.float64)

        medians = self.get_cached(log, "timestamp_medians", compute_medians)
        valid_mask = ~torch.isnan(medians)
        valid = torch.nonzero(valid_mask).view(-1)

        if len(valid) == 0:
            raise ValueError("No valid timestamps in log.")

        quantiles = self.get_cached(
            log,
            f"time_bins_{self.n_subsets}",
            lambda l: torch.quantile(
                medians[valid],
                torch.linspace(0, 1, self.n_subsets + 1, dtype=torch.float64),
            ),
        )

        lo, hi = quantiles[subset_idx], quantiles[subset_idx + 1]
        idx = (
            (valid_mask & (medians >= lo) & (medians <= hi))
            .nonzero()
            .view(-1)
            .tolist()
        )
        subset = [log[i] for i in idx]
        return TraceSubset(subset, indices=idx)


class VariantFrequencySampler(Sampler):
    def __init__(self, n_bins=3, **kwargs):
        super().__init__(n_subsets=n_bins, **kwargs)

    def sample_once(self, log, subset_idx):

        def compute_variant_freqs(l):
            variants = [";".join(e["concept:name"] for e in t) for t in l]
            freq = Counter(variants)
            freqs = torch.tensor(
                [freq[v] for v in variants], dtype=torch.float32
            )
            order = torch.argsort(freqs, descending=True)
            return {"freqs": freqs, "order": order}

        cached = self.get_cached(log, "variant_freqs", compute_variant_freqs)
        order = cached["order"]
        n = len(order)
        bin_size = max(1, n // self.n_subsets)
        start, end = subset_idx * bin_size, min((subset_idx + 1) * bin_size, n)
        idx = order[start:end].tolist()
        subset = [log[i] for i in idx]
        return TraceSubset(subset, indices=idx)


class ActivitySetSampler(Sampler):
    """
    Groups traces by their (unordered) set of activities.
    Each subset corresponds to one of the largest activity-set clusters.
    """

    def __init__(self, max_groups=5, **kwargs):
        super().__init__(n_subsets=max_groups, **kwargs)
        self.max_groups = max_groups

    def sample_once(self, log, subset_idx):
        def compute_activity_groups(l):
            # Build mapping activity_set -> list of (index, trace)
            groups = defaultdict(list)
            for i, trace in enumerate(l):
                acts = frozenset(
                    e["concept:name"] for e in trace if "concept:name" in e
                )
                groups[acts].append((i, trace))
            # Sort groups largest → smallest
            sorted_groups = sorted(groups.values(), key=len, reverse=True)
            return sorted_groups

        # Cache the grouping per-log
        sorted_groups = self.get_cached(
            log, "activity_groups", compute_activity_groups
        )
        if not sorted_groups:
            raise ValueError(
                "No activity groups could be formed (empty log?)."
            )

        # Wrap around if fewer groups than requested subsets
        g = sorted_groups[subset_idx % len(sorted_groups)]
        idx, traces = zip(*g)
        return TraceSubset(traces, indices=list(idx))


class AttributeClusterSampler(Sampler):
    def __init__(self, attribute, n_subsets=None, **kwargs):
        super().__init__(n_subsets=n_subsets or 3, **kwargs)
        self.attribute = attribute

    def sample_once(self, log, subset_idx):
        from collections import defaultdict

        groups = defaultdict(list)
        for i, trace in enumerate(log):
            val = trace.attributes.get(self.attribute, None)
            groups[val].append((i, trace))

        group_keys = list(groups.keys())
        if not group_keys:
            raise ValueError(
                f"Attribute '{self.attribute}' not found in any trace."
            )

        # cycle if more requested than existing
        key = group_keys[subset_idx % len(group_keys)]
        idx, traces = zip(*groups[key])
        return TraceSubset(traces, indices=idx)


class ProcessModelDataset(Dataset):
    """
    Dataset that yields Petri nets discovered from event logs via different
    process discovery algorithms and parameter configurations.
    """

    @dataclass
    class ItemType:
        pm: PetriNet
        im: Marking
        fm: Marking
        variant: str
        parameters: list[float | int | bool]
        sampler: str
        subset_idx: int
        trace_indices: list[int] | None

        def hash(self) -> str:
            d = {str(k): str(v) for k, v in self.__dict__.items()}
            return hashlib.sha1(
                json.dumps(d, sort_keys=True).encode()
            ).hexdigest()

    def __init__(
        self,
        log_dataset: BaseEventLogDataset,
        discovery_methods: Union[
            dict[str, Callable[[Any], tuple[PetriNet, Marking, Marking]]],
            "DISCOVERY_METHODS",
        ],
        param_grid: Union[dict[str, list[float | int | bool]], "PARAM_GRID"],
        sampler_specs: Union[
            dict[str, Callable[[Any], list[TraceSubset]]], "SAMPLER_SPECS"
        ],
        max_models=None,
        cached=False,
        cache_dir=None,
        num_workers=None,
        **kwargs,
    ):
        """
        Args:
            log_dataset (BaseEventLogDataset): Dataset of traces or a pm4py EventLog.
            discovery_methods (dict): Mapping of method name to pm4py discovery function.
                e.g. {"inductive": pm4py.discover_petri_net_inductive, "heuristic": pm4py.discover_petri_net_heuristics}
            param_grid (dict[str, list]): Dict of parameter names to list of values to sweep.
                e.g. {"noise_threshold": [0.0, 0.2, 0.5]}
            sampler_fn (Callable): Optional function controlling how to sample subsets of traces.
            max_models (int): Optional limit on total number of discovered models.
        """
        self.log = getattr(log_dataset, "log", log_dataset)

        if hasattr(discovery_methods, "value"):
            discovery_methods = discovery_methods.value
        if hasattr(param_grid, "value"):
            param_grid = param_grid.value
        if hasattr(sampler_specs, "value"):
            sampler_specs = sampler_specs.value

        self.discovery_methods: dict[
            str, Callable[[Any], tuple[PetriNet, Marking, Marking]]
        ] = discovery_methods
        self.param_grid: dict[str, list[float | int | bool]] = param_grid
        self.sampler_specs = sampler_specs or {
            "full": FullSampler(n_subsets=1)
        }
        self.max_models = max_models
        self.cached = cached
        self.num_workers = num_workers or os.cpu_count()

        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path('/'.join(log_dataset.source_path.split('/')[:-1]))
            / Path(".cache_process_models")
        )

        self.configurations = self._generate_configurations()

        if self.cached:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._populate_cache_parallel()

    def hash(self) -> str:
        base = {
            "methods": list(self.discovery_methods.keys()),
            "param_grid": self.param_grid,
            "sampler_specs": list(self.sampler_specs.keys()),
        }
        return hashlib.sha1(
            json.dumps(base, sort_keys=True).encode()
        ).hexdigest()

    # --- helper: deterministic key per configuration ---
    def _config_hash(
        self,
        method_name: str,
        params: dict[str, Any],
        sampler_name: str,
        subset_idx: int,
        subset: TraceSubset,
    ) -> str:
        base = {
            "method": method_name,
            "params": params,
            "sampler": sampler_name,
            "subset_idx": subset_idx,
            "indices": getattr(subset, "indices", None),
        }
        return hashlib.sha1(
            json.dumps(base, sort_keys=True).encode()
        ).hexdigest()

    def _cache_path(self, key):
        return self.cache_dir / f"{key}.pkl"

    # --- caching logic ---
    def _populate_cache_parallel(self):
        logging.info("Populating process model cache...")
        with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {}
            for cfg in self.configurations:
                method_name, fn, params, sampler_name, subset_idx, subset = cfg
                key = self._config_hash(
                    method_name, params, sampler_name, subset_idx, subset
                )
                path = self._cache_path(key)
                if not path.exists():
                    futures[pool.submit(self._discover_and_save, cfg, key)] = (
                        key,
                        cfg,
                    )

            for f in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Caching discovered models",
            ):
                key, cfg = futures[f]
                try:
                    f.result()
                    logging.debug("Cached model %s", key)
                except Exception as e:
                    logging.error("Failed to cache %s: %s", key, e, cfg)

        logging.info(
            "Cache population done (%d models).",
            len(os.listdir(self.cache_dir)),
        )

    def _generate_configurations(self):
        configs = []
        seen = set()

        # precompute sampler outputs once (they only depend on the log)
        precomputed_subsets = {
            s_name: sampler(self.log)
            for s_name, sampler in self.sampler_specs.items()
        }

        for method_name, fn in self.discovery_methods.items():
            # pick method-specific grid when present, else global grid
            method_grid = (
                self.param_grid.get(method_name)
                if isinstance(self.param_grid.get(method_name), dict)
                else self.param_grid
            )

            sig = inspect.signature(fn)
            sig_params = set(sig.parameters.keys())
            accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )

            if method_grid:
                relevant_grid = (
                    method_grid
                    if accepts_kwargs
                    else {
                        k: v for k, v in method_grid.items() if k in sig_params
                    }
                )
            else:
                relevant_grid = {}

            # build param combos
            if not relevant_grid:
                combos = [dict()]
            else:
                keys = list(relevant_grid.keys())
                lists = [
                    (
                        list(v)
                        if hasattr(v, "__iter__")
                        and not isinstance(v, (str, bytes))
                        else [v]
                    )
                    for v in (relevant_grid[k] for k in keys)
                ]
                combos = [dict(zip(keys, vals)) for vals in product(*lists)]

            for params in combos:
                params_key = tuple(sorted(params.items()))
                for sampler_name, subsets in precomputed_subsets.items():
                    for subset_idx, subset in enumerate(subsets):
                        key = (
                            method_name,
                            sampler_name,
                            subset_idx,
                            params_key,
                        )
                        logging.debug(key)
                        if key in seen:
                            continue
                        seen.add(key)
                        configs.append(
                            (
                                method_name,
                                fn,
                                params,
                                sampler_name,
                                subset_idx,
                                subset,
                            )
                        )

        if self.max_models:
            configs = configs[: self.max_models]

        logging.info(
            "Total discovery configurations generated: %d", len(configs)
        )
        return configs

    def _default_sampler(self, log):
        """Default subset sampler (full log)."""
        return log

    def _safe_discover(self, fn, log, params):
        """
        Call a pm4py discovery function with only the supported keyword arguments.
        """
        sig = inspect.signature(fn)
        valid_keys = sig.parameters.keys()
        filtered = {k: v for k, v in params.items() if k in valid_keys}
        return fn(log, **filtered)

    def _discover_and_save(self, cfg, key):
        method_name, fn, params, sampler_name, subset_idx, subset = cfg
        print("subset:", subset)
        subset = _normalize_log_input(subset)
        net, im, fm = self._safe_discover(fn, subset, params)
        data = {
            "pm": net,
            "im": im,
            "fm": fm,
            "variant": method_name,
            "parameters": params,
            "sampler": sampler_name,
            "subset_idx": subset_idx,
            "trace_indices": getattr(subset, "indices", None),
        }
        with open(self._cache_path(key), "wb") as f:
            pickle.dump(data, f)

    def __len__(self):
        return len(self.configurations)

    def __getitem__(self, idx: int) -> "ProcessModelDataset.ItemType":
        method_name, fn, params, sampler_name, subset_idx, subset = (
            self.configurations[idx]
        )
        key = self._config_hash(
            method_name, params, sampler_name, subset_idx, subset
        )
        path = self._cache_path(key)

        if self.cached and path.exists():
            with open(path, "rb") as f:
                return ProcessModelDataset.ItemType(**pickle.load(f))

        subset = _normalize_log_input(subset)
        net, im, fm = self._safe_discover(fn, subset, params)
        data = {
            "pm": net,
            "im": im,
            "fm": fm,
            "variant": method_name,
            "parameters": params,
            "sampler": sampler_name,
            "subset_idx": subset_idx,
            "trace_indices": getattr(subset, "indices", None),
        }
        if self.cached:
            with open(path, "wb") as f:
                pickle.dump(data, f)
        return ProcessModelDataset.ItemType(**data)


class DISCOVERY_METHODS(Enum):
    ALL = {
        "inductive": discover_petri_net_inductive,
        "heuristic": discover_petri_net_heuristics,
        "alpha": discover_petri_net_alpha,
        "alpha_plus": discover_petri_net_alpha_plus,
        "ilp": discover_petri_net_ilp,
    }

    GURANTEED_SOUND = {
        "inductive": discover_petri_net_inductive,
        "ilp": discover_petri_net_ilp,
    }

    PROBABLY_SOUND = {
        "inductive": discover_petri_net_inductive,
        "ilp": discover_petri_net_ilp,
        "heuristic": discover_petri_net_heuristics,
    }


class PARAM_GRID(Enum):
    STANDARD = {
        # Global sweeps (apply to all where relevant)
        "noise_threshold": [0.0, 0.1, 0.2, 0.3, 0.5],
        "dependency_threshold": [0.5, 0.6, 0.7, 0.8],
        "and_threshold": [0.6, 0.7, 0.8],
        "loop_two_threshold": [0.3, 0.5, 0.7],
        "alpha": [0.5, 0.8, 1.0, 1.2],
        "disable_fallthroughs": [False, True],
        "multi_processing": [False],
    }

    EXTENSIVE = {
        "noise_threshold": torch.linspace(0.0, 0.8, 9).tolist(),  # finer sweep
        "dependency_threshold": torch.linspace(0.4, 0.9, 6).tolist(),
        "and_threshold": torch.linspace(0.5, 0.9, 5).tolist(),
        "loop_two_threshold": torch.linspace(0.2, 0.8, 7).tolist(),
        "alpha": torch.linspace(0.5, 1.5, 6).tolist(),
        "disable_fallthroughs": [False, True],
        "multi_processing": [False],
    }


class SAMPLER_SPECS(Enum):
    STANDARD = {
        "full": FullSampler(n_subsets=1),
        "random20": RandomSubsetSampler(frac=0.2, n_subsets=3, seed=42),
    }

    EXTENSIVE = (
        {
            "full": FullSampler(n_subsets=1),
            "random20": RandomSubsetSampler(frac=0.2, n_subsets=10, seed=42),
            "random50": RandomSubsetSampler(frac=0.5, n_subsets=5, seed=1337),
            "activity_set5": ActivitySetSampler(max_groups=5),
            "activity_set15": ActivitySetSampler(max_groups=15),
            "activity_set25": ActivitySetSampler(max_groups=25),
            "temporal_drift3": TemporalDriftSampler(n_bins=3),
            "temporal_drift7": TemporalDriftSampler(n_bins=7),
            "temporal_drift15": TemporalDriftSampler(n_bins=15),
            "length_strat3": LengthStratifiedSampler(n_bins=3),
            "length_strat5": LengthStratifiedSampler(n_bins=3),
            "variant3": VariantFrequencySampler(n_bins=3),
            "variant7": VariantFrequencySampler(n_bins=7),
        },
    )

    SUBSETS_ONLY = {
        "activity_set5": ActivitySetSampler(max_groups=5),
        "activity_set15": ActivitySetSampler(max_groups=15),
        "activity_set25": ActivitySetSampler(max_groups=25),
        "temporal_drift3": TemporalDriftSampler(n_bins=3),
        "temporal_drift7": TemporalDriftSampler(n_bins=7),
        "temporal_drift15": TemporalDriftSampler(n_bins=15),
        "length_strat3": LengthStratifiedSampler(n_bins=3),
        "length_strat5": LengthStratifiedSampler(n_bins=3),
        "variant3": VariantFrequencySampler(n_bins=3),
        "variant7": VariantFrequencySampler(n_bins=7),
    }


def random_subset_sampler(log):
    n = len(log)
    indices = random.sample(range(n), k=n // 2)
    return [log[i] for i in indices]


if __name__ == "__main__":
    from dataloaders.base import make_feature_fn
    from dataloaders.csv_log import CSVEventLogDataset
    from dataloaders.xes_log import XESEventLogDataset
    from pm4py.vis import view_petri_net

    # path = "data/c3f3ba2d-e81e-4274-87c7-882fa1dbab0d/BPI2016_Werkmap_Messages.csv"
    path = "data/63a8435a-077d-4ece-97cd-2c76d394d99c/BPIC15_2.xes"

    log_dataset = XESEventLogDataset(
        path, attribute="concept:name", feature_fn=make_feature_fn
    )

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
        sampler_specs=SAMPLER_SPECS.SUBSETS_ONLY,
        cached=True,
    )

    for item in pm_dataset:
        print(item)
        view_petri_net(
            item.pm,
            item.im,
            item.fm,
        )
        break

    # subset = [190, 191, 317, 325, 327, 334, 346, 352, 355, 358, 361, 362, 365, 366, 369, 370, 371, 372, 373, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476]
    # print(len(subset))
    # trace_subset = EventLog([log_dataset.log[i] for i in subset])
    # print(type(trace_subset))
    # print(type(trace_subset[0]))
    # print(type(trace_subset[0][0]))
    # print(trace_subset)
    # pm = discover_petri_net_inductive(trace_subset)
    # print(pm)
    # pickle.dump({"pm": pm[0], "im": pm[1], "fm": pm[2]}, open("test.pkl", "wb"))
    # view_petri_net(*pm)
