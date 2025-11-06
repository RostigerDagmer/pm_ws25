from typing import Optional, Any
import torch
from torch.utils.data import Dataset
from dataloaders.base import BaseEventLogDataset
from dataloaders.util import _normalize_log_input
from pm4py.pm4py.discovery import (
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
from tqdm import tqdm
from enum import Enum
import logging
from abc import ABC, abstractmethod
from pm4py.pm4py.objects.log.obj import EventLog
from collections.abc import Sequence

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


class Sampler(ABC):
    def __init__(self, n_subsets=1, name=None, seed=None):
        self.n_subsets = n_subsets
        self.name = name or self.__class__.__name__
        self.seed = seed

    @abstractmethod
    def sample_once(self, log, subset_idx: int):
        """Return a TraceSubset."""
        raise NotImplementedError

    def __call__(self, log):
        subsets = []
        # deterministic per (seed, subset_idx)
        for i in range(self.n_subsets):
            if self.seed is not None:
                random.seed((hash((self.seed, i)) & 0xFFFFFFFF))
            subsets.append(self.sample_once(log, i))
        return subsets


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


class ProcessModelDataset(Dataset):
    """
    Dataset that yields Petri nets discovered from event logs via different
    process discovery algorithms and parameter configurations.
    """

    def __init__(
        self,
        log_dataset: BaseEventLogDataset,
        discovery_methods: dict,
        param_grid: dict[str, list],
        sampler_specs=None,
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

        self.discovery_methods = discovery_methods
        self.param_grid = param_grid
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

    # --- helper: deterministic key per configuration ---
    def _config_hash(
        self,
        method_name: str,
        params: dict[str, Any],
        sampler_name: str,
        subset_idx: int,
        subset,
    ):
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
                        key
                    )

            for f in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Caching discovered models",
            ):
                key = futures[f]
                try:
                    f.result()
                    logging.debug("Cached model %s", key)
                except Exception as e:
                    logging.error("Failed to cache %s: %s", key, e)

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

    def __getitem__(self, idx: int):
        method_name, fn, params, sampler_name, subset_idx, subset = (
            self.configurations[idx]
        )
        key = self._config_hash(
            method_name, params, sampler_name, subset_idx, subset
        )
        path = self._cache_path(key)

        if self.cached and path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)

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
        return data


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

    EXTENSIVE = {
        "full": FullSampler(n_subsets=1),
        "random20": RandomSubsetSampler(frac=0.2, n_subsets=10, seed=42),
        "random50": RandomSubsetSampler(frac=0.5, n_subsets=5, seed=1337),
    }


def random_subset_sampler(log):
    n = len(log)
    indices = random.sample(range(n), k=n // 2)
    return [log[i] for i in indices]


if __name__ == "__main__":
    from dataloaders.base import make_feature_fn
    from dataloaders.csv import CSVEventLogDataset
    from pm4py.vis import view_petri_net

    print(DISCOVERY_METHODS.ALL.value)

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
        sampler_specs=SAMPLER_SPECS.EXTENSIVE,
        cached=True,
    )

    for item in pm_dataset:
        print(item)
        view_petri_net(
            item["pm"],
            item["im"],
            item["fm"],
        )
        break
