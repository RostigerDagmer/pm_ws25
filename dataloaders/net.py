from dataclasses import dataclass
from typing import Callable, Generator, Optional, Any, Union
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

from pm4py.objects.log.obj import EventLog, Trace
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

logging.getLogger(None)
logging.basicConfig(level=logging.INFO)


class SerializedView:
    """
    Generic serialized view that provides access to serialized items
    without full deserialization.

    Works with any dataset that provides a get_serialized function.
    """

    @dataclass
    class ItemType:
        pm: str
        variant: str
        parameters: list[float | int | bool]
        sampler: str
        subset_idx: int
        trace_indices: list[int] | None

        def hash(self) -> str:
            d = {
                str(k): str(v)
                for k, v in self.__dict__.items()
                if k != "pm"
            }
            return hashlib.sha1(
                json.dumps(d, sort_keys=True).encode()
            ).hexdigest()

        def deserialize(self) -> "ProcessModelDataset.ItemType":
            pm, im, fm = pnml_importer.deserialize(self.pm.decode('utf-8'))
            return ProcessModelDataset.ItemType(
                pm=pm,
                im=im,
                fm=fm,
                variant=self.variant,
                parameters=self.parameters,
                sampler=self.sampler,
                subset_idx=self.subset_idx,
                trace_indices=self.trace_indices,
            )

    def __init__(self, dataset, get_serialized_fn: Callable[[int], ItemType]):
        """
        Args:
            dataset: The dataset to create a view for
            get_serialized_fn: Function that takes an index and returns SerializedView.ItemType
        """
        self.dataset = dataset
        self.get_serialized_fn = get_serialized_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> ItemType:
        return self.get_serialized_fn(idx)

    def __iter__(self) -> Generator[ItemType, None, None]:
        for i in range(len(self.dataset)):
            yield self.get_serialized_fn(i)


class TraceSubset(Sequence):
    """Lightweight, list-like wrapper that carries index metadata."""

    def __init__(
        self,
        data: EventLog,
        indices: Optional[list[int] | torch.Tensor] = None,
    ):
        self._data = list(data)
        self.indices = indices  # may be None

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i: int):
        return self._data[i]

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        return str(self.indices)


class Sampler(ABC):
    def __init__(
        self,
        n_subsets: int = 1,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.n_subsets = n_subsets
        self.name = name or self.__class__.__name__
        self.seed = seed
        torch.manual_seed(self.seed or 42)
        self._cache = {}  # (id(log)) -> dict of computed features

    @abstractmethod
    def sample_once(self, log: EventLog, subset_idx: int):
        raise NotImplementedError

    def __call__(self, log: EventLog):
        subsets = []
        for i in range(self.n_subsets):
            if self.seed is not None:
                # derive deterministic 32-bit seed from (sampler_name, seed, subset_idx)
                key = f"{self.name}:{self.seed}:{i}".encode()
                seed_int = int(hashlib.sha1(key).hexdigest(), 16) % (2**32)
                random.seed(seed_int)
            subsets.append(self.sample_once(log, i))
        return subsets

    def get_cached(
        self, log: EventLog, key: int, compute_fn: Callable[[EventLog], Any]
    ):
        """Cache helper: compute once per-log per-key."""
        lid = id(log)
        if lid not in self._cache:
            self._cache[lid] = {}
        cache = self._cache[lid]
        if key not in cache:
            cache[key] = compute_fn(log)
        return cache[key]


class FullSampler(Sampler):
    def sample_once(self, log: EventLog, subset_idx: int):
        # keep indices=None for full log
        return TraceSubset(log, indices=None)


class VariantRandomDistributionSampler(Sampler):
    def __init__(
        self,
        n_subsets: int = 1000,  # defines how often the log is sampled... basically
        max_len_subset: int = 100,  # limits the possible length of each sample
        min_len_subset: int = 10,
        len_distribution: torch.distributions.Distribution = torch.distributions.Exponential(
            torch.tensor([1.0 / 100.0])
        ),  # defines the distribution of lengths across samples
        freq_distribution: torch.distributions.Distribution = torch.distributions.Normal(
            10.0, 5.0
        ),  # defines the reordering of traces by defining the sampling distribution over index(index) -> p(frequency).
        reconstruct_frequency: bool = True,  # toggle whether to reconstruct the frequency of variants in the sampled subset (repeat variants according to sampled frequency)
        **kwargs,
    ):
        super().__init__(n_subsets=n_subsets, **kwargs)
        self.max_len_subset = max_len_subset
        self.min_len_subset = min_len_subset
        self.len_distribution = len_distribution
        self.freq_distribution = freq_distribution
        self.reconstruct_frequency = reconstruct_frequency

    def sample_once(self, log: EventLog, subset_idx: int):
        def compute_variants(l: EventLog):
            variants: dict[str, Trace] = {
                ";".join(e["concept:name"] for e in t): t for t in l
            }
            return {"vars": variants}

        cached = self.get_cached(log, "variants", compute_variants)
        variants = list(cached["vars"].values())
        len_set = int(self.len_distribution.sample((1,))[0].item())
        rand_freq = self.freq_distribution.sample((len(variants),))

        # normal returns shape (), exponential returns shape (1,)
        if rand_freq.dim() > 1:
            rand_freq = rand_freq.squeeze()

        # clamp to positive
        rand_freq = torch.clamp(rand_freq, min=1.0)
        new_order = torch.argsort(rand_freq, descending=True)
        ord_freq = torch.sort(rand_freq, descending=True).values

        # reorder variants by sampled frequency
        idx = new_order[
            : max(min(len_set, self.max_len_subset), self.min_len_subset)
        ].tolist()

        # reconstruct frequency in sampled subset
        if not self.reconstruct_frequency:
            subset = [variants[i] for i in idx]
        else:
            subset = [
                [variants[i]] * int(f.item()) for i, f in zip(idx, ord_freq)
            ]
            subset = [trace for traces in subset for trace in traces]
        return TraceSubset(subset, indices=idx)


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
            d = {str(k): str(v) for k, v in self.__dict__.items() if k != "pm"}
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

    @property
    def serialized(self):
        """Access serialized view without deserialization."""
        return SerializedView(self, self._get_serialized)

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

    def _safe_discover(self, fn, log: EventLog, params: dict[str, Any]):
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

        serialized = pnml_exporter.serialize(net, im, fm)

        data = {
            "pm": serialized,
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

    # --- skips deserialization for faster access ---
    def _get_serialized(self, idx: int) -> SerializedView.ItemType:
        method_name, fn, params, sampler_name, subset_idx, subset = (
            self.configurations[idx]
        )
        key = self._config_hash(
            method_name, params, sampler_name, subset_idx, subset
        )
        path = self._cache_path(key)

        if self.cached and path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
                return SerializedView.ItemType(**data)

        subset = _normalize_log_input(subset)
        net, im, fm = self._safe_discover(fn, subset, params)
        serialized = pnml_exporter.serialize(net, im, fm)

        data = {
            "pm": serialized,
            "variant": method_name,
            "parameters": params,
            "sampler": sampler_name,
            "subset_idx": subset_idx,
            "trace_indices": getattr(subset, "indices", None),
        }
        if self.cached:
            with open(path, "wb") as f:
                pickle.dump(data, f)
        return SerializedView.ItemType(**data)

    def __getitem__(self, idx: int) -> "ProcessModelDataset.ItemType":
        serialized_item = self._get_serialized(idx)
        return serialized_item.deserialize()


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

    TINY = {
        "noise_threshold": [0.0, 0.1, 0.25, 0.4],
        "disable_fallthroughs": [True],
        "multi_processing": [False],
    }


class SAMPLER_SPECS(Enum):
    STANDARD = {
        "variant_random": VariantRandomDistributionSampler(
            n_subsets=1000,  # number of subsets: defines how often the log is sampled... basically
            max_len_subset=100,
            min_len_subset=10,  # max_length_subset: limits the possible length of each sample (what is fed to the discovery algorithm)
            len_distribution=torch.distributions.Exponential(
                torch.tensor([1.0 / 100.0])
            ),  # subset length distribution: defines the distribution of lengths across samples
            freq_distribution=torch.distributions.Normal(
                10.0, 5.0
            ),  # (variant) freq_distribution: defines the reordering of traces/variants on every sampling call, by defining the sampling behavior over index(variant) -> frequency.
            # realistically this would more likely be an exponential... but harder to parametrize... e.g.:
            # freq_distribution=torch.distributions.Exponential(
            #     torch.tensor([1.0 / 20.0])
            # )
            reconstruct_frequency=True,  # toggle whether to reconstruct the frequency of variants in the sampled subset (repeat variants according to sampled frequency)
        )
    }


if __name__ == "__main__":
    from dataloaders.xes_log import XESEventLogDataset
    from dataloaders.unique_net import UniqueProcessModelDataset
    from deduplication.deduplicator import DeduplicationConfig
    from pm4py.vis import view_petri_net

    path = "data/63a8435a-077d-4ece-97cd-2c76d394d99c/BPIC15_2.xes"

    log_dataset = XESEventLogDataset(path, attribute="concept:name")

    '''
    Example usage of the "Dedup by Variant, then define an arbitrary distribution over variants" approach.
    In the following "subset" is equivalent to "EventLog"... every subset is also an EventLog, however it is only a subset of the FULL eventlog that the FULL XES Dataset describes.
    '''

    # Create base dataset with caching enabled
    pm_dataset = ProcessModelDataset(
        log_dataset=log_dataset,
        discovery_methods={"inductive": discover_petri_net_inductive},
        param_grid={
            "noise_threshold": [0.0, 0.1, 0.2, 0.3],
            "disable_fallthroughs": [True],
        },
        sampler_specs={
            "variant_random": VariantRandomDistributionSampler(
                n_subsets=1000,  # number of subsets: defines how often the log is sampled... basically
                max_len_subset=100,
                min_len_subset=10,  # max_length_subset: limits the possible length of each sample (what is fed to the discovery algorithm)
                len_distribution=torch.distributions.Exponential(
                    torch.tensor([1.0 / 100.0])
                ),  # subset length distribution: defines the distribution of lengths across samples
                freq_distribution=torch.distributions.Normal(
                    10.0, 5.0
                ),  # (variant) freq_distribution: defines the reordering of traces/variants on every sampling call, by defining the sampling behavior over index(variant) -> frequency.
                # realistically this would more likely be an exponential... but harder to parametrize... e.g.:
                # freq_distribution=torch.distributions.Exponential(
                #     torch.tensor([1.0 / 20.0])
                # )
                reconstruct_frequency=True,  # toggle whether to reconstruct the frequency of variants in the sampled subset (repeat variants according to sampled frequency)
            )
        },
        cached=True,
    )

    # Wrap with deduplication
    unique_dataset = UniqueProcessModelDataset(
        base_dataset=pm_dataset,
        dedup_config=DeduplicationConfig()
    )

    for i, item in enumerate(unique_dataset):
        print(item)
        view_petri_net(
            item.pm,
            item.im,
            item.fm,
        )
        if i >= 10:
            break
