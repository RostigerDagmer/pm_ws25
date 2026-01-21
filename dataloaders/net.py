import pebble
from util.distributions import deserialize
from dataclasses import asdict
from concurrent.futures import as_completed
from dataclasses import dataclass
from typing import Callable, Optional, Any, Union
import torch
from torch.utils.data import Dataset
from dataloaders.base import BaseEventLogDataset, _normalize_log_input
from pm4py.discovery import (
    discover_petri_net_alpha,
    discover_petri_net_alpha_plus,
    discover_petri_net_heuristics,
    discover_petri_net_ilp,
    discover_petri_net_inductive,
)
from pm4py.objects.petri_net.utils.check_soundness import (
    check_wfnet,
    check_easy_soundness_net_in_fin_marking,
)
from itertools import product
from pathlib import Path
import pickle

import hashlib
import json
import inspect
import os
import pandas as pd
from tqdm import tqdm
from enum import Enum
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataloaders.serializable import (
    WithSerializedView,
    Serializable,
    Deserializable,
)
from io import BufferedWriter

from pm4py.objects.log.obj import EventLog, Trace
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
import traceback
import gc

from util.rng import RNG
from util.distributions import (
    make_distribution,
    ExponentialSpec,
    NormalSpec,
    DistParam,
)


class TraceSubset(Sequence[Trace]):
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
        seed: int,
        n_subsets: int = 1,
        name: Optional[str] = None,
    ):
        self.n_subsets = n_subsets
        self.name = name or str(self.__class__.__name__)
        self.seed = seed
        self._cache = {}  # (id(log)) -> dict of computed features

    @abstractmethod
    def sample_once(self, log: EventLog, subset_idx: int) -> TraceSubset:
        raise NotImplementedError

    def __call__(self, log: EventLog):
        subsets = []
        for i in range(self.n_subsets):
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

    def serialize(self):
        return {
            "name": self.name,
            "seed": self.seed,
            "n_subsets": self.n_subsets,
        }

    @abstractmethod
    def deserialize(data):
        raise NotImplementedError

    def hash(self):
        return hashlib.sha1(
            json.dumps(self.serialize(), sort_keys=True).encode()
        ).hexdigest()


class FullSampler(Sampler):
    def __init__(self, seed: int, n_subsets: int = 1, **kwargs):
        super().__init__(seed=seed, n_subsets=n_subsets, **kwargs)

    def sample_once(self, log: EventLog, subset_idx: int) -> TraceSubset:
        # keep indices=None for full log
        return TraceSubset(log, indices=None)

    @staticmethod
    def deserialize(data):
        return FullSampler(seed=data["seed"], n_subsets=data["n_subsets"])


class VariantRandomDistributionSampler(Sampler):
    def __init__(
        self,
        seed: int,
        n_subsets: int = 1000,  # defines how often the log is sampled... basically
        max_len_subset: int = 100,  # limits the possible length of each sample
        min_len_subset: int = 10,
        len_distribution: DistParam = ExponentialSpec(
            torch.tensor([1.0 / 100.0])
        ),  # defines the distribution of lengths across samples
        freq_distribution: DistParam = NormalSpec(
            10.0, 5.0
        ),  # defines the reordering of traces by defining the sampling distribution over index(index) -> p(frequency).
        reconstruct_frequency: bool = True,  # toggle whether to reconstruct the frequency of variants in the sampled subset (repeat variants according to sampled frequency)
        **kwargs,
    ):
        super().__init__(seed=seed, n_subsets=n_subsets, **kwargs)
        self.max_len_subset = max_len_subset
        self.min_len_subset = min_len_subset
        self.len_spec = len_distribution
        self.freq_spec = freq_distribution
        self.len_distribution = make_distribution(len_distribution)
        self.freq_distribution = make_distribution(freq_distribution)
        self.reconstruct_frequency = reconstruct_frequency

    def serialize(self):
        return {
            "name": self.name,
            "seed": self.seed,
            "n_subsets": self.n_subsets,
            "max_len_subset": self.max_len_subset,
            "min_len_subset": self.min_len_subset,
            "len_spec": {
                "type": str(self.len_spec.__class__.__name__),
                "args": asdict(self.len_spec),
            },
            "freq_spec": {
                "type": str(self.freq_spec.__class__.__name__),
                "args": asdict(self.freq_spec),
            },
            "reconstruct_frequency": self.reconstruct_frequency,
        }

    @staticmethod
    def deserialize(data):
        return VariantRandomDistributionSampler(
            seed=data["seed"],
            n_subsets=data["n_subsets"],
            max_len_subset=data["max_len_subset"],
            min_len_subset=data["min_len_subset"],
            len_distribution=deserialize(data["len_spec"]),
            freq_distribution=deserialize(data["freq_spec"]),
            reconstruct_frequency=data["reconstruct_frequency"],
        )

    def hash(self):
        return hashlib.sha1(
            json.dumps(self.serialize(), sort_keys=True).encode()
        ).hexdigest()

    def sample_once(self, log: EventLog, subset_idx: int):
        def compute_variants(l: EventLog):
            variants: dict[str, Trace] = {
                ";".join(e["concept:name"] for e in t): t for t in l
            }
            return {"vars": variants}

        cached = self.get_cached(log, "variants", compute_variants)
        variants = list(cached["vars"].values())
        len_set = int(
            self.len_distribution.sample(
                generator=torch.Generator().manual_seed(self.seed + subset_idx)
            )[0].item()
        )
        rand_freq = self.freq_distribution.sample(
            (len(variants),),
            generator=torch.Generator().manual_seed(self.seed + subset_idx),
        )

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


def deserialize_sampler(data) -> Sampler:
    match data["name"]:
        case "FullSampler":
            return FullSampler.deserialize(data)
        case "VariantRandomDistributionSampler":
            return VariantRandomDistributionSampler.deserialize(data)
        case _:
            raise ValueError(f"Unknown sampler type: {data['name']}")


@dataclass
class ItemType(Serializable["SerializedItemType"]):
    pm: PetriNet
    im: Marking
    fm: Marking
    variant: str
    parameters: list[float | int | bool]
    sampler: dict[str, Any]
    subset_idx: int
    trace_indices: list[int] | None

    def hash(self) -> str:
        if not isinstance(self.sampler, dict):
            self.sampler = self.sampler.serialize()
        return ProcessModelDataset._config_hash(
            self.variant,
            self.parameters,
            self.sampler,
            self.subset_idx,
            self.trace_indices,
        )

    def serialize(self) -> "SerializedItemType":
        serialized: str = pnml_exporter.serialize(self.pm, self.im, self.fm)
        return SerializedItemType(
            pm=serialized,
            variant=self.variant,
            parameters=self.parameters,
            sampler=self.sampler,
            subset_idx=self.subset_idx,
            trace_indices=self.trace_indices,
        )


@dataclass
class SerializedItemType(Deserializable[ItemType]):
    pm: bytes  # serialized PNML
    variant: str
    parameters: list[float | int | bool]
    sampler: dict[str, Any]
    subset_idx: int
    trace_indices: list[int] | None

    def hash(self) -> str:
        return ProcessModelDataset._config_hash(
            self.variant,
            self.parameters,
            self.sampler,
            self.subset_idx,
            self.trace_indices,
        )

    def deserialize(self) -> ItemType:
        pm, im, fm = pnml_importer.deserialize(self.pm.decode('utf-8'))
        return ItemType(
            pm=pm,
            im=im,
            fm=fm,
            variant=self.variant,
            parameters=self.parameters,
            sampler=self.sampler,
            subset_idx=self.subset_idx,
            trace_indices=self.trace_indices,
        )


class ProcessModelDataset(
    Dataset[ItemType], WithSerializedView[ItemType, SerializedItemType]
):
    """
    Dataset that yields Petri nets discovered from event logs via different
    process discovery algorithms and parameter configurations.
    """

    def __init__(
        self,
        log_dataset: BaseEventLogDataset,
        discovery_methods: Union[
            dict[str, Callable[[Any], tuple[PetriNet, Marking, Marking]]],
            "DISCOVERY_METHODS",
        ],
        param_grid: Union[dict[str, list[float | int | bool]], "PARAM_GRID"],
        sampler_specs: dict[str, Sampler],
        max_models=None,
        cached=False,
        cache_dir=None,
        num_workers=None,
        timeout: float = 300.0,
        write_batch_size: int = 100,
        filter_unsound: bool = True,
        skip_init: bool = False,
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
        super().__init__()
        self.log = getattr(log_dataset, "log", log_dataset)
        self.log_uuid = log_dataset.log_uuid

        if hasattr(discovery_methods, "value"):
            discovery_methods = discovery_methods.value
        if hasattr(param_grid, "value"):
            param_grid = param_grid.value

        self.discovery_methods: dict[
            str, Callable[[Any], tuple[PetriNet, Marking, Marking]]
        ] = discovery_methods
        self.param_grid: dict[str, list[float | int | bool]] = param_grid
        if sampler_specs is None:
            raise ValueError("sampler_specs must be provided.")
        self.sampler_specs = sampler_specs
        self.max_models = max_models
        self.write_batch_size = write_batch_size
        self.cached = cached
        self.num_workers = num_workers or os.cpu_count()
        self.timeout = timeout
        self.filter_unsound = filter_unsound
        self.skip_init = skip_init
        if cached:
            self.cache_dir = (
                Path(cache_dir)
                if cache_dir
                else Path('./cache/.cache_process_models')
            )
        self.items = {}
        self.configurations = {}
        self._generate_configurations()

        if self.cached:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._populate_cache_parallel()

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # --- helper: deterministic key per configuration ---
    @staticmethod
    def _config_hash(
        variant: str,
        params: dict[str, Any],
        sampler: dict[str, Any],
        subset_idx: int,
        subset: TraceSubset | list[int],
    ) -> str:
        base = {
            "variant": variant,
            "params": params,
            "sampler": sampler,
            "subset_idx": subset_idx,
            "indices": (
                subset.indices if isinstance(subset, TraceSubset) else subset
            ),
        }
        return hashlib.sha1(
            json.dumps(base, sort_keys=True).encode()
        ).hexdigest()

    def save_path(self):
        return self.cache_dir / f"{self.log_uuid}.pkl"

    def _tryload(self):
        path = self.save_path()
        if not os.path.exists(path):
            return
        self.items = {}

        with open(path, "rb") as f:
            while True:
                try:
                    chunk = pickle.load(f)  # each chunk = dict of N items
                    self.items.update(chunk)
                except EOFError:
                    break
        self.index = list(self.items.keys())
        logging.info("Loaded %d models from cache.", len(self.items))

    @staticmethod
    def _flush_batch(
        f: BufferedWriter,
        batch: dict[str, "ProcessModelDataset.SerializedItemType"],
    ):
        pickle.dump(batch, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())

    def _populate_cache_parallel(self):
        self._tryload()
        logging.info("Populating process model cache...")
        os.makedirs(self.save_path().parent, exist_ok=True)
        with (
            pebble.ProcessPool(max_workers=self.num_workers) as pool,
            open(self.save_path(), "ab") as f,
        ):
            new_items = {}
            futures = {}
            batch = {}
            seen_hashes = set(self.items.keys())
            scheduled = tqdm(total=len(self.configurations), desc="Scheduled")
            discovered = tqdm(total=0, desc="Discovered")
            written = tqdm(total=0, desc="Written")
            irrelevant = set(self.items.keys())
            for key, cfg in self.configurations.items():
                (
                    method_name,
                    fn,
                    params,
                    sampler_serialized,
                    subset_idx,
                    subset,
                ) = cfg
                try:
                    irrelevant.remove(key)
                except KeyError:
                    pass
                if key in seen_hashes:
                    scheduled.total -= 1
                    scheduled.update(0)
                    continue

                seen_hashes.add(key)
                if self.skip_init:
                    continue
                fut = pool.schedule(
                    ProcessModelDataset._process_item,
                    args=(cfg, self.filter_unsound),
                    timeout=self.timeout,
                )
                futures[fut] = (
                    key,
                    cfg,
                )
                discovered.total += 1
                written.total += 1
                scheduled.update(1)

            scheduled.close()
            for fut in as_completed(futures):
                key, cfg = futures[fut]
                try:
                    item = fut.result()
                    if item is None:
                        discovered.total -= 1
                        written.total -= 1
                        discovered.update(0)
                        written.update(0)
                        continue
                    assert (
                        item.hash() == key
                    ), f"Hash mismatch: {item.hash()} != {key}: {item}"
                    new_items[key] = item
                    batch[key] = item
                    discovered.update(1)
                except Exception as e:
                    traceback.print_exc()
                    logging.error(
                        "Failed to cache %s: %s (config: %s)", key, e, cfg
                    )
                if len(batch) >= self.write_batch_size:
                    ProcessModelDataset._flush_batch(f, batch)
                    written.update(len(batch))
                    batch.clear()
            if batch:
                ProcessModelDataset._flush_batch(f, batch)
                written.update(len(batch))
                batch.clear()
            discovered.close()
            written.close()

        self.items = {
            k: v for k, v in self.items.items() if k not in irrelevant
        }
        self.items.update(new_items)
        self.index = list(self.items.keys())
        gc.collect()
        logging.info(
            "Cache population done (%d models).",
            len(self.items),
        )

    def _generate_configurations(self) -> None:

        # precompute sampler outputs once (they only depend on the log)
        precomputed_subsets = {
            s_name: sampler(self.log)
            for s_name, sampler in self.sampler_specs.items()
        }

        precomputed_samplers = {
            s_name: sampler for s_name, sampler in self.sampler_specs.items()
        }
        num_models = 0
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
                for sampler_name, subsets in precomputed_subsets.items():
                    for subset_idx, subset in enumerate(subsets):
                        if (
                            self.max_models is not None
                            and num_models >= self.max_models
                        ):
                            break
                        sampler_serialized = precomputed_samplers[
                            sampler_name
                        ].serialize()
                        key = ProcessModelDataset._config_hash(
                            method_name,
                            params,
                            sampler_serialized,
                            subset_idx,
                            subset,
                        )
                        logging.debug(key)
                        if key in self.configurations:
                            continue
                        num_models += 1
                        self.configurations[key] = (
                            method_name,
                            fn,
                            params,
                            sampler_serialized,
                            subset_idx,
                            subset,
                        )

        logging.info(
            "Total discovery configurations generated: %d",
            len(self.configurations),
        )

    @staticmethod
    def _safe_discover(
        fn: Callable[..., tuple[PetriNet, Marking, Marking]],
        log: EventLog | Trace | pd.DataFrame,
        params: dict[str, Any],
    ) -> tuple[PetriNet, Marking, Marking]:
        """
        Call a pm4py discovery function with only the supported keyword arguments.
        """
        sig = inspect.signature(fn)
        valid_keys = sig.parameters.keys()
        filtered = {k: v for k, v in params.items() if k in valid_keys}
        return fn(log, **filtered)

    @staticmethod
    def _process_item(
        cfg: tuple[
            str,
            Callable[..., tuple[PetriNet, Marking, Marking]],
            dict[str, Any],
            str,
            int,
            EventLog | Trace | pd.DataFrame,
        ],
        filter_unsound: bool = True,
    ) -> Optional[SerializedItemType]:
        method_name, fn, params, sampler, subset_idx, subset = cfg
        subset_log = _normalize_log_input(subset)
        net, im, fm = ProcessModelDataset._safe_discover(
            fn, subset_log, params
        )
        if filter_unsound:
            if not check_wfnet(
                net
            ) or not check_easy_soundness_net_in_fin_marking(net, im, fm):
                return None

        item = ItemType(
            pm=net,
            im=im,
            fm=fm,
            variant=method_name,
            parameters=params,
            sampler=sampler,
            subset_idx=subset_idx,
            trace_indices=subset.indices,
        )
        return item.serialize()

    def __len__(self):
        return len(self.items)

    # --- skips deserialization for faster access ---
    def _get_serialized(self, idx: int | str) -> SerializedItemType:
        if isinstance(idx, int):
            key = self.index[idx]
        else:
            key = idx
        if key in self.items:
            return self.items[key]
        print("keys: ", self.items.keys())
        raise KeyError(key)

    def __getitem__(self, idx: int | str) -> ItemType:
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

    GUARANTEED_SOUND = {
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


if __name__ == "__main__":
    from dataloaders.xes_log import XESEventLogDataset
    from dataloaders.unique_net import UniqueProcessModelDataset
    from deduplication.deduplicator import DeduplicationConfig
    from pm4py.vis import view_petri_net

    path = "data/63a8435a-077d-4ece-97cd-2c76d394d99c/BPIC15_2.xes"
    RNG.initialize(42)
    log_dataset = XESEventLogDataset(path, attribute="concept:name")

    '''
    Example usage of the "Dedup by Variant, then define an arbitrary distribution over variants" approach.
    In the following "subset" is equivalent to "EventLog"... every subset is also an EventLog, however it is only a subset of the FULL eventlog that the FULL XES Dataset describes.
    '''

    # Create base dataset with caching enabled
    pm_dataset = ProcessModelDataset(
        log_dataset=log_dataset,
        discovery_methods={
            "inductive": discover_petri_net_inductive,
            "heuristic": discover_petri_net_heuristics,
        },
        param_grid={
            "noise_threshold": [0.0, 0.2, 0.4],
            "dependency_threshold": [0.0, 0.1, 0.2],
            "and_threshold": [0.0, 0.2, 0.5],
            "loop_two_threshold": [0.0, 0.2, 0.3],
            "disable_fallthroughs": [True],
        },
        sampler_specs={
            "variant_random": VariantRandomDistributionSampler(
                seed=RNG.get_seed(),
                n_subsets=100,  # number of subsets: defines how often the log is sampled... basically
                max_len_subset=100,
                min_len_subset=10,  # max_length_subset: limits the possible length of each sample (what is fed to the discovery algorithm)
                len_distribution=ExponentialSpec(
                    1.0 / 100.0
                ),  # subset length distribution: defines the distribution of lengths across samples
                freq_distribution=NormalSpec(
                    10.0, 5.0
                ),  # (variant) freq_distribution: defines the reordering of traces/variants on every sampling call, by defining the sampling behavior over index(variant) -> frequency.
                # realistically this would more likely be an exponential... but harder to parametrize... e.g.:
                # freq_distribution=ExponentialSpec(
                #     torch.tensor([1.0 / 20.0])
                # )
                reconstruct_frequency=True,  # toggle whether to reconstruct the frequency of variants in the sampled subset (repeat variants according to sampled frequency)
            )
        },
        cached=True,
    )

    for i, item in enumerate(pm_dataset):
        item = pm_dataset[-(i + 1)]
        view_petri_net(
            item.pm,
            item.im,
            item.fm,
        )
        if i >= 2:
            break

    for item in pm_dataset.serialized:
        print(item)
        break
