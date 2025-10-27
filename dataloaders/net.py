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
import inspect
import random
from enum import Enum
import logging

logging.getLogger(None)
logging.basicConfig(level=logging.DEBUG)


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
        sampler_fn=None,
        max_models=None,
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

        # Resolve Enum -> dict
        if hasattr(discovery_methods, "value"):
            discovery_methods = discovery_methods.value

        self.discovery_methods = discovery_methods
        self.param_grid = param_grid
        self.sampler_fn = sampler_fn or self._default_sampler
        self.max_models = max_models

        self.configurations = self._generate_configurations()

    def _generate_configurations(self):
        configs = []

        # Resolve Enum -> dict
        if hasattr(self.param_grid, "value"):
            param_grid = self.param_grid.value
        else:
            param_grid = self.param_grid

        if not isinstance(param_grid, dict):
            raise TypeError(
                f"param_grid must be dict or Enum[dict], got {type(param_grid)}"
            )

        for method_name, fn in self.discovery_methods.items():
            logging.debug(
                f"Generating configurations for method: {method_name}"
            )

            # Case 1: Per-method grid defined explicitly
            if method_name in param_grid and isinstance(
                param_grid[method_name], dict
            ):
                method_grid = param_grid[method_name]
            # Case 2: Global grid (same for all)
            else:
                method_grid = param_grid

            if not method_grid:
                method_param_combos = [{}]
            else:
                keys, values = zip(*method_grid.items())
                method_param_combos = [
                    dict(zip(keys, combo)) for combo in product(*values)
                ]

            # Filter parameters by discovery function signature
            sig_params = set(inspect.signature(fn).parameters.keys())

            for params in method_param_combos:
                # Drop unknown keys for this method
                filtered_params = {
                    k: v for k, v in params.items() if k in sig_params
                }

                subset = self.sampler_fn(self.log)
                configs.append((method_name, fn, filtered_params, subset))

        if self.max_models:
            configs = configs[: self.max_models]

        logging.info(
            f"Total discovery configurations generated: {len(configs)}"
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

    def __len__(self):
        return len(self.configurations)

    def __getitem__(self, idx):
        method_name, fn, params, subset = self.configurations[idx]
        subset = _normalize_log_input(subset)
        net, im, fm = self._safe_discover(fn, subset, params)
        return {
            "pm": net,
            "im": im,
            "fm": fm,
            "variant": method_name,
            "parameters": params,
            "trace_indices": getattr(subset, "indices", None),
        }


class DISCOVERY_METHODS(Enum):
    ALL = {
        "inductive": discover_petri_net_inductive,
        "heuristic": discover_petri_net_heuristics,
        "alpha": discover_petri_net_alpha,
        "alpha_plus": discover_petri_net_alpha_plus,
        "ilp": discover_petri_net_ilp,
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


def random_subset_sampler(log):
    n = len(log)
    indices = random.sample(range(n), k=n // 2)
    return [log[i] for i in indices]


if __name__ == "__main__":
    from dataloaders.base import make_feature_fn
    from dataloaders.csv import CSVEventLogDataset
    from pm4py.vis import view_petri_net

    print(DISCOVERY_METHODS.ALL.value)

    path = "data/dx.doi.org_10.4121_uuid_c3f3ba2d-e81e-4274-87c7-882fa1dbab0d/BPI2016_Werkmap_Messages.csv"
    log_dataset = CSVEventLogDataset(
        path,
        # attribute="EventType",
        case_id_col="CustomerID",
        timestamp_col="EventDateTime",
        activity_col="HandlingChannelID",
        sep=";",
        feature_fn=make_feature_fn,
    )

    pm_dataset = ProcessModelDataset(
        log_dataset=log_dataset,
        discovery_methods=DISCOVERY_METHODS.ALL,
        param_grid=PARAM_GRID.STANDARD,
        sampler_fn=random_subset_sampler,
    )

    for item in pm_dataset:
        print(item)
        view_petri_net(
            item["pm"],
            item["im"],
            item["fm"],
        )
        break
