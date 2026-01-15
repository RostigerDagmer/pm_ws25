"""
UniqueProcessModelDataset - Wrapper for deduplication of process models.

This module provides a wrapper around ProcessModelDataset that handles
deduplication of structurally similar Petri nets using a multi-stage
comparison pipeline.
"""

from typing import Optional, Generator
from pathlib import Path
import logging
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from datetime import datetime

from dataloaders.net import ProcessModelDataset
from dataloaders.serializable import WithSerializedView
from deduplication.deduplicator import (
    PetriNetDeduplicator,
    DeduplicationConfig,
    PetriNetItem,
)
from deduplication.utils import duplicate_map_to_groups
from pm4py.visualization.petri_net import visualizer as pn_visualizer
import hashlib
import json
import pickle
import os


logging.basicConfig(level=logging.INFO)


class UniqueProcessModelDataset(
    Dataset[ProcessModelDataset.ItemType],
    WithSerializedView[
        ProcessModelDataset.ItemType, ProcessModelDataset.SerializedItemType
    ],
):
    """
    Dataset wrapper that deduplicates a ProcessModelDataset.

    This class wraps a ProcessModelDataset and removes duplicate Petri nets
    using an improved two-stage comparison pipeline:
    1. Transition label counts comparison (prefilter)
    2. Combined path-based edges + dual-score features comparison

    The deduplication is performed on cached models and updates the
    underlying dataset's configuration list.
    """

    def __init__(
        self,
        base_dataset: ProcessModelDataset,
        dedup_config: Optional[DeduplicationConfig] = None,
        dedup_cache_dir: Optional[Path] = None,
        use_cache: bool = True,
        force_recompute: bool = False,
    ):
        """
        Args:
            base_dataset: The ProcessModelDataset to deduplicate
            dedup_config: Configuration for deduplication thresholds
            cache_dir: Directory to save cache and reports
                      (defaults to parallel directory of base_dataset.cache_dir)
            use_cache: Whether to use caching for deduplication results
            force_recompute: Force recomputation even if cache exists
        """
        if not base_dataset.cached:
            raise ValueError(
                "UniqueProcessModelDataset requires a cached "
                "ProcessModelDataset. Set cached=True when creating "
                "the base dataset."
            )

        self.base_dataset = base_dataset
        self.dedup_config = dedup_config or DeduplicationConfig()
        self.use_cache = use_cache

        # Set cache directory (parallel to base_dataset cache)
        # Use _dedup_cache_dir internally to avoid conflict with cache_dir property
        if dedup_cache_dir:
            self._dedup_cache_dir = Path(dedup_cache_dir)
        else:
            parent_dir = str(base_dataset.cache_dir.parent)
            self._dedup_cache_dir = Path(
                os.path.join(parent_dir, ".cache_unique_models")
            )

        self.rep_hashes: list[str] = []
        self.dup_to_rep: dict[str, str] = {}
        self.processed_hashes: set[str] = set()

        cache_key = self._compute_cache_key()
        self.cache_file = Path(
            os.path.join(
                str(self._dedup_cache_dir), f"unique_dedup_{cache_key}.pkl"
            )
        )

        loaded = False
        if use_cache and not force_recompute and self.cache_file.exists():
            self._load_from_cache()
            loaded = True

        # incremental update every time (even if cache loaded)
        self._incremental_deduplicate()

        if use_cache:
            self._dedup_cache_dir.mkdir(parents=True, exist_ok=True)
            self._save_to_cache()

    def _compute_cache_key(self) -> str:
        """
        Compute cache key based on base_dataset configuration and
        deduplication config.

        Returns:
            16-character hash string
        """
        components = {
            'base_dataset_hash': self.base_dataset.log_uuid,
            'dedup_config': {
                'label_threshold': self.dedup_config.label_similarity_threshold,
                'combined_threshold': self.dedup_config.combined_similarity_threshold,
            },
        }
        full_hash = hashlib.sha1(
            json.dumps(components, sort_keys=True).encode()
        ).hexdigest()
        return full_hash[:16]

    def _save_to_cache(self):
        cache_data = {
            "rep_hashes": self.rep_hashes,
            "dup_to_rep": self.dup_to_rep,
            "processed_hashes": list(self.processed_hashes),
            "report": self.dedup_report,
        }
        with open(self.cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        logging.info(f"Saved deduplication cache to {self.cache_file}")

    def _load_from_cache(self):
        with open(self.cache_file, "rb") as f:
            cache_data = pickle.load(f)
        self.rep_hashes = cache_data.get("rep_hashes", [])
        self.dup_to_rep = cache_data.get("dup_to_rep", {})
        self.processed_hashes = set(cache_data.get("processed_hashes", []))
        self.dedup_report = cache_data.get("report", {})
        logging.info(
            f"Loaded {len(self.rep_hashes)} representatives from cache ({self.cache_file})"
        )

    def _incremental_deduplicate(self):
        """
        Incremental dedup keyed by stable item.hash().
        Only compares unseen hashes against current representatives.
        Also repairs missing representatives if base_dataset changed.
        """
        logging.info("Starting incremental deduplication of cached models...")

        # 1) Scan current base dataset to build current hash universe + index mapping
        current_hashes: list[str] = list(self.base_dataset.items.keys())
        hash_to_index: dict[str, int] = {
            k: v for v, k in enumerate(self.base_dataset.index)
        }

        if not current_hashes:
            logging.warning("No cached models found for deduplication")
            self.unique_indices = []
            return

        current_hash_set = set(current_hashes)

        # 2) Repair representatives that disappeared (base dataset changed)
        self._repair_missing_representatives(current_hash_set)

        # 3) Determine new items we have never processed
        new_hashes = [
            h for h in current_hashes if h not in self.processed_hashes
        ]

        # 4) Build PetriNetDeduplicator and current representative items
        deduplicator = PetriNetDeduplicator(config=self.dedup_config)

        # Build unique_nets list from representatives that are present
        unique_nets: list[PetriNetItem] = []
        idx_to_hash: dict[int, str] = {}

        for rep_h in self.rep_hashes:
            if rep_h not in hash_to_index:
                continue
            base_idx = hash_to_index[rep_h]
            base_item = self.base_dataset[base_idx]
            pn_item = PetriNetItem(
                net=base_item.pm,
                im=base_item.im,
                fm=base_item.fm,
                idx=base_idx,
                metadata={"hash": rep_h},
            )
            unique_nets.append(pn_item)
            idx_to_hash[base_idx] = rep_h

        # 5) Incrementally place new items
        for h in tqdm(
            new_hashes,
            disable=not self.dedup_config.verbose,
            desc="Incremental dedup",
        ):
            base_idx = hash_to_index[h]
            base_item = self.base_dataset[base_idx]
            candidate = PetriNetItem(
                net=base_item.pm,
                im=base_item.im,
                fm=base_item.fm,
                idx=base_idx,
                metadata={"hash": h},
            )

            is_dup, rep_idx = deduplicator._find_duplicate(
                candidate, unique_nets
            )

            if is_dup:
                rep_h = idx_to_hash[rep_idx]
                self.dup_to_rep[h] = rep_h
            else:
                self.rep_hashes.append(h)
                unique_nets.append(candidate)
                idx_to_hash[base_idx] = h

            self.processed_hashes.add(h)

        # 6) Derive unique_indices for *current* base_dataset ordering
        reps_present = [h for h in self.rep_hashes if h in hash_to_index]
        self.unique_indices = sorted([hash_to_index[h] for h in reps_present])

        # 7) Create/refresh report
        self.dedup_report = {
            "num_total_current": len(current_hashes),
            "num_unique_current": len(reps_present),
            "num_duplicates_current": len(current_hashes) - len(reps_present),
            "rep_hashes_present": reps_present,
            "dup_to_rep": self.dup_to_rep,  # hash->hash
            "thresholds": {
                "label_threshold": self.dedup_config.label_similarity_threshold,
                "combined_threshold": self.dedup_config.combined_similarity_threshold,
            },
            "stages_enabled": {
                "stage1": self.dedup_config.enable_stage1,
                "stage2": self.dedup_config.enable_stage2,
            },
            # optional: include deduplicator report only for *this run's new comparisons*
            "last_increment": deduplicator.get_report(),
        }

        # optional: save report
        report_path = Path(
            os.path.join(
                str(self.base_dataset.cache_dir.parent),
                "deduplication_report.json",
            )
        )
        with open(report_path, "w") as f:
            json.dump(self.dedup_report, f, indent=2)

        logging.info(
            f"Incremental dedup complete. Unique now: {len(reps_present)} / {len(current_hashes)}. "
            f"Processed hashes total: {len(self.processed_hashes)}."
        )

    def _repair_missing_representatives(self, current_hash_set: set[str]):
        """
        If a representative disappeared from the base dataset, choose a new representative
        from the remaining members of its group and rewrite mappings.
        """
        if not self.rep_hashes:
            return

        # Build groups: rep -> members (including rep)
        rep_to_members: dict[str, set[str]] = {
            rep: {rep} for rep in self.rep_hashes
        }
        for dup, rep in self.dup_to_rep.items():
            rep_to_members.setdefault(rep, set()).add(dup)

        new_rep_hashes: list[str] = []
        new_dup_to_rep: dict[str, str] = dict(self.dup_to_rep)

        for rep in self.rep_hashes:
            members = rep_to_members.get(rep, {rep})
            members_present = sorted(
                [m for m in members if m in current_hash_set]
            )

            if not members_present:
                # whole group vanished -> drop it
                continue

            if rep in current_hash_set:
                # rep still valid
                new_rep_hashes.append(rep)
                continue

            # rep missing: promote deterministic replacement
            promoted = members_present[0]
            new_rep_hashes.append(promoted)

            # rewrite all members (except promoted) to point to promoted
            for m in members_present:
                if m == promoted:
                    continue
                new_dup_to_rep[m] = promoted

            # promoted itself should not be marked duplicate
            if promoted in new_dup_to_rep:
                del new_dup_to_rep[promoted]

        self.rep_hashes = new_rep_hashes
        self.dup_to_rep = new_dup_to_rep

    def __len__(self):
        return len(self.unique_indices)

    def __getitem__(self, idx: int | str) -> ProcessModelDataset.ItemType:
        """
        Get item by index in the unique dataset.
        Maps to the corresponding index in the base dataset.
        """
        if isinstance(idx, str):
            return self.base_dataset[idx]
        if idx < 0 or idx >= len(self.unique_indices):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )
        base_idx = self.unique_indices[idx]
        return self.base_dataset[base_idx]

    def __iter__(self) -> Generator[ProcessModelDataset.ItemType, None, None]:
        """Iterate over unique items only."""
        for base_idx in self.unique_indices:
            yield self.base_dataset[base_idx]

    def _get_serialized(
        self, idx: int | str
    ) -> ProcessModelDataset.SerializedItemType:
        """
        Get serialized item for unique dataset.
        Maps unique index to base dataset index.
        """
        if isinstance(idx, str):
            return self.base_dataset.serialized[idx]
        if idx < 0 or idx >= len(self.unique_indices):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )
        base_idx = self.unique_indices[idx]
        return self.base_dataset.serialized[base_idx]

    @property
    def log(self):
        """Access the event log from the base dataset."""
        return self.base_dataset.log

    @property
    def log_uuid(self) -> str:
        """Access the log UUID from the base dataset."""
        return self.base_dataset.log_uuid

    @property
    def cache_dir(self):
        """Access the cache directory from the base dataset."""
        return self.base_dataset.cache_dir

    def save_duplicate_visualizations(
        self,
        output_dir: Optional[str] = None,
        bgcolor: str = "white",
        format: str = "png",
    ):
        """
        Save visualizations of duplicate groups.

        For each duplicate group, creates a folder containing visualizations
        of the unique (representative) net and all its duplicates.

        Args:
            output_dir: Directory to save visualizations. Defaults to
                       '<dataset_folder>/duplicate_visualizations_<timestamp>'.
            bgcolor: Background color for visualizations (default: "white")
            format: Image format (default: "png", options: "png", "svg", "pdf")
        """
        if not self.duplicate_map:
            logging.warning("No duplicates found. Nothing to visualize.")
            return

        # Set output directory
        if output_dir is None:
            dataset_folder = str(self.base_dataset.cache_dir.parent)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(
                dataset_folder, f"duplicate_visualizations_{timestamp}"
            )

        os.makedirs(output_dir, exist_ok=True)

        # Convert duplicate_map to groups
        groups = duplicate_map_to_groups(self.duplicate_map)

        logging.info(
            f"Saving visualizations for {len(groups)} duplicate groups to "
            f"{output_dir}"
        )

        # Process each group
        for group_idx, group in enumerate(
            tqdm(groups, desc="Visualizing groups")
        ):
            representative_idx = group[0]
            duplicate_indices = group[1:]

            # Create folder for this group
            group_dir = os.path.join(
                output_dir, f"group_{group_idx:04d}_repr_{representative_idx}"
            )
            os.makedirs(group_dir, exist_ok=True)

            # Visualize representative net
            repr_item = self.base_dataset[representative_idx]
            repr_path = os.path.join(
                group_dir, f"representative_{representative_idx}.{format}"
            )
            self._save_single_visualization(
                repr_item.pm,
                repr_item.im,
                repr_item.fm,
                repr_path,
                title=f"Representative {representative_idx}",
                bgcolor=bgcolor,
            )

            # Visualize all duplicates
            for dup_idx in duplicate_indices:
                dup_item = self.base_dataset[dup_idx]
                dup_path = os.path.join(
                    group_dir, f"duplicate_{dup_idx}.{format}"
                )
                self._save_single_visualization(
                    dup_item.pm,
                    dup_item.im,
                    dup_item.fm,
                    dup_path,
                    title=f"Duplicate {dup_idx}",
                    bgcolor=bgcolor,
                )

        logging.info(
            f"Saved visualizations for {len(groups)} groups to {output_dir}"
        )

    def save_unique_visualizations(
        self,
        output_dir: Optional[str] = None,
        bgcolor: str = "white",
        format: str = "png",
    ):
        """
        Save visualizations of all unique nets.

        Args:
            output_dir: Directory to save visualizations. Defaults to
                       '<dataset_folder>/unique_visualizations_<timestamp>'.
            bgcolor: Background color for visualizations (default: "white")
            format: Image format (default: "png", options: "png", "svg", "pdf")
        """
        if not self.unique_indices:
            logging.warning("No unique nets found. Nothing to visualize.")
            return

        # Set output directory
        if output_dir is None:
            dataset_folder = str(self.base_dataset.cache_dir.parent)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(
                dataset_folder, f"unique_visualizations_{timestamp}"
            )

        os.makedirs(output_dir, exist_ok=True)

        logging.info(
            f"Saving visualizations for {len(self.unique_indices)} unique nets to "
            f"{output_dir}"
        )

        # Process each unique net
        for idx, base_idx in enumerate(
            tqdm(self.unique_indices, desc="Visualizing unique nets")
        ):
            item = self.base_dataset[base_idx]
            file_path = os.path.join(
                output_dir, f"unique_{idx:04d}_base_{base_idx}.{format}"
            )
            self._save_single_visualization(
                item.pm,
                item.im,
                item.fm,
                file_path,
                title=f"Unique {idx} (Base Index {base_idx})",
                bgcolor=bgcolor,
            )

        logging.info(
            f"Saved {len(self.unique_indices)} unique nets to {output_dir}"
        )

    def _save_single_visualization(
        self,
        net,
        im,
        fm,
        file_path: str,
        title: str = None,
        bgcolor: str = "white",
    ):
        """
        Save a single Petri net visualization.

        Args:
            net: Petri net
            im: Initial marking
            fm: Final marking
            file_path: Path to save the visualization
            title: Optional title for the visualization
            bgcolor: Background color
        """
        # Extract format from file extension
        file_format = os.path.splitext(file_path)[1][1:]  # Remove leading dot

        parameters = {"format": file_format, "bgcolor": bgcolor}
        if title:
            parameters["graph_title"] = title

        gviz = pn_visualizer.apply(net, im, fm, parameters=parameters)
        pn_visualizer.save(gviz, file_path)


if __name__ == "__main__":
    from dataloaders.xes_log import XESEventLogDataset
    from deduplication.deduplicator import DeduplicationConfig
    from pm4py.discovery import discover_petri_net_inductive
    from dataloaders.net import VariantRandomDistributionSampler
    from util.rng import RNG
    from util.distributions import ExponentialSpec, NormalSpec
    import torch

    RNG.initialize(42)

    path = "data/6af6d5f0-f44c-49be-aac8-8eaa5fe4f6fd/Hospital%20Billing%20-%20Event%20Log.xes"
    # path = "data/6a0a26d2-82d0-4018-b1cd-89afb0e8627f/DomesticDeclarations.xes"
    # path = "data/3301445f-95e8-4ff0-98a4-901f1f204972/BPI%20Challenge%202018.xes"
    # path = "data/d9769f3d-0ab0-4fb8-803b-0d1120ffcf54/Hospital_log.xes"

    log_dataset = XESEventLogDataset(path, attribute="concept:name")

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
                seed=RNG.get_seed(),
                n_subsets=1000,  # number of subsets: defines how often the log is sampled... basically
                max_len_subset=100,
                min_len_subset=10,  # max_length_subset: limits the possible length of each sample (what is fed to the discovery algorithm)
                len_distribution=ExponentialSpec(
                    1.0 / 100.0
                ),  # subset length distribution: defines the distribution of lengths across samples
                freq_distribution=NormalSpec(
                    10.0, 5.0
                ),  # (variant) freq_distribution: defines the reordering of traces/variants on every sampling call, by defining the sampling behavior over index(variant) -> frequency.
                reconstruct_frequency=True,  # toggle whether to reconstruct the frequency of variants in the sampled subset (repeat variants according to sampled frequency)
            )
        },
        max_models=400,
        cached=True,
    )

    unique_dataset = UniqueProcessModelDataset(
        base_dataset=pm_dataset,
        dedup_config=DeduplicationConfig(),
        force_recompute=False,
    )
    unique_dataset.save_unique_visualizations()
    unique_dataset.save_duplicate_visualizations()
