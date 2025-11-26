"""
UniqueProcessModelDataset - Wrapper for deduplication of process models.

This module provides a wrapper around ProcessModelDataset that handles
deduplication of structurally similar Petri nets using a multi-stage
comparison pipeline.
"""

from typing import Optional, Generator
from pathlib import Path
import logging
import pickle
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from datetime import datetime

from dataloaders.net import ProcessModelDataset, SerializedView
from deduplication.deduplicator import (
    PetriNetDeduplicator,
    DeduplicationConfig,
    PetriNetItem
)
from deduplication.utils import duplicate_map_to_groups
from pm4py.visualization.petri_net import visualizer as pn_visualizer
import hashlib
import json
import pickle
import os


logging.basicConfig(level=logging.INFO)


class UniqueProcessModelDataset(Dataset):
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
        cache_dir: Optional[Path] = None,
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
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            parent_dir = str(base_dataset.cache_dir.parent)
            self.cache_dir = Path(
                os.path.join(parent_dir, ".cache_unique_models")
            )

        # Initialize attributes
        self.unique_indices = []
        self.duplicate_map = {}
        self.dedup_report = {}

        # Compute cache key and path
        cache_key = self._compute_cache_key()
        self.cache_file = Path(
            os.path.join(str(self.cache_dir), f"unique_dedup_{cache_key}.pkl")
        )

        # Load from cache or deduplicate
        if use_cache and not force_recompute and self.cache_file.exists():
            self._load_from_cache()
        else:
            self._deduplicate()
            if use_cache:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._save_to_cache()

    def _compute_cache_key(self) -> str:
        """
        Compute cache key based on base_dataset configuration and
        deduplication config.

        Returns:
            16-character hash string
        """
        components = {
            'base_dataset_hash': self.base_dataset.hash(),
            'dedup_config': {
                'label_threshold': self.dedup_config.label_similarity_threshold,
                'combined_threshold': self.dedup_config.combined_similarity_threshold,
            }
        }
        full_hash = hashlib.sha1(
            json.dumps(components, sort_keys=True).encode()
        ).hexdigest()
        return full_hash[:16]

    def _save_to_cache(self):
        """Save unique_indices, duplicate_map, and dedup_report to cache."""
        cache_data = {
            'unique_indices': self.unique_indices,
            'duplicate_map': self.duplicate_map,
            'report': self.dedup_report,
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        logging.info(f"Saved deduplication cache to {self.cache_file}")

    def _load_from_cache(self):
        """Load unique_indices, duplicate_map, and dedup_report from cache."""
        with open(self.cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        self.unique_indices = cache_data['unique_indices']
        self.duplicate_map = cache_data['duplicate_map']
        self.dedup_report = cache_data['report']
        logging.info(
            f"Loaded {len(self.unique_indices)} unique indices from cache "
            f"({self.cache_file})"
        )

    def _deduplicate(self):
        """
        Deduplicate all cached models.

        Pipeline:
            1. Load all cached models
            2. Create deduplicator (no normalizer needed)
            3. Iterative deduplication
            4. Update base_dataset.configurations
            5. Save report
        """
        logging.info("Starting deduplication of cached models...")

        all_items = self._load_all_cached_models()

        if not all_items:
            logging.warning("No cached models found for deduplication")
            return

        deduplicator = PetriNetDeduplicator(config=self.dedup_config)

        unique_nets, duplicate_map = deduplicator.deduplicate(all_items)

        # Store unique indices (sorted for consistent ordering)
        self.unique_indices = sorted([item.idx for item in unique_nets])

        # Store duplicate_map as class attribute
        self.duplicate_map = duplicate_map

        # Get report from deduplicator
        self.dedup_report = deduplicator.get_report()

        # Save deduplication report as JSON in base_dataset directory
        report_path = Path(
            os.path.join(
                str(self.base_dataset.cache_dir.parent),
                "deduplication_report.json"
            )
        )
        with open(report_path, 'w') as f:
            json.dump(self.dedup_report, f, indent=2)

        logging.info(
            f"Deduplication complete. Kept {len(unique_nets)} unique nets "
            f"out of {len(all_items)} total. Report saved to {report_path}"
        )

    def _load_all_cached_models(self):
        """
        Load all cached models and wrap in PetriNetItem.

        Returns:
            List of PetriNetItem instances
        """
        items = []
        for idx, item in enumerate(self.base_dataset):
            items.append(PetriNetItem(
                net=item.pm,
                im=item.im,
                fm=item.fm,
                idx=idx,
                metadata={
                    'variant': item.variant,
                    'params': item.parameters
                }
            ))
        return items

    def __len__(self):
        return len(self.unique_indices)

    def __getitem__(self, idx: int) -> ProcessModelDataset.ItemType:
        """
        Get item by index in the unique dataset.
        Maps to the corresponding index in the base dataset.
        """
        if idx < 0 or idx >= len(self.unique_indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        base_idx = self.unique_indices[idx]
        return self.base_dataset[base_idx]

    def __iter__(self) -> Generator[ProcessModelDataset.ItemType, None, None]:
        """Iterate over unique items only."""
        for base_idx in self.unique_indices:
            yield self.base_dataset[base_idx]

    def _get_serialized(self, idx: int) -> SerializedView.ItemType:
        """
        Get serialized item for unique dataset.
        Maps unique index to base dataset index.
        """
        if idx < 0 or idx >= len(self.unique_indices):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )
        base_idx = self.unique_indices[idx]
        return self.base_dataset._get_serialized(base_idx)

    @property
    def serialized(self):
        """Access serialized view of unique items."""
        return SerializedView(self, self._get_serialized)

    def save_duplicate_visualizations(
        self,
        output_dir: Optional[str] = None,
        bgcolor: str = "white",
        format: str = "png"
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
                dataset_folder,
                f"duplicate_visualizations_{timestamp}"
            )

        os.makedirs(output_dir, exist_ok=True)

        # Convert duplicate_map to groups
        groups = duplicate_map_to_groups(self.duplicate_map)

        logging.info(
            f"Saving visualizations for {len(groups)} duplicate groups to "
            f"{output_dir}"
        )

        # Process each group
        for group_idx, group in enumerate(tqdm(groups, desc="Visualizing groups")):
            representative_idx = group[0]
            duplicate_indices = group[1:]

            # Create folder for this group
            group_dir = os.path.join(
                output_dir,
                f"group_{group_idx:04d}_repr_{representative_idx}"
            )
            os.makedirs(group_dir, exist_ok=True)

            # Visualize representative net
            repr_item = self.base_dataset[representative_idx]
            repr_path = os.path.join(
                group_dir,
                f"representative_{representative_idx}.{format}"
            )
            self._save_single_visualization(
                repr_item.pm,
                repr_item.im,
                repr_item.fm,
                repr_path,
                title=f"Representative {representative_idx}",
                bgcolor=bgcolor
            )

            # Visualize all duplicates
            for dup_idx in duplicate_indices:
                dup_item = self.base_dataset[dup_idx]
                dup_path = os.path.join(
                    group_dir,
                    f"duplicate_{dup_idx}.{format}"
                )
                self._save_single_visualization(
                    dup_item.pm,
                    dup_item.im,
                    dup_item.fm,
                    dup_path,
                    title=f"Duplicate {dup_idx}",
                    bgcolor=bgcolor
                )

        logging.info(
            f"Saved visualizations for {len(groups)} groups to {output_dir}"
        )

    def save_unique_visualizations(
        self,
        output_dir: Optional[str] = None,
        bgcolor: str = "white",
        format: str = "png"
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
                dataset_folder,
                f"unique_visualizations_{timestamp}"
            )

        os.makedirs(output_dir, exist_ok=True)

        logging.info(
            f"Saving visualizations for {len(self.unique_indices)} unique nets to "
            f"{output_dir}"
        )

        # Process each unique net
        for idx, base_idx in enumerate(tqdm(self.unique_indices, desc="Visualizing unique nets")):
            item = self.base_dataset[base_idx]
            file_path = os.path.join(
                output_dir,
                f"unique_{idx:04d}_base_{base_idx}.{format}"
            )
            self._save_single_visualization(
                item.pm,
                item.im,
                item.fm,
                file_path,
                title=f"Unique {idx} (Base Index {base_idx})",
                bgcolor=bgcolor
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
        bgcolor: str = "white"
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

        parameters = {
            "format": file_format,
            "bgcolor": bgcolor
        }
        if title:
            parameters["graph_title"] = title

        gviz = pn_visualizer.apply(
            net, im, fm,
            parameters=parameters
        )
        pn_visualizer.save(gviz, file_path)