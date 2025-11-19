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

from dataloaders.net import ProcessModelDataset, SerializedView
from deduplication.deduplicator import (
    PetriNetDeduplicator,
    DeduplicationConfig,
    PetriNetItem
)
from deduplication.normalizers import ZScoreFeatureNormalizer
from deduplication.utils import save_duplicate_report
from features.extractors import ModelFeatureExtractor
import hashlib
import json
import pickle
import os


logging.basicConfig(level=logging.INFO)


class UniqueProcessModelDataset(Dataset):
    """
    Dataset wrapper that deduplicates a ProcessModelDataset.

    This class wraps a ProcessModelDataset and removes duplicate Petri nets
    using a three-stage comparison pipeline:
    1. Transition label counts comparison
    2. Transition edge structure comparison
    3. Feature vector comparison

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
                'edge_threshold': self.dedup_config.edge_similarity_threshold,
                'feature_threshold': self.dedup_config.feature_similarity_threshold,
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
            2. Extract features for all nets
            3. Compute feature normalizer (z-score parameters)
            4. Create deduplicator with normalizer
            5. Iterative deduplication
            6. Update base_dataset.configurations
            7. Save report
        """
        logging.info("Starting deduplication of cached models...")

        all_items = self._load_all_cached_models()

        if not all_items:
            logging.warning("No cached models found for deduplication")
            return

        feature_normalizer = self._compute_feature_normalizer(all_items)

        deduplicator = PetriNetDeduplicator(
            config=self.dedup_config,
            feature_normalizer=feature_normalizer
        )

        unique_nets, duplicate_map = deduplicator.deduplicate(all_items)

        # Store unique indices (sorted for consistent ordering)
        self.unique_indices = sorted([item.idx for item in unique_nets])

        # Store duplicate_map as class attribute
        self.duplicate_map = duplicate_map

        # Get report from deduplicator
        self.dedup_report = deduplicator.get_report()

        # Save deduplication report as JSON
        report_path = Path(
            os.path.join(str(self.cache_dir), "deduplication_report.json")
        )
        save_duplicate_report(
            unique_nets,
            duplicate_map,
            self.dedup_report['thresholds'],
            report_path
        )

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

    def _compute_feature_normalizer(self, all_items):
        """
        Compute feature normalizer over all nets.

        Args:
            all_items: List of PetriNetItem instances

        Returns:
            Fitted FeatureNormalizer
        """
        extractor = ModelFeatureExtractor()

        logging.info("Extracting features for normalization...")
        all_features = []
        for item in tqdm(all_items, desc="Extracting features"):
            feat = extractor.extract(
                item.net, item.im, item.fm, return_as_dict=False
            )
            all_features.append(feat)

        all_features = np.vstack(all_features)

        normalizer = ZScoreFeatureNormalizer(extractor.feature_names)
        normalizer.fit(all_features)

        return normalizer

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