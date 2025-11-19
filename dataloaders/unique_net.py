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

from dataloaders.net import ProcessModelDataset
from deduplication.deduplicator import (
    PetriNetDeduplicator,
    DeduplicationConfig,
    PetriNetItem
)
from deduplication.normalizers import ZScoreFeatureNormalizer
from deduplication.utils import save_duplicate_report
from features.extractors import ModelFeatureExtractor

from pm4py.objects.petri_net.importer import importer as pnml_importer


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
        report_dir: Optional[Path] = None,
    ):
        """
        Args:
            base_dataset: The ProcessModelDataset to deduplicate
            dedup_config: Configuration for deduplication thresholds
            report_dir: Directory to save deduplication report
                       (defaults to base_dataset.cache_dir)
        """
        if not base_dataset.cached:
            raise ValueError(
                "UniqueProcessModelDataset requires a cached "
                "ProcessModelDataset. Set cached=True when creating "
                "the base dataset."
            )

        self.base_dataset = base_dataset
        self.dedup_config = dedup_config or DeduplicationConfig()
        self.report_dir = report_dir or base_dataset.cache_dir

        # Store unique indices in base dataset
        self.unique_indices = []

        self._deduplicate()

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

        # Save deduplication report
        report_path = self.report_dir / "deduplication_report.json"
        save_duplicate_report(
            unique_nets,
            duplicate_map,
            {
                'label_threshold': self.dedup_config.label_similarity_threshold,
                'edge_threshold': self.dedup_config.edge_similarity_threshold,
                'feature_threshold': self.dedup_config.feature_similarity_threshold,
            },
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

    @property
    def serialized(self):
        """Access serialized view of unique items."""
        return self.UniqueSerializedView(self)

    class UniqueSerializedView:
        """Serialized view that only returns unique items."""

        def __init__(self, parent_dataset: "UniqueProcessModelDataset"):
            self.parent = parent_dataset

        def __len__(self):
            return len(self.parent.unique_indices)

        def __getitem__(self, idx: int):
            """Get serialized item without deserialization."""
            if idx < 0 or idx >= len(self.parent.unique_indices):
                raise IndexError(f"Index {idx} out of range")
            base_idx = self.parent.unique_indices[idx]
            return self.parent.base_dataset.serialized[base_idx]

        def __iter__(self):
            """Iterate over serialized unique items."""
            for base_idx in self.parent.unique_indices:
                yield self.parent.base_dataset.serialized[base_idx]