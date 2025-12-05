"""
Utility functions for multi-dataset support in classification models.
"""

from typing import Union, List
import logging
from dataloaders.runs import RunDataset


def normalize_datasets(
    datasets: Union[RunDataset, List[RunDataset]],
) -> List[RunDataset]:
    """Convert single dataset to list for uniform handling."""
    if isinstance(datasets, RunDataset):
        return [datasets]
    return datasets


def validate_aligner_consistency(datasets: List[RunDataset]):
    """Validate that all RunDatasets have identical aligners."""
    reference_aligners = set(a.name for a in datasets[0].aligners)

    for i, ds in enumerate(datasets[1:], start=1):
        ds_aligners = set(a.name for a in ds.aligners)
        if ds_aligners != reference_aligners:
            raise ValueError(
                f"Dataset {i} has inconsistent aligners. "
                f"Expected: {reference_aligners}, Got: {ds_aligners}"
            )

    logging.info(
        f"Validated {len(datasets)} datasets with "
        f"{len(reference_aligners)} aligners each"
    )


def iter_combined_datasets(datasets: List[RunDataset]):
    """
    Iterator over all combinations from multiple RunDatasets.

    Yields:
        Tuples of (model, trace, results_dict) for each unique combination
    """
    seen_comb_ids = {}  # comb_id -> dataset_index

    for ds_idx, dataset in enumerate(datasets):
        for model, trace, results_dict in dataset.iter_by_combination():
            # Compute comb_id
            comb_id = RunDataset._hash_comb(trace, model)

            if comb_id in seen_comb_ids:
                prev_ds_idx = seen_comb_ids[comb_id]
                logging.warning(
                    f"Duplicate comb_id {comb_id[:8]}... found in dataset {ds_idx} "
                    f"(already seen in dataset {prev_ds_idx}). Skipping duplicate."
                )
                continue

            seen_comb_ids[comb_id] = ds_idx
            yield (model, trace, results_dict)

    logging.info(
        f"Iterated over {len(seen_comb_ids)} unique combinations "
        f"from {len(datasets)} datasets"
    )
