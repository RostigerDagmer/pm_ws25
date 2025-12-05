"""
Utility functions for deduplication.
Includes reporting, data conversion, and analysis helpers.
"""

from typing import List, Dict
from collections import defaultdict
from pathlib import Path
import json


def duplicate_map_to_groups(duplicate_map: Dict[int, int]) -> List[List[int]]:
    """
    Convert duplicate_map to grouped format for visualization.

    Args:
        duplicate_map: Dict mapping duplicate_idx -> representative_idx
                      Example: {5: 0, 12: 0, 7: 3, 9: 3}

    Returns:
        List of groups where each group contains [representative, duplicates...]
        Example: [[0, 5, 12], [3, 7, 9]]
    """
    groups = defaultdict(list)
    for dup_idx, repr_idx in duplicate_map.items():
        groups[repr_idx].append(dup_idx)

    result = []
    for repr_idx, dup_indices in groups.items():
        result.append([repr_idx] + dup_indices)

    return result
