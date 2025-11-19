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


def save_duplicate_report(
    unique_nets: List,
    duplicate_map: Dict[int, int],
    config: Dict,
    output_path: Path
):
    """
    Save deduplication report as JSON.
    
    Args:
        unique_nets: List of unique PetriNetItems
        duplicate_map: Mapping of duplicate to representative indices
        config: Configuration dict with thresholds
        output_path: Path to save JSON report
    """
    total_input = len(unique_nets) + len(duplicate_map)
    
    report = {
        'summary': {
            'total_input': total_input,
            'total_unique': len(unique_nets),
            'total_duplicates': len(duplicate_map),
            'reduction_percent': (1 - len(unique_nets) / total_input) * 100
        },
        'unique_indices': [item.idx for item in unique_nets],
        'duplicate_map': duplicate_map,
        'config': config
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)


def load_duplicate_report(path: Path) -> Dict:
    """
    Load deduplication report from JSON.
    
    Args:
        path: Path to JSON report
    
    Returns:
        Report dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)