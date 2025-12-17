from typing import Dict, List, Union
import numpy as np
from collections import Counter
from pm4py.objects.petri_net.obj import PetriNet, Marking

from features.base_extractor import BaseFeatureExtractor


class TraceFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts features from trace Petri nets.
    Features include:
    - Trace length (number of activities/events); e.g. A -> A -> B -> B has length 4
    - Activity repetition statistics; e.g. for A -> A -> B -> B: [2, 2] -> mean=2, std=0
    """

    def _compute_cache_key(
        self, trace_net: PetriNet, trace_im: Marking, trace_fm: Marking
    ):
        """Use hash of the trace net as cache key."""
        return hash(trace_net)

    @property
    def feature_names(self) -> List[str]:
        return [
            'trace_length',
            'trace_activity_repeat_mean',
            'trace_activity_repeat_std',
        ]

    def _extract_features_internal(
        self, trace_net: PetriNet, trace_im: Marking, trace_fm: Marking
    ) -> Dict[str, float]:
        """Extract all trace features."""
        assert all(
            t.label is not None for t in trace_net.transitions
        ), "Trace net contains invisible transitions."
        labels = [t.label for t in trace_net.transitions]

        trace_length = len(labels)

        if trace_length == 0:
            return {
                'trace_length': 0,
                'trace_activity_repeat_mean': 0,
                'trace_activity_repeat_std': 0,
            }

        label_counts = Counter(labels)
        repeat_counts = list(label_counts.values())

        return {
            'trace_length': trace_length,
            'trace_activity_repeat_mean': np.mean(repeat_counts),
            'trace_activity_repeat_std': np.std(repeat_counts),
        }