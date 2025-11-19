"""
Comparison methods for Petri nets in deduplication pipeline.
Each comparator implements a stage of the deduplication process.
"""

from abc import ABC, abstractmethod
from collections import Counter
import numpy as np
from pm4py.objects.petri_net.obj import PetriNet, Marking
from features.extractors import BaseFeatureExtractor
from deduplication.normalizers import ZScoreFeatureNormalizer

class BaseComparator(ABC):
    """Interface for all comparators."""
    
    @abstractmethod
    def compare(
        self,
        net1: PetriNet, im1: Marking, fm1: Marking,
        net2: PetriNet, im2: Marking, fm2: Marking
    ) -> float:
        """
        Compare two Petri nets.
        
        Args:
            net1, im1, fm1: First Petri net with initial and final marking
            net2, im2, fm2: Second Petri net with initial and final marking
            
        Returns:
            Similarity score in [0, 1], where 1 = identical
        """
        pass


class TransitionLabelComparator(BaseComparator):
    """
    Stage 1: Compare transition label counts using Bray-Curtis similarity.

    Extracts count vectors of transition labels from both nets and compares them.
    Works on the union of labels from both nets, no prior knowledge required.
    """

    def __init__(self, use_cache: bool = True):
        """
        Initialize comparator.

        Args:
            use_cache: Whether to cache extracted label counts
        """
        self.use_cache = use_cache
        self._label_cache = {} if use_cache else None

    def _extract_label_counts(self, net: PetriNet) -> Counter:
        """Extract label counts from net transitions, using cache if enabled."""
        if self.use_cache:
            net_hash = hash(net)
            if net_hash in self._label_cache:
                return self._label_cache[net_hash]

        counts = Counter()
        for transition in net.transitions:
            label = transition.label if transition.label is not None else 'τ'
            counts[label] += 1

        if self.use_cache:
            self._label_cache[net_hash] = counts

        return counts

    def compare(
        self,
        net1: PetriNet, im1: Marking, fm1: Marking,
        net2: PetriNet, im2: Marking, fm2: Marking
    ) -> float:
        """Compare nets using Bray-Curtis similarity on label counts."""
        counts1 = self._extract_label_counts(net1)
        counts2 = self._extract_label_counts(net2)

        all_labels = set(counts1.keys()) | set(counts2.keys())

        if not all_labels:
            return 1.0

        numerator = sum(abs(counts1[label] - counts2[label]) for label in all_labels)
        denominator = sum(counts1[label] + counts2[label] for label in all_labels)

        bray_curtis_distance = numerator / denominator if denominator > 0 else 0.0
        return 1.0 - bray_curtis_distance


class TransitionEdgeComparator(BaseComparator):
    """
    Stage 2: Compare transition-to-transition edges using Bray-Curtis similarity.

    Extracts edges between transitions (via places) including START/END boundaries.
    Compares edge count distributions between two nets.
    """

    def __init__(self, use_cache: bool = True):
        """
        Initialize comparator.

        Args:
            use_cache: Whether to cache extracted edge counts
        """
        self.use_cache = use_cache
        self._edge_cache = {} if use_cache else None

    def _extract_transition_edges(
        self,
        net: PetriNet,
        im: Marking,
        fm: Marking
    ) -> Counter:
        """Extract transition-to-transition edges via places, using cache if enabled."""
        if self.use_cache:
            net_hash = hash(net)
            if net_hash in self._edge_cache:
                return self._edge_cache[net_hash]

        edges = []
        start_places = set(im.keys())
        end_places = set(fm.keys())
        
        for place in net.places:
            incoming = [
                arc.source for arc in place.in_arcs
                if isinstance(arc.source, PetriNet.Transition)
            ]
            outgoing = [
                arc.target for arc in place.out_arcs
                if isinstance(arc.target, PetriNet.Transition)
            ]
            
            if place in start_places:
                for t_out in outgoing:
                    label = t_out.label if t_out.label is not None else 'τ'
                    edges.append(('START', label))
            
            if place in end_places:
                for t_in in incoming:
                    label = t_in.label if t_in.label is not None else 'τ'
                    edges.append((label, 'END'))
            
            for t_in in incoming:
                for t_out in outgoing:
                    src = t_in.label if t_in.label is not None else 'τ'
                    tgt = t_out.label if t_out.label is not None else 'τ'
                    edges.append((src, tgt))

        edge_counts = Counter(edges)

        if self.use_cache:
            self._edge_cache[net_hash] = edge_counts

        return edge_counts
    
    def compare(
        self,
        net1: PetriNet, im1: Marking, fm1: Marking,
        net2: PetriNet, im2: Marking, fm2: Marking
    ) -> float:
        """Compare nets using Bray-Curtis similarity on edge counts."""
        edges1 = self._extract_transition_edges(net1, im1, fm1)
        edges2 = self._extract_transition_edges(net2, im2, fm2)

        all_edges = set(edges1.keys()) | set(edges2.keys())

        if not all_edges:
            return 1.0

        numerator = sum(abs(edges1[e] - edges2[e]) for e in all_edges)
        denominator = sum(edges1[e] + edges2[e] for e in all_edges)

        bray_curtis_distance = numerator / denominator if denominator > 0 else 0.0
        return 1.0 - bray_curtis_distance


class FeatureVectorComparator(BaseComparator):
    """
    Stage 3: Compare z-score normalized feature vectors using MAD.

    Extracts feature vectors, normalizes them using pre-computed z-score parameters,
    and computes weighted Mean Absolute Deviation (MAD) as distance metric.

    Returns MAD directly (not transformed to [0,1] range). Higher values indicate
    greater dissimilarity. MAD is interpreted as the weighted average deviation
    in standard deviations.

    Requires a pre-fitted FeatureNormalizer with statistics from all nets.
    """
    
    def __init__(
        self,
        feature_extractor: BaseFeatureExtractor,
        normalizer: ZScoreFeatureNormalizer,
        weights: dict = None
    ):
        """
        Initialize comparator.

        Args:
            feature_extractor: ModelFeatureExtractor instance
            normalizer: Pre-fitted FeatureNormalizer with z-score parameters
            weights: Optional weight dictionary for features. If None, default
                     weights are used (2.0 for basic counts, 1.0 for others)
        """
        self.extractor = feature_extractor
        self.normalizer = normalizer

        if weights is None:
            # Default weights: emphasize basic structural counts
            weights_dict = {
                'model_n_transitions': 1.0,
                'model_n_places': 1.0,
                'model_n_arcs': 1.0,
                'model_n_inv_transition': 2.0,
                'model_n_dup_transition': 1.0,
                'model_n_uniq_transition': 1.0,
                'model_n_and_split': 3.0,
                'model_n_xor_split': 3.0,
                'model_inv_tran_in_deg_mean': 1.0,
                'model_inv_tran_in_deg_std': 1.0,
                'model_inv_tran_out_deg_mean': 1.0,
                'model_inv_tran_out_deg_std': 1.0,
                'model_uniq_tran_in_deg_mean': 1.0,
                'model_uniq_tran_in_deg_std': 1.0,
                'model_uniq_tran_out_deg_mean': 1.0,
                'model_uniq_tran_out_deg_std': 1.0,
                'model_dup_tran_in_deg_mean': 1.0,
                'model_dup_tran_in_deg_std': 1.0,
                'model_dup_tran_out_deg_mean': 1.0,
                'model_dup_tran_out_deg_std': 1.0,
                'model_place_in_deg_mean': 1.0,
                'model_place_in_deg_std': 1.0,
                'model_place_out_deg_mean': 1.0,
                'model_place_out_deg_std': 1.0,
            }

            self.weights = self.extractor.dict_to_vector(weights_dict)
        else:
            # Convert provided weights dict to vector
            self.weights = self.extractor.dict_to_vector(weights)
    
    def compare(
        self,
        net1: PetriNet, im1: Marking, fm1: Marking,
        net2: PetriNet, im2: Marking, fm2: Marking
    ) -> float:
        """
        Compare nets using weighted MAD on z-score normalized features.

        Returns:
            MAD value (weighted average deviation in standard deviations).
            Higher values = greater dissimilarity. Range: [0, ∞)
        """
        feat1 = self.extractor.extract(net1, im1, fm1, return_as_dict=False)
        feat2 = self.extractor.extract(net2, im2, fm2, return_as_dict=False)

        feat1_norm = self.normalizer.normalize(feat1)
        feat2_norm = self.normalizer.normalize(feat2)

        weighted_diff = self.weights * np.abs(feat1_norm - feat2_norm)
        mad = np.sum(weighted_diff) / np.sum(self.weights)

        return mad