"""
Comparison methods for Petri nets in deduplication pipeline.
Each comparator implements a stage of the deduplication process.
"""

from abc import ABC, abstractmethod
from collections import Counter, deque
import numpy as np
from pm4py.objects.petri_net.obj import PetriNet, Marking
from features.extractors import BaseFeatureExtractor
from deduplication.normalizers import ZScoreFeatureNormalizer
from typing import *

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


class PathBasedTransitionEdgeComparator(BaseComparator):
    """
    Improved edge comparator that skips invisible transitions.
    
    Uses a single efficient forward-BFS to determine structural edges between
    visible transitions (Structural Directly-Follows Edges).
    """

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._edge_cache = {} if use_cache else None

    def _find_successors_and_end(
        self,
        start_places: List[PetriNet.Place],
        final_places: Set[PetriNet.Place]
    ) -> Tuple[List[str], bool]:
        """
        Performs BFS starting from given places to find next reachable visible transitions.
        Also checks if the final marking is reachable via invisible steps.
        """
        visible_labels = []
        reaches_end = False
        
        visited = set()
        queue = deque(start_places)

        while queue:
            curr_place = queue.popleft()
            
            if curr_place in visited:
                continue
            visited.add(curr_place)

            if curr_place in final_places:
                reaches_end = True

            for arc in curr_place.out_arcs:
                trans = arc.target
                
                if trans in visited: 
                    continue
                
                if trans.label is not None:
                    # Found visible transition: record and stop this branch
                    visible_labels.append(trans.label)
                else:
                    # Invisible transition: continue search
                    visited.add(trans)
                    for out_arc in trans.out_arcs:
                        queue.append(out_arc.target)
                        
        return visible_labels, reaches_end

    def _extract_transition_edges(
        self,
        net: PetriNet,
        im: Marking,
        fm: Marking
    ) -> Counter:
        """
        Extracts structural edges (A -> B) by skipping invisible transitions.
        """
        if self.use_cache:
            net_hash = hash(net)
            if net_hash in self._edge_cache:
                return self._edge_cache[net_hash]

        edges = []
        final_places = set(fm.keys())
        
        # 1. Handle START -> ...
        next_labels, reaches_end = self._find_successors_and_end(
            list(im.keys()), final_places
        )
        
        for label in next_labels:
            edges.append(('START', label))
        
        if reaches_end:
            edges.append(('START', 'END'))

        # 2. Handle Visible Transition -> ...
        for t in net.transitions:
            if t.label is not None:
                out_places = [arc.target for arc in t.out_arcs]
                
                next_labels, reaches_end = self._find_successors_and_end(
                    out_places, final_places
                )
                
                for target_label in next_labels:
                    edges.append((t.label, target_label))
                
                if reaches_end:
                    edges.append((t.label, 'END'))

        edge_counts = Counter(edges)

        if self.use_cache:
            self._edge_cache[net_hash] = edge_counts

        return edge_counts

    def compare(
        self,
        net1: PetriNet, im1: Marking, fm1: Marking,
        net2: PetriNet, im2: Marking, fm2: Marking
    ) -> float:
        edges1 = self._extract_transition_edges(net1, im1, fm1)
        edges2 = self._extract_transition_edges(net2, im2, fm2)

        all_edges = set(edges1.keys()) | set(edges2.keys())

        if not all_edges:
            return 1.0

        numerator = sum(abs(edges1[e] - edges2[e]) for e in all_edges)
        denominator = sum(edges1[e] + edges2[e] for e in all_edges)

        return 1.0 - (numerator / denominator if denominator > 0 else 0.0)


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


class DualScoreFeatureComparator(BaseComparator):
    """
    Improved feature comparator using Canberra Distance with dual scoring.

    Splits features into two groups:
    1. Structural features (model_n_transitions to model_n_xor_split)
    2. Degree statistics (all degree-related features)

    Computes Canberra Distance for each group separately and combines
    them with 50/50 weighting.

    Returns similarity score (higher = more similar). Range: [0, 1]
    """

    def __init__(self, feature_extractor: BaseFeatureExtractor, epsilon: float = 1e-10):
        """
        Initialize comparator.

        Args:
            feature_extractor: ModelFeatureExtractor instance
            epsilon: Small value to avoid division by zero in Canberra Distance
        """
        self.extractor = feature_extractor
        self.epsilon = epsilon

        # Define feature group indices based on ModelFeatureExtractor.feature_names
        # Structural: indices 0-7 (model_n_transitions to model_n_xor_split)
        # Degree stats: indices 8-23 (all degree features)
        self.structural_indices = list(range(0, 8))
        self.degree_indices = list(range(8, 24))

    def _canberra_distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        indices: list
    ) -> float:
        """
        Compute robust Canberra Distance for specified feature indices.

        Args:
            x: First feature vector
            y: Second feature vector
            indices: List of feature indices to include

        Returns:
            Normalized Canberra Distance in [0, 1]
        """
        x_subset = x[indices]
        y_subset = y[indices]

        numerator = np.abs(x_subset - y_subset)
        denominator = np.abs(x_subset) + np.abs(y_subset) + self.epsilon

        distances = numerator / denominator
        # Normalize by number of features to get value in [0, 1]
        return np.mean(distances)

    def compare(
        self,
        net1: PetriNet, im1: Marking, fm1: Marking,
        net2: PetriNet, im2: Marking, fm2: Marking
    ) -> float:
        """
        Compare nets using dual-score Canberra Distance.

        Returns:
            Similarity score in [0, 1], where higher = more similar
        """
        feat1 = self.extractor.extract(net1, im1, fm1, return_as_dict=False)
        feat2 = self.extractor.extract(net2, im2, fm2, return_as_dict=False)

        # Compute Canberra Distance for each feature group
        structural_dist = self._canberra_distance(
            feat1, feat2, self.structural_indices
        )
        degree_dist = self._canberra_distance(
            feat1, feat2, self.degree_indices
        )

        # Combine with 50/50 weighting
        combined_dissimilarity = 0.5 * structural_dist + 0.5 * degree_dist

        # Convert dissimilarity to similarity
        return 1.0 - combined_dissimilarity


class CombinedComparator(BaseComparator):
    """
    Combines path-based edge comparison with dual-score feature comparison.

    This comparator integrates two complementary comparison approaches:
    1. Path-based transition edge comparison (structural/behavioral similarity)
    2. Dual-score feature comparison (statistical similarity)

    Both similarity scores are combined with configurable weighting (default 50/50).

    Returns similarity score (higher = more similar). Range: [0, 1]
    """

    def __init__(
        self,
        edge_comparator: PathBasedTransitionEdgeComparator,
        feature_comparator: DualScoreFeatureComparator,
        edge_weight: float = 0.5
    ):
        """
        Initialize combined comparator.

        Args:
            edge_comparator: PathBasedTransitionEdgeComparator instance
            feature_comparator: DualScoreFeatureComparator instance
            edge_weight: Weight for edge score (feature weight = 1 - edge_weight)
        """
        self.edge_comparator = edge_comparator
        self.feature_comparator = feature_comparator
        self.edge_weight = edge_weight
        self.feature_weight = 1.0 - edge_weight

    def compare(
        self,
        net1: PetriNet, im1: Marking, fm1: Marking,
        net2: PetriNet, im2: Marking, fm2: Marking
    ) -> float:
        """
        Compare nets using combined edge and feature scores.

        Returns:
            Similarity score in [0, 1], where higher = more similar
        """
        # Get edge similarity (range [0, 1], 1 = identical)
        edge_similarity = self.edge_comparator.compare(
            net1, im1, fm1, net2, im2, fm2
        )

        # Get feature similarity (range [0, 1], 1 = identical)
        feature_similarity = self.feature_comparator.compare(
            net1, im1, fm1, net2, im2, fm2
        )

        # Combine similarities with weighted average
        combined_similarity = (
            self.edge_weight * edge_similarity +
            self.feature_weight * feature_similarity
        )

        return combined_similarity