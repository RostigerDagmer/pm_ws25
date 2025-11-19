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
    
    def _extract_label_counts(self, net: PetriNet) -> Counter:
        """Extract label counts from net transitions."""
        counts = Counter()
        for transition in net.transitions:
            label = transition.label if transition.label is not None else 'τ'
            counts[label] += 1
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
        
        numerator = sum(min(counts1[label], counts2[label]) for label in all_labels)
        denominator = sum(counts1[label] + counts2[label] for label in all_labels)
        
        return 2 * numerator / denominator if denominator > 0 else 0.0


class TransitionEdgeComparator(BaseComparator):
    """
    Stage 2: Compare transition-to-transition edges using Bray-Curtis similarity.
    
    Extracts edges between transitions (via places) including START/END boundaries.
    Compares edge count distributions between two nets.
    """
    
    def _extract_transition_edges(
        self,
        net: PetriNet,
        im: Marking,
        fm: Marking
    ) -> Counter:
        """Extract transition-to-transition edges via places."""
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
        
        return Counter(edges)
    
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
        
        numerator = sum(min(edges1[e], edges2[e]) for e in all_edges)
        denominator = sum(edges1[e] + edges2[e] for e in all_edges)
        
        return 2 * numerator / denominator if denominator > 0 else 0.0


class FeatureVectorComparator(BaseComparator):
    """
    Stage 3: Compare z-score normalized feature vectors using MAD.
    
    Extracts feature vectors, normalizes them using pre-computed z-score parameters,
    and computes Mean Absolute Deviation (MAD) as similarity metric.
    
    Requires a pre-fitted FeatureNormalizer with statistics from all nets.
    """
    
    def __init__(
        self,
        feature_extractor: BaseFeatureExtractor,
        normalizer: ZScoreFeatureNormalizer,
        weights: np.ndarray = None
    ):
        """
        Initialize comparator.
        
        Args:
            feature_extractor: ModelFeatureExtractor instance
            normalizer: Pre-fitted FeatureNormalizer with z-score parameters
            weights: Optional weight vector for features
        """
        self.extractor = feature_extractor
        self.normalizer = normalizer
        self.weights = weights
        
        if self.weights is None:
            n_features = len(self.extractor.feature_names)
            self.weights = np.ones(n_features)
            self.weights[:8] = 2.0
    
    def compare(
        self,
        net1: PetriNet, im1: Marking, fm1: Marking,
        net2: PetriNet, im2: Marking, fm2: Marking
    ) -> float:
        """Compare nets using MAD on z-score normalized features."""
        feat1 = self.extractor.extract(net1, im1, fm1, return_as_dict=False)
        feat2 = self.extractor.extract(net2, im2, fm2, return_as_dict=False)
        
        feat1_norm = self.normalizer.normalize(feat1)
        feat2_norm = self.normalizer.normalize(feat2)
        
        weighted_diff = self.weights * np.abs(feat1_norm - feat2_norm)
        mad = np.mean(weighted_diff)
        
        return 1.0 / (1.0 + mad)