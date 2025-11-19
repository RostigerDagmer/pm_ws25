"""
Iterative deduplicator for Petri nets.
Implements multi-stage comparison pipeline with early stopping.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm

from pm4py.objects.petri_net.obj import PetriNet, Marking
from deduplication.comparators import (
    TransitionLabelComparator,
    TransitionEdgeComparator,
    FeatureVectorComparator
)
from deduplication.normalizers import ZScoreFeatureNormalizer
from features.extractors import ModelFeatureExtractor


logger = logging.getLogger(__name__)


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication pipeline."""
    
    label_similarity_threshold: float = 0.80
    edge_similarity_threshold: float = 0.95
    feature_similarity_threshold: float = 0.98
    
    enable_stage1: bool = True
    enable_stage2: bool = True
    enable_stage3: bool = True
    
    verbose: bool = True


@dataclass
class PetriNetItem:
    """Wrapper for Petri net with metadata."""
    
    net: PetriNet
    im: Marking
    fm: Marking
    idx: int
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PetriNetDeduplicator:
    """
    Iterative deduplicator for Petri nets.
    
    Pipeline:
        1. First net → unique_nets
        2. For each subsequent net:
           - Compare with ALL nets already in unique_nets
           - If too similar → mark as duplicate
           - If unique → add to unique_nets
    
    Uses three-stage comparison with early stopping:
        Stage 1: Transition label counts (Bray-Curtis)
        Stage 2: Transition edges (Bray-Curtis)
        Stage 3: Feature vectors (MAD on z-scores)
    """
    
    def __init__(
        self,
        config: DeduplicationConfig,
        feature_normalizer: ZScoreFeatureNormalizer
    ):
        """
        Initialize deduplicator.
        
        Args:
            config: Deduplication configuration
            feature_normalizer: Pre-fitted normalizer with z-score parameters
        """
        self.config = config
        
        self.stage1 = TransitionLabelComparator()
        self.stage2 = TransitionEdgeComparator()
        self.stage3 = FeatureVectorComparator(
            feature_extractor=ModelFeatureExtractor(),
            normalizer=feature_normalizer
        )
        
        self.stats = {
            'total_input': 0,
            'stage1_filtered': 0,
            'stage2_filtered': 0,
            'stage3_filtered': 0,
            'final_unique': 0,
            'comparisons_performed': 0
        }

        self.report = None  # Will be populated after deduplication
    
    def deduplicate(
        self,
        nets: List[PetriNetItem]
    ) -> Tuple[List[PetriNetItem], Dict[int, int]]:
        """
        Deduplicate nets iteratively.
        
        Args:
            nets: List of all Petri nets to deduplicate
        
        Returns:
            Tuple of (unique_nets, duplicate_map)
            
            unique_nets: List of unique Petri nets
            duplicate_map: Dict mapping duplicate_idx -> representative_idx
                          Example: {5: 0, 12: 0} means nets 5 and 12 are duplicates of net 0
        """
        if not nets:
            return [], {}
        
        self.stats['total_input'] = len(nets)
        
        unique_nets = []
        duplicate_map = {}
        
        for current_net in tqdm(nets, disable=not self.config.verbose, desc="Deduplicating"):
            is_duplicate, representative_idx = self._find_duplicate(
                current_net,
                unique_nets
            )
            
            if is_duplicate:
                duplicate_map[current_net.idx] = representative_idx
            else:
                unique_nets.append(current_net)
        
        self.stats['final_unique'] = len(unique_nets)

        if self.config.verbose:
            self._print_stats()

        # Generate report
        self._generate_report(unique_nets, duplicate_map)

        return unique_nets, duplicate_map
    
    def _find_duplicate(
        self,
        candidate: PetriNetItem,
        unique_nets: List[PetriNetItem]
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if candidate is duplicate of any unique net.
        
        Args:
            candidate: Net to check
            unique_nets: List of already identified unique nets
        
        Returns:
            Tuple of (is_duplicate, representative_idx)
        """
        for unique_net in unique_nets:
            self.stats['comparisons_performed'] += 1
            
            if self.config.enable_stage1:
                sim1 = self.stage1.compare(
                    candidate.net, candidate.im, candidate.fm,
                    unique_net.net, unique_net.im, unique_net.fm
                )
                
                if sim1 < self.config.label_similarity_threshold:
                    continue
                
                self.stats['stage1_filtered'] += 1
            
            if self.config.enable_stage2:
                sim2 = self.stage2.compare(
                    candidate.net, candidate.im, candidate.fm,
                    unique_net.net, unique_net.im, unique_net.fm
                )
                
                if sim2 < self.config.edge_similarity_threshold:
                    continue
                
                self.stats['stage2_filtered'] += 1
            
            if self.config.enable_stage3:
                sim3 = self.stage3.compare(
                    candidate.net, candidate.im, candidate.fm,
                    unique_net.net, unique_net.im, unique_net.fm
                )
                
                if sim3 < self.config.feature_similarity_threshold:
                    continue
                
                self.stats['stage3_filtered'] += 1
            
            return True, unique_net.idx
        
        return False, None
    
    def _generate_report(
        self,
        unique_nets: List[PetriNetItem],
        duplicate_map: Dict[int, int]
    ):
        """
        Generate deduplication report.

        Args:
            unique_nets: List of unique nets
            duplicate_map: Mapping duplicate_idx -> representative_idx
        """
        self.report = {
            'num_unique': len(unique_nets),
            'num_total': self.stats['total_input'],
            'num_duplicates': self.stats['total_input'] - len(unique_nets),
            'reduction_percent': (
                (1 - len(unique_nets) / self.stats['total_input']) * 100
                if self.stats['total_input'] > 0 else 0.0
            ),
            'thresholds': {
                'label_threshold': self.config.label_similarity_threshold,
                'edge_threshold': self.config.edge_similarity_threshold,
                'feature_threshold': self.config.feature_similarity_threshold,
            },
            'stages_enabled': {
                'stage1': self.config.enable_stage1,
                'stage2': self.config.enable_stage2,
                'stage3': self.config.enable_stage3,
            },
            'duplicate_map': {
                str(k): v for k, v in duplicate_map.items()
            },
            'stats': self.stats.copy()
        }

    def get_report(self) -> Optional[Dict]:
        """
        Get the deduplication report.

        Returns:
            Report dict if deduplication was run, None otherwise
        """
        return self.report

    def _print_stats(self):
        """Print deduplication statistics."""
        print("\n" + "="*60)
        print("DEDUPLICATION STATISTICS")
        print("="*60)
        print(f"Total input nets:            {self.stats['total_input']}")
        print(f"Total comparisons performed: {self.stats['comparisons_performed']}")
        print(f"  Passed stage 1:            {self.stats['stage1_filtered']}")
        print(f"  Passed stage 2:            {self.stats['stage2_filtered']}")
        print(f"  Passed stage 3 (duplicates):{self.stats['stage3_filtered']}")
        print(f"Final unique nets:           {self.stats['final_unique']}")
        print(f"Duplicates found:            {self.stats['total_input'] - self.stats['final_unique']}")
        reduction = (1 - self.stats['final_unique']/self.stats['total_input']) * 100
        print(f"Reduction:                   {reduction:.1f}%")
        print("="*60 + "\n")