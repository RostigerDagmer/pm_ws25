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
    PathBasedTransitionEdgeComparator,
    DualScoreFeatureComparator,
    CombinedComparator
)
from features.extractors import ModelFeatureExtractor


logger = logging.getLogger(__name__)


@dataclass
class DeduplicationConfig:
    """
    Configuration for improved deduplication pipeline.

    Two-stage pipeline:
        Stage 1 (Prefilter): Transition label counts comparison
        Stage 2 (Combined): Path-based edges + dual-score features

    To be considered a duplicate, nets must pass both stages:
        similarity >= label_similarity_threshold (stage 1)
        similarity >= combined_similarity_threshold (stage 2)
    """

    # Stage 1: Label-based prefilter (similarity, higher = more similar)
    label_similarity_threshold: float = 0.95

    # Stage 2: Combined edge+feature comparison (similarity, higher = more similar)
    combined_similarity_threshold: float = 0.95

    # Enable/disable stages
    enable_stage1: bool = True
    enable_stage2: bool = True

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

    Uses improved two-stage comparison with early stopping:
        Stage 1: Transition label counts (Bray-Curtis, prefilter)
        Stage 2: Combined path-based edges + dual-score features
    """

    def __init__(self, config: DeduplicationConfig):
        """
        Initialize deduplicator.

        Args:
            config: Deduplication configuration
        """
        self.config = config

        # Initialize comparison logging
        self.comparison_log = []
        self._current_comparison = {}

        # Create callbacks for score collection
        def label_callback(scores):
            self._current_comparison.update(scores)

        def edge_callback(scores):
            self._current_comparison.update(scores)

        def feature_callback(scores):
            self._current_comparison.update(scores)

        def combined_callback(scores):
            self._current_comparison.update(scores)

        # Stage 1: Label-based prefilter
        self.stage1 = TransitionLabelComparator(debug_callback=label_callback)

        # Stage 2: Combined edge + feature comparison
        edge_comparator = PathBasedTransitionEdgeComparator(debug_callback=edge_callback)
        feature_comparator = DualScoreFeatureComparator(debug_callback=feature_callback)
        self.stage2 = CombinedComparator(
            edge_comparator=edge_comparator,
            feature_comparator=feature_comparator,
            debug_callback=combined_callback
        )

        self.stats = {
            'total_input': 0,
            'stage1_filtered': 0,
            'stage2_filtered': 0,
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

        # Generate report
        self._generate_report(unique_nets, duplicate_map)

        if self.config.verbose:
            self.print_report()

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

            # Reset current comparison dict
            self._current_comparison = {
                'candidate_idx': candidate.idx,
                'unique_idx': unique_net.idx,
                'passed_stage1': False,
                'passed_stage2': False,
                'is_duplicate': False
            }

            # Stage 1: Label-based prefilter (similarity metric)
            if self.config.enable_stage1:
                label_similarity = self.stage1.compare(
                    candidate.net, candidate.im, candidate.fm,
                    unique_net.net, unique_net.im, unique_net.fm
                )

                if label_similarity < self.config.label_similarity_threshold:
                    # Log and continue
                    self.comparison_log.append(self._current_comparison.copy())
                    continue

                self._current_comparison['passed_stage1'] = True
                self.stats['stage1_filtered'] += 1

            # Stage 2: Combined edge+feature comparison (similarity metric)
            if self.config.enable_stage2:
                combined_similarity = self.stage2.compare(
                    candidate.net, candidate.im, candidate.fm,
                    unique_net.net, unique_net.im, unique_net.fm
                )

                if combined_similarity < self.config.combined_similarity_threshold:
                    # Log and continue
                    self.comparison_log.append(self._current_comparison.copy())
                    continue

                self._current_comparison['passed_stage2'] = True
                self.stats['stage2_filtered'] += 1

            # Passed all enabled stages → is a duplicate
            self._current_comparison['is_duplicate'] = True
            self.comparison_log.append(self._current_comparison.copy())
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
                'combined_threshold': self.config.combined_similarity_threshold,
            },
            'stages_enabled': {
                'stage1': self.config.enable_stage1,
                'stage2': self.config.enable_stage2,
            },
            'duplicate_map': {
                str(k): v for k, v in duplicate_map.items()
            },
            'stats': self.stats.copy(),
            'comparison_log': self.comparison_log
        }

    def get_report(self) -> Optional[Dict]:
        """
        Get the deduplication report.

        Returns:
            Report dict if deduplication was run, None otherwise
        """
        return self.report

    def print_report(self):
        """Print formatted deduplication report."""
        if self.report is None:
            print("No deduplication report available. Run deduplicate() first.")
            return

        print("\n" + "="*70)
        print("DEDUPLICATION REPORT (Improved Pipeline)")
        print("="*70)
        print(f"Total input nets:            {self.report['num_total']}")
        print(f"Final unique nets:           {self.report['num_unique']}")
        print(f"Duplicates found:            {self.report['num_duplicates']}")
        print(f"Reduction:                   {self.report['reduction_percent']:.1f}%")
        print()
        print("COMPARISON STATISTICS:")
        print(f"  Total comparisons:         {self.report['stats']['comparisons_performed']}")
        print(f"  Passed stage 1 (labels):   {self.report['stats']['stage1_filtered']}")
        print(f"  Passed stage 2 (combined): {self.report['stats']['stage2_filtered']}")
        print()
        print("THRESHOLDS:")
        print(f"  Label similarity:          >= {self.report['thresholds']['label_threshold']:.2f}")
        print(f"  Combined similarity:       >= {self.report['thresholds']['combined_threshold']:.2f}")
        print()
        print("STAGES ENABLED:")
        print(f"  Stage 1 (prefilter):       {self.report['stages_enabled']['stage1']}")
        print(f"  Stage 2 (combined):        {self.report['stages_enabled']['stage2']}")
        print("="*70 + "\n")