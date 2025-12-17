from typing import Dict, List, Union
import numpy as np
from pm4py.objects.petri_net.obj import PetriNet, Marking
import traceback
import logging
            
from pm4py.objects.process_tree.obj import Operator, ProcessTree
from pm4py.convert import convert_to_process_tree
from features.base_extractor import BaseFeatureExtractor


class StateSpaceSizeExtractor(BaseFeatureExtractor):
    """
    Extracts state space size feature based on Process Tree conversion.
    
    Calculates a measure of state space size by converting the Petri net to a Process Tree
    and recursively combining values:
    - Leaf: 1
    - SEQ, XOR, LOOP: Sum of children
    - AND (PARALLEL): Product of children
    """

    def _compute_cache_key(self, net: PetriNet, im: Marking, fm: Marking):
        """Use hash of the Petri net as cache key."""
        return hash(net)

    @property
    def feature_names(self) -> List[str]:
        return ['state_space_size']

    def _extract_features_internal(
        self, net: PetriNet, im: Marking, fm: Marking
    ) -> Dict[str, float]:
        """Extract state space size feature."""
        try:
            tree = convert_to_process_tree(net, im, fm)
            self._check_unsupported_operators(tree)
            size = self._calculate_state_space(tree)
            
            return {'state_space_size': float(size)}
        except Exception as e:
            traceback.print_exc()
            logging.error(f"Conversion failed: {repr(e)}")
            # Return -1.0 if conversion fails
            return {'state_space_size': -1.0}

    def _check_unsupported_operators(self, node: ProcessTree):
        """Recursively check for unsupported operators in the process tree."""
        if node.operator not in {
            Operator.SEQUENCE,
            Operator.PARALLEL,
            Operator.XOR,
            Operator.LOOP,
            None, # Leaf nodes
        }:
            logging.warning(
                f"Process Tree contains unsupported operator {node.operator} for state space size calculation."
            )

        # Recursively check children
        for child in node.children:
            self._check_unsupported_operators(child)

    def _calculate_state_space(self, node: ProcessTree) -> float:
        if not node.children:
            # log(2) because log(1) = 0 would result in 0 for and-nodes with 
            # multiple child leaf nodes
            return np.log(2)

        child_values = [self._calculate_state_space(child) for child in node.children]

        if node.operator in [Operator.SEQUENCE, Operator.XOR, Operator.LOOP]:
            # LogSumExp
            # Child 1: log(c1), Child 2: log(c2) -> log(c1 + c2)
            return np.logaddexp.reduce(child_values)
        elif node.operator == Operator.PARALLEL:
            # Sum (equivalent to Product in original domain)
            # Child 1: log(c1), Child 2: log(c2) -> log(c1 * c2)
            return np.sum(child_values)
        else:
            # Default to LogSumExp for other operators (treating them as choice/sequence-like)
            return np.logaddexp.reduce(child_values)