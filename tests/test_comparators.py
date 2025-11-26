"""
Unit tests for Petri net comparators.

Tests TransitionLabelComparator and PathBasedTransitionEdgeComparator
by loading the running example, creating modified variants, and
comparing manual calculations with comparator outputs.
"""

import os
import unittest
from collections import Counter
from pm4py.objects.petri_net.importer import importer as petri_importer
from pm4py.objects.petri_net.utils.petri_utils import (
    add_transition, add_place, add_arc_from_to, remove_arc
)
from deduplication.comparators import (
    TransitionLabelComparator,
    PathBasedTransitionEdgeComparator
)
from pm4py.vis import view_petri_net


class PetriNetTestBase:
    """
    Base class providing common Petri net loading and modification methods.

    This class is not a TestCase itself, but provides utility methods
    for loading and modifying Petri nets used across multiple test classes.
    """

    @staticmethod
    def load_running_example():
        """
        Load the original running example Petri net.

        Returns:
            Tuple of (net, initial_marking, final_marking)
        """
        pnml_path = os.path.join(
            "pm4py", "tests", "input_data", "running-example.pnml"
        )
        return petri_importer.apply(pnml_path)

    @staticmethod
    def create_modified_net():
        """
        Create a modified version of the running example.

        Modifications:
        1. Add invisible transition after "register request"
        2. Add duplicate "register request" after "reinitiate request"
        3. Remove one occurrence of "examine casually"

        Returns:
            Tuple of (net, initial_marking, final_marking)
        """
        pnml_path = os.path.join(
            "pm4py", "tests", "input_data", "running-example.pnml"
        )
        net, marking, fmarking = petri_importer.apply(pnml_path)

        # Modification 1: Add invisible transition after "register request"
        t_register = next(
            t for t in net.transitions if t.label == "register request"
        )
        arc_t_p = next(arc for arc in net.arcs if arc.source == t_register)
        p_after_register = arc_t_p.target

        t_inv = add_transition(net, name="t_tau_1", label=None)
        p_new = add_place(net, name="p_new_1")
        add_arc_from_to(t_register, p_new, net)
        add_arc_from_to(p_new, t_inv, net)
        add_arc_from_to(t_inv, p_after_register, net)
        remove_arc(net, arc_t_p)

        # Modification 2: Add duplicate "register request" after "reinitiate request"
        t_reinitiate = next(
            t for t in net.transitions if t.label == "reinitiate request"
        )
        arc_t_p = next(arc for arc in net.arcs if arc.source == t_reinitiate)
        p_after_reinitiate = arc_t_p.target

        t_register_2 = add_transition(
            net, name="t_register_2", label="register request"
        )
        p_new_2 = add_place(net, name="p_new_2")
        add_arc_from_to(t_reinitiate, p_new_2, net)
        add_arc_from_to(p_new_2, t_register_2, net)
        add_arc_from_to(t_register_2, p_after_reinitiate, net)
        remove_arc(net, arc_t_p)

        # Modification 3: Remove one "examine casually" transition
        t_examine = next(
            t for t in net.transitions if t.label == "examine casually"
        )
        # Remove all arcs connected to this transition
        arcs_to_remove = [
            arc for arc in net.arcs
            if arc.source == t_examine or arc.target == t_examine
        ]
        for arc in arcs_to_remove:
            remove_arc(net, arc)
        # Remove the transition itself
        net.transitions.remove(t_examine)

        return net, marking, fmarking


class TestTransitionLabelComparator(unittest.TestCase):
    """
    Test suite for TransitionLabelComparator.

    Tests label-based similarity computation using Bray-Curtis distance
    on the running example and a modified version.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        cls.comparator = TransitionLabelComparator(use_cache=False)
        cls.net1, cls.im1, cls.fm1 = PetriNetTestBase.load_running_example()
        cls.net2, cls.im2, cls.fm2 = PetriNetTestBase.create_modified_net()
        view_petri_net(cls.net1, cls.im1, cls.fm1)
        view_petri_net(cls.net2, cls.im2, cls.fm2)

    def test_label_counts_extraction(self):
        """Test that label counts are correctly extracted from nets."""
        counts1 = self.comparator._extract_label_counts(self.net1)
        expected_counts1 = Counter({
            'register request': 1,
            'examine thoroughly': 1,
            'examine casually': 1,
            'check ticket': 1,
            'decide': 1,
            'reinitiate request': 1,
            'pay compensation': 1,
            'reject request': 1,
        })
        self.assertEqual(counts1, expected_counts1)

        counts2 = self.comparator._extract_label_counts(self.net2)
        expected_counts2 = Counter({
            'register request': 2,
            'examine thoroughly': 1,
            'check ticket': 1,
            'decide': 1,
            'reinitiate request': 1,
            'pay compensation': 1,
            'reject request': 1,
        })
        self.assertEqual(counts2, expected_counts2)

    def test_bray_curtis_similarity_manual_calculation(self):
        """
        Test Bray-Curtis similarity with manual calculation.

        Manual calculation:

        Label counts from net1 and net2:
        All labels union: {
            'register request', 'examine thoroughly', 'examine casually',
            'check ticket', 'decide', 'reinitiate request',
            'pay compensation', 'reject request', 'τ'
        }

        Count vectors:
        Label                  | net1 | net2 | |net1-net2| | net1+net2
        ----------------------|------|------|------------|----------
        register request      |   1  |   2  |     1      |    3
        examine thoroughly    |   1  |   1  |     0      |    2
        examine casually      |   1  |   0  |     1      |    1
        check ticket          |   1  |   1  |     0      |    2
        decide                |   1  |   1  |     0      |    2
        reinitiate request    |   1  |   1  |     0      |    2
        pay compensation      |   1  |   1  |     0      |    2
        reject request        |   1  |   1  |     0      |    2
        ----------------------|------|------|------------|----------
        SUM                   |  8   |  8   |     2      |   16

        Bray-Curtis distance = numerator / denominator = 2 / 16 = 0.125
        Bray-Curtis similarity = 1 - distance = 1 - 0.125 = 0.875
        """
        expected_similarity = 1.0 - (2.0 / 16.0)  # = 0.875

        actual_similarity = self.comparator.compare(
            self.net1, self.im1, self.fm1,
            self.net2, self.im2, self.fm2
        )

        self.assertAlmostEqual(
            actual_similarity,
            expected_similarity,
            places=10,
            msg="Bray-Curtis similarity mismatch"
        )

    def test_identical_nets_similarity(self):
        """Test that identical nets have similarity of 1.0."""
        similarity = self.comparator.compare(
            self.net1, self.im1, self.fm1,
            self.net1, self.im1, self.fm1
        )
        self.assertAlmostEqual(similarity, 1.0, places=10)

    def test_symmetry(self):
        """Test that similarity is symmetric: sim(A, B) == sim(B, A)."""
        sim_ab = self.comparator.compare(
            self.net1, self.im1, self.fm1,
            self.net2, self.im2, self.fm2
        )
        sim_ba = self.comparator.compare(
            self.net2, self.im2, self.fm2,
            self.net1, self.im1, self.fm1
        )
        self.assertAlmostEqual(sim_ab, sim_ba, places=10)


class TestPathBasedTransitionEdgeComparator(unittest.TestCase):
    """Test suite verifying structural edge extraction and BC distance."""

    @classmethod
    def setUpClass(cls):
        cls.comparator = PathBasedTransitionEdgeComparator(use_cache=False)
        cls.net1, cls.im1, cls.fm1 = PetriNetTestBase.load_running_example()
        cls.net2, cls.im2, cls.fm2 = PetriNetTestBase.create_modified_net()

    def test_compare_extracted_edges_with_manual(self):
        """
        Verifies that extracted edges match manual extraction from images.
        """
        # Manual Extraction for Image 1 (Running Example)
        expected_edges_net1 = Counter({
            ('START', 'register request'): 1,
            ('register request', 'check ticket'): 1,
            ('register request', 'examine casually'): 1,
            ('register request', 'examine thoroughly'): 1,
            ('check ticket', 'decide'): 1,
            ('examine casually', 'decide'): 1,
            ('examine thoroughly', 'decide'): 1,
            ('decide', 'reinitiate request'): 1,
            ('decide', 'pay compensation'): 1,
            ('decide', 'reject request'): 1,
            ('reinitiate request', 'check ticket'): 1,
            ('reinitiate request', 'examine casually'): 1,
            ('reinitiate request', 'examine thoroughly'): 1,
            ('pay compensation', 'END'): 1,
            ('reject request', 'END'): 1
        })

        # Manual Extraction for Image 2 (Modified)
        # Changes: No 'examine casually', Loop goes to 'register request'
        expected_edges_net2 = Counter({
            ('START', 'register request'): 1,
            ('register request', 'check ticket'): 2,
            ('register request', 'examine thoroughly'): 2,
            ('check ticket', 'decide'): 1,
            ('examine thoroughly', 'decide'): 1,
            ('decide', 'reinitiate request'): 1,
            ('decide', 'pay compensation'): 1,
            ('decide', 'reject request'): 1,
            ('reinitiate request', 'register request'): 1,
            ('pay compensation', 'END'): 1,
            ('reject request', 'END'): 1,
        })

        actual_edges1 = self.comparator._extract_transition_edges(self.net1, self.im1, self.fm1)
        actual_edges2 = self.comparator._extract_transition_edges(self.net2, self.im2, self.fm2)

        # Assertions
        self.assertEqual(actual_edges1, expected_edges_net1, "Net 1 edges mismatch")
        self.assertEqual(actual_edges2, expected_edges_net2, "Net 2 edges mismatch")

    
    def test_bray_curtis_calculation(self):
        """
        Verifies comparator similarity against manual BC calculation with updated edges.
        
        Manual Calculation:
        
        1. Edge Counts:
           Net 1 Total: 15 edges
           Net 2 Total: 13 edges (1+2+2+1+1+1+1+1+1+1+1)
           Denominator (Sum of Totals) = 15 + 13 = 28

        2. Absolute Differences (|Count1 - Count2|):
           - ('reg', 'check'):    |1 - 2| = 1  (Diff due to path split in Net 2)
           - ('reg', 'ex tho'):   |1 - 2| = 1  (Diff due to path split in Net 2)
           - ('reg', 'ex cas'):   |1 - 0| = 1  (Missing in Net 2)
           - ('ex cas', 'dec'):   |1 - 0| = 1  (Missing in Net 2)
           - ('reinit', 'check'): |1 - 0| = 1  (Old Loop path)
           - ('reinit', 'ex cas'):|1 - 0| = 1  (Old Loop path)
           - ('reinit', 'ex tho'):|1 - 0| = 1  (Old Loop path)
           - ('reinit', 'reg'):   |0 - 1| = 1  (New Loop path)
           - All others match:    |1 - 1| = 0
           
           Numerator (Sum of Diffs) = 8

        3. Similarity:
           BC Distance = 8 / 28
           Similarity  = 1.0 - (8 / 28) = 20 / 28 = 0.7142857...
        """
        expected_similarity = 1.0 - (8.0 / 28.0)

        actual_similarity = self.comparator.compare(
            self.net1, self.im1, self.fm1,
            self.net2, self.im2, self.fm2
        )

        self.assertAlmostEqual(
            actual_similarity, 
            expected_similarity, 
            places=7, 
            msg="Bray-Curtis similarity calculation mismatch"
        )

    def test_identical_nets_similarity(self):
        """Test that identical nets have similarity of 1.0."""
        similarity = self.comparator.compare(
            self.net1, self.im1, self.fm1,
            self.net1, self.im1, self.fm1
        )
        self.assertAlmostEqual(similarity, 1.0, places=10)

    def test_symmetry(self):
        """Test that similarity is symmetric: sim(A, B) == sim(B, A)."""
        sim_ab = self.comparator.compare(
            self.net1, self.im1, self.fm1,
            self.net2, self.im2, self.fm2
        )
        sim_ba = self.comparator.compare(
            self.net2, self.im2, self.fm2,
            self.net1, self.im1, self.fm1
        )
        self.assertAlmostEqual(sim_ab, sim_ba, places=10)

if __name__ == '__main__':
    unittest.main()
