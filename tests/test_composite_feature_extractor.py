"""
Unit tests for feature extraction from Petri nets and traces.
"""

import os
import unittest
import numpy as np
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as petri_importer
from pm4py.objects.petri_net.utils.petri_utils import (
    add_transition, add_place, add_arc_from_to, remove_arc, construct_trace_net
)
from pm4py.vis import view_petri_net
from features import CompositeFeatureExtractor


class TestCompositeFeatureExtractor(unittest.TestCase):
    """
    Test suite for CompositeFeatureExtractor.
    
    Tests feature extraction from modified Petri nets and traces,
    verifying model features, trace features, and interaction features.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        cls.extractor = CompositeFeatureExtractor()
        cls.net, cls.marking, cls.fmarking = cls._create_modified_petri_net()
        cls.trace_net, cls.trace_im, cls.trace_fm = cls._create_modified_trace_net()
        view_petri_net(cls.net, cls.marking, cls.fmarking)
        view_petri_net(cls.trace_net, cls.trace_im, cls.trace_fm)
    
    @classmethod
    def _create_modified_petri_net(cls):
        """
        Create and modify the Petri net according to test_existing_heuristics.py.
        
        Returns:
            Tuple of (net, marking, fmarking)
        """
        log_path = os.path.join("pm4py", "tests", "input_data", "running-example.xes")
        pnml_path = os.path.join("pm4py", "tests", "input_data", "running-example.pnml")
        net, marking, fmarking = petri_importer.apply(pnml_path)
        
        # Add invisible transition after "register request"
        t_register_request = next(t for t in net.transitions if t.label == "register request")
        arc_t_p = next(arc for arc in net.arcs if arc.source == t_register_request)
        p_after_register_request = arc_t_p.target
        t_inv_new = add_transition(net, name="t_tau", label=None)
        p_new = add_place(net, name="p_new")
        add_arc_from_to(t_register_request, p_new, net)
        add_arc_from_to(p_new, t_inv_new, net)
        add_arc_from_to(t_inv_new, p_after_register_request, net)
        remove_arc(net, arc_t_p)
        
        # Add transition "register request" again after "reinitiate request"
        t_reinitiate_request = next(t for t in net.transitions if t.label == "reinitiate request")
        arc_t_p = next(arc for arc in net.arcs if arc.source == t_reinitiate_request)
        p_after_reinitiate_request = arc_t_p.target
        t_register_request_2 = add_transition(net, name="t_register_request_2", 
                                               label="register request")
        p_new_2 = add_place(net, name="p_new_2")
        add_arc_from_to(t_reinitiate_request, p_new_2, net)
        add_arc_from_to(p_new_2, t_register_request_2, net)
        add_arc_from_to(t_register_request_2, p_after_reinitiate_request, net)
        remove_arc(net, arc_t_p)
        
        # Add another "register request" after "check ticket"
        t_check_ticket = next(t for t in net.transitions if t.label == "check ticket")
        arc_t_p = next(arc for arc in net.arcs if arc.source == t_check_ticket)
        p_after_check_ticket = arc_t_p.target
        t_register_request_3 = add_transition(net, name="t_register_request_3",
                                               label="register request")
        p_new_3 = add_place(net, name="p_new_3")
        add_arc_from_to(t_check_ticket, p_new_3, net)
        add_arc_from_to(p_new_3, t_register_request_3, net)
        add_arc_from_to(t_register_request_3, p_after_check_ticket, net)
        remove_arc(net, arc_t_p)
        
        return net, marking, fmarking
    
    @classmethod
    def _create_modified_trace_net(cls):
        """
        Create and modify the trace net from the first trace.
        
        Returns:
            Tuple of (trace_net, trace_im, trace_fm)
        """
        log_path = os.path.join("pm4py", "tests", "input_data", "running-example.xes")
        log = xes_importer.apply(log_path)
        trace = log._list[0]
        trace_net, trace_im, trace_fm = construct_trace_net(trace)
        
        # Add "send reminder" after "reinitiate request"
        t_reinitiate_request = next(t for t in trace_net.transitions 
                                    if t.label == "reinitiate request")
        arc_t_p = next(arc for arc in trace_net.arcs if arc.source == t_reinitiate_request)
        p_after_reinitiate_request = arc_t_p.target
        p_new = add_place(trace_net, name="p_new_trace")
        t_send_reminder = add_transition(trace_net, name="t_send_reminder", 
                                        label="send reminder")
        add_arc_from_to(t_reinitiate_request, p_new, trace_net)
        add_arc_from_to(p_new, t_send_reminder, trace_net)
        add_arc_from_to(t_send_reminder, p_after_reinitiate_request, trace_net)
        remove_arc(trace_net, arc_t_p)
        
        return trace_net, trace_im, trace_fm
    
    def test_composite_feature_extraction(self):
        """Test that CompositeFeatureExtractor produces correct features."""
        features = self.extractor.extract(
            self.net, self.marking, self.fmarking,
            self.trace_net, self.trace_im, self.trace_fm,
            return_as_dict=True
        )
        
        inv_trans_in_degrees = [1, 1, 1]
        inv_trans_out_degrees = [1, 2, 1]
        uniq_trans_in_degrees = [1, 1, 1, 2, 1, 1, 1]
        uniq_trans_out_degrees = [1, 1, 1, 1, 1, 1, 1]
        dup_trans_in_degrees = [1, 1, 1]
        dup_trans_out_degrees = [1, 1, 1]
        place_in_degrees = [0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
        place_out_degrees = [0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
        
        trace_activity_counts = [1, 1, 1, 1, 1, 1, 2, 2]
        
        expected_features = {
            'model_n_transitions': 13,
            'model_n_places': 12,
            'model_n_arcs': 28,
            'model_n_inv_transition': 3,
            'model_n_dup_transition': 3,
            'model_n_uniq_transition': 7,
            'model_n_and_split': 1,
            'model_n_xor_split': 3,
            'model_inv_tran_in_deg_mean': np.mean(inv_trans_in_degrees),
            'model_inv_tran_in_deg_std': np.std(inv_trans_in_degrees),
            'model_inv_tran_out_deg_mean': np.mean(inv_trans_out_degrees),
            'model_inv_tran_out_deg_std': np.std(inv_trans_out_degrees),
            'model_uniq_tran_in_deg_mean': np.mean(uniq_trans_in_degrees),
            'model_uniq_tran_in_deg_std': np.std(uniq_trans_in_degrees),
            'model_uniq_tran_out_deg_mean': np.mean(uniq_trans_out_degrees),
            'model_uniq_tran_out_deg_std': np.std(uniq_trans_out_degrees),
            'model_dup_tran_in_deg_mean': np.mean(dup_trans_in_degrees),
            'model_dup_tran_in_deg_std': np.std(dup_trans_in_degrees),
            'model_dup_tran_out_deg_mean': np.mean(dup_trans_out_degrees),
            'model_dup_tran_out_deg_std': np.std(dup_trans_out_degrees),
            'model_place_in_deg_mean': np.mean(place_in_degrees),
            'model_place_in_deg_std': np.std(place_in_degrees),
            'model_place_out_deg_mean': np.mean(place_out_degrees),
            'model_place_out_deg_std': np.std(place_out_degrees),
            'trace_length': 10,
            'trace_activity_repeat_mean': np.mean(trace_activity_counts),
            'trace_activity_repeat_std': np.std(trace_activity_counts),
            'interaction_n_activity_present_in_model': 9,
            'interaction_n_activity_not_in_model': 1,
        }
        
        for feature_name, expected_value in expected_features.items():
            with self.subTest(feature=feature_name):
                self.assertAlmostEqual(
                    features[feature_name], 
                    expected_value,
                    places=10,
                    msg=f"Feature {feature_name} mismatch"
                )
    
    def test_feature_extraction_as_array(self):
        """Test that feature extraction returns numpy array when return_as_dict=False."""
        features_array = self.extractor.extract(
            self.net, self.marking, self.fmarking,
            self.trace_net, self.trace_im, self.trace_fm,
            return_as_dict=False
        )
        
        self.assertIsInstance(features_array, np.ndarray)
        self.assertEqual(len(features_array), len(self.extractor.feature_names))
        self.assertFalse(np.any(np.isnan(features_array)))
    
    def test_dict_to_vector_conversion(self):
        """Test conversion between dict and vector representations."""
        features_dict = self.extractor.extract(
            self.net, self.marking, self.fmarking,
            self.trace_net, self.trace_im, self.trace_fm,
            return_as_dict=True
        )
        features_array = self.extractor.extract(
            self.net, self.marking, self.fmarking,
            self.trace_net, self.trace_im, self.trace_fm,
            return_as_dict=False
        )
        
        converted_array = self.extractor.dict_to_vector(features_dict)
        np.testing.assert_array_almost_equal(converted_array, features_array)
        
        converted_dict = self.extractor.vector_to_dict(features_array)
        for key in features_dict:
            self.assertAlmostEqual(converted_dict[key], features_dict[key], places=10)


if __name__ == '__main__':
    unittest.main()