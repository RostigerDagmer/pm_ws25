from typing import Dict, List, Union
import numpy as np
from collections import Counter
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.networkx_graph import (
    create_networkx_directed_graph,
)

from features.base_extractor import BaseFeatureExtractor


class ModelFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts structural features from Petri nets.
    Features include:
    - Basic counts (transitions, places, arcs)
    - Transition types (invisible, unique, duplicate)
    - Split patterns (AND, XOR)
    - Degree statistics per transition/place type
    """

    @property
    def feature_names(self) -> List[str]:
        return [
            'model_n_transitions',
            'model_n_places',
            'model_n_arcs',
            'model_n_inv_transition',
            'model_n_dup_transition',  # Number of transitions with a label that appears more than once
            'model_n_uniq_transition',
            'model_n_and_split',
            'model_n_xor_split',
            'model_inv_tran_in_deg_mean',
            'model_inv_tran_in_deg_std',
            'model_inv_tran_out_deg_mean',
            'model_inv_tran_out_deg_std',
            'model_uniq_tran_in_deg_mean',
            'model_uniq_tran_in_deg_std',
            'model_uniq_tran_out_deg_mean',
            'model_uniq_tran_out_deg_std',
            'model_dup_tran_in_deg_mean',
            'model_dup_tran_in_deg_std',
            'model_dup_tran_out_deg_mean',
            'model_dup_tran_out_deg_std',
            'model_place_in_deg_mean',
            'model_place_in_deg_std',
            'model_place_out_deg_mean',
            'model_place_out_deg_std',
            'model_tran_in_deg_mean',
            'model_tran_in_deg_std',
            'model_tran_out_deg_mean',
            'model_tran_out_deg_std',
            'model_and_split_avg_out_deg',
            'model_and_split_max_out_deg',
            'model_and_split_out_deg_std',
            'model_xor_split_avg_out_deg',
            'model_xor_split_max_out_deg',
            'model_xor_split_out_deg_std',
            'model_density_arcs_per_transition',
            'model_density_arcs_per_transition_plus_places',
            'model_density_arcs_per_place',
        ]

    def _extract_features_internal(
        self, net: PetriNet, im: Marking, fm: Marking
    ) -> Dict[str, float]:
        """Extract all model features."""
        features = {}

        G, inv_dict = create_networkx_directed_graph(net)
        node_ids = {v: k for k, v in inv_dict.items()}

        inv_trans = [t for t in net.transitions if t.label is None]
        visible_trans = [t for t in net.transitions if t.label is not None]

        labels = [t.label for t in visible_trans]
        label_counts = Counter(labels)
        uniq_trans = [t for t in visible_trans if label_counts[t.label] == 1]
        dup_trans = [t for t in visible_trans if label_counts[t.label] > 1]

        features.update(
            self._extract_counts(net, inv_trans, uniq_trans, dup_trans)
        )
        features.update(self._extract_split_patterns(net, G, node_ids))
        features.update(
            self._extract_degree_stats(
                G, node_ids, net, inv_trans, uniq_trans, dup_trans
            )
        )
        features.update(self._extract_density_features(net))

        return features

    def _extract_counts(self, net, inv_trans, uniq_trans, dup_trans):
        return {
            'model_n_transitions': len(net.transitions),
            'model_n_places': len(net.places),
            'model_n_arcs': len(net.arcs),
            'model_n_inv_transition': len(inv_trans),
            'model_n_dup_transition': len(dup_trans),
            'model_n_uniq_transition': len(uniq_trans),
        }

    def _extract_split_patterns(self, net, G, node_ids):
        and_splits = [t for t in net.transitions if len(t.out_arcs) > 1]
        xor_splits = [p for p in net.places if len(p.out_arcs) > 1]

        # AND split out degree statistics
        if and_splits:
            and_split_out_degs = [
                G.out_degree(node_ids[t]) for t in and_splits
            ]
            and_split_avg = np.mean(and_split_out_degs)
            and_split_max = np.max(and_split_out_degs)
            and_split_std = np.std(and_split_out_degs)
        else:
            and_split_avg = 0.0
            and_split_max = 0.0
            and_split_std = 0.0

        # XOR split out degree statistics
        if xor_splits:
            xor_split_out_degs = [
                G.out_degree(node_ids[p]) for p in xor_splits
            ]
            xor_split_avg = np.mean(xor_split_out_degs)
            xor_split_max = np.max(xor_split_out_degs)
            xor_split_std = np.std(xor_split_out_degs)
        else:
            xor_split_avg = 0.0
            xor_split_max = 0.0
            xor_split_std = 0.0

        return {
            'model_n_and_split': len(and_splits),
            'model_n_xor_split': len(xor_splits),
            'model_and_split_avg_out_deg': and_split_avg,
            'model_and_split_max_out_deg': and_split_max,
            'model_and_split_out_deg_std': and_split_std,
            'model_xor_split_avg_out_deg': xor_split_avg,
            'model_xor_split_max_out_deg': xor_split_max,
            'model_xor_split_out_deg_std': xor_split_std,
        }

    def _extract_degree_stats(
        self, G, node_ids, net, inv_trans, uniq_trans, dup_trans
    ):
        features = {}

        for prefix, trans_list in [
            ('inv_tran', inv_trans),
            ('uniq_tran', uniq_trans),
            ('dup_tran', dup_trans),
        ]:
            in_degs = (
                [G.in_degree(node_ids[t]) for t in trans_list]
                if trans_list
                else [0]
            )
            out_degs = (
                [G.out_degree(node_ids[t]) for t in trans_list]
                if trans_list
                else [0]
            )
            features[f'model_{prefix}_in_deg_mean'] = np.mean(in_degs)
            features[f'model_{prefix}_in_deg_std'] = np.std(in_degs)
            features[f'model_{prefix}_out_deg_mean'] = np.mean(out_degs)
            features[f'model_{prefix}_out_deg_std'] = np.std(out_degs)

        place_in_degs = [G.in_degree(node_ids[p]) for p in net.places]
        place_out_degs = [G.out_degree(node_ids[p]) for p in net.places]
        features['model_place_in_deg_mean'] = np.mean(place_in_degs)
        features['model_place_in_deg_std'] = np.std(place_in_degs)
        features['model_place_out_deg_mean'] = np.mean(place_out_degs)
        features['model_place_out_deg_std'] = np.std(place_out_degs)

        all_trans_in_degs = [G.in_degree(node_ids[t]) for t in net.transitions]
        all_trans_out_degs = [
            G.out_degree(node_ids[t]) for t in net.transitions
        ]
        features['model_tran_in_deg_mean'] = np.mean(all_trans_in_degs)
        features['model_tran_in_deg_std'] = np.std(all_trans_in_degs)
        features['model_tran_out_deg_mean'] = np.mean(all_trans_out_degs)
        features['model_tran_out_deg_std'] = np.std(all_trans_out_degs)

        return features

    def _extract_density_features(self, net):
        """Extract density features (ratios of arcs to nodes)."""
        n_arcs = len(net.arcs)
        n_transitions = len(net.transitions)
        n_places = len(net.places)
        n_total_nodes = n_transitions + n_places

        density_arcs_per_transition = (
            n_arcs / n_transitions if n_transitions > 0 else 0.0
        )
        density_arcs_per_place = n_arcs / n_places if n_places > 0 else 0.0
        density_arcs_per_total = (
            n_arcs / n_total_nodes if n_total_nodes > 0 else 0.0
        )

        return {
            'model_density_arcs_per_transition': density_arcs_per_transition,
            'model_density_arcs_per_transition_plus_places': density_arcs_per_total,
            'model_density_arcs_per_place': density_arcs_per_place,
        }
