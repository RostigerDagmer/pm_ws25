"""
Feature extraction for Petri nets and traces for alignment heuristic recommendation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Union
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from collections import Counter
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.networkx_graph import (
    create_networkx_directed_graph,
)
import traceback
import logging

from pm4py.objects.process_tree.obj import Operator, ProcessTree
from pm4py.convert import convert_to_process_tree


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for feature extraction from Petri nets and traces.

    Provides common interface for extracting features as numpy arrays or dicts,
    with automatic conversion between the two representations.

    Includes optional caching mechanism to avoid re-extracting features
    from the same Petri net multiple times.
    """

    def __init__(self, use_cache: bool = True):
        """
        Initialize feature extractor.

        Args:
            use_cache: Whether to cache extracted features
        """
        self.use_cache = use_cache
        self._feature_cache = {} if use_cache else None

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Returns ordered list of feature names."""
        pass

    @abstractmethod
    def _extract_features_internal(self, *args, **kwargs) -> Dict[str, float]:
        """Internal feature extraction method. Must return a flat dict."""
        pass

    @abstractmethod
    def _compute_cache_key(self, *args, **kwargs):
        """
        Compute cache key from arguments.

        Subclasses must implement this to define how to generate unique
        cache keys for their specific inputs.

        Returns:
            Hashable cache key (typically int from id(), or tuple of ids)
        """
        raise NotImplementedError(
            "Subclasses must implement _compute_cache_key"
        )

    def extract(
        self,
        *args,
        return_as_dict: bool = False,
        use_cache: bool = None,
        **kwargs,
    ) -> Union[np.ndarray, Dict[str, float]]:
        """
        Extract features from input.

        Args:
            return_as_dict: If True, return dict. Otherwise return numpy array.
            use_cache: Override instance cache setting. If None, uses self.use_cache.

        Returns:
            Feature vector as numpy array or dict.
        """
        should_cache = self.use_cache if use_cache is None else use_cache

        # Check cache if enabled
        if should_cache:
            cache_key = self._compute_cache_key(*args, **kwargs)
            if cache_key is not None and cache_key in self._feature_cache:
                cached_dict = self._feature_cache[cache_key]
                if return_as_dict:
                    return cached_dict
                return self.dict_to_vector(cached_dict)

        # Extract features
        feature_dict = self._extract_features_internal(*args, **kwargs)
        assert set(feature_dict.keys()) == set(self.feature_names), (
            f"Extracted features do not match expected feature names. "
            f"Expected: {self.feature_names}, but got: {feature_dict.keys()}"
        )

        # Cache if enabled
        if should_cache:
            cache_key = self._compute_cache_key(*args, **kwargs)
            if cache_key is not None:
                self._feature_cache[cache_key] = feature_dict

        if return_as_dict:
            return feature_dict
        return self.dict_to_vector(feature_dict)

    def extract_batched(
        self,
        *args,
        return_as_dict: bool = False,
        use_cache: bool = None,
        **kwargs,
    ) -> list[Dict[str, float]] | list[np.ndarray]:
        """Extract features for a batch of traces."""
        # TODO: caching
        feats = self._extract_features_batch(*args, **kwargs)
        if return_as_dict:
            return feats
        return [self.dict_to_vector(feat) for feat in feats]

    def dict_to_vector(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy array using feature_names order."""
        return np.nan_to_num(
            np.array([feature_dict[k] for k in self.feature_names])
        )

    def vector_to_dict(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """Convert feature vector to dict using feature_names order."""
        return dict(zip(self.feature_names, feature_vector))


class ModelFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts structural features from Petri nets.
    Features include:
    - Basic counts (transitions, places, arcs)
    - Transition types (invisible, unique, duplicate)
    - Split patterns (AND, XOR)
    - Degree statistics per transition/place type
    """

    def _compute_cache_key(self, net: PetriNet, im: Marking, fm: Marking):
        """Use hash of the Petri net as cache key."""
        return hash(net)

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
        features.update(self._extract_split_patterns(net))
        features.update(
            self._extract_degree_stats(
                G, node_ids, net, inv_trans, uniq_trans, dup_trans
            )
        )

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

    def _extract_split_patterns(self, net):
        and_splits = sum(1 for t in net.transitions if len(t.out_arcs) > 1)
        xor_splits = sum(1 for p in net.places if len(p.out_arcs) > 1)
        return {
            'model_n_and_split': and_splits,
            'model_n_xor_split': xor_splits,
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

        return features


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
            None,  # Leaf nodes
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

        child_values = [
            self._calculate_state_space(child) for child in node.children
        ]

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


class CompositeFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts features from both model and trace, including interaction features.

    Combines features from ModelFeatureExtractor and TraceFeatureExtractor,
    then adds interaction features that relate model and trace characteristics.
    """

    def __init__(self, use_cache: bool = True):
        super().__init__(use_cache=use_cache)
        self.model_extractor = ModelFeatureExtractor(use_cache=use_cache)
        self.trace_extractor = TraceFeatureExtractor(use_cache=use_cache)
        self.state_space_extractor = StateSpaceSizeExtractor(
            use_cache=use_cache
        )

    def _compute_cache_key(
        self,
        petri_net: PetriNet,
        petri_net_im: Marking,
        petri_net_fm: Marking,
        trace_net: PetriNet,
        trace_net_im: Marking,
        trace_net_fm: Marking,
    ):
        """Use tuple of both net hashes as cache key."""
        return (hash(petri_net), hash(trace_net))

    @property
    def feature_names(self) -> List[str]:
        return (
            self.model_extractor.feature_names
            + self.trace_extractor.feature_names
            + self.state_space_extractor.feature_names
            + [
                'interaction_n_activity_present_in_model',  # e.g. model has A,B,C and trace has A,B,A,D -> 3
                'interaction_n_activity_not_in_model',
            ]  # e.g. model has A,B,C and trace has A,B,A,D -> 1
        )

    def _extract_features_internal(
        self,
        petri_net: PetriNet,
        petri_net_im: Marking,
        petri_net_fm: Marking,
        trace_net: PetriNet,
        trace_net_im: Marking,
        trace_net_fm: Marking,
    ) -> Dict[str, float]:
        """Extract model, trace, and interaction features."""
        model_features = self.model_extractor.extract(
            petri_net, petri_net_im, petri_net_fm, return_as_dict=True
        )
        trace_features = self.trace_extractor.extract(
            trace_net, trace_net_im, trace_net_fm, return_as_dict=True
        )
        state_space_features = self.state_space_extractor.extract(
            petri_net, petri_net_im, petri_net_fm, return_as_dict=True
        )
        interaction_features = self._extract_interactions(petri_net, trace_net)

        return {
            **model_features,
            **trace_features,
            **state_space_features,
            **interaction_features,
        }

    def _extract_features_batch(
        self,
        petri_net: PetriNet,
        petri_net_im: Marking,
        petri_net_fm: Marking,
        trace_nets: list[tuple[PetriNet, Marking, Marking]],
    ) -> list[Dict[str, float]]:
        """Extract features for a batch of traces."""
        model_features = self.model_extractor.extract(
            petri_net, petri_net_im, petri_net_fm, return_as_dict=True
        )
        trace_features = [
            self.trace_extractor.extract(
                trace_net, trace_net_im, trace_net_fm, return_as_dict=True
            )
            for trace_net, trace_net_im, trace_net_fm in trace_nets
        ]
        state_space_features = self.state_space_extractor.extract(
            petri_net, petri_net_im, petri_net_fm, return_as_dict=True
        )
        interaction_features = [
            self._extract_interactions(petri_net, trace_net)
            for trace_net, _, _ in trace_nets
        ]

        return [
            {
                **model_features,
                **trace_features[i],
                **state_space_features,
                **interaction_features[i],
            }
            for i in range(len(trace_nets))
        ]

    def _extract_interactions(
        self, model_net: PetriNet, trace_net: PetriNet
    ) -> Dict[str, float]:
        """Extract interaction features between model and trace."""
        model_labels = {
            t.label for t in model_net.transitions if t.label is not None
        }
        trace_labels = [
            t.label for t in trace_net.transitions if t.label is not None
        ]

        present_in_model = sum(
            1 for label in trace_labels if label in model_labels
        )
        not_in_model = sum(
            1 for label in trace_labels if label not in model_labels
        )

        return {
            'interaction_n_activity_present_in_model': present_in_model,
            'interaction_n_activity_not_in_model': not_in_model,
        }


class SpectralFeatureExtractor(BaseFeatureExtractor):

    def __init__(self, d_model: int, n_coeffs: int, use_cache: bool = False):
        """
        Args:
            d_model: Number of spectral dimensions to use (k).
            n_coeffs: Number of DCT coefficients to keep (C).
            use_cache: Whether to cache extracted features.
        """
        super().__init__(use_cache=use_cache)
        self.d_model = d_model
        self.n_coeffs = n_coeffs
        self._dct_matrix_cache = {}

    @property
    def dim(self):
        return self.d_model * self.n_coeffs

    def _compute_cache_key(
        self,
        petri_net: PetriNet,
        petri_net_im: Marking,
        petri_net_fm: Marking,
        trace_net: PetriNet,
        trace_net_im: Marking,
        trace_net_fm: Marking,
    ):
        return (hash(petri_net), hash(trace_net))

    @property
    def feature_names(self) -> List[str]:
        # Output size is n_coeffs * (d_model + 1)
        # The +1 is for the "unknown activity" dimension
        names = []
        for c in range(self.n_coeffs):
            for d in range(self.d_model):
                names.append(f"spectral_dct_c{c}_d{d}")
        return names

    def _extract_features_internal(
        self,
        petri_net: PetriNet,
        petri_net_im: Marking,
        petri_net_fm: Marking,
        trace_net: PetriNet,
        trace_net_im: Marking,
        trace_net_fm: Marking,
    ) -> Dict[str, float]:

        tensors = self.extract_tensors(
            petri_net,
            petri_net_im,
            petri_net_fm,
            trace_net,
            trace_net_im,
            trace_net_fm,
        )

        flat_features = tensors["trace_embedding"].tolist()
        return dict(zip(self.feature_names, flat_features))

    def compute_basis_and_maps(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        labels: list[str],
    ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """
        Returns:
        basis_labels: List[str] length T_vocab (no silent labels)
        basis:        [T_vocab, d_basis]
        local_to_vocab: [T_local] with -1 for silent or not-in-vocab, else vocab idx
        """
        A = (post - pre).float()  # [T_local, P]
        T_local, P = A.shape

        # local_to_vocab always defined
        local_to_vocab = torch.full(
            (T_local,), -1, dtype=torch.long, device=A.device
        )

        assert T_local > 0, f"Invalid vocab size for net: {pre, post, labels}"

        U, S, Vh = torch.linalg.svd(A, full_matrices=False)

        # Take k components from U (rows correspond to transitions)
        k = min(self.d_model, U.shape[1])
        U_k = U[:, :k]
        if k < self.d_model:
            U_k = torch.cat(
                [
                    U_k,
                    torch.zeros(
                        (T_local, self.d_model - k),
                        device=A.device,
                        dtype=U_k.dtype,
                    ),
                ],
                dim=1,
            )

        # Merge by label (skip silent)
        # Keep deterministic label order: first occurrence order (not sorted)
        label_to_row = {}
        sums = []
        counts = []

        for j, lab in enumerate(labels):
            if lab == "":
                continue  # silent excluded from basis vocab
            if lab not in label_to_row:
                label_to_row[lab] = len(sums)
                sums.append(U_k[j].clone())
                counts.append(1)
            else:
                r = label_to_row[lab]
                sums[r] += U_k[j]
                counts[r] += 1

        if not sums:
            return (
                [],
                torch.zeros((0, self.d_model), device=A.device),
                local_to_vocab,
            )

        basis_labels = list(label_to_row.keys())  # first-occurrence order
        basis = torch.stack(
            [s / c for s, c in zip(sums, counts)], dim=0
        )  # [T_vocab, d_basis]

        # Build local_to_vocab: local index -> vocab id
        for j, lab in enumerate(labels):
            if lab == "":
                continue
            local_to_vocab[j] = label_to_row[lab]

        return basis_labels, basis, local_to_vocab
