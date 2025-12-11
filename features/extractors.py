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
        self, *args, return_as_dict: bool = False, use_cache: bool = None, **kwargs
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
        self.state_space_extractor = StateSpaceSizeExtractor(use_cache=use_cache)

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
        self.d_model = d_model - 1
        self.n_coeffs = n_coeffs
        self._dct_matrix_cache = {}

    @property
    def dim(self):
        return (self.d_model + 1) * self.n_coeffs

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
            for d in range(self.d_model + 1):
                names.append(f"spectral_dct_c{c}_d{d}")
        return names

    def _get_dct_matrix(self, N: int) -> torch.Tensor:
        """
        Compute DCT-II matrix of size (n_coeffs, N).
        D[k, n] = cos(pi * k * (2n + 1) / (2N))
        """
        if N in self._dct_matrix_cache:
            return self._dct_matrix_cache[N]

        k = torch.arange(self.n_coeffs).unsqueeze(1)  # [C, 1]
        n = torch.arange(N).unsqueeze(0)  # [1, N]

        # DCT-II formula
        # X_k = sum_{n=0}^{N-1} x_n * cos(pi * k * (2n + 1) / (2N))
        # We want the matrix D such that X = D @ x
        # So D[k, n] = cos(...)

        dct_mat = torch.cos(np.pi * k * (2 * n + 1) / (2 * N))
        self._dct_matrix_cache[N] = dct_mat
        return dct_mat

    def _compute_basis(self, net: PetriNet) -> Dict[str, torch.Tensor]:
        """
        Compute spectral basis for the Petri net.
        Returns a dictionary mapping transition labels to their spectral embeddings.
        """
        # 1. Build Incidence Matrix
        transitions = list(net.transitions)
        places = list(net.places)
        T = len(transitions)
        P = len(places)

        place_idx = {p: i for i, p in enumerate(places)}

        # Incidence matrix A: [P, T] (or [T, P], let's do [T, P] to match SVD on transitions)
        # Actually, usually A is [P, T]. Let's stick to [T, P] so rows are transitions.
        # A[j, i] = Post(t_j, p_i) - Pre(t_j, p_i)
        A = torch.zeros((T, P), dtype=torch.float)

        for j, t in enumerate(transitions):
            for arc in t.out_arcs:
                A[j, place_idx[arc.target]] += 1
            for arc in t.in_arcs:
                A[j, place_idx[arc.source]] -= 1

        # 2. Compute SVD
        # A = U S V^T
        # U: [T, T] - columns are left singular vectors (eigen-transitions)
        # We want the rows of U corresponding to transitions.
        # If we take top k components: U_k: [T, k]

        if T == 0:
            return {}

        try:
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        except RuntimeError:
            # Fallback for empty/singular cases if needed, though full_matrices=False usually handles it
            U = torch.zeros((T, min(T, P)))

        # Keep top k dimensions
        k = min(self.d_model, U.shape[1])
        U_k = U[:, :k]

        # Pad if k < d_model
        if k < self.d_model:
            padding = torch.zeros((T, self.d_model - k))
            U_k = torch.cat([U_k, padding], dim=1)

        # 3. Map labels to embeddings
        # Handle duplicate labels by averaging their embeddings
        label_embeddings = {}
        label_counts = {}

        for j, t in enumerate(transitions):
            if t.label is None:
                continue

            emb = U_k[j]
            if t.label not in label_embeddings:
                label_embeddings[t.label] = emb
                label_counts[t.label] = 1
            else:
                label_embeddings[t.label] += emb
                label_counts[t.label] += 1

        # Average
        for label in label_embeddings:
            label_embeddings[label] /= label_counts[label]

        return label_embeddings

    def extract_tensors(
        self,
        petri_net: PetriNet,
        petri_net_im: Marking,
        petri_net_fm: Marking,
        trace_net: PetriNet,
        trace_net_im: Marking,
        trace_net_fm: Marking,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features as structured tensors.
        Returns:
            dict with keys:
            - 'model_basis': [T, d_model] tensor
            - 'trace_embedding': [d_trace] tensor (flattened DCT features)
        """
        # 1. Compute Model Basis
        label_embeddings = self._compute_basis(petri_net)

        # Convert label embeddings to a tensor sequence [T, d_model]
        # We need a consistent ordering. The _compute_basis uses net.transitions iteration.
        # We should probably return the basis as it corresponds to the transitions in the net?
        # Or just the unique label embeddings?
        # The user said: "Take the spectral features of _compute_basis as an input sequence"
        # _compute_basis returns Dict[str, Tensor].
        # If we want a sequence, we probably want the embeddings for each transition in the model?
        # Or just the set of unique embeddings?
        # "Cross attend between that 'model sequence' and the single trace vector"
        # If we use unique embeddings, we lose structural info about duplicates (though they are averaged in _compute_basis).
        # Let's return the values of the dictionary as a sequence.
        # Ideally we'd want to keep them somewhat consistent, but since it's a set of labels...
        # Let's sort by label to be deterministic.

        sorted_labels = sorted(label_embeddings.keys())
        if not sorted_labels:
            # Handle empty case
            model_basis = torch.zeros((1, self.d_model))
        else:
            model_basis = torch.stack(
                [label_embeddings[l] for l in sorted_labels]
            )

        # 2. Map Trace to Trajectory
        trace_labels = [
            t.label for t in trace_net.transitions if t.label is not None
        ]
        L = len(trace_labels)

        if L == 0:
            trace_embedding = torch.zeros(self.dim)
        else:
            # Trajectory: [L, d_model + 1]
            trajectory = torch.zeros((L, self.d_model + 1))

            for i, label in enumerate(trace_labels):
                if label in label_embeddings:
                    # Known: [emb, 0]
                    trajectory[i, : self.d_model] = label_embeddings[label]
                    trajectory[i, self.d_model] = 0.0
                else:
                    # Unknown: [0...0, 1]
                    trajectory[i, : self.d_model] = 0.0
                    trajectory[i, self.d_model] = 1.0

            # 3. DCT Compression
            D = self._get_dct_matrix(L)  # [C, L]
            dct_features = D @ trajectory  # [C, D+1]
            trace_embedding = dct_features.flatten()

        return {"model_basis": model_basis, "trace_embedding": trace_embedding}

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

    def _compute_basis_from_tensors(
        self, pre: torch.Tensor, post: torch.Tensor, labels: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute spectral basis from net tensors.
        Args:
            pre: [T, P]
            post: [T, P]
            labels: List of transition labels
        """
        # Incidence matrix A: [T, P]
        # A[j, i] = Post(t_j, p_i) - Pre(t_j, p_i)
        A = (post - pre).float()

        T, P = A.shape

        if T == 0:
            return {}

        try:
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        except RuntimeError:
            U = torch.zeros((T, min(T, P)), device=A.device)

        # Keep top k dimensions
        k = min(self.d_model, U.shape[1])
        U_k = U[:, :k]

        # Pad if k < d_model
        if k < self.d_model:
            padding = torch.zeros((T, self.d_model - k), device=A.device)
            U_k = torch.cat([U_k, padding], dim=1)

        # Map labels to embeddings
        label_embeddings = {}
        label_counts = {}

        for j, label in enumerate(labels):
            if label == "":  # Skip silent transitions
                continue

            emb = U_k[j]
            if label not in label_embeddings:
                label_embeddings[label] = emb
                label_counts[label] = 1
            else:
                label_embeddings[label] += emb
                label_counts[label] += 1

        # Average
        for label in label_embeddings:
            label_embeddings[label] /= label_counts[label]

        return label_embeddings

    def extract_batch_tensors(
        self,
        net_tensors: tuple[torch.Tensor, torch.Tensor],
        labels: List[str],
        logs_tensor: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features for a batch of traces from the same model.
        Args:
            net_tensors: (pre, post) tuple
            labels: List of transition labels
            logs_tensor: [B, Steps] tensor of label indices (0-indexed into labels, -1 for padding/silent).
                         Assumes logs are compacted (no -1 in the middle) and padded with -1 at the end.

        Returns:
            dict with keys:
            - 'model_basis': [1, T_unique, d_model] tensor (broadcastable)
            - 'trace_embeddings': [B, d_trace] tensor
        """
        pre, post = net_tensors
        device = pre.device

        # 1. Compute Model Basis
        label_embeddings = self._compute_basis_from_tensors(pre, post, labels)

        sorted_labels = sorted(label_embeddings.keys())
        if not sorted_labels:
            model_basis = torch.zeros((1, self.d_model), device=device)
        else:
            model_basis = torch.stack(
                [label_embeddings[l] for l in sorted_labels]
            )

        # Add batch dim [1, T_unique, d_model]
        model_basis = model_basis.unsqueeze(0)

        # 2. Map Traces to Trajectories
        B, Steps = logs_tensor.shape

        # Create embedding matrix for all labels + unknown
        # labels are 0-indexed in logs_tensor corresponding to labels list
        # We need to map from label index to embedding

        # Create a lookup tensor: [num_labels + 1, d_model + 1]
        # +1 for unknown/padding (mapped to 0 vector or specific unknown vector?)
        # In original code:
        # Known: [emb, 0]
        # Unknown: [0...0, 1]

        num_labels = len(labels)
        # We need to handle the case where a label in the log is not in the basis (shouldn't happen if basis is from same model, but possible if basis filtering is aggressive)
        # But here basis is computed from the same model, so all visible labels should be in basis.
        # However, logs_tensor might have padding (-1).

        # Let's build the lookup table
        # Index i -> embedding for labels[i]
        # We need to map labels[i] -> embedding vector

        # Initialize with Unknown vector [0...0, 1]
        lookup = torch.zeros((num_labels + 1, self.d_model + 1), device=device)
        lookup[:, self.d_model] = 1.0  # Default to unknown

        for i, label in enumerate(labels):
            if label in label_embeddings:
                emb = label_embeddings[label]
                lookup[i, : self.d_model] = emb
                lookup[i, self.d_model] = 0.0

        # Handle padding/silent in logs_tensor
        # logs_tensor has -1 for padding/silent.
        # We can map -1 to the last index (num_labels) which is "Unknown" or just 0?
        # If it's padding, we probably want 0?
        # The original code doesn't handle padding explicitly, it iterates over trace.
        # If we have padding, we should probably mask it out or treat it as 0.
        # But DCT requires fixed length?
        # Original code: "If L < n_coeffs... D @ trajectory".
        # Here we have fixed Steps.
        # If we treat padding as 0 vector, it contributes 0 to DCT sum?
        # DCT is sum_n x_n * cos(...)
        # If x_n is 0, it adds nothing.
        # But the DCT matrix depends on L (length of trace).
        # In batch mode, traces have different lengths?
        # logs_tensor is padded.
        # We should probably compute DCT per trace based on its actual length?
        # Or just use max length (Steps)?
        # If we use Steps, then padding (0) acts as silence.
        # But the frequency content changes if we consider the signal to be length Steps vs length L.
        # Original code uses L = len(trace).
        # So we need actual lengths.

        # Calculate actual lengths
        # Assuming -1 is padding
        mask = logs_tensor != -1
        lengths = mask.sum(dim=1)  # [B]

        # Map indices to embeddings
        # Replace -1 with num_labels (which is the last row of lookup, currently Unknown)
        # We want padding to be 0 vector?
        # Let's add a row for Padding: [0...0, 0]
        # So lookup: [num_labels, Unknown, Padding]

        lookup = torch.zeros((num_labels + 2, self.d_model + 1), device=device)
        # Default Unknown (index num_labels)
        lookup[num_labels, : self.d_model] = 0.0
        lookup[num_labels, self.d_model] = 1.0

        # Padding (index num_labels + 1) -> All zeros
        lookup[num_labels + 1, :] = 0.0

        for i, label in enumerate(labels):
            if label in label_embeddings:
                emb = label_embeddings[label]
                lookup[i, : self.d_model] = emb
                lookup[i, self.d_model] = 0.0
            else:
                # Label in list but not in embeddings (e.g. silent?)
                # If silent, it shouldn't be in logs_tensor (filtered out by simulate_batch compact=True?)
                # simulate_batch returns label IDs, 0 for silent? No, it returns label index.
                # simulate_batch compact=True removes silent transitions.
                # So we only have visible labels.
                # If a visible label has no embedding (e.g. disconnected?), it maps to Unknown.
                pass

        # Map logs_tensor to embeddings
        # logs_tensor indices are 0..num_labels-1. -1 is padding.
        # Map -1 to num_labels + 1 (Padding)
        indices = logs_tensor.clone()
        indices[indices == -1] = num_labels + 1

        # [B, Steps, d_model+1]
        trajectory = F.embedding(indices, lookup)

        # 3. DCT Compression
        # Vectorized approach:
        # We construct a batch of DCT matrices D_batch: [B, C, Steps]
        # D_batch[b] corresponds to the DCT matrix for length L = lengths[b], padded with zeros.
        # Then we do D_batch @ trajectory (where trajectory is also padded/masked).

        # Precompute DCT matrices for all possible lengths 0..Steps
        # We can cache this stack if Steps is constant, but it might vary per batch.
        # For now, compute on the fly.

        # We only only need to compute D for lengths that actually appear in the batch
        unique_lengths = torch.unique(lengths)

        # D_stack: [Steps + 1, C, Steps]
        D_stack = torch.zeros((Steps + 1, self.n_coeffs, Steps), device=device)

        for L_val in unique_lengths:
            L = int(L_val.item())
            if L == 0:
                continue

            # Get DCT matrix [C, L]
            D_L = self._get_dct_matrix(L).to(device)

            # Place in stack
            D_stack[L, :, :L] = D_L

        # Gather D matrices for the batch: [B, C, Steps]
        D_batch = D_stack[lengths]

        # Apply DCT: [B, C, Steps] @ [B, Steps, D+1] -> [B, C, D+1]
        # trajectory is [B, Steps, D+1].
        # We treat padding in trajectory as 0 (lookup mapped padding to 0 vector).
        # D_batch has 0s where n >= L.
        # So the sum is correctly limited to 0..L-1.

        dct_features = torch.bmm(D_batch, trajectory)

        # Flatten: [B, C * (D+1)]
        trace_embeddings = dct_features.flatten(start_dim=1)

        return {
            "model_basis": model_basis,
            "trace_embedding": trace_embeddings,
        }

        return {
            "model_basis": model_basis,
            "trace_embedding": trace_embeddings,
        }
