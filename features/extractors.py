"""
Feature extraction for Petri nets and traces for alignment heuristic recommendation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Union
import torch
import numpy as np
import networkx as nx
from collections import Counter
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.networkx_graph import (
    create_networkx_directed_graph,
)


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
        interaction_features = self._extract_interactions(petri_net, trace_net)

        return {**model_features, **trace_features, **interaction_features}

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

    def _extract_features_internal(
        self,
        petri_net: PetriNet,
        petri_net_im: Marking,
        petri_net_fm: Marking,
        trace_net: PetriNet,
        trace_net_im: Marking,
        trace_net_fm: Marking,
    ) -> Dict[str, float]:

        # 1. Compute Model Basis
        # Note: In a real scenario, we might want to cache this per model
        # independent of the trace. But our architecture passes both.
        # Since we have a cache key that includes the model hash,
        # we could optimize this further, but for now we recompute or rely on the main cache.
        label_embeddings = self._compute_basis(petri_net)

        # 2. Map Trace to Trajectory
        # Trace net is a sequence of transitions
        trace_labels = [
            t.label for t in trace_net.transitions if t.label is not None
        ]
        L = len(trace_labels)

        if L == 0:
            return {name: 0.0 for name in self.feature_names}

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
        # We want [n_coeffs, d_model + 1]
        # X_dct = D @ X_traj
        D = self._get_dct_matrix(L)  # [C, L]

        # If L < n_coeffs, D will be [C, L]. The matrix mult works,
        # but high freq coeffs might be aliased or meaningless.
        # Standard DCT is defined for N points returning N coeffs.
        # Here we project L points to C coeffs.
        # If L < C, we are upsampling? Or just getting what we can.
        # Our _get_dct_matrix handles any L.

        dct_features = D @ trajectory  # [C, L] @ [L, D+1] -> [C, D+1]

        # Flatten
        flat_features = dct_features.flatten().tolist()

        return dict(zip(self.feature_names, flat_features))
