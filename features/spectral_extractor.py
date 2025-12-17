from typing import Dict, List, Union
import torch
import torch.nn.functional as F
import numpy as np
from pm4py.objects.petri_net.obj import PetriNet, Marking

from features.base_extractor import BaseFeatureExtractor


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
