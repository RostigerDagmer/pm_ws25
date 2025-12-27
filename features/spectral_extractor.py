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
