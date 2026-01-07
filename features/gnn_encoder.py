import torch
import torch.nn as nn


class PetriGNNLayer(nn.Module):
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.dropout = nn.Dropout(dropout)

        # Messages: P->T (pre, post)
        self.p2t_pre = nn.Linear(d, d, bias=False)
        self.p2t_post = nn.Linear(d, d, bias=False)

        # Messages: T->P (pre, post)
        self.t2p_pre = nn.Linear(d, d, bias=False)
        self.t2p_post = nn.Linear(d, d, bias=False)

        # Updates
        self.upd_t = nn.Sequential(
            nn.Linear(d * 3, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        self.upd_p = nn.Sequential(
            nn.Linear(d * 3, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        self.norm_t = nn.LayerNorm(d)
        self.norm_p = nn.LayerNorm(d)

    def forward(
        self,
        h_t: torch.Tensor,
        h_p: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
    ):
        """
        h_t: [T, d] transition embeddings
        h_p: [P, d] place embeddings
        pre/post: [T, P] integer weights
        """
        # Cast arc weights to float for matmul
        pre_f = pre.float()
        post_f = post.float()

        # ---- Place -> Transition aggregation ----
        # For each transition t, aggregate from places p with weights pre[t,p] / post[t,p]
        # [T,P] @ [P,d] -> [T,d]
        deg_t_pre = pre_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        deg_t_post = post_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        m_t_pre = pre_f @ self.p2t_pre(h_p) / deg_t_pre
        m_t_post = post_f @ self.p2t_post(h_p) / deg_t_post

        # ---- Transition -> Place aggregation ----
        # For each place p, aggregate from transitions t (note transpose)
        # [P,T] @ [T,d] -> [P,d]
        deg_p_pre = pre_f.t().sum(dim=1, keepdim=True).clamp_min(1.0)
        deg_p_post = post_f.t().sum(dim=1, keepdim=True).clamp_min(1.0)
        m_p_pre = pre_f.t() @ self.t2p_pre(h_t) / deg_p_pre
        m_p_post = post_f.t() @ self.t2p_post(h_t) / deg_p_post

        # ---- Update with residual ----
        t_in = torch.cat([h_t, m_t_pre, m_t_post], dim=-1)
        p_in = torch.cat([h_p, m_p_pre, m_p_post], dim=-1)

        h_t_new = self.norm_t(h_t + self.dropout(self.upd_t(t_in)))
        h_p_new = self.norm_p(h_p + self.dropout(self.upd_p(p_in)))

        return h_t_new, h_p_new


class PetriNetGNNEncoder(nn.Module):
    """
    Drop-in replacement for SpectralFeatureExtractor.compute_basis_and_maps
    Produces:
      basis_labels: List[str] length T_vocab (silent excluded, duplicates merged)
      basis: [T_vocab, d_model]
      local_to_vocab: [T_local] maps transition index -> vocab index (or -1 if silent)
    """

    def __init__(self, d_model: int, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [PetriGNNLayer(d_model, dropout) for _ in range(n_layers)]
        )

        # Learnable initial embeddings
        self.init_t = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.init_p = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # Trans feats: log degs (4), is init (1), is final (1) => 6
        self.p_feat_dim = 6
        # Trans feats: log degs (4), silent (1) => 5
        self.t_feat_dim = 5

        self.p_mlp = nn.Sequential(
            nn.Linear(self.p_feat_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.t_mlp = nn.Sequential(
            nn.Linear(self.t_feat_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    @torch.no_grad()
    def _build_local_to_vocab(self, labels: list[str], device) -> tuple[list[str], torch.Tensor]:
        """
        Vocab construction:
        - visible transitions: merged by label (first-occurrence order)
        - silent transitions: each gets its own unique vocab entry (per transition id)
        Returns:
        basis_labels: [V] list of string identifiers
        local_to_vocab: [T] maps every transition -> vocab id (no -1)
        """
        T_local = len(labels)
        local_to_vocab = torch.empty((T_local,), dtype=torch.long, device=device)

        label_to_row: dict[str, int] = {}
        basis_labels: list[str] = []

        # First: allocate visible label slots (merged)
        for j, lab in enumerate(labels):
            if lab == "":
                continue
            if lab not in label_to_row:
                label_to_row[lab] = len(basis_labels)
                basis_labels.append(lab)

        # Second: allocate one slot per silent transition
        silent_rows = {}
        for j, lab in enumerate(labels):
            if lab != "":
                continue
            # unique name per silent transition
            key = f"<tau:t{j:06d}>"
            silent_rows[j] = len(basis_labels)
            basis_labels.append(key)

        # Fill local_to_vocab
        for j, lab in enumerate(labels):
            if lab == "":
                local_to_vocab[j] = silent_rows[j]
            else:
                local_to_vocab[j] = label_to_row[lab]

        return basis_labels, local_to_vocab

    def _init_features(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        labels: list[str],
        init_place_idx: int | None,
        final_place_idx: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          p_feats: [P, p_feat_dim]
          t_feats: [T, t_feat_dim]
        """
        device = pre.device
        T, P = pre.shape

        pre_f = pre.float()
        post_f = post.float()

        # weighted token degrees
        p_in_tok = post_f.sum(dim=0)  # [P]
        p_out_tok = pre_f.sum(dim=0)  # [P]
        t_pre_tok = pre_f.sum(dim=1)  # [T]
        t_post_tok = post_f.sum(dim=1)  # [T]

        # unweighted arc degrees
        p_in_arc = (post > 0).sum(dim=0).float()
        p_out_arc = (pre > 0).sum(dim=0).float()
        t_pre_arc = (pre > 0).sum(dim=1).float()
        t_post_arc = (post > 0).sum(dim=1).float()

        # flags
        is_init = torch.zeros((P,), device=device)
        is_final = torch.zeros((P,), device=device)
        if init_place_idx is not None and 0 <= init_place_idx < P:
            is_init[init_place_idx] = 1.0
        if final_place_idx is not None and 0 <= final_place_idx < P:
            is_final[final_place_idx] = 1.0

        is_silent = torch.tensor(
            [1.0 if lab == "" else 0.0 for lab in labels], device=device
        )

        # assemble features
        p_feats = torch.stack(
            [
                torch.log1p(p_in_tok),
                torch.log1p(p_out_tok),
                torch.log1p(p_in_arc),
                torch.log1p(p_out_arc),
                is_init,
                is_final,
            ],
            dim=1,
        )  # [P,10]

        t_feats = torch.stack(
            [
                torch.log1p(t_pre_tok),
                torch.log1p(t_post_tok),
                torch.log1p(t_pre_arc),
                torch.log1p(t_post_arc),
                is_silent,
            ],
            dim=1,
        )  # [T,9]

        return p_feats, t_feats

    def compute_basis_and_maps(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        labels: list[str],
        im: int,
        fm: int,
    ):
        """
        pre/post: [T,P]
        labels: list[str] length T, aligned with transitions
        """
        device = pre.device
        T, P = pre.shape

        # Build mapping label->vocab index (silent excluded)
        basis_labels, local_to_vocab = self._build_local_to_vocab(
            labels, device
        )
        V = len(basis_labels)

        if V == 0:
            return (
                [],
                torch.zeros((0, self.d_model), device=device),
                local_to_vocab,
            )
        # init embeddings with features
        p_feats, t_feats = self._init_features(pre, post, labels, im, fm)

        # Initialize node embeddings
        h_t = self.init_t.expand(T, self.d_model).contiguous() + self.t_mlp(
            t_feats
        )
        h_p = self.init_p.expand(P, self.d_model).contiguous() + self.p_mlp(
            p_feats
        )

        # Message passing
        for layer in self.layers:
            h_t, h_p = layer(h_t, h_p, pre, post)

        # Merge transitions by label (average duplicates)
        # basis[v] = mean_{t: local_to_vocab[t]=v} h_t[t]
        
        basis = torch.zeros((V, self.d_model), device=device, dtype=h_t.dtype)
        counts = torch.zeros((V,), device=device, dtype=h_t.dtype)

        idx = local_to_vocab  # [T], all valid
        basis.index_add_(0, idx, h_t)
        counts.index_add_(0, idx, torch.ones_like(idx, dtype=h_t.dtype))
        basis = basis / counts.clamp_min(1.0).unsqueeze(-1)

        return basis_labels, basis, local_to_vocab


if __name__ == "__main__":
    from pm4py.vis import view_petri_net
    from util.distributions import (
        BernoulliDepthLinearSpec,
        CategoricalSpec,
        PoissonSpec,
    )
    from experiments.simulation import models
    from util.rng import RNG
    import logging
    import matplotlib.pyplot as plt

    plt.switch_backend("qtagg")

    logging.basicConfig(level=logging.DEBUG)
    RNG.initialize(4)

    dist_params = {
        "op": CategoricalSpec([0.1, 0.5, 0.3, 0.1]),
        "seq_len": PoissonSpec(4),
        "p_stop": BernoulliDepthLinearSpec(base=0.15, slope=0.1),
        "width": PoissonSpec(10),
    }
    stnet = models.sample_net(
        dist_params, max_depth=3, generator=RNG.torch_generator()
    )

    # view_petri_net(stnet.net, stnet.im, stnet.fm, format="svg")

    encoder = PetriNetGNNEncoder(d_model=128, n_layers=8, dropout=0.2)
    t_net = stnet.to_tensor()
    b_labels, b, vocab = encoder.compute_basis_and_maps(
        t_net.pre, t_net.post, t_net.labels, t_net.init, t_net.final
    )

    print(b_labels)
    print("basis")
    print(b)
    plt.imshow(b.detach().cpu())
    plt.show()
    print("vocab")
    print(vocab)
    print(b.shape)
