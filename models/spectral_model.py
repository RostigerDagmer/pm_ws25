from experiments.simulation.structured_net import TensorNet
from features.gnn_encoder import PetriNetGNNEncoder
from dataloaders.runs import SyntheticTraceSampler
from pathlib import Path
from dataloaders.runs import AlignerSpec
from dataloaders.synthetic import SyntheticProcessModelDataset
from sklearn.preprocessing._label import LabelEncoder
from models.base import ClassificationModel, PredictionResult
from experiments.simulation.structured_net import StructuredNet
from pm4py.objects.log.obj import Trace
from dataloaders.runs import RunDataset
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.rotary_embedding import RotaryEmbedding
import zlib


def traces_to_tensors(
    traces,
    net_labels: list[str],
    device: torch.device,
    unk_buckets: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    B = len(traces)
    S = max(len(t) for t in traces) if B else 0
    L = len(net_labels)

    label_to_idx = {l: i for i, l in enumerate(net_labels)}

    tok_idx = torch.full((B, S), -1, dtype=torch.long, device=device)
    unk_bucket = torch.full((B, S), -1, dtype=torch.long, device=device)

    for b, trace in enumerate(traces):
        for s, event in enumerate(trace):
            name = event["concept:name"]
            idx = label_to_idx.get(name, L)  # L == unknown sentinel
            tok_idx[b, s] = idx
            if idx == L:
                # stable deterministic bucket id
                bucket = zlib.crc32(name.encode("utf-8")) % unk_buckets
                unk_bucket[b, s] = bucket

    return tok_idx, unk_bucket


def prepare_masked_batch(
    extractor,
    model,  # SpectralModel
    tensor_net,  # TensorNet
    tok_idx: torch.Tensor,  # [B,S] values: -1 pad, 0..T-1 transition-id, T unknown sentinel (or L unknown; see below)
    unk_bucket: torch.Tensor,  # [B,S] values: -1 for non-unk, else 0..K-1
    device: torch.device,
    mask_prob: float = 0.15,
):
    """
    Updated for "silent transitions are present in basis vocab" and for traces that contain
    only *visible* transitions (but basis includes silent transitions as memory slots).

    Assumptions:
      - tensor_net.labels is length T (transition-aligned)
      - tok_idx contains transition indices in [0, T-1], -1 for PAD.
      - Unknown sentinel is == T (recommended). If you still use L, set unknown_id accordingly.

    Returns:
        model_basis:   [1, T_vocab, d_model]
        embedded_log:  [B, S, d_trace]
        final_targets: [B, S] vocab indices for masked known tokens, else -100
        local_to_vocab: [T] maps transition-id -> vocab-id (includes silent)
    """
    pre = tensor_net.pre
    post = tensor_net.post
    labels = tensor_net.labels
    im = tensor_net.init
    fm = tensor_net.final

    B, S = tok_idx.shape
    T_local = pre.shape[0]  # number of transitions
    d_basis = model.d_model
    d_trace = model.d_trace

    # Define unknown sentinel.
    unknown_id = T_local

    # -------------------------
    # A) Compute Basis + local->vocab map (silent INCLUDED; visible labels may be merged)
    # -------------------------
    basis_labels, basis, local_to_vocab = extractor.compute_basis_and_maps(
        pre, post, labels, im, fm
    )
    basis = basis.to(device)
    local_to_vocab = local_to_vocab.to(
        device
    )  # [T_local], all >=0 now (includes silent)
    vocab_size = basis.shape[0]

    # model_basis: [1, T_vocab, d_basis]
    if vocab_size == 0:
        model_basis = torch.zeros((1, 0, d_basis), device=device)
    else:
        # pad/truncate basis to d_basis defensively
        assert basis.shape[1] == model.d_model, (basis.shape, model.d_model)
        model_basis = basis.unsqueeze(0)
    # -------------------------
    # B) Sequence masks
    # -------------------------
    pad_mask = tok_idx == -1
    unk_mask = tok_idx == unknown_id

    # "known local" now means: a valid transition id
    known_local_mask = (tok_idx >= 0) & (tok_idx < T_local)

    # Safe clamp for indexing
    clamped_local = tok_idx.clamp(0, max(T_local - 1, 0))

    # -------------------------
    # C) vocab indices for learnable known tokens
    # -------------------------
    vocab_idx = torch.full((B, S), -1, dtype=torch.long, device=device)
    if known_local_mask.any():
        mapped = local_to_vocab[
            clamped_local
        ]  # [B,S], valid for all transitions (incl silent)
        vocab_idx[known_local_mask] = mapped[known_local_mask]

    predictable_mask = vocab_idx >= 0

    # -------------------------
    # D) Build embedded_log [B,S,d_trace]
    # -------------------------
    embedded_log = torch.empty((B, S, d_trace), device=device)

    pad_vec = model.pad_token.to(device).view(-1)  # [d_trace]

    # initialize all as PAD
    embedded_log[:] = pad_vec

    # UNK buckets
    if unk_mask.any():
        bkt = unk_bucket.clamp(min=0)
        embedded_log[unk_mask] = model.unk_buckets(bkt[unk_mask]).to(device)

    # KNOWN predictable tokens -> embed from basis into trace space
    if predictable_mask.any() and vocab_size > 0:
        vocab_lookup = torch.zeros(
            (vocab_size, d_trace), device=device, dtype=embedded_log.dtype
        )
        vocab_lookup[:, :d_basis] = model_basis[0]  # [T_vocab, d_basis]
        embedded_log[predictable_mask] = vocab_lookup[
            vocab_idx[predictable_mask]
        ]

    # NOTE:
    # We no longer emit a special "silent token" in the trace, because traces are assumed to
    # contain only visible transitions. Silent transitions are instead accessible via basis memory.

    # -------------------------
    # E) Targets + masking (MLM over trace-visible transition tokens)
    # -------------------------
    final_targets = torch.full((B, S), -100, dtype=torch.long, device=device)

    if model.pretraining and vocab_size > 0 and mask_prob > 0.0:
        rand = torch.rand((B, S), device=device)
        mask_bool = (
            (rand < mask_prob) & predictable_mask & (~pad_mask) & (~unk_mask)
        )

        final_targets[mask_bool] = vocab_idx[mask_bool]

        mask_tok = model.mask_token.to(device).expand(B, S, d_trace)
        embedded_log = torch.where(
            mask_bool.unsqueeze(-1), mask_tok, embedded_log
        )

    return model_basis, embedded_log, final_targets, local_to_vocab


class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        pos_enc: RotaryEmbedding | None = None,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(
                E_q, E_total * 3, bias=bias, **factory_kwargs
            )
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert (
            E_total % nheads == 0
        ), "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias
        self.pos_enc = pos_enc

    def _apply_pos_enc(self, q, k):
        if self.pos_enc is None:
            return q, k
        if isinstance(self.pos_enc, RotaryEmbedding):
            return self.pos_enc.rotate_queries_and_keys(q, k, seq_dim=1)
        raise ValueError(
            f"unknown positional encoding type: {type(self.pos_enc)}"
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        is_causal=False,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
            attn_mask (torch.Tensor, optional): attention mask of shape (``N``, ``L_q``, ``L_kv``) to pass to SDPA. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # apply positional encodings
        query, key = self._apply_pos_enc(query, key)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout, is_causal=is_causal
        )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output


class SpectralModel(nn.Module, ClassificationModel):
    def __init__(
        self,
        d_model: int,
        d_trace: int,
        hidden_dim: int,
        mlp_hidden_dim: int,
        n_classes: int,
        num_heads: int = 1,
        n_layers: int = 1,
        n_self_attn: int = 1,
        dropout: float = 0.1,
        pretraining: bool = False,
        num_unk_buckets: int = 32,
    ):
        """
        Args:
            d_model: Dimension of model basis vectors (k).
            d_trace: Dimension of flattened trace embedding (C * (k+1)).
            hidden_dim: Dimension for attention mechanism.
            n_classes: Number of output classes.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()

        self.d_model = d_model
        self.d_trace = d_trace
        self.hidden_dim = hidden_dim
        self.pretraining = pretraining
        self.n_classes = n_classes
        self.n_self_attn = n_self_attn

        self.positional_encoding = RotaryEmbedding(
            hidden_dim, use_xpos=True, learned_freq=True
        )
        self.feature_extractor = PetriNetGNNEncoder(
            d_model=d_model, n_layers=12, dropout=0.2
        )
        self.label_encoder = LabelEncoder()

        # Input Projections
        self.model_proj = nn.Linear(d_model, hidden_dim)
        self.trace_proj = nn.Linear(d_trace, hidden_dim)

        # Seq Self Attention
        self.self_attn_blocks = nn.ModuleList(
            [
                MultiHeadAttention(
                    E_q=hidden_dim,
                    E_k=hidden_dim,
                    E_v=hidden_dim,
                    E_total=hidden_dim,
                    nheads=num_heads,
                    dropout=dropout,
                    pos_enc=self.positional_encoding,
                )
                for _ in range(n_self_attn)
            ]
        )
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_mlp_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, mlp_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(n_self_attn)
            ]
        )

        # Net <-> Seq Cross Attention
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=mlp_hidden_dim,
                dropout=dropout,
                activation=nn.GELU(),
                batch_first=True,
            ),
            num_layers=n_layers,
        )

        # # MLP Head
        self.mlp_head = nn.Sequential(
            # nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

        self.pooler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.mask_token = nn.Parameter(torch.randn(1, 1, d_trace))
        self.unk_buckets = nn.Embedding(num_unk_buckets, d_trace)
        self.pad_token = nn.Parameter(torch.randn(1, 1, d_trace))
        self.silent_token = nn.Parameter(torch.randn(1, 1, d_trace))

        self.num_unk_buckets = num_unk_buckets
        if not self.pretraining:
            self.mask_token.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def label_map(self):
        return {
            0: 'VERSION_DIJKSTRA_NO_HEURISTICS',
            1: 'VERSION_REMAINING_TRACE',
            2: 'VERSION_REQUIRED_ACTIVITIES',
            3: 'VERSION_REQUIRED_MODEL_MOVE',
            4: 'VERSION_STATE_EQUATION_A_STAR',
            5: 'VERSION_STATE_EQUATION_A_STAR_ILP',
        }

    @property
    def inv_label_map(self):
        return {v: k for k, v in self.label_map.items()}

    def _default_hyperparameters(self):
        pass

    def _predict_proba(self):
        pass

    def _train_classifier(self):
        pass

    def pool(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        if pad_mask is None:
            return x.mean(dim=1)
        keep = (~pad_mask).float()  # 1 where real tokens
        denom = keep.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * keep.unsqueeze(-1)).sum(dim=1) / denom

    def pool_logits(
        self, logits: torch.Tensor, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        if pad_mask is not None:
            logits = logits.masked_fill(pad_mask.unsqueeze(-1), float("-inf"))
        pooled = torch.logsumexp(logits, dim=1)  # [B,C]
        return pooled

    def attn_pool(
        self, x: torch.Tensor, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        # x: [B,S,H]
        scores = self.pooler(x).squeeze(-1)  # [B,S]
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, float("-inf"))
        w = scores.softmax(dim=1).unsqueeze(-1)  # [B,S,1]
        return (x * w).sum(dim=1)  # [B,H]

    def forward(
        self,
        model_basis: torch.Tensor,
        trace_embedding: torch.Tensor,
        trace_mask: torch.Tensor | None = None,
        net_mask: torch.Tensor | None = None,
        attn_pool: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            model_basis: [Batch, T, d_model]
            trace_embedding: [Batch, S, d_trace]

        Returns:
            logits: [Batch, n_classes]
        """
        # 1. Project inputs
        # model_basis: [B, T, d_model] -> [B, T, hidden_dim]
        k_v = self.model_proj(model_basis)

        # trace_embedding: [B, S, d_trace] -> [B, S, hidden_dim]
        q = self.trace_proj(trace_embedding)

        # 2. Cross Attention
        # Query: Trace, Key/Value: Model
        # attn_output: [B, 1, hidden_dim]
        for attn_block, mlp_block in zip(
            self.self_attn_blocks, self.self_mlp_blocks
        ):
            # print(f"q shape: {q.shape}, k_v shape: {k_v.shape}")
            attn_output = attn_block(query=q, key=q, value=q)
            attn_output = self.self_attn_norm(attn_output)
            attn_output = self.self_attn_dropout(attn_output)
            attn_output = mlp_block(attn_output) + attn_output
            q = attn_output + q  # skip connection

        # 3. Cross Attention
        q = self.decoder(
            tgt=q,
            memory=k_v,
            tgt_key_padding_mask=trace_mask,
            memory_mask=net_mask,
        )

        if self.pretraining:
            k_v_t = k_v.transpose(1, 2)

            # Compute logits via Dot Product
            # logits: [B, Seq_Len, T_vocab]
            logits = torch.bmm(q, k_v_t) / torch.sqrt(
                torch.tensor(self.hidden_dim, dtype=torch.float32)
            )
        # 4. MLP Head
        else:
            B, S, V = q.shape
            if attn_pool:
                q_ = self.attn_pool(q, trace_mask)
                return self.mlp_head(q_)
            logits = self.mlp_head(q.view(B * S, V))
            logits = logits.view(B, S, self.n_classes)

        return logits

    def get_feature_importance(self) -> None:
        return None

    def predict_batched(
        self, model_item: RunDataset.ItemType, traces: list[Trace]
    ) -> list[PredictionResult]:
        """Predict best heuristics for a batch of traces for a given model."""

        t_fe_start = time.perf_counter()
        if hasattr(model_item, "net"):
            tensor_net = model_item.net.to(device=self.device)
        else:
            # safe path for non-synthetic models
            tensor_net = StructuredNet(
                "sample", model_item.pm, model_item.im, model_item.fm
            ).to_tensor(device=self.device)

        tok_ids, unk_ids = traces_to_tensors(
            traces,
            tensor_net.labels,
            device=self.device,
            unk_buckets=self.num_unk_buckets,
        )
        trace_mask = tok_ids == -1
        trace_mask[:, 0] = False  # unmask at least one position

        model_basis, trace_embedding, _, _ = prepare_masked_batch(
            extractor=self.feature_extractor,
            model=self,
            tensor_net=tensor_net,
            tok_idx=tok_ids,
            unk_bucket=unk_ids,
            device=self.device,
            mask_prob=0.0,
        )

        t_fe_end = time.perf_counter()
        feature_extraction_time = t_fe_end - t_fe_start

        t_clf_start = time.perf_counter()
        try:
            logits = self(
                model_basis.repeat(trace_embedding.shape[0], 1, 1),
                trace_embedding,
                trace_mask=trace_mask,
            )
        except:
            print(
                f"Failed to infer for model:\n{tensor_net}\nand traces:\n{traces}"
            )
            return [
                PredictionResult(
                    predicted_heuristic=self.label_map[0],
                    confidence=float(0.0),
                    feature_extraction_time=feature_extraction_time
                    / len(traces),
                    classification_time=0.0,
                )
                for _ in range(len(traces))
            ]

        logits = self.pool_logits(logits, trace_mask).detach()
        logits = F.softmax(logits, dim=-1)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        t_clf_end = time.perf_counter()
        classification_time = t_clf_end - t_clf_start

        pred = torch.argmax(logits, dim=1)
        confs = torch.max(logits, dim=1).values

        return [
            PredictionResult(
                predicted_heuristic=self.label_map[p.item()],
                confidence=float(logit),
                feature_extraction_time=feature_extraction_time / len(traces),
                classification_time=classification_time / len(traces),
            )
            for p, logit in zip(pred, confs)
        ]


if __name__ == "__main__":
    from util.rng import RNG
    from util.distributions import (
        CategoricalSpec,
        PoissonSpec,
        BernoulliDepthLinearSpec,
    )
    from experiments.simulation.models import sample_net
    from dataloaders.synthetic import SyntheticEventLogDataset

    RNG.initialize(42)

    synthetic_dataset = SyntheticProcessModelDataset(
        param_grid=[
            (
                {  # and dominant
                    "dist_params": {
                        "op": CategoricalSpec([0.1, 0.6, 0.2, 0.1]),
                        "seq_len": PoissonSpec(4),
                        "p_stop": BernoulliDepthLinearSpec(
                            base=0.2, slope=0.1
                        ),
                        "width": PoissonSpec(3),
                    },
                    "min_depth": 1,
                    "max_depth": 2,
                },
                5,  # Number of models per config
            ),
        ],
    )

    trace_sampler = SyntheticTraceSampler(
        ds=synthetic_dataset,
        seed=RNG.get_seed(),
        batch_size=16,
        slice=range(0, 8),
        steps=40,
        device="cuda",
    )

    dataset = RunDataset(
        Path("tmp/test"),
        synthetic_dataset,
        AlignerSpec.A_STAR.value,
        trace_sampler,
        n_runs=5,
        n_workers=20,
    )

    model = SpectralModel(
        d_model=64,
        d_trace=64,
        hidden_dim=64,
        mlp_hidden_dim=128,
        n_classes=6,
        num_heads=4,
        n_layers=2,
        dropout=0.25,
        pretraining=False,  # Important!
    )

    limit = 5
    for pm, items in dataset.iter_by_model():
        if limit <= 0:
            break
        tensor_net = pm.net  # .to_tensor()

        # tok_ids, unk_ids = traces_to_tensors(
        #     [item.trace for item in items],
        #     tensor_net.labels,
        #     model.device,
        #     model.num_unk_buckets,
        # )

        # print((unk_ids == -1).all())
        # print(unk_ids.shape)
        # print(tok_ids.shape)

        # model_basis, embedded_log, targets = prepare_masked_batch(
        #     model.feature_extractor,
        #     model,
        #     tensor_net,
        #     tok_ids,
        #     unk_ids,
        #     model.device,
        # )
        # print(f"model: {model_basis.shape}")
        # print(f"embedded_log: {embedded_log.shape}")
        # y = model(model_basis.repeat(embedded_log.shape[0], 1, 1), embedded_log)
        # print(f"y: {y}")
        pred = model.predict_batched(pm, [item.trace for item in items])
        print(pred)

        print("feature extraction times")
        print([p.feature_extraction_time for p in pred])
        print("classifier times")
        print([p.classification_time for p in pred])
        limit -= 1
