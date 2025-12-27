from dataloaders.runs import SyntheticTraceSampler
from pathlib import Path
from dataloaders.runs import AlignerSpec
from dataloaders.synthetic import SyntheticProcessModelDataset
from sklearn.preprocessing._label import LabelEncoder
from features.spectral_extractor import SpectralFeatureExtractor
from models.base import ClassificationModel, PredictionResult
from experiments.simulation.structured_net import StructuredNet
from pm4py.objects.log.obj import Trace
from dataloaders.runs import RunDataset
import time
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
    pre: torch.Tensor,
    post: torch.Tensor,
    labels: list[str],
    tok_idx: torch.Tensor,  # [B,S] values: -1 pad, 0..L-1 known/silent, L unknown
    unk_bucket: torch.Tensor,  # [B,S] values: -1 for non-unk, else 0..K-1
    device: torch.device,
    mask_prob: float = 0.15,
):
    """
    Returns:
        model_basis:   [1, T_vocab, d_model]
        embedded_log:  [B, S, d_trace]
        final_targets: [B, S] vocab indices for masked known tokens, else -100
    """
    B, S = tok_idx.shape
    L = len(labels)  # local label vocab size; unknown sentinel is == L
    d_basis = model.d_model
    d_trace = model.d_trace

    # -------------------------
    # A) Compute Basis + local->vocab map (silent excluded, duplicates merged)
    # -------------------------
    basis_labels, basis, local_to_vocab = extractor.compute_basis_and_maps(
        pre, post, labels
    )
    basis = basis.to(device)
    local_to_vocab = local_to_vocab.to(
        device
    )  # [L_local], -1 for silent/excluded
    vocab_size = basis.shape[0]

    # model_basis: [1, T_vocab, d_basis]
    if vocab_size == 0:
        model_basis = torch.zeros((1, 0, d_basis), device=device)
    else:
        # pad/truncate basis to d_basis defensively
        if basis.shape[1] < d_basis:
            pad = torch.zeros(
                (vocab_size, d_basis - basis.shape[1]),
                device=device,
                dtype=basis.dtype,
            )
            basis = torch.cat([basis, pad], dim=1)
        elif basis.shape[1] > d_basis:
            basis = basis[:, :d_basis]
        model_basis = basis.unsqueeze(0)

    # Silent local indices
    silent_local_mask = torch.tensor(
        [lab == "" for lab in labels], dtype=torch.bool, device=device
    )

    # B) Sequence masks
    pad_mask = tok_idx == -1
    unk_mask = tok_idx == L
    valid_local = (tok_idx >= 0) & (tok_idx < L)
    clamped_local = tok_idx.clamp(0, max(L - 1, 0))
    silent_mask = valid_local & silent_local_mask[clamped_local]
    known_local_mask = valid_local & (~silent_mask)

    # C) vocab indices for predictable known tokens
    vocab_idx = torch.full((B, S), -1, dtype=torch.long, device=device)
    if known_local_mask.any():
        mapped = local_to_vocab[clamped_local]  # [B,S], -1 if excluded
        vocab_idx[known_local_mask] = mapped[known_local_mask]
    predictable_mask = vocab_idx >= 0

    # D) Build embedded_log [B,S,d_trace]
    embedded_log = torch.empty((B, S, d_trace), device=device)

    pad_vec = model.pad_token.to(device).view(-1)  # [d_trace]
    silent_vec = model.silent_token.to(device).view(-1)  # [d_trace]

    # initialize all as PAD
    embedded_log[:] = pad_vec

    # SILENT token positions
    if silent_mask.any():
        embedded_log[silent_mask] = silent_vec

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

    # print(f"predictable mask: {predictable_mask}")
    # E) Targets + masking
    final_targets = torch.full((B, S), -100, dtype=torch.long, device=device)

    if model.pretraining and vocab_size > 0 and mask_prob > 0.0:
        rand = torch.rand((B, S), device=device)
        mask_bool = (rand < mask_prob) & predictable_mask
        # print(f"mask_bool: {mask_bool}")
        final_targets[mask_bool] = vocab_idx[mask_bool]

        # mask token is [1,1,d_trace] -> broadcast to [B,S,d_trace]
        mask_tok = model.mask_token.to(device).expand(B, S, d_trace)
        embedded_log = torch.where(
            mask_bool.unsqueeze(-1), mask_tok, embedded_log
        )
    # print(f"final targets: {final_targets[final_targets != -100]}")
    return model_basis, embedded_log, final_targets


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

        self.positional_encoding = RotaryEmbedding(hidden_dim)
        self.feature_extractor = SpectralFeatureExtractor(
            d_model=d_model, n_coeffs=8
        )
        self.label_encoder = LabelEncoder()

        # Projections
        self.model_proj = nn.Linear(d_model, hidden_dim)
        self.trace_proj = nn.Linear(d_trace, hidden_dim)

        # Cross Attention
        # batch_first=True means input is (batch, seq, feature)
        self.attn_blocks = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.mlp_blocks = nn.ModuleList(
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
                for _ in range(n_layers)
            ]
        )

        # MLP Head
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

        self.mask_token = nn.Parameter(torch.randn(1, 1, d_trace))
        self.unk_buckets = nn.Embedding(num_unk_buckets, d_trace)
        self.pad_token = nn.Parameter(torch.randn(1, 1, d_trace))
        self.silent_token = nn.Parameter(torch.randn(1, 1, d_trace))

        self.num_unk_buckets = num_unk_buckets
        if self.pretraining:
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

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)  # mean pooling

    def forward(
        self, model_basis: torch.Tensor, trace_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            model_basis: [Batch, T, d_model]
            trace_embedding: [Batch, d_trace]

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
        for attn_block, mlp_block in zip(self.attn_blocks, self.mlp_blocks):
            # print(f"q shape: {q.shape}, k_v shape: {k_v.shape}")
            q = self.positional_encoding.rotate_queries_or_keys(q)
            k_v = self.positional_encoding.rotate_queries_or_keys(k_v)
            attn_output, _ = attn_block(query=q, key=k_v, value=k_v)
            attn_output = self.attn_norm(attn_output)
            attn_output = self.attn_dropout(attn_output)
            attn_output = mlp_block(attn_output) + attn_output
            q = attn_output + q  # skip connection

        if self.pretraining:
            k_v_t = k_v.transpose(1, 2)

            # Compute logits via Dot Product
            # logits: [B, Seq_Len, T_vocab]
            logits = torch.bmm(q, k_v_t) / torch.sqrt(
                torch.tensor(self.d_model, dtype=torch.float32)
            )
        # 3. MLP Head
        else:
            B, S, V = q.shape
            logits = self.mlp_head(q.view(B * S, V))
            logits = logits.view(B, S, self.n_classes)

        return logits

    def get_feature_importance(self) -> None:
        return None

    def predict_batched(
        self, model_item: RunDataset.ItemType, traces: list[Trace]
    ) -> list[PredictionResult]:
        """Predict best heuristics for a batch of traces for a given model."""

        t_start = time.perf_counter()
        max_len = max(len(trace) for trace in traces)

        if hasattr(model_item, "net"):
            net_tensor = model_item.net.to(device=self.device)
        else:
            # safe path for non-synthetic models
            net_tensor = StructuredNet(
                "sample", model_item.pm, model_item.im, model_item.fm
            ).to_tensor(device=self.device)

        tok_ids, unk_ids = traces_to_tensors(
            traces,
            net_tensor.labels,
            device=self.device,
            unk_buckets=self.num_unk_buckets,
        )

        t_fe_start = time.perf_counter()

        model_basis, trace_embedding, _ = prepare_masked_batch(
            extractor=self.feature_extractor,
            model=self,  # Need model to access the learnable mask token
            pre=net_tensor.pre,
            post=net_tensor.post,
            labels=net_tensor.labels,
            tok_idx=tok_ids,
            unk_bucket=unk_ids,
            device=self.device,
            mask_prob=0.0,
        )

        t_fe_end = time.perf_counter()
        feature_extraction_time = t_fe_end - t_fe_start

        t_clf_start = time.perf_counter()
        logits = self(
            model_basis.repeat(trace_embedding.shape[0], 1, 1), trace_embedding
        )

        logits = self.pool(logits).detach()
        logits = F.softmax(logits, dim=-1)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        t_clf_end = time.perf_counter()
        classification_time = t_clf_end - t_clf_start

        pred = torch.argmax(logits, dim=1)
        confs = torch.max(logits, dim=1).values
        t_end = time.perf_counter()
        total_time = t_end - t_start

        return [
            PredictionResult(
                predicted_heuristic=self.label_map[p.item()],
                confidence=float(logit),
                total_prediction_time=total_time / len(traces),
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
                1,  # Number of models per config
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

    pm, items = next(dataset.iter_by_model())
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
    #     tensor_net.pre,
    #     tensor_net.post,
    #     tensor_net.labels,
    #     tok_ids,
    #     unk_ids,
    #     model.device,
    # )
    # print(f"model: {model_basis.shape}")
    # print(f"embedded_log: {embedded_log.shape}")
    # y = model(model_basis.repeat(embedded_log.shape[0], 1, 1), embedded_log)
    # print(f"y: {y}")
    print(model.predict_batched(pm, [item.trace for item in items]))
