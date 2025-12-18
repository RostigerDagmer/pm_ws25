from dataloaders.synthetic import SyntheticProcessModelDataset
from models.spectral_model import SpectralModel, prepare_masked_batch
from experiments.simulation.simulate import simulate_batch
import torch
import torch.nn.functional as F
import torch.nn as nn
from features.extractors import SpectralFeatureExtractor
from util.distributions import (
    CategoricalSpec,
    PoissonSpec,
    BernoulliDepthLinearSpec,
)
from itertools import product
from tqdm import tqdm

MIN_DEPTH = 1
MAX_DEPTH = 4

PARAM_GRID = [
    (
        {
            "dist_params": {
                "op": CategoricalSpec(cat),
                "seq_len": PoissonSpec(l),
                "p_stop": BernoulliDepthLinearSpec(base=0.2, slope=0.1),
                "width": PoissonSpec(w),
            },
            "min_depth": MIN_DEPTH,
            "max_depth": MAX_DEPTH,
        },
        5,
    )  # Number of models per config
    for cat in product(*([torch.linspace(0.1, 0.6, 5).tolist()] * 4))
    for l in [2, 4, 6]
    for w in [2, 3, 4]
]

# print(f"Total Configs: {PARAM_GRID}")
# print(f"Total Models: {len(PARAM_GRID)}")


@torch.no_grad()
def inject_pretraining_noise(
    tok_idx: torch.Tensor,  # [B,S] ints in {-1, 0..L-1}
    L: int,  # len(net.labels); unknown sentinel will be L
    K: int,  # number of unk buckets
    unk_bucket: torch.Tensor | None = None,  # [B,S] or None
    p_unk: float = 0.07,  # replace known token with UNK
    p_drop: float = 0.03,  # replace known token with PAD (-1)
    p_swap: float = 0.03,  # swap adjacent tokens
) -> tuple[torch.Tensor, torch.Tensor]:
    device = tok_idx.device
    B, S = tok_idx.shape

    if unk_bucket is None:
        unk_bucket = torch.full_like(tok_idx, -1)

    # Work on copies
    tok = tok_idx.clone()
    bkt = unk_bucket.clone()

    pad_mask = tok == -1
    valid_mask = ~pad_mask

    # 1) Replace some valid tokens with UNK
    if p_unk > 0.0:
        m_unk = (torch.rand((B, S), device=device) < p_unk) & valid_mask
        tok[m_unk] = L
        bkt[m_unk] = torch.randint(0, K, (m_unk.sum().item(),), device=device)

    # 2) Dropout (delete) some remaining valid (non-UNK) tokens to PAD
    if p_drop > 0.0:
        valid_nonunk = (tok >= 0) & (tok < L)
        m_drop = (torch.rand((B, S), device=device) < p_drop) & valid_nonunk
        tok[m_drop] = -1
        bkt[m_drop] = -1

    # 3) Adjacent swap (local reorder); avoid pads
    if p_swap > 0.0 and S > 1:
        # Decide swap positions i (swap i and i+1)
        swap_i = torch.rand((B, S - 1), device=device) < p_swap

        # Only swap where both positions are not PAD
        left_ok = tok[:, :-1] != -1
        right_ok = tok[:, 1:] != -1
        swap_i = swap_i & left_ok & right_ok

        # Perform swaps
        # tok
        left = tok[:, :-1].clone()
        right = tok[:, 1:].clone()
        tok[:, :-1] = torch.where(swap_i, right, left)
        tok[:, 1:] = torch.where(swap_i, left, right)

        # buckets must swap alongside (only meaningful for UNK positions, but safe always)
        left_b = bkt[:, :-1].clone()
        right_b = bkt[:, 1:].clone()
        bkt[:, :-1] = torch.where(swap_i, right_b, left_b)
        bkt[:, 1:] = torch.where(swap_i, left_b, right_b)

    return tok, bkt


def train():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    synthetic_dataset = SyntheticProcessModelDataset(
        param_grid=PARAM_GRID,
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
        pretraining=True,  # Important!
    ).to(device)

    extractor = SpectralFeatureExtractor(d_model=64, n_coeffs=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Standard Cross Entropy
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    for epoch in range(5):  # Single epoch for demo
        print(f"Epoch {epoch + 1}")
        for item in tqdm(synthetic_dataset, total=len(synthetic_dataset)):
            pm = item.stnet
            net_tensor = pm.to_tensor(device=device)

            # 1. Simulate Logs (Ground Truth)
            logs_tensor = simulate_batch(
                (net_tensor.pre, net_tensor.post),
                net_tensor.M0,
                net_tensor.Mf,
                labels=net_tensor.labels,
                steps=50,
                batch_size=64,
            )

            # 1.1 Inject noise
            tok_idx, unk_bucket = inject_pretraining_noise(
                tok_idx=logs_tensor,
                L=len(net_tensor.labels),
                K=model.unk_buckets.num_embeddings,  # if it's nn.Embedding
                unk_bucket=None,
                p_unk=0.07,
                p_drop=0.03,
                p_swap=0.03,
            )

            # 2. Prepare Masked Inputs & Targets
            model_basis, embedded_log, targets = prepare_masked_batch(
                extractor,
                model,
                net_tensor.pre,
                net_tensor.post,
                net_tensor.labels,
                tok_idx,
                unk_bucket,
                device,
                mask_prob=0.15,
            )

            # 3. Batchify Model Basis
            # The basis is unique to the Petri net, but needs to be repeated for the batch
            # Basis: [1, T_vocab, d_model] -> [B, T_vocab, d_model]
            batch_model_basis = model_basis.repeat(embedded_log.shape[0], 1, 1)

            # 4. Forward Pass
            optimizer.zero_grad()

            # Logits: [B, Seq_Len, T_vocab]
            logits = model(batch_model_basis, embedded_log)

            # 5. Calculate Loss
            # Flatten for CrossEntropy:
            # Logits -> [B * Seq_Len, T_vocab]
            # Targets -> [B * Seq_Len]

            try:
                flat_logits = logits.view(-1, logits.size(-1))
                flat_targets = targets.view(-1)
            except Exception as e:
                print(f"e: {e}")
                print(f"model_basis: {model_basis}; {logits}")
                optimizer.zero_grad()
                continue

            # print(f"flat_logits: {flat_logits}")
            # print(f"flat_logits.shape: {flat_logits.shape}")
            # print(f"flat_targets: {flat_targets.tolist()}")
            # print(f"flat_targets.shape: {flat_targets.shape}")

            loss = loss_fn(flat_logits, flat_targets)

            loss.backward()
            optimizer.step()

            # print(f"Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), "spectral_model_pretrained.pth")


if __name__ == "__main__":
    from util.rng import RNG

    RNG.initialize(42)
    train()
