import os
import time
from datetime import datetime
from pathlib import Path

from dataloaders.synthetic import SyntheticProcessModelDataset
from models.spectral_model import SpectralModel, prepare_masked_batch
from experiments.simulation.simulate import simulate_batch
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from util.distributions import (
    CategoricalSpec,
    PoissonSpec,
    BernoulliDepthLinearSpec,
)
from itertools import product
from tqdm import tqdm
import gc

MIN_DEPTH = 2
MAX_DEPTH = 4


def unique_rows_eps(x: torch.Tensor, eps: float):
    # Quantize
    q = torch.round(x / eps)

    # Unique rows (exact, integer-like)
    uq, inverse = torch.unique(q, dim=0, return_inverse=True)

    # Find first occurrence per unique row
    num_unique = uq.shape[0]
    first_idx = torch.full((num_unique,), x.shape[0], device=x.device)

    idx = torch.arange(x.shape[0], device=x.device)
    first_idx.scatter_reduce_(
        0, inverse, idx, reduce="amin", include_self=True
    )
    return x[first_idx]


def op_dist_grid():
    t = torch.tensor(
        list(product(*([torch.linspace(0.1, 0.8, 6).tolist()] * 4))),
        dtype=float,
    )
    t = t / t.sum(dim=0, keepdim=True)
    unique = unique_rows_eps(t, 1e-10)
    return unique


PARAM_GRID = [
    (
        {
            "dist_params": {
                "op": CategoricalSpec(cat.tolist()),
                "seq_len": PoissonSpec(l),
                "p_stop": BernoulliDepthLinearSpec(base=0.2, slope=0.1),
                "width": PoissonSpec(w),
            },
            "min_depth": MIN_DEPTH,
            "max_depth": MAX_DEPTH,
        },
        12,
    )  # Number of models per config
    for cat in op_dist_grid()
    for l in [1, 2, 3, 4]
    for w in [1, 2, 3, 4, 5]
]

print(f"Number of unique model sampling configurations: {len(PARAM_GRID)}")


@torch.no_grad()
def enabled_trans_to_vocab(
    enabled_hist: torch.Tensor, local_to_vocab: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    """
    enabled_hist: [B,S,T] bool
    local_to_vocab: [T] long, -1 for silent/excluded
    returns enabled_vocab: [B,S,V] bool
    """
    device = enabled_hist.device
    B, S, T = enabled_hist.shape
    V = vocab_size

    # Build a sparse-ish mapping matrix Mtv: [T,V], where Mtv[t,v]=1 if local_to_vocab[t]==v
    Mtv = torch.zeros((T, V), device=device, dtype=torch.float32)
    valid = local_to_vocab >= 0
    t_idx = torch.arange(T, device=device)[valid]
    v_idx = local_to_vocab[valid]
    Mtv[t_idx, v_idx] = 1.0

    # enabled_hist.float(): [B,S,T] @ [T,V] => [B,S,V]
    enabled_vocab = (enabled_hist.float() @ Mtv) > 0.0
    return enabled_vocab


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
    """
    Vectorized equivalent to experiments.simulation.inject_noise_trace
    """

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


def is_cuda_oom(e: BaseException) -> bool:
    msg = str(e).lower()
    return ("out of memory" in msg) or ("cuda error: out of memory" in msg)


def train():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    epochs = 5
    print(f"Using device: {device}")

    synthetic_dataset = SyntheticProcessModelDataset(
        param_grid=PARAM_GRID,
    )

    model = SpectralModel(
        d_model=512,
        d_trace=512,
        hidden_dim=512,
        mlp_hidden_dim=1536,
        n_classes=6,
        num_heads=4,
        n_layers=6,
        n_self_attn=2,
        dropout=0.1,
        pretraining=True,  # Important!
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    lambda_feas = 0.5  # feasibility loss scaling
    total_steps = len(synthetic_dataset) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )

    # --- Loss Fn's ---
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    # --- TensorBoard ---
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("outputs") / "runs" / f"pretrain_{run_id}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # Logging cadence
    log_every = 20  # steps
    lambda_feas = 0.2

    global_step = 0
    total_seq_tokens = 0
    total_net_tokens = 0

    # Quiet console: disable tqdm progress bar if you want zero clutter
    pbar = tqdm(total=len(synthetic_dataset) * epochs, disable=False)

    try:
        for epoch in range(epochs):
            epoch_loss_sum = 0.0
            epoch_mlm_sum = 0.0
            epoch_feas_sum = 0.0
            epoch_steps = 0

            t_epoch_start = time.perf_counter()

            for idx in torch.randperm(len(synthetic_dataset)):
                try:
                    item = synthetic_dataset[idx]
                    pm = item.stnet
                    net_tensor = pm.to_tensor(device=device)

                    # 1) Simulate
                    t_sim0 = time.perf_counter()
                    logs_tensor, enabled_hist, lengths = simulate_batch(
                        (net_tensor.pre, net_tensor.post),
                        net_tensor.M0,
                        net_tensor.Mf,
                        labels=net_tensor.labels,
                        steps=128,
                        batch_size=128,
                        record_enabled_history=True,
                        compact=True,
                    )
                    t_sim1 = time.perf_counter()

                    sum_tokens = lengths.sum().item()
                    total_seq_tokens += sum_tokens

                    # 1.1) Noise (keep swap/drop at 0 when using feasibility)
                    tok_idx, unk_bucket = inject_pretraining_noise(
                        tok_idx=logs_tensor,
                        L=len(net_tensor.labels),
                        K=model.unk_buckets.num_embeddings,
                        unk_bucket=None,
                        p_unk=0.07,
                        p_drop=0.00,
                        p_swap=0.00,
                    )

                    # 2) Prepare batch
                    t_prep0 = time.perf_counter()
                    model_basis, embedded_log, targets, local_to_vocab = (
                        prepare_masked_batch(
                            model.feature_extractor,
                            model,
                            net_tensor,
                            tok_idx,
                            unk_bucket,
                            device,
                            mask_prob=0.15,
                        )
                    )
                    total_net_tokens += model_basis.shape[-2]

                    t_prep1 = time.perf_counter()
                    trace_mask = logs_tensor == -1

                    # 3) Batchify basis
                    batch_model_basis = model_basis.repeat(
                        embedded_log.shape[0], 1, 1
                    )

                    # 4) Forward
                    optimizer.zero_grad(set_to_none=True)

                    t_fwd0 = time.perf_counter()
                    logits = model(
                        batch_model_basis, embedded_log, trace_mask=trace_mask
                    )  # [B,S,V]
                    t_fwd1 = time.perf_counter()

                    # 5) Losses
                    # MLM

                    try:
                        flat_logits = logits.view(-1, logits.size(-1))
                        flat_targets = targets.view(-1)
                    except Exception as e:
                        print(f"e: {e}")
                        print(f"model_basis: {model_basis}; {logits}")
                        optimizer.zero_grad()
                        continue

                    mlm_loss = ce_loss_fn(flat_logits, flat_targets)

                    # Feasibility (label-level): enabled_hist is [B,S,T], local_to_vocab is [T]
                    enabled_vocab = enabled_trans_to_vocab(
                        enabled_hist=enabled_hist,
                        local_to_vocab=local_to_vocab,
                        vocab_size=logits.size(-1),
                    )
                    enabled_target = enabled_vocab.float()

                    # Position mask: within compacted lengths AND not PAD after noise
                    B, S = tok_idx.shape
                    pos_mask = torch.arange(S, device=device).unsqueeze(
                        0
                    ) < lengths.unsqueeze(1)
                    pos_mask &= tok_idx != -1

                    feas = bce_loss_fn(logits, enabled_target)  # [B,S,V]
                    feas_loss = feas[pos_mask].mean()

                    loss = mlm_loss + lambda_feas * feas_loss

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    # --- TB logging ---
                    epoch_loss_sum += float(loss.detach())
                    epoch_mlm_sum += float(mlm_loss.detach())
                    epoch_feas_sum += float(feas_loss.detach())
                    epoch_steps += 1

                    if global_step % log_every == 0:
                        # scalars
                        writer.add_scalar(
                            "train/loss_total", loss.item(), global_step
                        )
                        writer.add_scalar(
                            "train/loss_mlm", mlm_loss.item(), global_step
                        )
                        writer.add_scalar(
                            "train/loss_feas", feas_loss.item(), global_step
                        )
                        writer.add_scalar(
                            "train/lr", scheduler.get_last_lr()[0], global_step
                        )
                        writer.add_scalar(
                            "train/total_seq_tokens",
                            total_seq_tokens,
                            global_step,
                        )
                        writer.add_scalar(
                            "train/total_net_tokens",
                            total_net_tokens,
                            global_step,
                        )
                        writer.add_scalar(
                            "train/total_tokens",
                            total_seq_tokens + total_net_tokens,
                            global_step,
                        )

                        # timings (ms)
                        writer.add_scalar(
                            "time/sim_ms",
                            (t_sim1 - t_sim0) * 1000.0,
                            global_step,
                        )
                        writer.add_scalar(
                            "time/prepare_ms",
                            (t_prep1 - t_prep0) * 1000.0,
                            global_step,
                        )
                        writer.add_scalar(
                            "time/forward_ms",
                            (t_fwd1 - t_fwd0) * 1000.0,
                            global_step,
                        )

                        # simple stats
                        avg_len = lengths.float().mean().item()
                        writer.add_scalar(
                            "data/avg_seq_len", avg_len, global_step
                        )
                        writer.add_scalar(
                            "data/vocab_size", logits.size(-1), global_step
                        )

                    global_step += 1
                    pbar.update(1)
                except (RuntimeError, torch.OutOfMemoryError) as e:
                    if not is_cuda_oom(e):
                        raise  # not an OOM -> real bug

                    optimizer.zero_grad(set_to_none=True)

                    # free cached blocks + python refs
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    gc.collect()

                    print(f"[WARN] CUDA OOM at index {idx}, skipping batch")
                    continue

            t_epoch_end = time.perf_counter()

            # epoch aggregates
            if epoch_steps > 0:
                writer.add_scalar(
                    "epoch/loss_total", epoch_loss_sum / epoch_steps, epoch
                )
                writer.add_scalar(
                    "epoch/loss_mlm", epoch_mlm_sum / epoch_steps, epoch
                )
                writer.add_scalar(
                    "epoch/loss_feas", epoch_feas_sum / epoch_steps, epoch
                )
            writer.add_scalar(
                "epoch/duration_s", t_epoch_end - t_epoch_start, epoch
            )

            # checkpoint per epoch
            ckpt_path = (
                log_dir / f"spectral_model_pretrained_epoch{epoch + 1}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)

        # final checkpoint
        torch.save(
            model.state_dict(), log_dir / "spectral_model_pretrained_final.pth"
        )

    finally:
        pbar.close()
        writer.flush()
        writer.close()


if __name__ == "__main__":
    from util.rng import RNG

    RNG.initialize(42)
    train()
