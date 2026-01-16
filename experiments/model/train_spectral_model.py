from typing import Union, Literal, Tuple, Dict, List, Sequence, Optional
from collections import defaultdict
import gc
from pathlib import Path
from experiments.simulation.structured_net import StructuredNet
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import random
from dataclasses import dataclass
from tqdm import tqdm
from dataloaders.labels import LabelDataset
from dataloaders.util import (
    get_natural_dataset,
    get_synthetic_dataset,
)
from models.spectral_model import (
    SpectralModel,
    traces_to_tensors,
    prepare_masked_batch,
)
from util.rng import RNG

# Configuration
LOGGING_LEVEL = logging.INFO
SEED = 1
EPOCHS = 50


@torch.no_grad()
def evaluate(model, batches, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in batches:
            model_basis, trace_embedding, batch_y, trace_mask = batch

            # Expand model basis for batch
            model_basis = model_basis.expand(len(batch_y), -1, -1)

            logits = model(model_basis, trace_embedding, trace_mask=trace_mask)
            # pool logits
            logits = model.pool_logits(logits, trace_mask)
            loss = criterion(logits, batch_y)
            # loss = loss.mean()

            total_loss += loss.item() * len(batch_y)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total_samples += len(batch_y)

    if total_samples == 0:
        return 0.0, 0.0

    return total_loss / total_samples, correct / total_samples

Mode = Literal["oversample", "undersample", "both"]
Target = Union[Literal["max", "min", "mean", "median"], int]


@dataclass
class BalanceStats:
    before: Dict[int, int]
    after: Dict[int, int]
    target_per_class: Dict[int, int]
    kept_batches: int
    total_samples_before: int
    total_samples_after: int


@torch.no_grad()
def rebalance_label_distribution(
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    mode: Mode = "oversample",
    target: Target = "max",
    classes: Optional[Sequence[int]] = None,
    seed: int = 0,
    drop_empty_labels: bool = True,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], BalanceStats]:
    """
    Rebalances label distribution across `batches` by sampling indices within each original batch.

    Input batch tuple:
      (model_basis [1, T, d_model], trace_embedding [B, S, d_trace], batch_y [B], trace_mask [B, S] or [B, ...])

    Key property:
      We *never* mix samples across different original batches when reconstructing tensors.
      That avoids shape issues (e.g. different T/S per net/model).

    Parameters
    ----------
    mode:
      - "oversample": upsample minority labels to match the largest label count
      - "undersample": downsample majority labels to match the smallest label count
      - "both": match an explicit `target` (or statistic) by oversampling some labels and cutting others
    target:
      - "max", "min", "mean", "median" (computed over label counts)
      - or an explicit integer target count per class
      Note: for mode="oversample", target defaults to "max" behavior; for "undersample", defaults to "min".
    classes:
      Optional explicit list of class ids to consider (e.g. range(num_classes)).
      If None: inferred from present labels in data.
    seed:
      RNG seed for reproducibility.
    drop_empty_labels:
      If `classes` is provided and some classes are absent, either:
        - True: ignore absent labels (common, avoids generating nothing/degenerate behavior)
        - False: will raise (because you can't sample absent labels)

    Returns
    -------
    new_batches, stats
    """

    # --- Collect all sample locations per label ---
    # label -> list[(batch_idx, item_idx)]
    locs_by_label: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    total_samples_before = 0
    for b_idx, (_, trace_embedding, batch_y, _) in enumerate(batches):
        if batch_y.ndim != 1:
            raise ValueError(f"Expected batch_y to be 1D [B], got shape {tuple(batch_y.shape)} in batch {b_idx}")
        B = int(batch_y.shape[0])
        if trace_embedding.shape[0] != B:
            raise ValueError(
                f"trace_embedding[0] != batch_y[0] in batch {b_idx}: "
                f"{trace_embedding.shape[0]} vs {B}"
            )
        total_samples_before += B
        y_cpu = batch_y.detach().to("cpu")
        for i in range(B):
            locs_by_label[int(y_cpu[i].item())].append((b_idx, i))

    present_labels = sorted(locs_by_label.keys())

    if classes is None:
        labels = present_labels
    else:
        labels = list(classes)
        missing = [c for c in labels if c not in locs_by_label]
        if missing:
            if drop_empty_labels:
                labels = [c for c in labels if c in locs_by_label]
            else:
                raise ValueError(f"Requested classes contain absent labels: {missing}")

    if not labels:
        # Nothing to do
        stats = BalanceStats(
            before={},
            after={},
            target_per_class={},
            kept_batches=0,
            total_samples_before=total_samples_before,
            total_samples_after=0,
        )
        return [], stats

    counts = {c: len(locs_by_label[c]) for c in labels}

    # --- Decide target count per label ---
    def _stat_target(name: str) -> int:
        vals = torch.tensor([counts[c] for c in labels], dtype=torch.float32)
        if name == "max":
            return int(vals.max().item())
        if name == "min":
            return int(vals.min().item())
        if name == "mean":
            # round to nearest int; you can change to floor/ceil if you prefer
            return int(torch.round(vals.mean()).item())
        if name == "median":
            return int(torch.median(vals).item())
        raise ValueError(f"Unknown target statistic: {name}")

    if isinstance(target, int):
        tgt = int(target)
        if tgt < 0:
            raise ValueError("target must be >= 0")
    else:
        # Default behavior if user picks a mode but leaves target at something odd:
        if mode == "oversample" and target == "min":
            tgt = _stat_target("max")
        elif mode == "undersample" and target == "max":
            tgt = _stat_target("min")
        else:
            tgt = _stat_target(target)

    # For pure oversample/undersample, force sensible bounds:
    if mode == "oversample":
        tgt = max(tgt, max(counts.values()))
    elif mode == "undersample":
        tgt = min(tgt, min(counts.values()))
    # mode == "both": use tgt as-is

    target_per_class = {c: tgt for c in labels}

    # --- Sample locations to match target ---
    g = torch.Generator()
    g.manual_seed(seed)

    selected_locs: List[Tuple[int, int]] = []
    for c in labels:
        locs = locs_by_label[c]
        n = len(locs)
        t = tgt

        if t == 0:
            continue

        if mode == "oversample":
            # always sample with replacement to reach t (but if already >t, keep all; oversample doesn't cut)
            if n >= t:
                selected_locs.extend(locs)
            else:
                idx = torch.randint(low=0, high=n, size=(t,), generator=g)
                selected_locs.extend([locs[int(j)] for j in idx.tolist()])

        elif mode == "undersample":
            # always cut down to t (but if already <t, keep all; undersample doesn't repeat)
            if n <= t:
                selected_locs.extend(locs)
            else:
                perm = torch.randperm(n, generator=g)[:t]
                selected_locs.extend([locs[int(j)] for j in perm.tolist()])

        elif mode == "both":
            # match exactly t using cut or repeat
            if n == t:
                selected_locs.extend(locs)
            elif n > t:
                perm = torch.randperm(n, generator=g)[:t]
                selected_locs.extend([locs[int(j)] for j in perm.tolist()])
            else:  # n < t
                idx = torch.randint(low=0, high=n, size=(t,), generator=g)
                selected_locs.extend([locs[int(j)] for j in idx.tolist()])
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # --- Group selections by original batch and reconstruct tensors ---
    by_batch: Dict[int, List[int]] = defaultdict(list)
    for b_idx, i_idx in selected_locs:
        by_batch[b_idx].append(i_idx)

    new_batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    after_counts: Dict[int, int] = {c: 0 for c in labels}

    for b_idx, (model_basis, trace_embedding, batch_y, trace_mask) in enumerate(batches):
        sel = by_batch.get(b_idx)
        if not sel:
            continue

        sel_t = torch.tensor(sel, dtype=torch.long, device=batch_y.device)

        # index_select supports repeats -> oversampling works naturally
        new_trace_embedding = trace_embedding.index_select(0, sel_t)
        new_batch_y = batch_y.index_select(0, sel_t)

        # trace_mask is [B, ...], so index on dim0 as well
        new_trace_mask = trace_mask.index_select(0, sel_t)

        # Update after-counts on CPU
        y_cpu = new_batch_y.detach().to("cpu")
        for v in y_cpu.tolist():
            if int(v) in after_counts:
                after_counts[int(v)] += 1

        new_batches.append((model_basis, new_trace_embedding, new_batch_y, new_trace_mask))

    stats = BalanceStats(
        before=counts,
        after=after_counts,
        target_per_class=target_per_class,
        kept_batches=len(new_batches),
        total_samples_before=total_samples_before,
        total_samples_after=sum(after_counts.values()),
    )
    return new_batches, stats

@torch.no_grad()
def get_batches(
    label_dataset: LabelDataset,
    model: SpectralModel,
    device: str,
):
    all_batches = []
    for batch in label_dataset.iter_by_model():
        pmodel = batch[0][1].model.deserialize()
        if hasattr(pmodel, "net"):  # synthetic item type
            stnet = pmodel.net
        else:
            stnet = StructuredNet(
                name=pmodel.hash(), net=pmodel.pm, im=pmodel.im, fm=pmodel.fm
            )
        tensor_net = stnet.to_tensor(device)
        items = [e for ds_id, e in batch]
        tok_idx, unk_idx = traces_to_tensors(
            [item.trace for item in items],
            tensor_net.labels,
            model.device,
            model.num_unk_buckets,
        )
        model_basis, trace_embedding, _, _ = prepare_masked_batch(
            extractor=model.feature_extractor,
            model=model,  # Need model to access the learnable mask token
            tensor_net=tensor_net,
            tok_idx=tok_idx,
            unk_bucket=unk_idx,
            device=device,
            mask_prob=0.0,
        )
        trace_mask = tok_idx == -1
        labels = [item.algo for item in items]

        batch_y = torch.tensor(
            [model.inv_label_map[l] for l in labels], dtype=torch.long, device=model.device
        )

        # model_basis [1, T, d_model]
        # trace_embedding [B, S, d_trace]
        if trace_embedding.shape[1] < 2:
            continue

        if (
            trace_embedding.shape[0] != batch_y.shape[0]
            or trace_embedding.shape[-1] != model.d_trace
            or trace_embedding.ndim != 3
        ):
            print(f"Mismatch in batch sizes: {tensor_net}")
            print(f"token ids tensor: {tok_idx}")
            print(f"unk ids tensor: {unk_idx}")
            print(f"Model basis: {model_basis}")
            print(f"Model basis shape: {model_basis.shape}")
            print(f"Trace embedding: {trace_embedding}")
            print(f"Trace embedding shape: {trace_embedding.shape}")
            print(f"Batch y: {batch_y}")
            raise ValueError("Batch size mismatch")

        all_batches.append((model_basis, trace_embedding, batch_y, trace_mask))

    gc.collect()

    return all_batches


def train():
    logging.basicConfig(level=LOGGING_LEVEL)
    RNG.initialize(SEED)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config_path = "./configs/default.yaml"

    DATASETS = {
        # 'a6f651a7-5ce0-4bc6-8be1-a7747effa1cc': ['RequestForPayment.xes'],
        # '33632f3c-5c48-40cf-8d8f-2db57f5a6ce7': [
        #     'Sepsis%20Cases%20-%20Event%20Log.xes'
        # ],
        # '500573e6-accc-4b0c-9576-aa5468b10cee': [
        #     'BPI_Challenge_2013_incidents.xes'
        # ],
        # '91fd1fa8-4df4-4b1a-9a3f-0116c412378f': [
        #     'InternationalDeclarations.xes'
        # ],
        # 'fb84cf2d-166f-4de2-87be-62ee317077e5': ['PrepaidTravelCost.xes'],
        # '12683249': ['Road_Traffic_Fine_Management_Process.xes'],
        # '5f3067df-f10b-45da-b98b-86ae4c7a310b': ['BPI%20Challenge%202017.xes'],
        # 'db35afac-2133-40f3-a565-2dc77a9329a3': ['PermitLog.xes'],
        # '6a0a26d2-82d0-4018-b1cd-89afb0e8627f': ['DomesticDeclarations.xes'],
        "synthetic": ["synthetic"],
    }
    # 3. Model Initialization
    model = SpectralModel(
        d_model=128,
        d_trace=128,
        hidden_dim=256,
        mlp_hidden_dim=512,
        n_classes=6,
        num_heads=4,
        n_layers=2,
        n_self_attn=2,
        dropout=0.1,
        pretraining=False,  # Important!
    ).to(device)

    run_datasets = []
    for dataset_uuid, files in list(DATASETS.items()):
        if dataset_uuid.endswith("synthetic"):
            run_dataset = get_synthetic_dataset(
                Path("cache/.runs"),
                seed=SEED,
                device=device,
                num_models=200,
                num_traces=32,
                min_depth=2,
                max_depth=3
            )
        else:
            run_dataset = get_natural_dataset(
                str(Path("data") / dataset_uuid / files[0]),
                config_path,
                "cache/.runs",
                seed=SEED,
            )
        run_datasets.append(run_dataset)

    label_dataset = LabelDataset(run_datasets)
    print(f"Label distribution: {label_dataset.df["algo"].value_counts()}")
    batches = get_batches(label_dataset, model, device)
    batches, stats = rebalance_label_distribution(batches, mode="oversample", target="max", seed=SEED)

    random.shuffle(batches)
    split = [int(q * len(batches)) for q in [0.8, 0.1, 0.1]]
    train_batches, test_batches, val_batches = (
        batches[: split[0]],
        batches[split[0] : split[1]],
        batches[split[1] :],
    )

    aligners = label_dataset.labels
    print(f"Aligner classes: {aligners}")

    model.load_state_dict(
        torch.load("outputs/runs/pretrain_20260105_011924/spectral_model_pretrained_epoch1.pth", map_location=device), strict=False
    )

    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    lr_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_batches) * EPOCHS)

    criterion = nn.CrossEntropyLoss(reduction="mean")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0

        random.shuffle(train_batches)

        pbar = tqdm(train_batches, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in pbar:
            model_basis, trace_embeddings, batch_y, trace_mask = batch
            trace_mask[:, 0] = False
            # Expand model basis for batch
            model_basis = model_basis.repeat(trace_embeddings.shape[0], 1, 1)

            # Forward
            optimizer.zero_grad()
            logits = model(model_basis, trace_embeddings, trace_mask=trace_mask)
            # pool logits
            logits = model.pool_logits(logits, trace_mask)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_schedule.step()

            total_loss += loss.item() * len(batch_y)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total_samples += len(batch_y)

        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples

        # Validation
        val_loss, val_acc = evaluate(model, val_batches, criterion, device)

        pbar.set_postfix(
            {
                "loss": avg_loss,
                "acc": accuracy,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        logging.info(
            f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )

    torch.save(model.state_dict(), "transformer_model.pth")
    print("Training complete.")


if __name__ == "__main__":
    print("Script loaded")
    train()
