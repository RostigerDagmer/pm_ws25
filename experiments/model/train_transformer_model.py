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
import pickle
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
from dataloaders.labels import LabelDataset
from dataloaders.util import (
    get_natural_dataset,
    get_synthetic_dataset,
    find_existing_tables,
)
from models.gnn_transformer_model import (
    GNNTransformer,
    traces_to_tensors,
    prepare_masked_batch,
)
from util.rng import RNG

# Configuration
LOGGING_LEVEL = logging.INFO
SEED = 1
EPOCHS = 40


def is_cuda_oom(e: BaseException) -> bool:
    msg = str(e).lower()
    return ("out of memory" in msg) or ("cuda error: out of memory" in msg)


@torch.no_grad()
def evaluate(model, batches, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in batches:
            tensor_net, tok_idx, unk_idx, batch_y = batch

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
            if tok_idx.shape[-1] <= 0:
                logging.warning("empty trace in test batch. skipping.")
                continue

            trace_mask[:, 0] = False
            # Expand model basis for batch
            model_basis = model_basis.repeat(trace_embedding.shape[0], 1, 1)

            logits = model(model_basis, trace_embedding, trace_mask=trace_mask)
            # pool logits
            logits = model.pool_logits(logits, trace_mask)
            loss = criterion(logits, batch_y)

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
    batches: Sequence[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    *,
    mode: Mode = "oversample",
    target: Target = "max",
    classes: Optional[Sequence[int]] = None,
    seed: int = 0,
    drop_empty_labels: bool = True,
) -> Tuple[
    List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    BalanceStats,
]:
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
    for b_idx, (tnet, tok_idx, unk_idx, batch_y) in enumerate(batches):
        if batch_y.ndim != 1:
            raise ValueError(
                f"Expected batch_y to be 1D [B], got shape {tuple(batch_y.shape)} in batch {b_idx}"
            )
        B = int(batch_y.shape[0])
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
                raise ValueError(
                    f"Requested classes contain absent labels: {missing}"
                )

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

    new_batches: List[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = []
    after_counts: Dict[int, int] = {c: 0 for c in labels}

    for b_idx, (tnet, tok_idx, unk_idx, batch_y) in enumerate(batches):
        sel = by_batch.get(b_idx)
        if not sel:
            continue

        sel_t = torch.tensor(sel, dtype=torch.long, device=batch_y.device)

        # index_select supports repeats -> oversampling works naturally
        new_tok_ids = tok_idx.index_select(0, sel_t)
        new_batch_y = batch_y.index_select(0, sel_t)

        # trace_mask is [B, ...], so index on dim0 as well
        new_unk_ids = unk_idx.index_select(0, sel_t)

        # Update after-counts on CPU
        y_cpu = new_batch_y.detach().to("cpu")
        for v in y_cpu.tolist():
            if int(v) in after_counts:
                after_counts[int(v)] += 1

        new_batches.append((tnet, new_tok_ids, new_unk_ids, new_batch_y))

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
    train_tables: dict[str, pd.DataFrame],
    test_tables: dict[str, pd.DataFrame],
    val_tables: dict[str, pd.DataFrame],
    model: GNNTransformer,
    device: str,
):
    train_batches, test_batches, val_batches = [], [], []
    for batch in label_dataset.iter_by_model():
        dataset_id = batch[0][0]
        pmodel = batch[0][1].model.deserialize()

        if hasattr(pmodel, "net"):  # synthetic item type
            stnet = pmodel.net
        else:
            stnet = StructuredNet(
                name=pmodel.hash(), net=pmodel.pm, im=pmodel.im, fm=pmodel.fm
            )
        tensor_net = stnet.to_tensor(device)
        items = [e for ds_id, e in batch]

        labels = [item.algo for item in items]
        train, test, val = [], [], []

        for i, (ds_id, e) in enumerate(batch):
            if e.item_id in train_tables[ds_id]["item_id"].values:
                train.append(i)
            elif e.item_id in test_tables[ds_id]["item_id"].values:
                test.append(i)
            elif e.item_id in val_tables[ds_id]["item_id"].values:
                val.append(i)
            else:
                logging.warn(
                    f"item ds_id: {ds_id}; item_id: {e.item_id} not in any test, train, val table"
                )

        if train:
            train_batch_y = torch.tensor(
                [model.inv_label_map[l] for l in [labels[i] for i in train]],
                dtype=torch.long,
                device=model.device,
            )
            tok_idx, unk_idx = traces_to_tensors(
                [item.trace for item in [items[i] for i in train]],
                tensor_net.labels,
                model.device,
                model.num_unk_buckets,
            )
            train_batches.append((tensor_net, tok_idx, unk_idx, train_batch_y))

        if test:
            test_batch_y = torch.tensor(
                [model.inv_label_map[l] for l in [labels[i] for i in test]],
                dtype=torch.long,
                device=model.device,
            )
            tok_idx, unk_idx = traces_to_tensors(
                [item.trace for item in [items[i] for i in test]],
                tensor_net.labels,
                model.device,
                model.num_unk_buckets,
            )
            test_batches.append((tensor_net, tok_idx, unk_idx, test_batch_y))

        if val:
            val_batch_y = torch.tensor(
                [model.inv_label_map[l] for l in [labels[i] for i in val]],
                dtype=torch.long,
                device=model.device,
            )
            tok_idx, unk_idx = traces_to_tensors(
                [item.trace for item in [items[i] for i in val]],
                tensor_net.labels,
                model.device,
                model.num_unk_buckets,
            )
            val_batches.append((tensor_net, tok_idx, unk_idx, val_batch_y))

    gc.collect()

    return train_batches, test_batches, val_batches


def get_tables(
    cache_path: Path, uuid: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_tables, test_tables, val_tables = find_existing_tables(
        cache_path, selection=[uuid]
    )
    return train_tables[uuid], test_tables[uuid], val_tables[uuid]


def train():
    logging.basicConfig(level=LOGGING_LEVEL)
    RNG.initialize(SEED)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config_path = "./configs/default.yaml"
    cache_path = "cache/.runs"

    DATASETS = {
        'ed445cdd-27d5-4d77-a1f7-59fe7360cfbe': ['BPIC15_3.xes'],
        '679b11cf-47cd-459e-a6de-9ca614e25985': ['BPIC15_4.xes'],
        '3301445f-95e8-4ff0-98a4-901f1f204972': ['BPI%20Challenge%202018.xes'],
        '3926db30-f712-4394-aebc-75976070e91f': ['BPI_Challenge_2012.xes'],
        '6af6d5f0-f44c-49be-aac8-8eaa5fe4f6fd': [
            'Hospital%20Billing%20-%20Event%20Log.xes'
        ],
        'd06aff4b-79f0-45e6-8ec8-e19730c248f1': ['BPI_Challenge_2019.xes'],
        '3537c19d-6c64-4b1d-815d-915ab0e479da': [
            'BPI_Challenge_2013_open_problems.xes'
        ],
        # 'a0addfda-2044-4541-a450-fdcc9fe16d17': ['BPIC15_1.xes'],
        'a6f651a7-5ce0-4bc6-8be1-a7747effa1cc': ['RequestForPayment.xes'],
        '500573e6-accc-4b0c-9576-aa5468b10cee': [
            'BPI_Challenge_2013_incidents.xes'
        ],
        '91fd1fa8-4df4-4b1a-9a3f-0116c412378f': [
            'InternationalDeclarations.xes'
        ],
        'fb84cf2d-166f-4de2-87be-62ee317077e5': ['PrepaidTravelCost.xes'],
        '12683249': ['Road_Traffic_Fine_Management_Process.xes'],
        '33632f3c-5c48-40cf-8d8f-2db57f5a6ce7': [
            'Sepsis%20Cases%20-%20Event%20Log.xes'
        ],
        "synthetic": ["synthetic"],
    }
    # 3. Model Initialization
    model = GNNTransformer(
        d_model=768,
        d_trace=768,
        hidden_dim=768,
        mlp_hidden_dim=768 * 2,
        n_classes=6,
        num_heads=6,
        n_layers=4,
        n_gnn_layers=10,
        n_self_attn=2,
        dropout=0.1,
        pretraining=False,
    ).to(device)

    run_datasets = []
    if (
        Path("cache/train_batches.pkl").exists()
        and Path("cache/test_batches.pkl").exists()
        and Path("cache/val_batches.pkl").exists()
    ):
        train_batches, test_batches, val_batches = (
            pickle.load(open(Path("cache/train_batches.pkl"), 'rb')),
            pickle.load(open(Path("cache/test_batches.pkl"), 'rb')),
            pickle.load(open(Path("cache/val_batches.pkl"), 'rb')),
        )
    else:
        train_tables, test_tables, val_tables = {}, {}, {}
        for dataset_uuid, files in list(DATASETS.items()):
            if dataset_uuid.endswith("synthetic"):
                run_dataset = get_synthetic_dataset(
                    Path(cache_path),
                    seed=SEED,
                    num_models=200,
                    num_traces=32,
                    min_depth=2,
                    max_depth=3,
                )

            else:
                run_dataset = get_natural_dataset(
                    str(Path("data") / dataset_uuid / files[0]),
                    config_path,
                    cache_path,
                    seed=SEED,
                    skip_init=True,
                )

            train, test, val = get_tables(Path(cache_path), dataset_uuid)
            train_tables[dataset_uuid] = train
            test_tables[dataset_uuid] = test
            val_tables[dataset_uuid] = val

            run_datasets.append(run_dataset)

        label_dataset = LabelDataset(run_datasets)
        print(f"Label distribution: {label_dataset.df["algo"].value_counts()}")

        train_batches, test_batches, val_batches = get_batches(
            label_dataset, train_tables, test_tables, val_tables, model, device
        )
        train_batches, stats = rebalance_label_distribution(
            train_batches, mode="oversample", target="max", seed=SEED
        )

        pickle.dump(train_batches, open("cache/train_batches.pkl", "wb"))
        print("wrote batches to: cache/train_batches.pkl")
        pickle.dump(test_batches, open("cache/test_batches.pkl", "wb"))
        print("wrote batches to: cache/test_batches.pkl")
        pickle.dump(val_batches, open("cache/val_batches.pkl", "wb"))
        print("wrote batches to: cache/val_batches.pkl")

    print(
        f"Num params: {sum([p.numel() for p in model.parameters() if p.requires_grad])}"
    )

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    lr_schedule = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_batches) * EPOCHS, eta_min=3e-7
    )

    criterion = nn.CrossEntropyLoss(reduction="mean")
    skip = set()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        permutation = list(range(len(train_batches)))

        random.shuffle(permutation)

        pbar = tqdm(permutation, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for index in pbar:
            if index in skip:
                continue
            batch = train_batches[index]
            try:
                tensor_net, tok_idx, unk_idx, batch_y = batch

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
                if tok_idx.shape[-1] <= 0:
                    logging.warning("empty trace in train batch. skipping.")
                    logging.warning(
                        f"trace_embedding.shape: {trace_embedding.shape}"
                    )
                    logging.warning(f"model_basis: {model_basis}")
                    logging.warning(f"tensornet: {tensor_net}")
                    logging.warning(f"tok_idx: {tok_idx}")
                    logging.warning(f"batch_y: {batch_y}")
                    continue

                trace_mask[:, 0] = False
                # Expand model basis for batch
                model_basis = model_basis.repeat(
                    trace_embedding.shape[0], 1, 1
                )

                # Forward
                optimizer.zero_grad()
                logits = model(
                    model_basis, trace_embedding, trace_mask=trace_mask
                )
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
            except (RuntimeError, torch.OutOfMemoryError) as e:
                if not is_cuda_oom(e):
                    raise  # not an OOM -> real bug
                skip.add(index)
                optimizer.zero_grad(set_to_none=True)

                # free cached blocks + python refs
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                gc.collect()

                print("[WARN] CUDA OOM skipping batch")
                continue

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

    torch.save(model.state_dict(), "cache/models/transformer_model.pth")
    print("Training complete.")


if __name__ == "__main__":
    print("Script loaded")
    train()
