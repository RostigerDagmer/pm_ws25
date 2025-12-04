from configs.schema import SliceType
from typing import Optional
from scripts.create_labels import split_dataframes
from typing import Callable
from torch.utils.data.dataloader import DataLoader
import yaml
from configs.schema import PipelineConfig
from scripts.generate_dataset import build_pipeline
from dataloaders.runs import PerfCounter
from experiments.simulation.structured_net import StructuredNet
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from dataloaders.runs import RunDataset, AlignerSpec, SyntheticTraceSampler
from dataloaders.synthetic import SyntheticProcessModelDataset
from features.extractors import SpectralFeatureExtractor
from experiments.model.spectral_model import SpectralModel
from util.rng import RNG
from util.distributions import (
    CategoricalSpec,
    PoissonSpec,
    BernoulliDepthLinearSpec,
)

# Configuration
LOGGING_LEVEL = logging.INFO
SEED = 1
N_RUNS = 3
BATCH_SIZE = 128
STEPS = 50
EPOCHS = 10
LEARNING_RATE = 1e-3
D_MODEL = 64
N_COEFFS = 8
HIDDEN_DIM = 256
MLP_HIDDEN_DIM = 512
NUM_HEADS = 4
N_LAYERS = 4
DROPOUT = 0.1
TIMEOUT = 20.0


def get_synthetic_dataset(device: str) -> RunDataset:
    # 1. Dataset Generation
    print("Generating synthetic dataset...")
    MAX_DEPTH = 2
    MIN_DEPTH = 1

    synthetic_dataset = SyntheticProcessModelDataset(
        param_grid=[
            (
                {
                    "dist_params": {
                        "op": CategoricalSpec([0.3, 0.3, 0.3, 0.1]),
                        "seq_len": PoissonSpec(4),
                        "p_stop": BernoulliDepthLinearSpec(
                            base=0.2, slope=0.1
                        ),
                        "width": PoissonSpec(3),
                    },
                    "min_depth": MIN_DEPTH,
                    "max_depth": MAX_DEPTH,
                },
                5,  # Number of models per config
            ),
            (
                {
                    "dist_params": {
                        "op": CategoricalSpec([0.2, 0.2, 0.2, 0.4]),
                        "seq_len": PoissonSpec(4),
                        "p_stop": BernoulliDepthLinearSpec(
                            base=0.2, slope=0.1
                        ),
                        "width": PoissonSpec(3),
                    },
                    "min_depth": MIN_DEPTH,
                    "max_depth": MAX_DEPTH,
                },
                5,  # Number of models per config
            ),
            (
                {
                    "dist_params": {
                        "op": CategoricalSpec([0.2, 0.4, 0.2, 0.2]),
                        "seq_len": PoissonSpec(4),
                        "p_stop": BernoulliDepthLinearSpec(
                            base=0.2, slope=0.1
                        ),
                        "width": PoissonSpec(3),
                    },
                    "min_depth": MIN_DEPTH,
                    "max_depth": MAX_DEPTH,
                },
                5,  # Number of models per config
            ),
            (
                {
                    "dist_params": {
                        "op": CategoricalSpec([0.2, 0.2, 0.4, 0.2]),
                        "seq_len": PoissonSpec(4),
                        "p_stop": BernoulliDepthLinearSpec(
                            base=0.2, slope=0.1
                        ),
                        "width": PoissonSpec(3),
                    },
                    "min_depth": MIN_DEPTH,
                    "max_depth": MAX_DEPTH,
                },
                5,  # Number of models per config
            ),
        ],
    )

    trace_sampler = SyntheticTraceSampler(
        seed=RNG.get_seed(),
        ds=synthetic_dataset,
        slice=None,  # Traces per model
        steps=STEPS,
        batch_size=BATCH_SIZE,
        device=device,
    )

    run_dataset = RunDataset(
        Path('./data/runs_synthetic_train'),
        synthetic_dataset,
        AlignerSpec.A_STAR.value,
        trace_sampler,
        n_runs=N_RUNS,
        n_workers=0,
    )
    return run_dataset


def collate_fn(batch: list[RunDataset.SerializedItemType]) -> pd.DataFrame:
    data = []

    for run in batch:
        # avoid from_tensor conversion during deserialization
        # We want average total time
        durations = [PerfCounter.from_dict(p).duration for p in run.perf]
        durations = [d for d in durations if d is not None]
        mean_duration = np.mean(durations) if durations else float('inf')

        data.append(
            {
                "combination_id": run.comb_id,
                "model_hash": run.model.hash(),
                "trace_hash": RunDataset._hash_trace(run.trace),
                "aligner": run.algo,
                "duration": mean_duration,
            }
        )

    df = pd.DataFrame(data)
    return df


def get_natural_dataset(
    log_path: str,
    config: str,
    base_path: Optional[str] = None,
    slice: Optional[range] = None,
) -> RunDataset:
    cfg_dict = yaml.safe_load(open(config))
    cfg = PipelineConfig.model_validate(cfg_dict)
    cfg.log_path = log_path
    cfg.alignment.cache_path = base_path
    if slice is not None:
        cfg.alignment.sampler.slice = SliceType(
            **{"from": slice.start, "to": slice.stop}
        )
    cfg.seed = SEED
    cfg.alignment.workers = 24
    return build_pipeline(
        cfg,
    )


def create_label_df(
    run_dataset: RunDataset,
    schema: list[str],
    collate: Callable[[list[RunDataset.SerializedItemType]], pd.DataFrame],
):
    df = pd.DataFrame(columns=schema)

    dataloader = DataLoader(
        run_dataset.serialized,
        batch_size=512,
        shuffle=False,
        num_workers=4,  # cfg.alignment.workers if cfg.alignment.workers > 0 else os.cpu_count(),
        persistent_workers=True,
        collate_fn=collate,
    )

    for df_batch in tqdm(dataloader, desc="Extracting features from runs"):
        df = pd.concat(
            [df, df_batch],
            ignore_index=True,
        )

    return df


def prepare_batches(
    run_dataset, label_map, extractor, device, desc="Preparing Data"
):
    print(f"Pre-computing batches: {desc}...")
    models = run_dataset.pm_dataset.serialized
    all_batches = []

    pbar = tqdm(models, desc=desc)
    for model_item in pbar:
        # Deserialize model
        if hasattr(model_item, "net"):
            net_tensor = model_item.net.to(device=device)
        else:
            # save path for non-synthetic models
            proc_model = model_item.deserialize()
            net_tensor = StructuredNet(
                "sample", proc_model.pm, proc_model.im, proc_model.fm
            ).to_tensor(device=device)
        model_hash = model_item.hash()

        N = net_tensor
        labels = N.labels

        # Inner loop over traces
        model_traces = list(
            run_dataset.trace_sampler.iter_for_model(model_item)
        )
        # This gives us all traces for the model.

        # Get targets
        model_targets = []
        model_keys = []
        for t in model_traces:
            t_hash = RunDataset._hash_trace(t)
            if (model_hash, t_hash) in label_map:
                model_targets.append(label_map[(model_hash, t_hash)])
                model_keys.append((model_hash, t_hash))
            else:
                # This trace belongs to a different split (e.g. val or test) if we are preparing train
                # Or it is just missing.
                # Since we iterate over ALL traces for the model, we need to skip those not in the current label_map
                model_targets.append(-1)
                model_keys.append(None)

        # Filter valid
        valid_indices = [i for i, t in enumerate(model_targets) if t != -1]
        model_traces = [model_traces[i] for i in valid_indices]
        model_targets = [model_targets[i] for i in valid_indices]
        model_keys = [model_keys[i] for i in valid_indices]

        # Batching
        for i in range(0, len(model_traces), BATCH_SIZE):
            batch_trace_objs = model_traces[i : i + BATCH_SIZE]
            batch_y = torch.tensor(
                model_targets[i : i + BATCH_SIZE], device=device
            )
            batch_keys_chunk = model_keys[i : i + BATCH_SIZE]

            # Convert Traces to Tensor
            # We need to map labels to indices
            # labels list is N.labels
            label_to_idx = {l: i for i, l in enumerate(labels)}

            # Find max length
            max_len = max(len(t) for t in batch_trace_objs)

            # Pad with -1
            logs_tensor = torch.full(
                (len(batch_trace_objs), max_len),
                -1,
                dtype=torch.long,
                device=device,
            )

            for b_idx, t in enumerate(batch_trace_objs):
                for s_idx, event in enumerate(t):
                    name = event["concept:name"]
                    if name in label_to_idx:
                        logs_tensor[b_idx, s_idx] = label_to_idx[name]
                    else:
                        logs_tensor[b_idx, s_idx] = len(
                            labels
                        )  # Map to Unknown

            # Extract Features
            tensors = extractor.extract_batch_tensors(
                (net_tensor.pre, net_tensor.post),
                net_tensor.labels,
                logs_tensor,
            )

            model_basis = tensors["model_basis"]  # [1, T, d_model]
            trace_embedding = tensors["trace_embedding"]  # [B, d_trace]

            # Get keys for this batch
            batch_keys = []
            for i in range(len(batch_trace_objs)):
                # Re-construct key or store it?
                # We filtered model_traces and model_targets using valid_indices.
                # We need the corresponding hashes.
                # Let's store hashes when we filter.
                pass

            # Actually, let's refactor the loop slightly to keep keys

            all_batches.append(
                (model_basis, trace_embedding, batch_y, batch_keys_chunk)
            )

    return all_batches


def evaluate(model, batches, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in batches:
            model_basis, trace_embedding, batch_y, _ = batch

            # Expand model basis for batch
            model_basis = model_basis.expand(len(batch_y), -1, -1)

            logits = model(model_basis, trace_embedding)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * len(batch_y)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total_samples += len(batch_y)

    if total_samples == 0:
        return 0.0, 0.0

    return total_loss / total_samples, correct / total_samples


def in_depth_eval(model, batches, df, aligners, device):
    model.eval()

    # Pre-compute stats
    print("Computing duration stats...")
    stats_df = df.groupby(["model_hash", "trace_hash"])["duration"].agg(
        ["mean", "max", "min"]
    )
    stats_lookup = stats_df.to_dict(orient="index")

    # Lookup for specific durations: (model_hash, trace_hash, aligner) -> duration
    duration_lookup = df.set_index(["model_hash", "trace_hash", "aligner"])[
        "duration"
    ].to_dict()

    total_avg_saved = 0.0
    total_max_saved = 0.0
    total_avg_lost = 0.0
    count = 0

    with torch.no_grad():
        for batch in batches:
            model_basis, trace_embedding, batch_y, batch_keys = batch

            model_basis = model_basis.expand(len(batch_y), -1, -1)
            logits = model(model_basis, trace_embedding)
            preds = logits.argmax(dim=1)

            for i, pred_idx in enumerate(preds):
                key = batch_keys[i]  # (model_hash, trace_hash)
                pred_aligner = aligners[pred_idx.item()]

                if key not in stats_lookup:
                    continue

                stats = stats_lookup[key]
                # replace inf with our timeout (20.0)
                stats["mean"] = (
                    stats["mean"] if stats["mean"] != float("inf") else TIMEOUT
                )
                stats["max"] = (
                    stats["max"] if stats["max"] != float("inf") else TIMEOUT
                )
                stats["min"] = (
                    stats["min"] if stats["min"] != float("inf") else TIMEOUT
                )

                avg_duration = stats["mean"]
                max_duration = stats["max"]
                min_duration = stats["min"]

                # Get predicted duration
                # If the predicted aligner failed or is missing for this trace, what do we do?
                # The dataset should be complete?
                lookup_key = (key[0], key[1], pred_aligner)
                if lookup_key in duration_lookup:
                    pred_duration = duration_lookup[lookup_key]

                    total_avg_saved += avg_duration - pred_duration
                    total_max_saved += max_duration - pred_duration
                    total_avg_lost += pred_duration - min_duration
                    count += 1
                else:
                    # Predicted aligner not found for this trace
                    pass

    if count == 0:
        return 0.0, 0.0, 0.0

    return (
        total_avg_saved / count,
        total_max_saved / count,
        total_avg_lost / count,
    )


def train():
    logging.basicConfig(level=LOGGING_LEVEL)
    RNG.initialize(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config = "./configs/default.yaml"
    log_path = "./data/5f3067df-f10b-45da-b98b-86ae4c7a310b/BPI%20Challenge%202017.xes"
    log_path = "./data/6af6d5f0-f44c-49be-aac8-8eaa5fe4f6fd/Hospital%20Billing%20-%20Event%20Log.xes"
    log_path = "./data/a0addfda-2044-4541-a450-fdcc9fe16d17/BPIC15_1.xes"

    # run_dataset = get_natural_dataset(log_path, config, slice=range(0, 10))
    run_dataset = get_synthetic_dataset(device="cuda:0")

    # 2. Label Extraction
    print("Extracting labels from RunDataset...")
    # We need to map (model_hash, trace_hash) -> best_aligner_index

    df = create_label_df(
        run_dataset,
        ["combination_id", "model_hash", "trace_hash", "aligner", "duration"],
        collate_fn,
    )

    # Find best aligner for each combination
    best_indices = df.groupby("combination_id")["duration"].idxmin()
    timeout_indices = (
        df[df["duration"] == float('inf')]
        .groupby("combination_id")["duration"]
        .idxmin()
    )
    labels_df = df.loc[best_indices]
    print(labels_df["aligner"].value_counts())
    print("==== TIMEOUTS ====")
    print(df.loc[timeout_indices]["aligner"].value_counts())

    # Create label map
    # We need a mapping from aligner name to class index
    aligners = sorted(df["aligner"].unique())
    aligner_to_idx = {a: i for i, a in enumerate(aligners)}
    print(f"Aligner classes: {aligner_to_idx}")

    label_map = {}  # (model_hash, trace_hash) -> class_idx
    print(f"Generated {len(label_map)} labels.")

    train_df, test_df, eval_df = split_dataframes(labels_df, 0.8, 0.1)

    def build_map(df_split):
        m = {}
        for _, row in df_split.iterrows():
            key = (row["model_hash"], row["trace_hash"])
            m[key] = aligner_to_idx[row["aligner"]]
        return m

    train_label_map = build_map(train_df)
    val_label_map = build_map(
        test_df
    )  # split_dataframes returns train, test(val), eval(test)
    test_label_map = build_map(eval_df)

    print(
        f"Split sizes: Train={len(train_label_map)}, Val={len(val_label_map)}, Test={len(test_label_map)}"
    )

    # 3. Model Initialization
    extractor = SpectralFeatureExtractor(d_model=D_MODEL, n_coeffs=N_COEFFS)

    model = SpectralModel(
        d_model=extractor.d_model,  # Pass internal d_model (63)
        d_trace=extractor.dim,
        hidden_dim=HIDDEN_DIM,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        n_classes=len(aligners),
        num_heads=NUM_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    torch.use_deterministic_algorithms(False)
    print("Starting training...")
    model.train()

    # We iterate over models, then generate traces for each model
    # We can use the dataset's serialized items to get models

    # Pre-compute data
    train_batches = prepare_batches(
        run_dataset, train_label_map, extractor, device, desc="Train Data"
    )
    val_batches = prepare_batches(
        run_dataset, val_label_map, extractor, device, desc="Val Data"
    )
    test_batches = prepare_batches(
        run_dataset, test_label_map, extractor, device, desc="Test Data"
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0

        random.shuffle(train_batches)

        pbar = tqdm(train_batches, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in pbar:
            model_basis, trace_embedding, batch_y, _ = batch

            # Expand model basis for batch
            model_basis = model_basis.expand(len(batch_y), -1, -1)

            # Forward
            optimizer.zero_grad()
            logits = model(model_basis, trace_embedding)

            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

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

    # Final Test
    test_loss, test_acc = evaluate(model, test_batches, criterion, device)
    print(f"Test Results: Loss={test_loss:.4f}, Acc={test_acc:.4f}")

    # In-Depth Eval
    avg_saved, max_saved, avg_lost = in_depth_eval(
        model, test_batches, df, aligners, device
    )
    print(
        f"In-Depth Eval: Avg Time Saved={avg_saved:.4f}s, Max Time Saved={max_saved:.4f}s, Avg Time Lost={avg_lost:.4f}s"
    )

    # save model
    torch.save(model.state_dict(), "synthetic_model.pth")

    print("Training complete.")


if __name__ == "__main__":
    print("Script loaded")
    train()
