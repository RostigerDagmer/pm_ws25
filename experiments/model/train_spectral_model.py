import gc
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
from features.base_extractor import SpectralFeatureExtractor
from experiments.model.spectral_model import SpectralModel
from util.rng import RNG
from util.distributions import (
    CategoricalSpec,
    PoissonSpec,
    BernoulliDepthLinearSpec,
)

import os

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
    skip_init: bool = True,
) -> RunDataset:
    RNG.initialize(SEED)
    cfg_dict = yaml.safe_load(open(config))
    cfg = PipelineConfig.model_validate(cfg_dict)
    cfg.log_path = log_path
    cfg.alignment.cache_path = base_path
    cfg.seed = SEED
    cfg.alignment.workers = 16
    return build_pipeline(cfg, skip_init)


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
        num_workers=4,
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
    run_dataset,
    label_map,
    timing_map,
    extractor,
    device,
    dataset_id,
    desc="Preparing Data",
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

        # Get targets
        model_targets = []
        model_keys = []
        model_timings = []
        for t in model_traces:
            t_hash = RunDataset._hash_trace(t)
            if (dataset_id, model_hash, t_hash) in label_map:
                model_targets.append(
                    label_map[(dataset_id, model_hash, t_hash)]
                )
                model_timings.append(
                    timing_map[(dataset_id, model_hash, t_hash)]
                )
                model_keys.append((dataset_id, model_hash, t_hash))

        # Filter valid
        valid_indices = [i for i, t in enumerate(model_targets) if t != -1]
        model_traces = [model_traces[i] for i in valid_indices]
        model_targets = [model_targets[i] for i in valid_indices]
        model_keys = [model_keys[i] for i in valid_indices]
        model_timings = [model_timings[i] for i in valid_indices]

        # Batching
        for i in range(0, len(model_traces), BATCH_SIZE):
            batch_trace_objs = model_traces[i : i + BATCH_SIZE]
            batch_y = torch.tensor(
                model_targets[i : i + BATCH_SIZE], device=device
            )
            batch_keys_chunk = model_keys[i : i + BATCH_SIZE]
            batch_timings_chunk = torch.tensor(
                model_timings[i : i + BATCH_SIZE], device=device
            )

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

            all_batches.append(
                (
                    model_basis,
                    trace_embedding,
                    batch_y,
                    batch_keys_chunk,
                    batch_timings_chunk,
                )
            )

    return all_batches


def evaluate(model, batches, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in batches:
            model_basis, trace_embedding, batch_y, _, batch_timings = batch

            # Expand model basis for batch
            model_basis = model_basis.expand(len(batch_y), -1, -1)

            logits = model(model_basis, trace_embedding)
            loss = criterion(logits, batch_y)
            loss = (batch_timings * loss).mean()

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
    stats_df = df.groupby(["dataset_id", "model_hash", "trace_hash"])[
        "duration"
    ].agg(["mean", "max", "min"])
    stats_lookup = stats_df.to_dict(orient="index")

    # Lookup for specific durations: (dataset_id, model_hash, trace_hash, aligner) -> duration
    duration_lookup = df.set_index(
        ["dataset_id", "model_hash", "trace_hash", "aligner"]
    )["duration"].to_dict()

    total_avg_saved = 0.0
    total_max_saved = 0.0
    total_avg_lost = 0.0
    predictions_timeout = 0
    count = 0

    pred_aligners = {}

    with torch.no_grad():
        for batch in batches:
            (
                model_basis,
                trace_embedding,
                batch_y,
                batch_keys,
                batch_timings,
            ) = batch

            model_basis = model_basis.expand(len(batch_y), -1, -1)
            logits = model(model_basis, trace_embedding)
            preds = logits.argmax(dim=1)

            for i, pred_idx in enumerate(preds):
                key = batch_keys[i]  # (dataset_id, model_hash, trace_hash)
                pred_aligner = aligners[pred_idx.item()]

                if pred_aligner not in pred_aligners:
                    pred_aligners[pred_aligner] = 1
                else:
                    pred_aligners[pred_aligner] += 1

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
                lookup_key = (key[0], key[1], key[2], pred_aligner)
                if lookup_key in duration_lookup:
                    pred_duration = duration_lookup[lookup_key]
                    if pred_duration == float("inf"):
                        pred_duration = TIMEOUT
                        predictions_timeout += 1

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
        pred_aligners,
        total_avg_saved / count,
        total_max_saved / count,
        total_avg_lost / count,
        predictions_timeout,
    )


TRAIN_DATASETS = {
    'd9769f3d-0ab0-4fb8-803b-0d1120ffcf54': ['Hospital_log.xes'],
    '63a8435a-077d-4ece-97cd-2c76d394d99c': ['BPIC15_2.xes'],
    'ed445cdd-27d5-4d77-a1f7-59fe7360cfbe': ['BPIC15_3.xes'],
    '679b11cf-47cd-459e-a6de-9ca614e25985': ['BPIC15_4.xes'],
    '3301445f-95e8-4ff0-98a4-901f1f204972': ['BPI%20Challenge%202018.xes'],
    '3926db30-f712-4394-aebc-75976070e91f': ['BPI_Challenge_2012.xes'],
    'a6f651a7-5ce0-4bc6-8be1-a7747effa1cc': ['RequestForPayment.xes'],
    '6af6d5f0-f44c-49be-aac8-8eaa5fe4f6fd': [
        'Hospital%20Billing%20-%20Event%20Log.xes'
    ],
    '33632f3c-5c48-40cf-8d8f-2db57f5a6ce7': [
        'Sepsis%20Cases%20-%20Event%20Log.xes'
    ],
    'd06aff4b-79f0-45e6-8ec8-e19730c248f1': ['BPI_Challenge_2019.xes'],
    '3537c19d-6c64-4b1d-815d-915ab0e479da': [
        'BPI_Challenge_2013_open_problems.xes'
    ],
    '500573e6-accc-4b0c-9576-aa5468b10cee': [
        'BPI_Challenge_2013_incidents.xes'
    ],
    '91fd1fa8-4df4-4b1a-9a3f-0116c412378f': ['InternationalDeclarations.xes'],
    'fb84cf2d-166f-4de2-87be-62ee317077e5': ['PrepaidTravelCost.xes'],
    '12683249': ['Road_Traffic_Fine_Management_Process.xes'],
}

TEST_DATASETS = {
    'a0addfda-2044-4541-a450-fdcc9fe16d17': ['BPIC15_1.xes'],
    'b32c6fe5-f212-4286-9774-58dd53511cf8': ['BPIC15_5.xes'],
    '5f3067df-f10b-45da-b98b-86ae4c7a310b': ['BPI%20Challenge%202017.xes'],
    'db35afac-2133-40f3-a565-2dc77a9329a3': ['PermitLog.xes'],
    '6a0a26d2-82d0-4018-b1cd-89afb0e8627f': ['DomesticDeclarations.xes'],
    'c2c3b154-ab26-4b31-a0e8-8f2350ddac11': [
        'BPI_Challenge_2013_closed_problems.xes'
    ],
}


def get_batches(
    path: str,
    config: str,
    feature_extractor: SpectralFeatureExtractor,
    device: str,
    # dataset_id: str,
):
    run_dataset = get_natural_dataset(path, config)
    # 2. Label Extraction
    print(f"Extracting labels from RunDataset {path}")

    # We need to map (model_hash, trace_hash) -> best_aligner_index
    df = create_label_df(
        run_dataset,
        ["combination_id", "model_hash", "trace_hash", "aligner", "duration"],
        collate_fn,
    )

    # insert column containing the run_dataset.hash()
    dataset_id = run_dataset.hash()
    df["dataset_id"] = dataset_id

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

    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Create label map
    # We need a mapping from aligner name to class index
    aligners = sorted(df["aligner"].unique())
    aligner_to_idx = {a: i for i, a in enumerate(aligners)}
    print(f"Aligner classes: {aligner_to_idx}")

    train_df, test_df, eval_df = split_dataframes(labels_df, 0.8, 0.1)

    def build_map(df_split, aligner_to_idx_map, timeout_val):
        m = {}
        mt = {}
        for _, row in df_split.iterrows():
            key = (row["dataset_id"], row["model_hash"], row["trace_hash"])
            m[key] = aligner_to_idx_map[row["aligner"]]
            mt[key] = (
                row["duration"]
                if row["duration"] != float('inf')
                else timeout_val
            )
        return m, mt

    train_label_map, train_timing_map = build_map(
        train_df, aligner_to_idx, TIMEOUT
    )
    val_label_map, val_timing_map = build_map(test_df, aligner_to_idx, TIMEOUT)
    test_label_map, test_timing_map = build_map(
        eval_df, aligner_to_idx, TIMEOUT
    )

    print(
        f"Split sizes: Train={len(train_label_map)}, Val={len(val_label_map)}, Test={len(test_label_map)}"
    )

    torch.use_deterministic_algorithms(False)

    # We iterate over models, then generate traces for each model
    # We can use the dataset's serialized items to get models

    # Pre-compute data
    train_batches = prepare_batches(
        run_dataset,
        train_label_map,
        train_timing_map,
        feature_extractor,
        device,
        dataset_id,
        desc="Train Data",
    )
    val_batches = prepare_batches(
        run_dataset,
        val_label_map,
        val_timing_map,
        feature_extractor,
        device,
        dataset_id,
        desc="Val Data",
    )
    test_batches = prepare_batches(
        run_dataset,
        test_label_map,
        test_timing_map,
        feature_extractor,
        device,
        dataset_id,
        desc="Test Data",
    )

    del run_dataset
    del train_label_map
    del val_label_map
    del test_label_map
    del train_df
    del test_df
    del eval_df
    del labels_df
    del best_indices
    del timeout_indices
    del aligner_to_idx

    gc.collect()

    return df, aligners, (train_batches, val_batches, test_batches)


def train():
    logging.basicConfig(level=LOGGING_LEVEL)
    RNG.initialize(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config = "./configs/default.yaml"

    DATASETS = {
        **TRAIN_DATASETS,
        **TEST_DATASETS,
    }
    # 3. Model Initialization
    extractor = SpectralFeatureExtractor(d_model=D_MODEL, n_coeffs=N_COEFFS)

    train_batches = []
    val_batches = []
    test_batches = []
    df = pd.DataFrame()
    aligners = []
    i = 22
    for dataset_name, files in DATASETS.items():
        for file in files:
            log_path = os.path.join("data/", dataset_name, file)
            batch_df, algs, bts = get_batches(
                log_path, config, extractor, device
            )
            train_batches.extend(bts[0])
            val_batches.extend(bts[1])
            test_batches.extend(bts[2])
            df = pd.concat([df, batch_df], ignore_index=True)
            print(f"len(df): {len(df)}")
            aligners.extend(algs)
        i -= 1
        if i == 0:
            break

    aligners = sorted(set(aligners))
    print(f"Aligner classes: {aligners}")

    model = SpectralModel(
        d_model=extractor.d_model,  # Pass internal d_model (64)
        d_trace=extractor.dim,
        hidden_dim=HIDDEN_DIM,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        n_classes=len(aligners),
        num_heads=NUM_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(reduction="none")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0

        random.shuffle(train_batches)

        pbar = tqdm(train_batches, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in pbar:
            model_basis, trace_embedding, batch_y, _, batch_timings = batch

            # Expand model basis for batch
            model_basis = model_basis.expand(len(batch_y), -1, -1)

            # Forward
            optimizer.zero_grad()
            logits = model(model_basis, trace_embedding)

            loss = criterion(logits, batch_y)

            # scale loss by timing per category
            loss = (loss * batch_timings).mean()

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
    pred_aligners, avg_saved, max_saved, avg_lost, predictions_timeout = (
        in_depth_eval(model, test_batches, df, aligners, device)
    )
    print(
        f"In-Depth Eval: Avg Time Saved={avg_saved:.4f}s, Max Time Saved={max_saved:.4f}s, Avg Time Lost={avg_lost:.4f}s"
    )
    print(
        f"Number of predictions that would have timed out: {predictions_timeout}"
    )
    print(f"Prediction distribution of aligners: {pred_aligners}")

    # save model
    torch.save(model.state_dict(), "synthetic_model.pth")

    print("Training complete.")


if __name__ == "__main__":
    print("Script loaded")
    train()
