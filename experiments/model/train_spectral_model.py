import gc
from pathlib import Path
from typing import Optional
from typing import Callable
from torch.utils.data.dataloader import DataLoader
import yaml
from configs.schema import PipelineConfig
from dataloaders.runs import PerfCounter
from experiments.simulation.structured_net import StructuredNet
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

from dataloaders.runs import RunDataset
from dataloaders.labels import LabelDataset
from dataloaders.util import (
    get_natural_dataset,
    get_synthetic_dataset,
    build_pipeline,
)
from features.extractors import SpectralFeatureExtractor
from models.spectral_model import (
    SpectralModel,
    traces_to_tensors,
    prepare_masked_batch,
)
from util.rng import RNG

import os

# Configuration
LOGGING_LEVEL = logging.INFO
SEED = 1
EPOCHS = 10


@torch.no_grad()
def evaluate(model, batches, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in batches:
            model_basis, trace_embedding, batch_y = batch

            # Expand model basis for batch
            model_basis = model_basis.expand(len(batch_y), -1, -1)

            logits = model(model_basis, trace_embedding)
            # pool logits
            logits = model.pool(logits)
            loss = criterion(logits, batch_y)
            loss = loss.mean()

            total_loss += loss.item() * len(batch_y)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total_samples += len(batch_y)

    if total_samples == 0:
        return 0.0, 0.0

    return total_loss / total_samples, correct / total_samples


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
        net_tensors = stnet.to_tensor(device)
        items = [e for ds_id, e in batch]
        tok_idx, unk_idx = traces_to_tensors(
            [item.trace for item in items],
            net_tensors.labels,
            model.device,
            model.num_unk_buckets,
        )
        model_basis, trace_embedding, _ = prepare_masked_batch(
            extractor=model.feature_extractor,
            model=model,  # Need model to access the learnable mask token
            pre=net_tensors.pre,
            post=net_tensors.post,
            labels=net_tensors.labels,
            tok_idx=tok_idx,
            unk_bucket=unk_idx,
            device=device,
            mask_prob=0.0,
        )
        labels = [item.algo for item in items]

        batch_y = torch.tensor(
            [model.inv_label_map[l] for l in labels], device=model.device
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
            print(f"Mismatch in batch sizes: {net_tensors}")
            print(f"token ids tensor: {tok_idx}")
            print(f"unk ids tensor: {unk_idx}")
            print(f"Model basis: {model_basis}")
            print(f"Model basis shape: {model_basis.shape}")
            print(f"Trace embedding: {trace_embedding}")
            print(f"Trace embedding shape: {trace_embedding.shape}")
            print(f"Batch y: {batch_y}")
            raise ValueError("Batch size mismatch")

        all_batches.append((model_basis, trace_embedding, batch_y))

    gc.collect()

    return all_batches


def train():
    logging.basicConfig(level=LOGGING_LEVEL)
    RNG.initialize(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config_path = "./configs/default.yaml"

    DATASETS = {
        'a6f651a7-5ce0-4bc6-8be1-a7747effa1cc': ['RequestForPayment.xes'],
        '33632f3c-5c48-40cf-8d8f-2db57f5a6ce7': [
            'Sepsis%20Cases%20-%20Event%20Log.xes'
        ],
        '500573e6-accc-4b0c-9576-aa5468b10cee': [
            'BPI_Challenge_2013_incidents.xes'
        ],
        '91fd1fa8-4df4-4b1a-9a3f-0116c412378f': [
            'InternationalDeclarations.xes'
        ],
        'fb84cf2d-166f-4de2-87be-62ee317077e5': ['PrepaidTravelCost.xes'],
        '12683249': ['Road_Traffic_Fine_Management_Process.xes'],
        '5f3067df-f10b-45da-b98b-86ae4c7a310b': ['BPI%20Challenge%202017.xes'],
        'db35afac-2133-40f3-a565-2dc77a9329a3': ['PermitLog.xes'],
        '6a0a26d2-82d0-4018-b1cd-89afb0e8627f': ['DomesticDeclarations.xes'],
        "synthetic": ["synthetic"],
    }
    # 3. Model Initialization
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
    ).to(device)

    run_datasets = []
    for dataset_uuid, files in list(DATASETS.items()):
        if dataset_uuid.endswith("synthetic"):
            run_dataset = get_synthetic_dataset(
                Path("./cache/.runs"), seed=SEED, device=device
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

    batches = get_batches(label_dataset, model, device)

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
        torch.load("spectral_model_pretrained.pth", map_location=device)
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(reduction="none")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0

        random.shuffle(train_batches)

        pbar = tqdm(train_batches, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in pbar:
            model_basis, trace_embeddings, batch_y = batch

            # Expand model basis for batch
            model_basis = model_basis.repeat(trace_embeddings.shape[0], 1, 1)

            # Forward
            optimizer.zero_grad()
            logits = model(model_basis, trace_embeddings)
            # pool logits
            logits = model.pool(logits)
            loss = criterion(logits, batch_y)

            # scale loss by timing per category
            loss = loss.mean()

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

    torch.save(model.state_dict(), "transformer_model.pth")
    print("Training complete.")


if __name__ == "__main__":
    print("Script loaded")
    train()
