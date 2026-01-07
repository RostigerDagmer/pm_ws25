import json
import pebble
import hashlib
import pandas as pd
from tqdm import tqdm
from typing import Iterator
from dataloaders.util import get_natural_dataset
from dataloaders.runs import RunDataset
from torch.utils.data import Dataset


class LabelDataset(Dataset):

    def __init__(self, run_datasets: list[RunDataset]):
        self.run_datasets = {
            run_ds.log_uuid: run_ds for run_ds in run_datasets
        }

        self.df = pd.DataFrame()
        with pebble.ProcessPool(max_workers=len(run_datasets)) as pool:
            futures = pool.map(
                LabelDataset._get_labels, self.run_datasets.values()
            )
            self.df = pd.concat(futures.result())

    def hash(self) -> str:
        return hashlib.sha1(
            json.dumps(list(self.run_datasets.keys()), sort_keys=True).encode()
        ).hexdigest()

    @property
    def labels(self) -> list[str]:
        return sorted(self.df.algo.unique().tolist())

    @staticmethod
    def label_criterion(item: RunDataset.SerializedItemType) -> float:
        if all([p["duration"] == float('inf') for p in item.perf]):
            return 20.0
        return sum(
            [
                20.0 if p["duration"] == float('inf') else p["duration"]
                for p in item.perf
            ]
        ) / len(item.perf)

    @staticmethod
    def _get_labels(run_ds: RunDataset) -> pd.DataFrame:
        labels = []
        dataset_id = run_ds.log_uuid
        for comb_id in tqdm(
            run_ds.combinations, desc=f"labeling {run_ds.log_uuid}"
        ):
            items = [
                run_ds.serialized[item_id]
                for item_id in run_ds.combinations[comb_id]
            ]
            model_id = items[0].model.hash()
            min_item = min(items, key=LabelDataset.label_criterion)
            if all([p["duration"] == float('inf') for p in min_item.perf]):
                # this run always timed out (ignore)
                continue
            run_id = min_item.item_id
            algo = min_item.algo
            labels.append(
                {
                    "dataset_id": dataset_id,
                    "model_id": model_id,
                    "comb_id": comb_id,
                    "run_id": run_id,
                    "algo": algo,
                }
            )

        return pd.DataFrame(labels)

    def __len__(self):
        return len(self.df)

    def __iter__(self) -> Iterator[tuple[str, RunDataset.SerializedItemType]]:
        for i in range(len(self.df)):
            yield self.__getitem__(i)

    def get_combination_results(
        self, dataset_id: str, comb_id: str
    ) -> list[RunDataset.SerializedItemType]:
        return self.run_datasets[dataset_id].get_combination(comb_id)

    def iter_by_model(
        self,
    ) -> Iterator[list[tuple[str, RunDataset.SerializedItemType]]]:
        for model_id in self.df.model_id.unique():
            yield [
                (
                    row["dataset_id"],
                    self.run_datasets[row["dataset_id"]].serialized[
                        row["run_id"]
                    ],
                )
                for _, row in self.df[
                    self.df["model_id"] == model_id
                ].iterrows()
            ]

    def __getitem__(self, idx) -> tuple[str, RunDataset.SerializedItemType]:
        idx = self.df.iloc[idx]
        return (
            idx["dataset_id"],
            self.run_datasets[idx["dataset_id"]].serialized[idx["run_id"]],
        )


if __name__ == "__main__":
    from pathlib import Path
    from util.rng import RNG

    SEED = 1
    RNG.initialize(SEED)

    TEST_DATASETS = {
        'db35afac-2133-40f3-a565-2dc77a9329a3': ['PermitLog.xes'],
        # '6a0a26d2-82d0-4018-b1cd-89afb0e8627f': ['DomesticDeclarations.xes'],
    }
    config_path = "configs/default.yaml"
    cache_path = "cache/.runs"

    test_run_datasets = []
    for dataset_uuid, files in TEST_DATASETS.items():
        for filename in files:
            print(f"Loading: {filename}")
            run_dataset = get_natural_dataset(
                str(Path("data") / dataset_uuid / filename),
                config_path,
                cache_path,
                seed=SEED,
            )
            if run_dataset is not None:
                test_run_datasets.append(run_dataset)
                print(f"  ✓ Loaded {len(run_dataset)} runs from cache")

    label_dataset = LabelDataset(test_run_datasets)
    print(label_dataset.df.head())

    for batch in label_dataset.iter_by_model():
        for dataset_id, item in batch:
            print(item.trace, item.algo)
            model_id = item.model.hash()
            print(label_dataset.df[label_dataset.df["model_id"] == model_id])
            test = label_dataset.get_combination_results(
                dataset_id, item.comb_id
            )
            print(test)
            print(
                label_dataset.df[
                    (label_dataset.df["dataset_id"] == dataset_id)
                    & (label_dataset.df["comb_id"] == item.comb_id)
                ]
            )
            break
        break
