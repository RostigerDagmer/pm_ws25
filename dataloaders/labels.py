import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Iterator, Optional
from dataloaders.runs import RunDataset
from torch.utils.data import Dataset


class LabelDataset(Dataset):

    def __init__(self, run_datasets: list[RunDataset]):
        self.run_datasets = {
            run_ds.log_uuid: run_ds for run_ds in run_datasets
        }

        self.df = pd.DataFrame()
        self.df = pd.concat(
            [LabelDataset._get_labels(ds) for ds in self.run_datasets.values()]
        )

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


@dataclass
class TableSampleItem:
    """Lightweight sample for table-based evaluation (no model/trace objects)."""

    dataset_id: str
    model_id: str
    combination_id: str
    feature_vector: np.ndarray
    algo: str  # best heuristic (label)


class TableLabelDataset(Dataset):
    """Dataset for i.i.d. evaluation using pre-computed CSV tables."""

    def __init__(
        self,
        test_tables: list[pd.DataFrame],
        runs_tables: Optional[list[pd.DataFrame]] = None,
    ):
        """
        Args:
            test_tables: List of test split DataFrames (.test.csv)
            runs_tables: List of full runs DataFrames (.runs.csv) for
                all heuristic timings. If None, only labels are available.
        """
        self.test_df = pd.concat(test_tables, ignore_index=True)
        self._parse_feature_vectors()

        # Build runs lookup for all heuristic times (for performance ratio)
        self.runs_by_combination: dict[str, pd.DataFrame] = {}
        if runs_tables:
            runs_df = pd.concat(runs_tables, ignore_index=True)
            for comb_id, group in runs_df.groupby("combination_id"):
                self.runs_by_combination[comb_id] = group

    def _parse_feature_vectors(self):
        """Parse feature_vector strings to numpy arrays."""
        def parse_fv(fv_str):
            if isinstance(fv_str, np.ndarray):
                return fv_str
            fv_str = fv_str.replace("[", "").replace("]", "").replace("\n", " ")
            return np.fromstring(fv_str, sep=" ")

        self.test_df["feature_vector_parsed"] = self.test_df[
            "feature_vector"
        ].apply(parse_fv)

    def hash(self) -> str:
        return hashlib.sha1(
            str(len(self.test_df)).encode()
        ).hexdigest()[:16]

    @property
    def labels(self) -> list[str]:
        return sorted(self.test_df["aligner"].unique().tolist())

    def __len__(self):
        return len(self.test_df)

    def __getitem__(self, idx) -> tuple[str, TableSampleItem]:
        row = self.test_df.iloc[idx]
        item = TableSampleItem(
            dataset_id=row["dataset_id"],
            model_id=row["model_id"],
            combination_id=row["combination_id"],
            feature_vector=row["feature_vector_parsed"],
            algo=row["aligner"],
        )
        return item.dataset_id, item

    def iter_by_model(self) -> Iterator[list[tuple[str, TableSampleItem]]]:
        """Iterate samples grouped by model_id."""
        for model_id in self.test_df["model_id"].unique():
            model_rows = self.test_df[self.test_df["model_id"] == model_id]
            yield [
                (
                    row["dataset_id"],
                    TableSampleItem(
                        dataset_id=row["dataset_id"],
                        model_id=row["model_id"],
                        combination_id=row["combination_id"],
                        feature_vector=row["feature_vector_parsed"],
                        algo=row["aligner"],
                    ),
                )
                for _, row in model_rows.iterrows()
            ]

    def get_combination_times(
        self, combination_id: str, timeout_value: float = 20.0
    ) -> dict[str, float]:
        """Get execution times for all heuristics for a combination."""
        if combination_id not in self.runs_by_combination:
            return {}
        group = self.runs_by_combination[combination_id]
        return {
            row["aligner"]: (
                row["time_total_mean"]
                if row["time_total_mean"] != float("inf")
                else timeout_value
            )
            for _, row in group.iterrows()
        }


if __name__ == "__main__":
    from dataloaders.util import get_natural_dataset
    from util.rng import RNG

    SEED = 1
    RNG.initialize(SEED)

    TEST_DATASETS = {
        'db35afac-2133-40f3-a565-2dc77a9329a3': ['PermitLog.xes'],
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
                print(f"  Loaded {len(run_dataset)} runs from cache")

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
