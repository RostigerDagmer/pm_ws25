import sys
import os
import json
import argparse
import numpy as np
from collections import Counter
from dataloaders.base import make_feature_fn, BaseEventLogDataset
from dataloaders.csv import CSVEventLogDataset
from dataloaders.xes import XESEventLogDataset
from dataloaders.util import (
    CONSTRUCTION_PARAMS,
    DEFAULT_PARAMS_CSV,
    DEFAULT_PARAMS_XES,
)

DATA_DIR = "data"
datasets = os.listdir(DATA_DIR)


def filter_files(fs: list[str]) -> list[str]:
    return list(filter(lambda f: f.endswith('.csv') or f.endswith('.xes'), fs))


files = {d: filter_files(os.listdir(f"{DATA_DIR}/{d}")) for d in datasets}
files


def summarize(ds: BaseEventLogDataset):
    return {
        "n_traces": len(ds),
        "avg_trace_len": np.mean([len(t) for t in ds.log]),
        "n_unique_activities": len(
            ds.vocab[getattr(ds, "activity_col", "concept:name")]
        ),
        "activity_freq": Counter(e["concept:name"] for t in ds.log for e in t),
    }


def get_ds(path: str):
    print(path)
    dataset_ftype = path.split(".")[-1].strip()
    match dataset_ftype:
        case "csv":
            doi = path.split('/')[-2]
            params = CONSTRUCTION_PARAMS.get(doi, DEFAULT_PARAMS_CSV)
            ensure_type = params.get("rtype")
            if ensure_type is not None:
                if ensure_type != "csv":
                    raise ValueError(
                        f"Something went wrong... params says this Dataset is a different type than csv: {ensure_type}"
                    )
            return CSVEventLogDataset(
                source_path=path, feature_fn=make_feature_fn, **params
            )
        case "xes":
            doi = path.split('/')[-2]
            params = CONSTRUCTION_PARAMS.get(doi, DEFAULT_PARAMS_XES)
            ensure_type = params.get("rtype")
            if ensure_type is not None:
                if ensure_type != "xes":
                    raise ValueError(
                        f"Something went wrong... params says this Dataset is a different type than xes: {ensure_type}"
                    )
            return XESEventLogDataset(
                source_path=path, feature_fn=make_feature_fn, **params
            )
        case _:
            raise ValueError(f"unknown dataset source type: '{dataset_ftype}'")


def make_path(doi: str):
    for f in files[doi]:
        yield f"{DATA_DIR}/{doi}/{f}"


def make_summary(doi: str):
    pth = list(make_path(doi))[0]
    test_ds = get_ds(pth)
    s = summarize(test_ds)

    summary_path = '/'.join(pth.split('/')[:-1] + ["summary.json"])
    with open(summary_path, mode="w", encoding="utf-8") as f:
        f.write(json.dumps(s, indent=4))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--doi", default=None, help="Doi of the dataset to summarize"
    )
    args = parser.parse_args()

    dois = datasets

    if args.doi is not None:
        dois = [args.doi]

    for doi in dois:
        make_summary(doi)
