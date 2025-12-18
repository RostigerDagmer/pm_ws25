import logging
import os
import yaml
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Callable
from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader
from pm4py.objects.petri_net.utils.petri_utils import construct_trace_net
from dataloaders.net import ProcessModelDataset
from dataloaders.unique_net import UniqueProcessModelDataset
from dataloaders.csv_log import CSVEventLogDataset
from dataloaders.xes_log import XESEventLogDataset
from dataloaders.synthetic import SyntheticProcessModelDataset
from dataloaders.runs import RunDataset, AlignerSpec, SyntheticTraceSampler
from features.extractors import BaseFeatureExtractor
from configs.schema import PipelineConfig
from util.rng import RNG

CONSTRUCTION_PARAMS = {
    # HEADER:
    # CustomerID;AgeCategory;Gender;Office_U;Office_W;ContactDate;ContactTimeStart;ContactTimeEnd;QuestionThemeID;QuestionSubthemeID;QuestionTopicID;QuestionTheme;QuestionSubtheme;QuestionTopic;QuestionTheme_EN;QuestionSubtheme_EN;QuestionTopic_EN
    "2b02709f-9a84-4538-a76a-eb002eacf8d1": {
        "rtype": "csv",
        "case_id_col": "CustomerID",
        "timestamp_col": "ContactTimeStart",
        "activity_col": "QuestionTopic_EN",
        "sep": ";",
    },
    # HEADER:
    # CI Name (aff);CI Type (aff);CI Subtype (aff);Service Component WBS (aff);Incident ID;Status;Impact;Urgency;Priority;Category;KM number;Alert Status;# Reassignments;Open Time;Reopen Time;Resolved Time;Close Time;Handle Time (Hours);Closure Code;# Related Interactions;Related Interaction;# Related Incidents;# Related Changes;Related Change;CI Name (CBy);CI Type (CBy);CI Subtype (CBy);ServiceComp WBS (CBy)
    "3cfa2260-f5c5-44be-afe1-b70d35288d6d": {
        "rtype": "csv",
        "case_id_col": "CI Name (aff)",
        "timestamp_col": "Open Time",
        "activity_col": "ServiceComp WBS (CBy)",
        "sep": ";",
    },
    # HEADER:
    # CI Name (aff);CI Type (aff);CI Subtype (aff);Service Comp WBS (aff);Interaction ID;Status;Impact;Urgency;Priority;Category;KM number;Open Time (First Touch);Close Time;Closure Code;First Call Resolution;Handle Time (secs);Related Incident
    "3d5ae0ce-198c-4b5c-b0f9-60d3035d07bf": {
        "rtype": "csv",
        "case_id_col": "CI Name (aff)",
        "timestamp_col": "Open Time (First Touch)",
        "activity_col": "CI Type (aff)",
        "sep": ";",
    },
    # HEADER:
    # SessionID;IPID;TIMESTAMP;VHOST;URL_FILE;PAGE_NAME;REF_URL_category;page_load_error;page_action_detail;tip;service_detail;xps_info;page_action_detail_EN;service_detail_EN;tip_EN
    "9b99a146-51b5-48df-aa70-288a76c82ec4": {
        "rtype": "csv",
        "case_id_col": "SessionID",
        "timestamp_col": "TIMESTAMP",
        "activity_col": "page_action_detail_EN",
        "sep": ";",
    },
    # HEADER:
    # CustomerID;AgeCategory;Gender;Office_U;Office_W;SessionID;IPID;TIMESTAMP;VHOST;URL_FILE;PAGE_NAME;REF_URL_category;page_load_error;page_action_detail;tip;service_detail;xps_info;page_action_detail_EN;service_detail_EN;tip_EN
    "01345ac4-7d1d-426e-92b8-24933a079412": {
        "rtype": "csv",
        "case_id_col": "CustomerID",
        "timestamp_col": "TIMESTAMP",
        "activity_col": "page_action_detail_EN",
        "sep": ";",
    },
    # HEADER:
    # Incident ID;DateStamp;IncidentActivity_Number;IncidentActivity_Type;Assignment Group;KM number;Interaction ID
    "86977bac-f874-49cf-8337-80f26bf5d2ef": {
        "rtype": "csv",
        "case_id_col": "Incident ID",
        "timestamp_col": "DateStamp",
        "activity_col": "IncidentActivity_Type",
        "sep": ";",
    },
    # HEADER:
    # CustomerID;AgeCategory;Gender;Office_U;Office_W;EventDateTime;EventType;HandlingChannelID
    "c3f3ba2d-e81e-4274-87c7-882fa1dbab0d": {
        "rtype": "csv",
        "case_id_col": "CustomerID",
        "timestamp_col": "EventDateTime",
        "activity_col": "Office_U",
        "sep": ";",
    },
    # HEADER:
    # CustomerID;AgeCategory;Gender;Office_U;Office_W;ComplaintDossierID;ComplaintID;ContactDate;ContactChannelID;ComplaintThemeID;ComplaintSubthemeID;ComplaintTopicID;ComplaintTheme;ComplaintSubtheme;ComplaintTopic;ComplaintTheme_EN;ComplaintSubtheme_EN;ComplaintTopic_EN
    "e30ba0c8-0039-4835-a493-6e3aa2301d3f": {
        "rtype": "csv",
        "case_id_col": "CustomerID",
        "timestamp_col": "ContactDate",
        "activity_col": "ComplaintTopic_EN",
        "sep": ";",
    },
}

DEFAULT_PARAMS_CSV = {
    "case_id_col": "case:concept:name",
    "activity_col": "concept:name",
    "timestamp_col": "time:timestamp",
    "sep": ",",
}

DEFAULT_PARAMS_XES = {"attribute": "concept:name"}


def build_dataset(path: str):
    ext = Path(path).suffix.lower()

    if ext == ".xes":
        return XESEventLogDataset(path)

    if ext == ".csv":
        dataset_id = Path(path).parent.name  # or another identifier
        if dataset_id not in CONSTRUCTION_PARAMS:
            raise KeyError(
                f"No CSV event log constructor params for '{dataset_id}'"
            )
        return CSVEventLogDataset(path, **CONSTRUCTION_PARAMS[dataset_id])

    raise ValueError(f"Unsupported log format: {ext}")


def build_pipeline(cfg: PipelineConfig, skip_init: bool = False) -> RunDataset:
    RNG.initialize(cfg.seed)
    log_dataset = build_dataset(cfg.log_path)

    pm_dataset = ProcessModelDataset(
        log_dataset=log_dataset,
        discovery_methods=cfg.discovery.resolve(),
        param_grid=cfg.discovery.params,
        sampler_specs={
            sampler.name: sampler.build() for sampler in cfg.discovery.samplers
        },
        cached=True,
        num_workers=cfg.discovery.workers or cfg.alignment.workers,
    )

    if cfg.deduplication:
        pm_dataset = UniqueProcessModelDataset(
            base_dataset=pm_dataset,
            dedup_config=cfg.deduplication.config,
            force_recompute=cfg.deduplication.force_recompute,
        )

    return RunDataset(
        base_path=cfg.alignment.cache_path or Path("cache/.runs"),
        process_model_dataset=pm_dataset,
        aligners=cfg.alignment.resolve(),
        trace_sampler=cfg.alignment.sampler.build(ds=pm_dataset),
        n_runs=cfg.alignment.runs,
        n_workers=cfg.alignment.workers,
        write_batch_size=cfg.alignment.write_batch_size,
        skip_init=skip_init,
    )


def get_natural_dataset(
    log_path: str,
    config: str,
    base_path: Optional[str] = None,
    skip_init: bool = False,
    seed: int = 42,
) -> RunDataset:
    RNG.initialize(seed)
    cfg_dict = yaml.safe_load(open(config))
    cfg = PipelineConfig.model_validate(cfg_dict)
    cfg.log_path = log_path
    cfg.alignment.cache_path = base_path

    cfg.seed = seed
    # Use SLURM_CPUS_PER_TASK if available, otherwise default to auto
    cfg.alignment.workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))

    # Skip config <-> cache check
    # (This is unsafe if process models are being referenced in the cache that are not included in the current config)
    return build_pipeline(cfg, skip_init=skip_init)


def get_synthetic_dataset(
    cache_path: Path, seed: int = 1, count: int = 20, device: str = "cpu"
) -> RunDataset:
    RNG.initialize(seed)

    N_RUNS = 5  # set number of runs here

    from util.distributions import (
        CategoricalSpec,
        PoissonSpec,
        BernoulliDepthLinearSpec,
    )

    MAX_DEPTH = 2
    MIN_DEPTH = 1

    synthetic_dataset = SyntheticProcessModelDataset(
        param_grid=[
            (
                {  # and dominant
                    "dist_params": {
                        "op": CategoricalSpec([0.1, 0.6, 0.2, 0.1]),
                        "seq_len": PoissonSpec(4),
                        "p_stop": BernoulliDepthLinearSpec(
                            base=0.2, slope=0.1
                        ),
                        "width": PoissonSpec(3),
                    },
                    "min_depth": MIN_DEPTH,
                    "max_depth": MAX_DEPTH,
                },
                count,  # Number of models per config
            ),
            (
                {  # xor dominant
                    "dist_params": {
                        "op": CategoricalSpec([0.6, 0.1, 0.2, 0.1]),
                        "seq_len": PoissonSpec(4),
                        "p_stop": BernoulliDepthLinearSpec(
                            base=0.2, slope=0.1
                        ),
                        "width": PoissonSpec(3),
                    },
                    "min_depth": MIN_DEPTH,
                    "max_depth": MAX_DEPTH,
                },
                count,  # Number of models per config
            ),
            (
                {  # shallow and wide xor dominant
                    "dist_params": {
                        "op": CategoricalSpec([0.6, 0.1, 0.2, 0.1]),
                        "seq_len": PoissonSpec(1),
                        "p_stop": BernoulliDepthLinearSpec(
                            base=0.5, slope=0.3
                        ),
                        "width": PoissonSpec(10),
                    },
                    "min_depth": MIN_DEPTH,
                    "max_depth": MAX_DEPTH,
                },
                count,  # Number of models per config
            ),
            (
                {  # shallow and wide xor / loop dominant
                    "dist_params": {
                        "op": CategoricalSpec([0.4, 0.1, 0.4, 0.1]),
                        "seq_len": PoissonSpec(1),
                        "p_stop": BernoulliDepthLinearSpec(
                            base=0.5, slope=0.3
                        ),
                        "width": PoissonSpec(10),
                    },
                    "min_depth": MIN_DEPTH,
                    "max_depth": MAX_DEPTH,
                },
                count,  # Number of models per config
            ),
            (
                {  # loop dominant
                    "dist_params": {
                        "op": CategoricalSpec([0.15, 0.15, 0.6, 0.1]),
                        "seq_len": PoissonSpec(4),
                        "p_stop": BernoulliDepthLinearSpec(
                            base=0.2, slope=0.1
                        ),
                        "width": PoissonSpec(3),
                    },
                    "min_depth": MIN_DEPTH,
                    "max_depth": MAX_DEPTH,
                },
                count,  # Number of models per config
            ),
        ],
    )
    trace_sampler = SyntheticTraceSampler(
        ds=synthetic_dataset,
        seed=RNG.get_seed(),
        batch_size=128,
        slice=range(0, 8),
        steps=40,
        device=device,
    )

    return RunDataset(
        cache_path,
        synthetic_dataset,
        AlignerSpec.A_STAR.value,
        trace_sampler,
        n_runs=N_RUNS,
        n_workers=20,
    )


def find_existing_tables(
    root: Path,
):
    # find files ending in .train.csv / .test.csv and .eval.csv
    train_tables = []
    test_tables = []
    eval_tables = []
    for table_path in root.glob("**/*.train.csv"):
        train_tables.append(table_path)
    for table_path in root.glob("**/*.test.csv"):
        test_tables.append(table_path)
    for table_path in root.glob("**/*.eval.csv"):
        eval_tables.append(table_path)

    train_tables = [pd.read_csv(table_path) for table_path in train_tables]
    test_tables = [pd.read_csv(table_path) for table_path in test_tables]
    eval_tables = [pd.read_csv(table_path) for table_path in eval_tables]

    return train_tables, test_tables, eval_tables


def split_dataframes(
    labels: pd.DataFrame,
    train_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_ids = labels["combination_id"].unique()
    random.shuffle(unique_ids)

    n = len(unique_ids)
    n_train = int(n * train_ratio)
    n_test = int(n * test_ratio)

    train_ids = set(unique_ids[:n_train])
    test_ids = set(unique_ids[n_train : n_train + n_test])
    eval_ids = set(unique_ids[n_train + n_test :])  # residual into eval

    train_df = labels[labels["combination_id"].isin(train_ids)]
    test_df = labels[labels["combination_id"].isin(test_ids)]
    eval_df = labels[labels["combination_id"].isin(eval_ids)]

    print(
        f"SPLIT SIZES → train:{len(train_df)}  test:{len(test_df)}  eval:{len(eval_df)}"
    )
    return train_df, test_df, eval_df


def collate(
    batch: list[RunDataset.SerializedItemType],
    fe: BaseFeatureExtractor,
    schema: list[str],
    fmt_row: Callable[
        [RunDataset.ItemType, np.typing.NDArray[np.float32]], pd.Series
    ],
) -> pd.DataFrame:
    df_local = pd.DataFrame(columns=schema)
    for run in batch:
        run = run.deserialize()
        model, trace, item, perf, algo = (
            run.model,
            run.trace,
            run.item,
            run.perf,
            run.algo,
        )
        trace_net, trace_im, trace_fm = construct_trace_net(trace)
        fv = fe.extract(
            model.pm, model.im, model.fm, trace_net, trace_im, trace_fm
        )
        row = fmt_row(run, fv)
        # supress future warning
        df_local = (
            pd.DataFrame([row])
            if df_local.empty
            else pd.concat([df_local, pd.DataFrame([row])], ignore_index=True)
        )

    return df_local


def create_tables(
    run_dataset: RunDataset,
    train_ratio: float,
    test_ratio: float,
    schema: list[str],
    fe: BaseFeatureExtractor,
    fmt_row: Callable[
        [RunDataset.ItemType, np.typing.NDArray[np.float32]], pd.Series
    ],
    batch_size: int = 512,
    num_workers: int = 8,
    persistent_workers: bool = True,
    force_recompute: bool = True,
):
    df = pd.DataFrame(columns=schema)
    base = run_dataset.save_path.with_suffix('')
    train_path = base.with_suffix('.train.csv')
    test_path = base.with_suffix('.test.csv')
    eval_path = base.with_suffix('.eval.csv')

    if not force_recompute and all(
        p.exists() for p in [train_path, test_path, eval_path]
    ):
        logging.info(
            f"Found existing tables for {run_dataset.save_path}, loading..."
        )
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        eval_df = pd.read_csv(eval_path)
        return train_df, test_df, eval_df

    collate_fn = partial(collate, fe=fe, schema=schema, fmt_row=fmt_row)

    dataloader = DataLoader(
        run_dataset.serialized,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    for df_batch in tqdm(dataloader, desc="Extracting features from runs"):
        df = pd.concat(
            [df, df_batch],
            ignore_index=True,
        )

    best_indices = df.groupby("combination_id")["time_total_mean"].idxmin()
    labels = df.loc[best_indices]

    base = run_dataset.save_path.with_suffix('')
    train_df, test_df, eval_df = split_dataframes(
        labels, train_ratio, test_ratio
    )
    train_df.to_csv(f"{base}.train.csv", index=False)
    test_df.to_csv(f"{base}.test.csv", index=False)
    eval_df.to_csv(f"{base}.eval.csv", index=False)

    df.to_csv(f"{base}.runs.csv", index=False)
    labels.to_csv(f"{base}.labels.csv", index=False)
    return train_df, test_df, eval_df
