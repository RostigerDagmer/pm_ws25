'''
    PM4Py â€“ A Process Mining Library for Python
Copyright (C) 2024 Process Intelligence Solutions UG (haftungsbeschrÃ¤nkt)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see this software project's root or
visit <https://www.gnu.org/licenses/>.

Website: https://processintelligence.solutions
Contact: info@processintelligence.solutions
'''
"""Utilities for enriching event data stored in Polars LazyFrames."""

from enum import Enum
from typing import Optional, Dict, Any

import polars as pl

from pm4py.util import constants, xes_constants, exec_utils


class Parameters(Enum):
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY
    START_TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    COMPUTE_EXTRA_TEMPORAL_FEATURES = "compute_extra_temporal_features"


def _prepare_case_features(
    df: pl.LazyFrame,
    case_id_key: str,
    start_timestamp_key: str,
    timestamp_key: str,
    compute_extra_temporal_features: bool,
) -> pl.LazyFrame:
    """Compute per-case aggregates needed for feature enrichment."""

    case_summary = df.group_by(case_id_key).agg(
        [
            pl.col(start_timestamp_key).first().alias("__case_start"),
            pl.col(timestamp_key).last().alias("__case_end"),
        ]
    )

    case_summary = case_summary.with_columns(
        (
            (
                pl.col("__case_end").dt.timestamp("ns")
                - pl.col("__case_start").dt.timestamp("ns")
            )
            / 1_000_000_000
        ).alias("@@case_throughput")
    )

    if compute_extra_temporal_features:
        case_summary = case_summary.with_columns(
            [
                pl.col("__case_start").dt.strftime("%Y").alias("@@case_start_year"),
                pl.col("__case_start").dt.strftime("%Y-%m").alias("@@case_start_ymonth"),
                pl.concat_str(
                    pl.lit("M"), pl.col("__case_start").dt.strftime("%m")
                ).alias("@@case_start_month"),
                pl.concat_str(
                    pl.lit("W"),
                    pl.col("__case_start")
                    .dt.week()
                    .cast(pl.Utf8)
                    .str.pad_start(2, "0"),
                ).alias("@@case_start_week"),
                pl.col("__case_end").dt.strftime("%Y").alias("@@case_end_year"),
                pl.col("__case_end").dt.strftime("%Y-%m").alias("@@case_end_ymonth"),
                pl.concat_str(
                    pl.lit("M"), pl.col("__case_end").dt.strftime("%m")
                ).alias("@@case_end_month"),
                pl.concat_str(
                    pl.lit("W"),
                    pl.col("__case_end")
                    .dt.week()
                    .cast(pl.Utf8)
                    .str.pad_start(2, "0"),
                ).alias("@@case_end_week"),
            ]
        )

    select_columns = [pl.col(case_id_key), pl.col("@@case_throughput")]

    if compute_extra_temporal_features:
        select_columns.extend(
            [
                pl.col("@@case_start_year"),
                pl.col("@@case_start_ymonth"),
                pl.col("@@case_start_month"),
                pl.col("@@case_start_week"),
                pl.col("@@case_end_year"),
                pl.col("@@case_end_ymonth"),
                pl.col("@@case_end_month"),
                pl.col("@@case_end_week"),
            ]
        )

    return case_summary.select(select_columns)


def compute_extra_columns(
    dataframe: pl.LazyFrame,
    parameters: Optional[Dict[Any, Any]] = None,
) -> pl.LazyFrame:
    """Enrich a Polars LazyFrame with additional case-level columns."""

    if parameters is None:
        parameters = {}

    case_id_key = exec_utils.get_param_value(
        Parameters.CASE_ID_KEY, parameters, constants.CASE_CONCEPT_NAME
    )
    start_timestamp_key = exec_utils.get_param_value(
        Parameters.START_TIMESTAMP_KEY,
        parameters,
        xes_constants.DEFAULT_TIMESTAMP_KEY,
    )
    timestamp_key = exec_utils.get_param_value(
        Parameters.TIMESTAMP_KEY,
        parameters,
        xes_constants.DEFAULT_TIMESTAMP_KEY,
    )
    compute_extra_temporal_features = exec_utils.get_param_value(
        Parameters.COMPUTE_EXTRA_TEMPORAL_FEATURES, parameters, True
    )

    df = dataframe.with_columns(pl.lit(1).alias("@@count"))

    case_features = _prepare_case_features(
        df,
        case_id_key,
        start_timestamp_key,
        timestamp_key,
        compute_extra_temporal_features,
    )

    enriched = df.join(case_features, on=case_id_key, how="left")
    return enriched


__all__ = ["Parameters", "compute_extra_columns"]
