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
"""Polars-based classic process cube variant."""

from bisect import bisect_right
import math
from enum import Enum
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import polars as pl

from pm4py.util import constants, exec_utils

CASE_ID_COL = constants.CASE_CONCEPT_NAME

# Accepted aggregate functions for Polars pivot
_AGG_FN_ALIASES = {
    "avg": "mean",
    "average": "mean",
    "mean": "mean",
    "sum": "sum",
    "min": "min",
    "max": "max",
    "count": "count",
    "len": "count",
    "median": "median",
    "first": "first",
    "last": "last",
}


class Parameters(Enum):
    MAX_DIVISIONS_X = "max_divisions_x"
    MAX_DIVISIONS_Y = "max_divisions_y"
    AGGREGATION_FUNCTION = "aggregation_function"
    X_BINS = "x_bins"
    Y_BINS = "y_bins"


_NUMERIC_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}


def _ensure_polars_df(feature_table: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    if isinstance(feature_table, pl.LazyFrame):
        return feature_table.collect()
    if isinstance(feature_table, pl.DataFrame):
        return feature_table.clone()
    raise TypeError(
        "feature_table must be a Polars LazyFrame or DataFrame"
    )


def _is_numeric_dtype(dtype: pl.DataType) -> bool:
    if dtype in _NUMERIC_DTYPES:
        return True
    dtype_str = str(dtype).lower()
    return any(token in dtype_str for token in ("int", "uint", "float"))


def _prepare_bins(
    series: pl.Series,
    manual_bins: Optional[List[float]],
    max_divisions: int,
) -> List[float]:
    if manual_bins:
        bins = sorted(set(float(b) for b in manual_bins))
        if len(bins) >= 2:
            return bins

    values = series.drop_nulls()
    if values.is_empty():
        return []

    min_val = float(values.min())
    max_val = float(values.max())
    if math.isfinite(min_val) and math.isfinite(max_val):
        if math.isclose(min_val, max_val):
            return [min_val - 0.5, max_val + 0.5]

        max_divisions = max(int(max_divisions), 1)
        bins_array = np.linspace(min_val, max_val, max_divisions + 1)
        bins = sorted(set(float(b) for b in bins_array))
        if len(bins) >= 2:
            return bins
        return [min_val - 0.5, max_val + 0.5]

    return []


def _assign_bins(series: pl.Series, bins: List[float], name: str) -> pl.Series:
    length = len(series)
    if len(bins) < 2:
        return pl.Series(name, [None] * length)

    labels = [f"[{bins[i]}, {bins[i + 1]}]" for i in range(len(bins) - 1)]
    last_edge = bins[-1]

    assigned: List[Optional[str]] = []
    for value in series.to_list():
        if value is None:
            assigned.append(None)
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            assigned.append(None)
            continue
        if not math.isfinite(numeric_value):
            assigned.append(None)
            continue

        idx = bisect_right(bins, numeric_value) - 1
        if idx == len(labels) and math.isclose(numeric_value, last_edge, rel_tol=1e-9, abs_tol=1e-9):
            idx -= 1
        if idx < 0 or idx >= len(labels):
            assigned.append(None)
        else:
            assigned.append(labels[idx])

    return pl.Series(name, assigned)


def _normalize_agg_fn(agg_fn: Any) -> str:
    if isinstance(agg_fn, str):
        key = agg_fn.lower()
        return _AGG_FN_ALIASES.get(key, key)
    return "mean"


def _aggregation_expression(column: str, agg_fn: str) -> pl.Expr:
    if agg_fn == "sum":
        return pl.col(column).sum()
    if agg_fn == "min":
        return pl.col(column).min()
    if agg_fn == "max":
        return pl.col(column).max()
    if agg_fn == "median":
        return pl.col(column).median()
    if agg_fn == "first":
        return pl.col(column).first()
    if agg_fn == "last":
        return pl.col(column).last()
    if agg_fn == "count":
        return pl.count()
    # default mean
    return pl.col(column).mean()


def _prefix_columns(df: pl.DataFrame, prefix: str) -> List[str]:
    return [col for col in df.columns if col.startswith(prefix+"_")]


def _numeric_numeric_case(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    agg_col: str,
    x_bins: List[float],
    y_bins: List[float],
) -> pl.DataFrame:
    x_bin_series = _assign_bins(df[x_col], x_bins, "x_bin")
    y_bin_series = _assign_bins(df[y_col], y_bins, "y_bin")

    df = df.with_columns([x_bin_series, y_bin_series])

    temp_df = (
        df.filter(pl.col("x_bin").is_not_null() & pl.col("y_bin").is_not_null())
        .select([pl.col(CASE_ID_COL), pl.col("x_bin"), pl.col("y_bin"), pl.col(agg_col)])
    )
    return temp_df


def _numeric_prefix_case(
    df: pl.DataFrame,
    x_col: str,
    agg_col: str,
    bins: List[float],
    prefix_cols: List[str],
    prefix_label: str,
) -> pl.DataFrame:
    x_bin_series = _assign_bins(df[x_col], bins, "x_bin")
    df = df.with_columns(x_bin_series)

    if not prefix_cols:
        return pl.DataFrame({CASE_ID_COL: [], "x_bin": [], "y_bin": [], agg_col: []})

    melted = (
        df.select([CASE_ID_COL, agg_col, "x_bin", *prefix_cols])
        .melt(
            id_vars=[CASE_ID_COL, agg_col, "x_bin"],
            value_vars=prefix_cols,
            variable_name=prefix_label,
            value_name="__value",
        )
        .filter(pl.col("x_bin").is_not_null() & (pl.col("__value").fill_null(0) >= 1))
        .select([
            pl.col(CASE_ID_COL),
            pl.col("x_bin"),
            pl.col(prefix_label).alias("y_bin"),
            pl.col(agg_col),
        ])
    )
    return melted


def _prefix_numeric_case(
    df: pl.DataFrame,
    y_col: str,
    agg_col: str,
    bins: List[float],
    prefix_cols: List[str],
    prefix_label: str,
) -> pl.DataFrame:
    y_bin_series = _assign_bins(df[y_col], bins, "y_bin")
    df = df.with_columns(y_bin_series)

    if not prefix_cols:
        return pl.DataFrame({CASE_ID_COL: [], "x_bin": [], "y_bin": [], agg_col: []})

    melted = (
        df.select([CASE_ID_COL, agg_col, "y_bin", *prefix_cols])
        .melt(
            id_vars=[CASE_ID_COL, agg_col, "y_bin"],
            value_vars=prefix_cols,
            variable_name=prefix_label,
            value_name="__value",
        )
        .filter(pl.col("y_bin").is_not_null() & (pl.col("__value").fill_null(0) >= 1))
        .select([
            pl.col(CASE_ID_COL),
            pl.col(prefix_label).alias("x_bin"),
            pl.col("y_bin"),
            pl.col(agg_col),
        ])
    )
    return melted


def _prefix_prefix_case(
    df: pl.DataFrame,
    agg_col: str,
    x_prefix_cols: List[str],
    y_prefix_cols: List[str],
) -> pl.DataFrame:
    if not x_prefix_cols or not y_prefix_cols:
        return pl.DataFrame({CASE_ID_COL: [], "x_bin": [], "y_bin": [], agg_col: []})

    x_melt = (
        df.select([CASE_ID_COL, agg_col, *x_prefix_cols])
        .melt(
            id_vars=[CASE_ID_COL, agg_col],
            value_vars=x_prefix_cols,
            variable_name="x_bin",
            value_name="__x_value",
        )
        .filter(pl.col("__x_value").fill_null(0) >= 1)
        .select([pl.col(CASE_ID_COL), pl.col("x_bin"), pl.col(agg_col)])
    )

    y_melt = (
        df.select([CASE_ID_COL, *y_prefix_cols])
        .melt(
            id_vars=[CASE_ID_COL],
            value_vars=y_prefix_cols,
            variable_name="y_bin",
            value_name="__y_value",
        )
        .filter(pl.col("__y_value").fill_null(0) >= 1)
        .select([pl.col(CASE_ID_COL), pl.col("y_bin")])
    )

    temp_df = (
        x_melt.join(y_melt, on=CASE_ID_COL, how="inner")
        .select([pl.col(CASE_ID_COL), pl.col("x_bin"), pl.col("y_bin"), pl.col(agg_col)])
    )
    return temp_df


def apply(
    feature_table: pl.LazyFrame | pl.DataFrame,
    x_col: str,
    y_col: str,
    agg_col: str,
    parameters: Optional[Dict[Any, Any]] = None,
) -> Tuple[pl.DataFrame, Dict[Any, Any]]:
    """Construct a process cube using Polars data structures."""

    if parameters is None:
        parameters = {}

    df = _ensure_polars_df(feature_table)
    if df.is_empty():
        return pl.DataFrame(), {}

    agg_fn_param = exec_utils.get_param_value(
        Parameters.AGGREGATION_FUNCTION, parameters, "mean"
    )
    agg_fn = _normalize_agg_fn(agg_fn_param)

    max_divisions_x = exec_utils.get_param_value(Parameters.MAX_DIVISIONS_X, parameters, 4)
    max_divisions_y = exec_utils.get_param_value(Parameters.MAX_DIVISIONS_Y, parameters, 4)
    x_bins_param = exec_utils.get_param_value(Parameters.X_BINS, parameters, None)
    y_bins_param = exec_utils.get_param_value(Parameters.Y_BINS, parameters, None)

    numeric_x = x_col in df.columns and _is_numeric_dtype(df.schema[x_col])
    numeric_y = y_col in df.columns and _is_numeric_dtype(df.schema[y_col])

    x_prefix_cols: List[str] = []
    y_prefix_cols: List[str] = []

    if not numeric_x:
        x_prefix_cols = _prefix_columns(df, x_col)
    if not numeric_y:
        y_prefix_cols = _prefix_columns(df, y_col)

    temp_df: pl.DataFrame

    if numeric_x and numeric_y:
        x_bins = _prepare_bins(df[x_col], x_bins_param, max_divisions_x)
        y_bins = _prepare_bins(df[y_col], y_bins_param, max_divisions_y)
        if len(x_bins) < 2 or len(y_bins) < 2:
            return pl.DataFrame(), {}
        temp_df = _numeric_numeric_case(df, x_col, y_col, agg_col, x_bins, y_bins)
    elif numeric_x and not numeric_y:
        x_bins = _prepare_bins(df[x_col], x_bins_param, max_divisions_x)
        if len(x_bins) < 2 or not y_prefix_cols:
            return pl.DataFrame(), {}
        temp_df = _numeric_prefix_case(df, x_col, agg_col, x_bins, y_prefix_cols, "y_bin")
    elif not numeric_x and numeric_y:
        y_bins = _prepare_bins(df[y_col], y_bins_param, max_divisions_y)
        if len(y_bins) < 2 or not x_prefix_cols:
            return pl.DataFrame(), {}
        temp_df = _prefix_numeric_case(df, y_col, agg_col, y_bins, x_prefix_cols, "x_bin")
    else:
        if not x_prefix_cols or not y_prefix_cols:
            return pl.DataFrame(), {}
        temp_df = _prefix_prefix_case(df, agg_col, x_prefix_cols, y_prefix_cols)

    if temp_df.is_empty():
        return pl.DataFrame(), {}

    temp_df = temp_df.filter(
        pl.col("x_bin").is_not_null() & pl.col("y_bin").is_not_null()
    )
    if temp_df.is_empty():
        return pl.DataFrame(), {}

    agg_expr = _aggregation_expression(agg_col, agg_fn).alias("value")
    grouped_values = (
        temp_df.group_by(["x_bin", "y_bin"], maintain_order=True)
        .agg(agg_expr)
    )

    pivot_df = grouped_values.pivot(
        values="value",
        index="x_bin",
        columns="y_bin",
        sort_columns=False,
    )

    case_groups = (
        temp_df.group_by(["x_bin", "y_bin"], maintain_order=True)
        .agg(pl.col(CASE_ID_COL).unique().alias("case_ids"))
    )
    cell_case_dict = {
        (row[0], row[1]): set(row[2])
        for row in case_groups.iter_rows()
    }

    return pivot_df, cell_case_dict


__all__ = ["Parameters", "apply"]
