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
from typing import Optional, Dict, Any, List

import polars as pl

from pm4py.objects.log.util.dataframe_utils import Parameters
from pm4py.util import constants, exec_utils, pandas_utils
from pm4py.util import xes_constants


def _sanitize_feature_name(prefix: str, value: Any) -> str:
    sanitized = (
        str(value)
        .encode("ascii", errors="ignore")
        .decode("ascii")
    )
    if not sanitized:
        sanitized = "value"
    return f"{prefix}_{sanitized}"


def _scalar_from_lazy(lf: pl.LazyFrame, expr: pl.Expr) -> Any:
    result = lf.select(expr.alias("__scalar")).collect()
    if result.height == 0 or result.width == 0:
        return None
    return result.to_series(0)[0]


def _lazy_schema(lf: pl.LazyFrame) -> pl.Schema:
    return lf.collect_schema()


def _lazy_columns(lf: pl.LazyFrame) -> List[str]:
    return _lazy_schema(lf).names()


def _is_numeric_dtype(dtype: pl.DataType) -> bool:
    dtype_str = str(dtype).lower()
    if any(token in dtype_str for token in ("int", "uint", "float")):
        return True
    if dtype_str.startswith("decimal"):
        return True
    if dtype_str.startswith("duration"):
        return True
    if dtype_str == "boolean":
        return True
    return False


def _is_string_dtype(dtype: pl.DataType) -> bool:
    dtype_str = str(dtype).lower()
    return dtype_str in {"utf8", "string"} or dtype_str.startswith("categorical")


def automatic_feature_selection_df(
    df: pl.LazyFrame, parameters: Optional[Dict[Any, Any]] = None
) -> pl.LazyFrame:
    """Selects useful features from a Polars lazyframe for ML purposes."""
    if parameters is None:
        parameters = {}

    schema = _lazy_schema(df)
    available_columns = set(schema.names())

    case_id_key = exec_utils.get_param_value(
        Parameters.CASE_ID_KEY, parameters, constants.CASE_CONCEPT_NAME
    )
    timestamp_key = exec_utils.get_param_value(
        Parameters.TIMESTAMP_KEY,
        parameters,
        xes_constants.DEFAULT_TIMESTAMP_KEY,
    )
    activity_key = exec_utils.get_param_value(
        Parameters.ACTIVITY_KEY, parameters, xes_constants.DEFAULT_NAME_KEY
    )

    default_mandatory = available_columns.intersection(
        {case_id_key, activity_key, timestamp_key}
    )
    mandatory_attributes = exec_utils.get_param_value(
        Parameters.MANDATORY_ATTRIBUTES,
        parameters,
        default_mandatory,
    )
    mandatory_set = set(mandatory_attributes)

    min_different_occ_str_attr = exec_utils.get_param_value(
        Parameters.MIN_DIFFERENT_OCC_STR_ATTR, parameters, 5
    )
    max_different_occ_str_attr = exec_utils.get_param_value(
        Parameters.MAX_DIFFERENT_OCC_STR_ATTR, parameters, 50
    )
    consider_all_attributes = exec_utils.get_param_value(
        Parameters.CONSIDER_ALL_ATTRIBUTES, parameters, True
    )

    other_attributes_to_retain = set()

    total_cases = _scalar_from_lazy(df, pl.col(case_id_key).n_unique())
    total_cases = int(total_cases or 0)

    for col, dtype in schema.items():
        if col == case_id_key:
            continue

        cases_with_value = _scalar_from_lazy(
            df.filter(pl.col(col).is_not_null()),
            pl.col(case_id_key).n_unique(),
        )
        cases_with_value = int(cases_with_value or 0)

        if cases_with_value != total_cases and not consider_all_attributes:
            continue

        if _is_numeric_dtype(dtype):
            other_attributes_to_retain.add(col)
        elif _is_string_dtype(dtype):
            unique_val_count = _scalar_from_lazy(
                df.filter(pl.col(col).is_not_null()),
                pl.col(col).n_unique(),
            )
            unique_val_count = int(unique_val_count or 0)
            if (
                min_different_occ_str_attr
                <= unique_val_count
                <= max_different_occ_str_attr
            ):
                other_attributes_to_retain.add(col)

    attributes_to_retain = mandatory_set.union(other_attributes_to_retain)
    selected_columns = [
        col_name for col_name in schema.names() if col_name in attributes_to_retain
    ]

    return df.select(selected_columns)


def select_number_column(
    df: pl.LazyFrame,
    fea_df: pl.LazyFrame,
    col: str,
    case_id_key: str = constants.CASE_CONCEPT_NAME,
) -> pl.LazyFrame:
    """Adds a numeric column to the feature lazyframe."""
    df_numeric = (
        df.with_row_count("__row_nr")
        .select(pl.col(case_id_key), pl.col(col), pl.col("__row_nr"))
        .drop_nulls(subset=[col])
        .group_by(case_id_key)
        .agg(pl.col(col).sort_by(pl.col("__row_nr")).last().alias(col))
    )

    return (
        fea_df.join(df_numeric, on=case_id_key, how="left")
        .with_columns(pl.col(col).cast(pl.Float32))
    )


def _collect_categorical_values(
    df: pl.LazyFrame, columns: List[str]
) -> Dict[str, List[Any]]:
    """Collects formatted unique values for the provided categorical columns."""

    collected: Dict[str, List[Any]] = {}
    for col in columns:
        unique_values = (
            df.select(pl.col(col))
            .drop_nulls(subset=[col])
            .unique()
            .collect()
            .get_column(col)
            .to_list()
        )
        formatted = [
            value for value in pandas_utils.format_unique(unique_values) if value is not None
        ]
        if formatted:
            collected[col] = formatted

    return collected


def _select_string_columns(
    df: pl.LazyFrame,
    fea_df: pl.LazyFrame,
    columns: List[str],
    case_id_key: str,
    count_occurrences: bool,
) -> pl.LazyFrame:
    """Adds one-hot or count encoded columns for the provided categorical attributes."""

    if not columns:
        return fea_df

    unique_values_map = _collect_categorical_values(df, columns)
    if not unique_values_map:
        return fea_df

    agg_exprs: List[pl.Expr] = []
    fill_exprs: List[pl.Expr] = []

    for column, unique_values in unique_values_map.items():
        for value in unique_values:
            column_name = _sanitize_feature_name(column, value)

            comparison = pl.col(column).eq(value)

            if count_occurrences:
                agg_expr = comparison.cast(pl.Int64).sum().alias(column_name)
            else:
                agg_expr = comparison.cast(pl.Int8).max().alias(column_name)

            agg_exprs.append(agg_expr)
            fill_exprs.append(
                pl.col(column_name).cast(pl.Float32)
            )

    feature_chunk = (
        df.select(
            [pl.col(case_id_key)] + [pl.col(col) for col in unique_values_map.keys()]
        )
        .group_by(case_id_key)
        .agg(agg_exprs)
        # Materialize all encoded columns in one go to minimize separate joins.
        .with_columns(fill_exprs)
    )

    return fea_df.join(feature_chunk, on=case_id_key, how="left")


def select_string_column(
    df: pl.LazyFrame,
    fea_df: pl.LazyFrame,
    col: str,
    case_id_key: str = constants.CASE_CONCEPT_NAME,
    count_occurrences: bool = False,
) -> pl.LazyFrame:
    """Adds one-hot or count encoded columns for a categorical attribute."""

    return _select_string_columns(
        df,
        fea_df,
        [col],
        case_id_key=case_id_key,
        count_occurrences=count_occurrences,
    )


def select_string_columns(
    df: pl.LazyFrame,
    fea_df: pl.LazyFrame,
    columns: List[str],
    case_id_key: str = constants.CASE_CONCEPT_NAME,
    count_occurrences: bool = False,
) -> pl.LazyFrame:
    """Adds one-hot or count encoded columns for the provided categorical attributes."""

    return _select_string_columns(
        df,
        fea_df,
        columns,
        case_id_key=case_id_key,
        count_occurrences=count_occurrences,
    )


def get_features_df(
    df: pl.LazyFrame,
    list_columns: List[str],
    parameters: Optional[Dict[Any, Any]] = None,
) -> pl.LazyFrame:
    """Performs automatic feature extraction on a Polars LazyFrame."""
    if parameters is None:
        parameters = {}

    case_id_key = exec_utils.get_param_value(
        Parameters.CASE_ID_KEY, parameters, constants.CASE_CONCEPT_NAME
    )
    add_case_identifier_column = exec_utils.get_param_value(
        Parameters.ADD_CASE_IDENTIFIER_COLUMN, parameters, False
    )
    count_occurrences = exec_utils.get_param_value(
        Parameters.COUNT_OCCURRENCES, parameters, False
    )

    fea_df = df.select(pl.col(case_id_key)).unique().sort(case_id_key)

    schema = _lazy_schema(df)
    numeric_columns: List[str] = []
    string_columns: List[str] = []

    for col in list_columns:
        if col == case_id_key or col not in schema:
            continue
        dtype = schema[col]
        if _is_numeric_dtype(dtype):
            numeric_columns.append(col)
        elif _is_string_dtype(dtype):
            string_columns.append(col)

    for col in numeric_columns:
        fea_df = select_number_column(
            df, fea_df, col, case_id_key=case_id_key
        )

    fea_df = select_string_columns(
        df,
        fea_df,
        string_columns,
        case_id_key=case_id_key,
        count_occurrences=count_occurrences,
    )

    fea_df = fea_df.sort(case_id_key)
    if not add_case_identifier_column:
        fea_df = fea_df.drop(case_id_key)

    return fea_df


def automatic_feature_extraction_df(
    df: pl.LazyFrame, parameters: Optional[Dict[Any, Any]] = None
) -> pl.LazyFrame:
    """Wrapper that performs automatic feature extraction on a Polars lazyframe."""
    if parameters is None:
        parameters = {}

    case_id_key = exec_utils.get_param_value(
        Parameters.CASE_ID_KEY, parameters, constants.CASE_CONCEPT_NAME
    )
    timestamp_key = exec_utils.get_param_value(
        Parameters.TIMESTAMP_KEY,
        parameters,
        xes_constants.DEFAULT_TIMESTAMP_KEY,
    )

    fea_sel_df = automatic_feature_selection_df(df, parameters=parameters)
    columns = set(_lazy_columns(fea_sel_df))

    if case_id_key in columns:
        columns.remove(case_id_key)

    if timestamp_key in columns:
        columns.remove(timestamp_key)

    return get_features_df(fea_sel_df, list(columns), parameters=parameters)
