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
from pm4py.util.constants import CASE_CONCEPT_NAME, PARAMETER_CONSTANT_CASEID_KEY, PARAMETER_CONSTANT_ACTIVITY_KEY
from enum import Enum
from pm4py.util import exec_utils, constants, xes_constants
from typing import Optional, Dict, Any, Union, List
import polars as pl


class Parameters(Enum):
    CASE_ID_KEY = PARAMETER_CONSTANT_CASEID_KEY
    ACTIVITY_KEY = PARAMETER_CONSTANT_ACTIVITY_KEY
    POSITIVE = "positive"


def apply(
    df: pl.LazyFrame,
    admitted_variants: List[List[str]],
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
) -> pl.LazyFrame:
    """
    Filter on variants

    Parameters
    -----------
    df
        LazyFrame
    admitted_variants
        List of variants to include/exclude
    parameters
        Parameters of the algorithm

    Returns
    -----------
    df
        Filtered LazyFrame
    """
    if parameters is None:
        parameters = {}

    case_id_glue = exec_utils.get_param_value(
        Parameters.CASE_ID_KEY, parameters, CASE_CONCEPT_NAME
    )
    activity_key = exec_utils.get_param_value(
        Parameters.ACTIVITY_KEY, parameters, xes_constants.DEFAULT_NAME_KEY
    )
    positive = exec_utils.get_param_value(Parameters.POSITIVE, parameters, True)

    # Create variants
    variants_df = (
        df.sort([case_id_glue, "time:timestamp"])
        .group_by(case_id_glue, maintain_order=True)
        .agg(
            pl.col(activity_key)
            .str.concat(constants.DEFAULT_VARIANT_SEP)
            .alias("variant")
        )
    )

    # Convert admitted variants to strings
    variant_strings = [constants.DEFAULT_VARIANT_SEP.join(v) for v in admitted_variants]
    
    # Filter for matching variants
    matching_cases = variants_df.filter(
        pl.col("variant").is_in(variant_strings)
    ).select(case_id_glue)

    if positive:
        ret = df.join(matching_cases, on=case_id_glue, how="inner")
    else:
        ret = df.join(matching_cases, on=case_id_glue, how="anti")

    return ret


def apply_auto_filter(
    df: pl.LazyFrame,
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
) -> pl.LazyFrame:
    """
    Apply auto filter on variants

    Parameters
    -----------
    df
        LazyFrame
    parameters
        Parameters of the algorithm

    Returns
    -----------
    df
        Filtered LazyFrame
    """
    if parameters is None:
        parameters = {}

    case_id_glue = exec_utils.get_param_value(
        Parameters.CASE_ID_KEY, parameters, CASE_CONCEPT_NAME
    )
    activity_key = exec_utils.get_param_value(
        Parameters.ACTIVITY_KEY, parameters, constants.DEFAULT_NAME_KEY
    )
    
    # Get variant counts
    variant_counts = (
        df.sort([case_id_glue, "time:timestamp"])
        .group_by(case_id_glue, maintain_order=True)
        .agg(
            pl.col(activity_key)
            .str.concat(constants.DEFAULT_VARIANT_SEP)
            .alias("variant")
        )
        .group_by("variant")
        .count()
        .sort("count", descending=True)
    )
    
    # Calculate cumulative percentage
    total_cases = variant_counts.select(pl.sum("count")).collect()[0, 0]
    variant_counts = variant_counts.with_columns(
        (pl.col("count").cumsum() / total_cases).alias("cumulative_percentage")
    )
    
    # Keep variants covering 80% of cases
    selected_variants = (
        variant_counts
        .filter(pl.col("cumulative_percentage") <= 0.8)
        .select("variant")
        .collect()["variant"]
        .to_list()
    )
    
    # Apply filter
    return apply_from_variant_list(df, selected_variants, parameters)


def apply_from_variant_list(
    df: pl.LazyFrame,
    variants: List[str],
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
) -> pl.LazyFrame:
    """
    Apply filter from variant list

    Parameters
    -----------
    df
        LazyFrame
    variants
        List of variants (as strings)
    parameters
        Parameters

    Returns
    -----------
    df
        Filtered LazyFrame
    """
    if parameters is None:
        parameters = {}

    case_id_glue = exec_utils.get_param_value(
        Parameters.CASE_ID_KEY, parameters, CASE_CONCEPT_NAME
    )
    activity_key = exec_utils.get_param_value(
        Parameters.ACTIVITY_KEY, parameters, constants.DEFAULT_NAME_KEY
    )
    positive = exec_utils.get_param_value(Parameters.POSITIVE, parameters, True)

    # Create variants
    variants_df = (
        df.sort([case_id_glue, "time:timestamp"])
        .group_by(case_id_glue, maintain_order=True)
        .agg(
            pl.col(activity_key)
            .str.concat(constants.DEFAULT_VARIANT_SEP)
            .alias("variant")
        )
    )
    
    # Filter for matching variants
    matching_cases = variants_df.filter(
        pl.col("variant").is_in(variants)
    ).select(case_id_glue)

    if positive:
        ret = df.join(matching_cases, on=case_id_glue, how="inner")
    else:
        ret = df.join(matching_cases, on=case_id_glue, how="anti")

    return ret
