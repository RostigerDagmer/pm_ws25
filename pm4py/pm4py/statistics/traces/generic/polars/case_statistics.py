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
from enum import Enum
from typing import Optional, Dict, Any, Union, Tuple, List

import numpy as np
import polars as pl

from pm4py.statistics.traces.generic.common import (
    case_duration as case_duration_commons,
)
from pm4py.util import exec_utils, constants
from pm4py.util import xes_constants as xes
from pm4py.util.business_hours import soj_time_business_hours_diff
from pm4py.util.constants import CASE_CONCEPT_NAME
from pm4py.util.xes_constants import DEFAULT_TIMESTAMP_KEY
from collections import Counter
from statistics import median


class Parameters(Enum):
    ATTRIBUTE_KEY = constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY
    START_TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY

    MAX_VARIANTS_TO_RETURN = "max_variants_to_return"
    VARIANTS_DF = "variants_df"
    ENABLE_SORT = "enable_sort"
    SORT_BY_COLUMN = "sort_by_column"
    SORT_ASCENDING = "sort_ascending"
    MAX_RET_CASES = "max_ret_cases"

    BUSINESS_HOURS = "business_hours"
    BUSINESS_HOUR_SLOTS = "business_hour_slots"
    WORKCALENDAR = "workcalendar"


def get_variant_statistics(
    lf: pl.LazyFrame,
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Get variants from a Polars LazyFrame

    Parameters
    -----------
    lf
        LazyFrame
    parameters
        Parameters of the algorithm, including:
            Parameters.CASE_ID_KEY -> Column that contains the Case ID
            Parameters.ACTIVITY_KEY -> Column that contains the activity
            Parameters.MAX_VARIANTS_TO_RETURN -> Maximum number of variants to return

    Returns
    -----------
    variants_list
        List of variants inside the LazyFrame
    """
    if parameters is None:
        parameters = {}
    
    case_id_glue = exec_utils.get_param_value(
        Parameters.CASE_ID_KEY, parameters, CASE_CONCEPT_NAME
    )
    activity_key = exec_utils.get_param_value(
        Parameters.ACTIVITY_KEY, parameters, xes.DEFAULT_NAME_KEY
    )
    max_variants_to_return = exec_utils.get_param_value(
        Parameters.MAX_VARIANTS_TO_RETURN, parameters, None
    )

    # Get variants by grouping activities per case
    variants_df = (
        lf.group_by(case_id_glue, maintain_order=True)
        .agg(pl.col(activity_key).alias("variant"))
        .collect()
    )
    
    # Count occurrences of each variant
    variant_counts = Counter()
    for row in variants_df.iter_rows():
        variant = tuple(row[1])  # Convert list to tuple for hashing
        variant_counts[variant] += 1
    
    # Convert to list format
    variants_list = [
        {"variant": variant, case_id_glue: count}
        for variant, count in variant_counts.items()
    ]
    
    # Sort by count and variant
    variants_list = sorted(
        variants_list,
        key=lambda x: (x[case_id_glue], x["variant"]),
        reverse=True,
    )

    if max_variants_to_return:
        variants_list = variants_list[
            : min(len(variants_list), max_variants_to_return)
        ]
    
    return variants_list


def get_variants_df_and_list(
    lf: pl.LazyFrame,
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
) -> Tuple[pl.DataFrame, List[List[Any]]]:
    """
    (Technical method) Provides variants_df and variants_list out of the box

    Parameters
    ------------
    lf
        LazyFrame
    parameters
        Parameters of the algorithm, including:
            Parameters.CASE_ID_KEY -> Column that contains the Case ID
            Parameters.ACTIVITY_KEY -> Column that contains the activity

    Returns
    ------------
    variants_df
        Variants dataframe
    variants_list
        List of variants sorted by their count
    """
    if parameters is None:
        parameters = {}
    
    case_id_glue = exec_utils.get_param_value(
        Parameters.CASE_ID_KEY, parameters, CASE_CONCEPT_NAME
    )

    variants_df = get_variants_df(lf, parameters=parameters)
    variants_stats = get_variant_statistics(lf, parameters=parameters)
    
    variants_list = []
    for vd in variants_stats:
        variant = vd["variant"]
        count = vd[case_id_glue]
        variants_list.append([variant, count])
    
    variants_list = sorted(
        variants_list, key=lambda x: (x[1], x[0]), reverse=True
    )
    
    return variants_df, variants_list


def get_cases_description(
    lf: pl.LazyFrame,
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Get a description of traces in the dataframe

    Parameters
    -----------
    lf
        LazyFrame
    parameters
        Parameters of the algorithm, including:
            Parameters.CASE_ID_KEY -> Column that contains the Case ID
            Parameters.TIMESTAMP_KEY -> Column that contains the timestamp

    Returns
    -----------
    ret
        Dictionary of cases with their start time, end time, and case duration
    """
    if parameters is None:
        parameters = {}

    case_id_glue = exec_utils.get_param_value(
        Parameters.CASE_ID_KEY, parameters, CASE_CONCEPT_NAME
    )
    timestamp_key = exec_utils.get_param_value(
        Parameters.TIMESTAMP_KEY, parameters, DEFAULT_TIMESTAMP_KEY
    )

    # Get case statistics
    case_stats = (
        lf.group_by(case_id_glue)
        .agg([
            pl.col(timestamp_key).min().alias("startTime"),
            pl.col(timestamp_key).max().alias("endTime")
        ])
        .with_columns(
            (pl.col("endTime") - pl.col("startTime")).dt.total_seconds().alias("caseDuration")
        )
        .collect()
    )

    ret = []
    for row in case_stats.iter_rows():
        case_id = row[0]
        start_time = row[1]
        end_time = row[2]
        duration = row[3]
        
        ret.append({
            "caseid": case_id,
            "startTime": start_time,
            "endTime": end_time,
            "caseDuration": duration
        })

    return ret


def get_variants_df(
    lf: pl.LazyFrame,
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
) -> pl.DataFrame:
    """
    Gets the variants dataframe

    Parameters
    -----------
    lf
        LazyFrame
    parameters
        Parameters of the algorithm, including:
            Parameters.CASE_ID_KEY -> Column that contains the Case ID
            Parameters.ACTIVITY_KEY -> Column that contains the activity

    Returns
    -----------
    variants_df
        Variants dataframe
    """
    if parameters is None:
        parameters = {}

    case_id_glue = exec_utils.get_param_value(
        Parameters.CASE_ID_KEY, parameters, CASE_CONCEPT_NAME
    )
    activity_key = exec_utils.get_param_value(
        Parameters.ACTIVITY_KEY, parameters, xes.DEFAULT_NAME_KEY
    )

    # Create variants dataframe
    variants_df = (
        lf.group_by(case_id_glue, maintain_order=True)
        .agg(pl.col(activity_key).alias("variant"))
        .collect()
    )

    return variants_df


def get_all_case_durations(
    lf: pl.LazyFrame,
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
) -> List[float]:
    """
    Gets all case durations

    Parameters
    -----------
    lf
        LazyFrame
    parameters
        Parameters of the algorithm

    Returns
    -----------
    case_durations
        List of case durations
    """
    if parameters is None:
        parameters = {}

    case_id_glue = exec_utils.get_param_value(
        Parameters.CASE_ID_KEY, parameters, CASE_CONCEPT_NAME
    )
    timestamp_key = exec_utils.get_param_value(
        Parameters.TIMESTAMP_KEY, parameters, DEFAULT_TIMESTAMP_KEY
    )

    # Get case durations
    durations_df = (
        lf.group_by(case_id_glue)
        .agg([
            pl.col(timestamp_key).min().alias("start_time"),
            pl.col(timestamp_key).max().alias("end_time")
        ])
        .with_columns(
            (pl.col("end_time") - pl.col("start_time")).dt.total_seconds().alias("duration")
        )
        .collect()
    )

    return durations_df["duration"].to_list()


def get_median_case_duration(
    lf: pl.LazyFrame,
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
) -> float:
    """
    Gets the median case duration

    Parameters
    -----------
    lf
        LazyFrame
    parameters
        Parameters of the algorithm

    Returns
    -----------
    median_case_duration
        Median case duration
    """
    durations = get_all_case_durations(lf, parameters)
    return median(durations)


def get_first_quartile_case_duration(
    lf: pl.LazyFrame,
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
) -> float:
    """
    Gets the first quartile case duration

    Parameters
    -----------
    lf
        LazyFrame
    parameters
        Parameters of the algorithm

    Returns
    -----------
    first_quartile_case_duration
        First quartile case duration
    """
    durations = get_all_case_durations(lf, parameters)
    return float(np.percentile(durations, 25))


def get_kde_caseduration(
    lf: pl.LazyFrame,
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
) -> Tuple[List[float], List[float]]:
    """
    Gets the KDE estimation for case durations

    Parameters
    -----------
    lf
        LazyFrame
    parameters
        Parameters of the algorithm

    Returns
    -----------
    x, y
        X and Y values for KDE
    """
    durations = get_all_case_durations(lf, parameters)
    return case_duration_commons.get_kde_caseduration(durations, parameters)


def get_kde_caseduration_json(
    lf: pl.LazyFrame,
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
) -> Dict[str, Any]:
    """
    Gets the KDE estimation for case durations (JSON format)

    Parameters
    -----------
    lf
        LazyFrame
    parameters
        Parameters of the algorithm

    Returns
    -----------
    json_data
        JSON representation of KDE
    """
    durations = get_all_case_durations(lf, parameters)
    return case_duration_commons.get_kde_caseduration_json(durations, parameters)