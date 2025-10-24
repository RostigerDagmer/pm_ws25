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
import polars as pl

from pm4py.util.xes_constants import DEFAULT_TIMESTAMP_KEY
from pm4py.util.constants import CASE_CONCEPT_NAME
from pm4py.util import exec_utils
from pm4py.util import constants
from enum import Enum
from typing import Optional, Dict, Any, Union


class Parameters(Enum):
    ATTRIBUTE_KEY = constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    START_TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY
    MAX_NO_POINTS_SAMPLE = "max_no_of_points_to_sample"
    KEEP_ONCE_PER_CASE = "keep_once_per_case"


def get_case_arrival_avg(
    lf: pl.LazyFrame,
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
) -> float:
    """
    Gets the average time interlapsed between case starts

    Parameters
    --------------
    lf
        Polars LazyFrame
    parameters
        Parameters of the algorithm, including:
            Parameters.TIMESTAMP_KEY -> attribute of the log to be used as timestamp

    Returns
    --------------
    case_arrival_avg
        Average time interlapsed between case starts
    """
    if parameters is None:
        parameters = {}

    caseid_glue = exec_utils.get_param_value(
        Parameters.CASE_ID_KEY, parameters, CASE_CONCEPT_NAME
    )
    timest_key = exec_utils.get_param_value(
        Parameters.TIMESTAMP_KEY, parameters, DEFAULT_TIMESTAMP_KEY
    )

    # Get first timestamp per case
    first_df = (
        lf.group_by(caseid_glue)
        .agg(pl.col(timest_key).first().alias("first_timestamp"))
        .sort("first_timestamp")
        .collect()
    )

    # Calculate time differences between consecutive case starts
    time_diffs = (
        first_df.with_columns([
            pl.col("first_timestamp").shift(-1).alias("next_timestamp")
        ])
        .filter(pl.col("next_timestamp").is_not_null())
        .with_columns(
            (pl.col("next_timestamp") - pl.col("first_timestamp"))
            .dt.total_seconds()
            .alias("interlapsed_time")
        )
    )

    return time_diffs["interlapsed_time"].mean()


def get_case_dispersion_avg(
    lf: pl.LazyFrame, 
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None
) -> float:
    """
    Gets the average time interlapsed between case ends

    Parameters
    --------------
    lf
        Polars LazyFrame
    parameters
        Parameters of the algorithm, including:
            Parameters.TIMESTAMP_KEY -> attribute of the log to be used as timestamp

    Returns
    --------------
    case_dispersion_avg
        Average time interlapsed between the completion of cases
    """
    if parameters is None:
        parameters = {}

    caseid_glue = exec_utils.get_param_value(
        Parameters.CASE_ID_KEY, parameters, CASE_CONCEPT_NAME
    )
    timest_key = exec_utils.get_param_value(
        Parameters.TIMESTAMP_KEY, parameters, DEFAULT_TIMESTAMP_KEY
    )

    # Get last timestamp per case
    last_df = (
        lf.group_by(caseid_glue)
        .agg(pl.col(timest_key).last().alias("last_timestamp"))
        .sort("last_timestamp")
        .collect()
    )

    # Calculate time differences between consecutive case ends
    time_diffs = (
        last_df.with_columns([
            pl.col("last_timestamp").shift(-1).alias("next_timestamp")
        ])
        .filter(pl.col("next_timestamp").is_not_null())
        .with_columns(
            (pl.col("next_timestamp") - pl.col("last_timestamp"))
            .dt.total_seconds()
            .alias("interlapsed_time")
        )
    )

    return time_diffs["interlapsed_time"].mean()