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
from typing import Optional, Dict, Any, Union, List

import polars as pl

from pm4py.util import exec_utils, constants, xes_constants


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY
    SUBCASE_CONCAT_STR = "subcase_concat_str"
    CUT_MODE = "cut_mode"


def apply(
    df: pl.LazyFrame,
    activity: Union[str, List[str]],
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
) -> pl.LazyFrame:
    """
    Splits the cases of a log (Polars LazyFrame) into subcases based on the provision of an activity.
    There are as many subcases as many occurrences of a given activity occur.

    Example:
    Original log:

    [['register request', 'examine casually', 'check ticket', 'decide', 'reinitiate request', 'examine thoroughly',
    'check ticket', 'decide', 'pay compensation'],
    ['register request', 'examine casually', 'check ticket', 'decide', 'reinitiate request', 'check ticket',
    'examine casually', 'decide', 'reinitiate request', 'examine casually', 'check ticket', 'decide', 'reject request']]


    Log filtered using the activity split filter on 'reinitiate request' with cut_mode='this':

    [['register request', 'examine casually', 'check ticket', 'decide'],
    ['reinitiate request', 'examine thoroughly', 'check ticket', 'decide', 'pay compensation'],
    ['register request', 'examine casually', 'check ticket', 'decide'],
    ['reinitiate request', 'check ticket', 'examine casually', 'decide'],
    ['reinitiate request', 'examine casually', 'check ticket', 'decide', 'reject request']]


    Log filtered using the activity split filter on 'reinitiate request' with cut_mode='next':

    [['register request', 'examine casually', 'check ticket', 'decide', 'reinitiate request'],
    ['examine thoroughly', 'check ticket', 'decide', 'pay compensation'],
    ['register request', 'examine casually', 'check ticket', 'decide', 'reinitiate request'],
    ['check ticket', 'examine casually', 'decide', 'reinitiate request'],
    ['examine casually', 'check ticket', 'decide', 'reject request']]


    Parameters
    ----------------
    df
        LazyFrame
    activity
        Activity (or collection of activities)
    parameters
        Parameters of the algorithm, including:
        - Parameters.ACTIVITY_KEY => activity key
        - Parameters.CASE_ID_KEY => case id
        - Parameters.SUBCASE_CONCAT_STR => concatenator between the case id and the subtrace index in the filtered df
        - Parameters.CUT_MODE => mode of cut:
            - "this" means that an event with the specified activity goes to the next subcase
            - "next" means that the following event (to the given activity) goes to the next subcase.

    Returns
    ----------------
    filtered_df
        LazyFrame in which the cases are split into subcases

    """
    if parameters is None:
        parameters = {}

    activity_key = exec_utils.get_param_value(
        Parameters.ACTIVITY_KEY, parameters, xes_constants.DEFAULT_NAME_KEY
    )
    case_id_key = exec_utils.get_param_value(
        Parameters.CASE_ID_KEY, parameters, constants.CASE_CONCEPT_NAME
    )
    subcase_concat_str = exec_utils.get_param_value(
        Parameters.SUBCASE_CONCAT_STR, parameters, "##@@"
    )
    cut_mode = exec_utils.get_param_value(
        Parameters.CUT_MODE, parameters, "this"
    )

    activities = [activity] if isinstance(activity, str) else activity

    # Add row number within each case
    df = df.with_columns(
        pl.col(case_id_key).cum_count().over(case_id_key).alias("_row_within_case")
    )

    # Mark rows where the activity occurs
    df = df.with_columns(
        pl.col(activity_key).is_in(activities).alias("_is_split_activity")
    )

    # Calculate the subcase number for each event
    if cut_mode == "this":
        # Events with split activity start a new subcase
        df = df.with_columns(
            pl.col("_is_split_activity").cast(pl.Int32).cum_sum().over(case_id_key).alias("_subcase_idx")
        )
    else:  # cut_mode == "next"
        # Events after split activity start a new subcase
        df = df.with_columns(
            pl.col("_is_split_activity").shift(1, fill_value=False).over(case_id_key).alias("_shifted_split")
        )
        df = df.with_columns(
            pl.col("_shifted_split").cast(pl.Int32).cum_sum().over(case_id_key).alias("_subcase_idx")
        )
        df = df.drop("_shifted_split")

    # Create new case IDs by concatenating original case ID with subcase index
    df = df.with_columns(
        pl.concat_str([
            pl.col(case_id_key).cast(pl.Utf8),
            pl.lit(subcase_concat_str),
            pl.col("_subcase_idx").cast(pl.Utf8)
        ]).alias(case_id_key)
    )

    # Drop helper columns
    df = df.drop(["_row_within_case", "_is_split_activity", "_subcase_idx"])

    return df