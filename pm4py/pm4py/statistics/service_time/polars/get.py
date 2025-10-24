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
from enum import Enum

from pm4py.util import exec_utils, constants, xes_constants
from pm4py.util.business_hours import soj_time_business_hours_diff
from typing import Optional, Dict, Any, Union


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    START_TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    AGGREGATION_MEASURE = "aggregationMeasure"
    BUSINESS_HOURS = "business_hours"
    BUSINESS_HOUR_SLOTS = "business_hour_slots"
    WORKCALENDAR = "workcalendar"


def apply(
    lf: pl.LazyFrame,
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
) -> Dict[str, float]:
    """
    Gets the service time per activity on a Polars LazyFrame

    Parameters
    --------------
    lf
        Polars LazyFrame
    parameters
        Parameters of the algorithm, including:
        - Parameters.ACTIVITY_KEY => activity key
        - Parameters.START_TIMESTAMP_KEY => start timestamp key
        - Parameters.TIMESTAMP_KEY => timestamp key
        - Parameters.BUSINESS_HOURS => calculates the difference of time based on the business hours, not the total time.
                                        Default: False
        - Parameters.BUSINESS_HOURS_SLOTS =>
        work schedule of the company, provided as a list of tuples where each tuple represents one time slot of business
        hours. One slot i.e. one tuple consists of one start and one end time given in seconds since week start, e.g.
        [
            (7 * 60 * 60, 17 * 60 * 60),
            ((24 + 7) * 60 * 60, (24 + 12) * 60 * 60),
            ((24 + 13) * 60 * 60, (24 + 17) * 60 * 60),
        ]
        meaning that business hours are Mondays 07:00 - 17:00 and Tuesdays 07:00 - 12:00 and 13:00 - 17:00
        - Parameters.AGGREGATION_MEASURE => performance aggregation measure (sum, min, max, mean, median)

    Returns
    --------------
    soj_time_dict
        Service time dictionary
    """
    if parameters is None:
        parameters = {}

    business_hours = exec_utils.get_param_value(
        Parameters.BUSINESS_HOURS, parameters, False
    )
    business_hours_slots = exec_utils.get_param_value(
        Parameters.BUSINESS_HOUR_SLOTS,
        parameters,
        constants.DEFAULT_BUSINESS_HOUR_SLOTS,
    )
    workcalendar = exec_utils.get_param_value(
        Parameters.WORKCALENDAR,
        parameters,
        constants.DEFAULT_BUSINESS_HOURS_WORKCALENDAR,
    )

    activity_key = exec_utils.get_param_value(
        Parameters.ACTIVITY_KEY, parameters, xes_constants.DEFAULT_NAME_KEY
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
    aggregation_measure = exec_utils.get_param_value(
        Parameters.AGGREGATION_MEASURE, parameters, "mean"
    )

    if business_hours:
        # For business hours calculation, we need to collect the data and apply Python function
        if start_timestamp_key == timestamp_key:
            # Handle case where both keys are the same - use 0 duration
            df_collected = lf.collect()
            diff_values = [0.0] * len(df_collected)
        else:
            df_collected = lf.collect()
            diff_values = []
            
            for row in df_collected.iter_rows():
                start_idx = df_collected.columns.index(start_timestamp_key)
                end_idx = df_collected.columns.index(timestamp_key)
                
                diff = soj_time_business_hours_diff(
                    row[start_idx],
                    row[end_idx],
                    business_hours_slots,
                    workcalendar,
                )
                diff_values.append(diff)
        
        df_with_diff = df_collected.with_columns(
            pl.Series("__diff", diff_values)
        ).lazy()
    else:
        # Regular time difference calculation
        if start_timestamp_key == timestamp_key:
            # Handle case where both keys are the same - use 0 duration
            df_with_diff = lf.with_columns(pl.lit(0.0).alias("__diff"))
        else:
            df_with_diff = lf.with_columns(
                (pl.col(timestamp_key) - pl.col(start_timestamp_key))
                .dt.total_seconds()
                .alias("__diff")
            )

    # Group by activity and apply aggregation measure
    if aggregation_measure == "median":
        result = df_with_diff.group_by(activity_key).agg(pl.col("__diff").median())
    elif aggregation_measure == "min":
        result = df_with_diff.group_by(activity_key).agg(pl.col("__diff").min())
    elif aggregation_measure == "max":
        result = df_with_diff.group_by(activity_key).agg(pl.col("__diff").max())
    elif aggregation_measure == "sum":
        result = df_with_diff.group_by(activity_key).agg(pl.col("__diff").sum())
    else:  # mean
        result = df_with_diff.group_by(activity_key).agg(pl.col("__diff").mean())

    result_df = result.collect()
    
    # Convert to dictionary
    ret_dict = {}
    for row in result_df.iter_rows():
        activity = row[0]
        value = row[1]
        ret_dict[activity] = float(value) if value is not None else 0.0

    return ret_dict