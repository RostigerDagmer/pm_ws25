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
from typing import Optional, Dict, Any, List, Union

import polars as pl

from pm4py.statistics.overlap.utils import compute
from pm4py.util import constants, xes_constants, exec_utils


class Parameters(Enum):
    START_TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY


def apply(
    lf: pl.LazyFrame,
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
) -> List[int]:
    """
    Counts the intersections of each interval event with the other interval events of the log
    (all the events are considered, not looking at the activity)

    Parameters
    ----------------
    lf
        Polars LazyFrame
    parameters
        Parameters of the algorithm, including:
        - Parameters.START_TIMESTAMP_KEY => the attribute to consider as start timestamp
        - Parameters.TIMESTAMP_KEY => the attribute to consider as timestamp

    Returns
    -----------------
    overlap
        For each interval event, ordered by the order of appearance in the log, associates the number
        of intersecting events.
    """
    if parameters is None:
        parameters = {}

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

    # Extract timestamps and convert to tuples
    if start_timestamp_key == timestamp_key:
        # Handle case where both keys are the same - use same timestamp for both start and end
        df_collected = lf.select([
            pl.col(timestamp_key).alias("start_ts"),
            pl.col(timestamp_key).alias("end_ts")
        ]).collect()
        points = []
        for row in df_collected.iter_rows():
            ts_val = row[0].timestamp() if hasattr(row[0], 'timestamp') else row[0]
            points.append((ts_val, ts_val))
    else:
        df_collected = lf.select([start_timestamp_key, timestamp_key]).collect()
        points = []
        for row in df_collected.iter_rows():
            start_ts = row[0].timestamp() if hasattr(row[0], 'timestamp') else row[0]
            end_ts = row[1].timestamp() if hasattr(row[1], 'timestamp') else row[1]
            points.append((start_ts, end_ts))

    return compute.apply(points)