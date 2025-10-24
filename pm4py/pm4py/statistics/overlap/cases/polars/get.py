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
from typing import Dict, Optional, Any, List, Union

import polars as pl

from pm4py.statistics.overlap.utils import compute
from pm4py.util import exec_utils, constants, xes_constants


class Parameters(Enum):
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    START_TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY


def apply(
    lf: pl.LazyFrame,
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
) -> List[int]:
    """
    Computes the case overlap statistic from a Polars LazyFrame

    Parameters
    -----------------
    lf
        LazyFrame
    parameters
        Parameters of the algorithm, including:
        - Parameters.TIMESTAMP_KEY => attribute representing the completion timestamp
        - Parameters.START_TIMESTAMP_KEY => attribute representing the start timestamp

    Returns
    ----------------
    case_overlap
        List associating to each case the number of open cases during the life of a case
    """
    if parameters is None:
        parameters = {}

    timestamp_key = exec_utils.get_param_value(
        Parameters.TIMESTAMP_KEY,
        parameters,
        xes_constants.DEFAULT_TIMESTAMP_KEY,
    )
    start_timestamp_key = exec_utils.get_param_value(
        Parameters.START_TIMESTAMP_KEY,
        parameters,
        xes_constants.DEFAULT_TIMESTAMP_KEY,
    )
    case_id_key = exec_utils.get_param_value(
        Parameters.CASE_ID_KEY, parameters, constants.CASE_CONCEPT_NAME
    )

    columns = list({timestamp_key, start_timestamp_key, case_id_key})
    
    # Get min and max timestamps per case
    case_ranges = (
        lf.select(columns)
        .group_by(case_id_key)
        .agg([
            pl.col(start_timestamp_key).min().dt.timestamp().alias("min_start"),
            pl.col(timestamp_key).max().dt.timestamp().alias("max_end")
        ])
        .collect()
    )
    
    # Convert to list of tuples for the compute function
    points = []
    for row in case_ranges.iter_rows():
        points.append((row[1], row[2]))  # (min_start, max_end)

    return compute.apply(points, parameters=parameters)