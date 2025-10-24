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
from typing import Optional, Dict, Any
from enum import Enum
from pm4py.util import constants, xes_constants, exec_utils
import pandas as pd


class Parameters(Enum):
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY
    START_TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    COMPUTE_EXTRA_TEMPORAL_FEATURES = "compute_extra_temporal_features"


def compute_extra_columns(dataframe: pd.DataFrame, parameters: Optional[Dict[Any, Any]]=None):
    if parameters is None:
        parameters = {}

    case_id_key = exec_utils.get_param_value(Parameters.CASE_ID_KEY, parameters, constants.CASE_CONCEPT_NAME)
    start_timestamp_key = exec_utils.get_param_value(Parameters.START_TIMESTAMP_KEY, parameters, xes_constants.DEFAULT_TIMESTAMP_KEY)
    timestamp_key = exec_utils.get_param_value(Parameters.TIMESTAMP_KEY, parameters, xes_constants.DEFAULT_TIMESTAMP_KEY)
    compute_extra_temporal_features = exec_utils.get_param_value(Parameters.COMPUTE_EXTRA_TEMPORAL_FEATURES, parameters, True)

    dataframe["@@count"] = 1

    # Calculate first and last timestamps for each case
    case_first_timestamp = dataframe.groupby(case_id_key)[start_timestamp_key].first()
    case_last_timestamp = dataframe.groupby(case_id_key)[timestamp_key].last()

    # Calculate case throughput using the already computed timestamps
    case_throughput = (case_last_timestamp - case_first_timestamp).dt.total_seconds()

    # Map the throughput back to the dataframe
    dataframe["@@case_throughput"] = dataframe[case_id_key].map(case_throughput)

    # Only compute extra temporal features if the parameter is True
    if compute_extra_temporal_features:
        # Compute all start temporal attributes at once on the Series level
        start_years = case_first_timestamp.dt.year.astype(str)
        start_months = case_first_timestamp.dt.month
        start_ymonths = start_years + '-' + start_months.astype(str).str.zfill(2)
        start_month_labels = 'M' + start_months.astype(str).str.zfill(2)
        start_weeks = 'W' + case_first_timestamp.apply(lambda x: str(x.isocalendar()[1]).zfill(2))

        # Compute all end temporal attributes at once on the Series level
        end_years = case_last_timestamp.dt.year.astype(str)
        end_months = case_last_timestamp.dt.month
        end_ymonths = end_years + '-' + end_months.astype(str).str.zfill(2)
        end_month_labels = 'M' + end_months.astype(str).str.zfill(2)
        end_weeks = 'W' + case_last_timestamp.apply(lambda x: str(x.isocalendar()[1]).zfill(2))

        # Map all computed values back to the dataframe in one go for each attribute
        case_col = dataframe[case_id_key]
        dataframe["@@case_start_year"] = case_col.map(start_years)
        dataframe["@@case_start_ymonth"] = case_col.map(start_ymonths)
        dataframe["@@case_start_month"] = case_col.map(start_month_labels)
        dataframe["@@case_start_week"] = case_col.map(start_weeks)

        dataframe["@@case_end_year"] = case_col.map(end_years)
        dataframe["@@case_end_ymonth"] = case_col.map(end_ymonths)
        dataframe["@@case_end_month"] = case_col.map(end_month_labels)
        dataframe["@@case_end_week"] = case_col.map(end_weeks)

    return dataframe
