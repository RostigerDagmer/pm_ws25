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
from typing import Optional, Dict, Any, List, Union
from pm4py.util import exec_utils, constants, pandas_utils
from pm4py.objects.log.obj import EventLog, EventStream
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.transformation.log_to_features import algorithm as log_to_features
from enum import Enum
import numpy as np


class Parameters(Enum):
    INCLUDE_HEADER = "include_header"
    MAX_LEN = "max_len"


def __transform_to_string(stru: str) -> str:
    if stru == "@@max_concurrent_activities_general":
        return "Maximum Number of Concurrent Events"
    elif stru.startswith("@@max_concurrent_activities_like_"):
        return (
            "Maximum Number of Concurrent '"
            + stru.split("@@max_concurrent_activities_like_")[-1]
            + "'"
        )
    elif stru.startswith("event:"):
        stru = stru.split("event:")[-1]
        if "@" in stru:
            attr = stru.split("@")[0]
            value = stru.split("@")[-1]
            return "Value '" + value + "' for Event Attribute '" + attr + "'"
        else:
            return "Values for Event Attribute '" + stru + "'"
    elif stru.startswith("trace:"):
        stru = stru.split("trace:")[-1]
        if "@" in stru:
            attr = stru.split("@")[0]
            value = stru.split("@")[-1]
            return "Value '" + value + "' for Case Attribute '" + attr + "'"
        else:
            return "Values for Case Attribute '" + stru + "'"
    elif stru.startswith("succession:"):
        stru = stru.split("succession:")[-1]
        attr = stru.split("@")[0]
        stru = stru.split("@")[-1]
        val1 = stru.split("#")[0]
        val2 = stru.split("#")[-1]
        return (
            "Succession '"
            + val1
            + "' -> '"
            + val2
            + "' for the Values of the Attribute '"
            + attr
            + "'"
        )
    elif stru == "@@caseDuration":
        return "Case Duration"
    elif stru.startswith("firstIndexAct@@"):
        return (
            "First Position of the Activity '"
            + stru.split("@@")[-1]
            + "' in the Case"
        )
    elif stru.startswith("lastIndexAct@@"):
        return (
            "Last Position of the Activity '"
            + stru.split("@@")[-1]
            + "' in the Case"
        )
    elif stru.startswith("startToLastOcc@@"):
        return (
            "Time from Case Start to Last Occurrence of the Activity '"
            + stru.split("@@")[-1]
            + "'"
        )
    elif stru.startswith("lastOccToEnd@@"):
        return (
            "Time from Last Occurrence of the Activity '"
            + stru.split("@@")[-1]
            + "' to Case End"
        )
    elif stru.startswith("startToFirstOcc@@"):
        return (
            "Time from Case Start to First Occurrence of the Activity '"
            + stru.split("@@")[-1]
            + "'"
        )
    elif stru.startswith("firstOccToEnd@@"):
        return (
            "Time from First Occurrence of the Activity '"
            + stru.split("@@")[-1]
            + "' to Case End"
        )
    elif stru.startswith("directPathPerformanceLastOcc@@"):
        stru = stru.split("@@")[-1].split("##")
        return (
            "Directly-Follows Paths Throughput between '"
            + stru[0]
            + "' and '"
            + stru[1]
            + "' (last occurrence of the path in the case)"
        )
    elif stru.startswith("indirectPathPerformanceLastOcc@@"):
        stru = stru.split("@@")[-1].split("##")
        return (
            "Eventually-Follows Paths Throughput between '"
            + stru[0]
            + "' and '"
            + stru[1]
            + "' (last occurrence of the path in the case)"
        )
    elif stru.startswith("resource_workload@@"):
        return "Resource Workload of '" + stru.split("@@")[-1] + "'"
    elif stru == "@@work_in_progress":
        return "Work in Progress"

    return stru


def _categorize_features(
    fea_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Scans the feature table once and returns a list of dictionaries, each of which
    fully describes one *relevant* feature.

    Keys of every dictionary:
        - col (str)  : original column name
        - desc (str) : human label (via __transform_to_string)
        - n_non_zero : how many cases have a non‑zero value
        - std_rel    : relative std‑dev w.r.t. mean (for ranking)
        - series     : the non‑zero part of the column (pd.Series)
    """
    categories: List[Dict[str, Any]] = []

    for col in fea_df.columns:
        ser = fea_df[col]
        non_zero = ser[ser > 0]
        if non_zero.empty:  # skip silent columns
            continue

        mean_val = np.average(non_zero)
        std_rel = 0 if mean_val == 0 or len(non_zero) == 1 else np.std(non_zero) / mean_val
        categories.append(
            dict(
                col=col,
                desc=__transform_to_string(col),
                n_non_zero=len(non_zero),
                std_rel=std_rel,
                series=non_zero,
            )
        )

    # Sort exactly like before: many occurrences first, then variability, then name
    categories.sort(key=lambda d: (d["n_non_zero"], d["std_rel"], d["desc"]), reverse=True)
    return categories


def _features_to_text(
    categories: List[Dict[str, Any]],
    include_header: bool,
    max_len: int,
) -> str:
    parts = ["\n"]
    if include_header:
        parts.append("Given the following features:\n\n")
    text = " ".join(parts)

    for feat in categories:
        if len(text) >= max_len:
            break
        q = feat["series"].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
        block = (
            f"{feat['desc']}:    "
            f"number of non-zero values: {feat['n_non_zero']} ; "
            f"quantiles of the non-zero: {q}\n"
        )
        text += block
    return text


def _features_to_dct(categories: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    dct = {}

    for feat in categories:
        q = feat['series'].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
        dct[feat['col']] = {'non_zero': feat['n_non_zero'], 'quantiles': q}

    return dct


def textual_abstraction_from_fea_df(
    fea_df: pd.DataFrame, parameters: Optional[Dict[Any, Any]] = None
) -> str:
    if parameters is None:
        parameters = {}

    include_header = exec_utils.get_param_value(Parameters.INCLUDE_HEADER, parameters, True)
    max_len = exec_utils.get_param_value(Parameters.MAX_LEN, parameters, constants.OPENAI_MAX_LEN)

    # –– STEP 1: categorise ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    categories = _categorize_features(fea_df)

    # –– STEP 2: text rendering ––––––––––––––––––––––––––––––––––––––––––––––––––––––
    return _features_to_text(categories, include_header, max_len)


def dct_abstraction_from_fea_df(fea_df: pd.DataFrame, parameters: Optional[Dict[Any, Any]] = None) -> Dict[str, Dict[str, Any]]:
    if parameters is None:
        parameters = {}

    categories = _categorize_features(fea_df)

    return _features_to_dct(categories)


def apply(
    log: Union[EventLog, EventStream, pd.DataFrame],
    parameters: Optional[Dict[Any, Any]] = None,
) -> str:
    if parameters is None:
        parameters = {}

    log = log_converter.apply(
        log, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters
    )

    data, feature_names = log_to_features.apply(log, parameters=parameters)
    fea_df = pandas_utils.instantiate_dataframe(data, columns=feature_names)

    return textual_abstraction_from_fea_df(fea_df, parameters=parameters)
