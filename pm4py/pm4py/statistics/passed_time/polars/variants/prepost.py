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
from pm4py.statistics.passed_time.polars.variants import pre, post
from typing import Optional, Dict, Any
import polars as pl


def apply(
    lf: pl.LazyFrame,
    activity: str,
    parameters: Optional[Dict[Any, Any]] = None,
) -> Dict[str, Any]:
    """
    Gets the time passed from/to each preceding/succeeding activity

    Parameters
    -------------
    lf
        LazyFrame
    activity
        Activity that we are considering
    parameters
        Possible parameters of the algorithm

    Returns
    -------------
    dictio
        Dictionary containing both 'pre' and 'post' keys with the
        list of aggregates times from/to each preceding/succeeding activity
    """
    if parameters is None:
        parameters = {}

    # Get pre statistics
    pre_stats = pre.apply(lf, activity, parameters=parameters)
    
    # Get post statistics  
    post_stats = post.apply(lf, activity, parameters=parameters)
    
    # Combine results
    result = {}
    result.update(pre_stats)
    result.update(post_stats)
    
    return result