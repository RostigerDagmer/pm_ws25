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
from typing import Optional, Dict, Any, Tuple

import polars as pl

from pm4py.statistics.process_cube.polars.variants import classic
from pm4py.util import exec_utils


class Variants(Enum):
    CLASSIC = classic


def apply(
    feature_table: pl.LazyFrame | pl.DataFrame,
    x_col: str,
    y_col: str,
    agg_col: str,
    variant=Variants.CLASSIC,
    parameters: Optional[Dict[Any, Any]] = None,
) -> Tuple[pl.DataFrame, Dict[Any, Any]]:
    """Applies the selected process cube variant using Polars data structures."""

    return exec_utils.get_variant(variant).apply(
        feature_table, x_col, y_col, agg_col, parameters=parameters
    )
