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
from typing import Dict, Union, Any, Optional, List
from pm4py.objects.log.obj import EventLog
import pandas as pd
from enum import Enum
from pm4py.util import exec_utils
from pm4py.statistics.chaotic_activities.variants import niek_sidorova


class Variants(Enum):
    NIEK_SIDOROVA = niek_sidorova


def apply(log: Union[pd.DataFrame, EventLog], variant=Variants.NIEK_SIDOROVA,
          parameters: Optional[Dict[Any, Any]] = None) -> List[Dict[str, Any]]:
    """
    Compute metrics used to detect *chaotic activities* in an event log.

    Parameters
    -----------------
    log
        Event log or Pandas dataframe
    variant
        Variant of the algorithm: Variants.NIEK_SIDOROVA
    parameters
        Variant-specific parameters

    Returns
    -----------------
    chaotic_activities
        List of dictionaries, each representing an activity, sorted decreasingly based on the chaotic score (less is better).
    """
    return exec_utils.get_variant(variant).apply(log, parameters)
