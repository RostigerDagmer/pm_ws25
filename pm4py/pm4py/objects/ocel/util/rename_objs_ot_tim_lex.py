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
from typing import Optional, Dict, Any

from copy import deepcopy
from pm4py.objects.ocel.obj import OCEL
from pm4py.util import exec_utils


class Parameters(Enum):
    EXCLUDE_OBJECT_TYPES = "exclude_object_types"


def apply(ocel: OCEL, parameters: Optional[Dict[Any, Any]] = None) -> OCEL:
    """
    Rename objects given their object type, lifecycle start/end timestamps, and lexicographic order,

    Parameters
    -----------------
    ocel
        Object-centric event log
    parameters
        Parameters of the algorithm, including:
        - Parameters.EXCLUDE_OBJECT_TYPES => set/list of object types to exclude from renaming

    Returns
    ----------------
    renamed_ocel
        Object-centric event log with renaming
    """
    if parameters is None:
        parameters = {}

    exclude_object_types = exec_utils.get_param_value(
        Parameters.EXCLUDE_OBJECT_TYPES, parameters, set()
    )
    # normalize to a Python set for fast membership checks
    try:
        exclude_object_types = set(exclude_object_types)
    except Exception:
        exclude_object_types = set()

    objects_start = (
        ocel.relations.groupby(ocel.object_id_column)[ocel.event_timestamp]
        .first()
        .to_dict()
    )
    objects_end = (
        ocel.relations.groupby(ocel.object_id_column)[ocel.event_timestamp]
        .last()
        .to_dict()
    )
    objects_ot0 = ocel.objects[
        [ocel.object_id_column, ocel.object_type_column]
    ].to_dict("records")
    objects_ot0 = [
        (x[ocel.object_id_column], x[ocel.object_type_column])
        for x in objects_ot0
    ]
    objects_ot1 = {}

    for el in objects_ot0:
        if not el[1] in objects_ot1:
            objects_ot1[el[1]] = []
        objects_ot1[el[1]].append(el[0])

    overall_objects = {}
    keys = sorted(list(objects_ot1))
    for ot in keys:
        objects = objects_ot1[ot]
        objects.sort(key=lambda x: (objects_start[x], objects_end[x], x))
        if ot in exclude_object_types:
            # skip renaming for this object type
            continue
        mapped = {objects[i]: ot + "_" + str(i + 1) for i in range(len(objects))}
        overall_objects.update(mapped)

    ocel = deepcopy(ocel)
    # map and keep original IDs for excluded object types
    oid_col = ocel.object_id_column
    ocel.objects[oid_col] = ocel.objects[oid_col].map(overall_objects).fillna(
        ocel.objects[oid_col]
    )
    ocel.relations[oid_col] = ocel.relations[oid_col].map(overall_objects).fillna(
        ocel.relations[oid_col]
    )
    if ocel.o2o is not None:
        ocel.o2o[oid_col] = ocel.o2o[oid_col].map(overall_objects).fillna(
            ocel.o2o[oid_col]
        )
        ocel.o2o[oid_col + "_2"] = ocel.o2o[oid_col + "_2"].map(overall_objects).fillna(
            ocel.o2o[oid_col + "_2"]
        )
    if ocel.object_changes is not None:
        ocel.object_changes[oid_col] = ocel.object_changes[oid_col].map(
            overall_objects
        ).fillna(ocel.object_changes[oid_col])

    return ocel
