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
from pm4py.objects.ocel.obj import OCEL
from enum import Enum
from pm4py.util import exec_utils
from typing import Optional, Dict, Any, Tuple


class Parameters(Enum):
    INCLUDE_TIMESTAMPS = "include_timestamps"


def apply(
    ocel: OCEL, parameters: Optional[Dict[Any, Any]] = None
) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    """
    Provides a tuple-based description of the provided object-centric event log

    Parameters
    --------------
    ocel
        Object-centric event log
    parameters
        Possible parameters of the algorithm, including:
        - Parameters.INCLUDE_TIMESTAMPS => include the timestamps (or not) in the representation

    Returns
    -------------
    Tuple[Tuple[Any, ...], Tuple[Any, ...]]
        Tuple-based representation of events and objects
    """
    if parameters is None:
        parameters = {}

    include_timestamps = exec_utils.get_param_value(
        Parameters.INCLUDE_TIMESTAMPS, parameters, True
    )

    object_ots = ocel.objects[
        [ocel.object_id_column, ocel.object_type_column]
    ].to_dict("records")
    object_ots = {
        x[ocel.object_id_column]: x[ocel.object_type_column]
        for x in object_ots
    }
    events = ocel.events.sort_values(
        [ocel.event_timestamp, ocel.event_activity, ocel.event_id_column]
    ).to_dict("records")
    objects = ocel.objects.sort_values(ocel.object_id_column).to_dict(
        "records"
    )
    relations = ocel.relations.sort_values(
        [
            ocel.event_timestamp,
            ocel.event_activity,
            ocel.object_id_column,
            ocel.event_id_column,
        ]
    )
    tdf = relations.groupby(ocel.object_id_column)[ocel.event_timestamp]
    objects_start = tdf.first().to_dict()
    objects_end = tdf.last().to_dict()
    objects_lifecycle = {
        x: objects_end[x].timestamp() - objects_start[x].timestamp()
        for x in objects_start
    }

    relations = (
        relations.groupby(ocel.event_id_column)[ocel.object_id_column]
        .agg(list)
        .to_dict()
    )

    event_descriptions = []

    for ev in events:
        related_objects = tuple(relations[ev[ocel.event_id_column]])
        event_tuple = (ev[ocel.event_activity],) + related_objects
        if include_timestamps:
            event_tuple = event_tuple + (ev[ocel.event_timestamp],)
        event_descriptions.append(event_tuple)

    object_descriptions = []

    for obj in objects:
        obj_id = obj[ocel.object_id_column]
        object_tuple = (obj_id, object_ots[obj_id])
        if include_timestamps:
            object_tuple = (
                object_tuple
                + (
                    objects_start[obj_id],
                    objects_end[obj_id],
                    objects_lifecycle[obj_id],
                )
            )
        object_descriptions.append(object_tuple)

    return tuple(event_descriptions), tuple(object_descriptions)
