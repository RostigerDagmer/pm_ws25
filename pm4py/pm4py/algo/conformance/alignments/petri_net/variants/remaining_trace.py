'''
    PM4Py – A Process Mining Library for Python
Copyright (C) 2024 Process Intelligence Solutions UG (haftungsbeschränkt)

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

Remaining Log Moves Heuristic Variant
'''
import heapq
import sys
import time
from copy import copy
from enum import Enum
from typing import Optional, Dict, Any, Union

from pm4py.objects.log import obj as log_implementation
from pm4py.objects.log.obj import Trace
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import align_utils as utils
from pm4py.objects.petri_net.utils.incidence_matrix import construct as inc_mat_construct
from pm4py.objects.petri_net.utils.petri_utils import (
    construct_trace_net_cost_aware,
    decorate_places_preset_trans,
    decorate_transitions_prepostset,
)
from pm4py.objects.petri_net.utils.synchronous_product import (
    construct_cost_aware,
    construct,
)
from pm4py.util import exec_utils
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.util.xes_constants import DEFAULT_NAME_KEY

# We define a Unit Cost for synchronous moves
# Standard PM4Py uses 0, but using 1 allows the 'Remaining Trace' heuristic
# to act as a valid lower bound (h = remaining_steps * 1).
DEFAULT_UNIT_SYNC_COST = 1


class Parameters(Enum):
    PARAM_TRACE_COST_FUNCTION = "trace_cost_function"
    PARAM_MODEL_COST_FUNCTION = "model_cost_function"
    PARAM_SYNC_COST_FUNCTION = "sync_cost_function"
    PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE = "ret_tuple_as_trans_desc"
    PARAM_TRACE_NET_COSTS = "trace_net_costs"
    TRACE_NET_CONSTR_FUNCTION = "trace_net_constr_function"
    TRACE_NET_COST_AWARE_CONSTR_FUNCTION = "trace_net_cost_aware_constr_function"
    PARAM_MAX_ALIGN_TIME_TRACE = "max_align_time_trace"
    ACTIVITY_KEY = PARAMETER_CONSTANT_ACTIVITY_KEY

PARAM_TRACE_COST_FUNCTION = Parameters.PARAM_TRACE_COST_FUNCTION.value
PARAM_MODEL_COST_FUNCTION = Parameters.PARAM_MODEL_COST_FUNCTION.value
PARAM_SYNC_COST_FUNCTION = Parameters.PARAM_SYNC_COST_FUNCTION.value

def get_best_worst_cost(
    petri_net, initial_marking, final_marking, parameters=None
):
    """
    Gets the best worst cost of an alignment

    Parameters
    -----------
    petri_net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking

    Returns
    -----------
    best_worst_cost
        Best worst cost of alignment
    """
    if parameters is None:
        parameters = {}
    trace = log_implementation.Trace()

    best_worst = apply(
        trace, petri_net, initial_marking, final_marking, parameters=parameters
    )

    if best_worst is not None:
        return best_worst["cost"]

    return None





def apply(trace: Trace, petri_net: PetriNet, initial_marking: Marking, final_marking: Marking,
          parameters: Optional[Dict[Union[str, Parameters], Any]] = None):
    if parameters is None: parameters = {}

    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, DEFAULT_NAME_KEY)

    # Check if costs are provided
    trace_cost_function = exec_utils.get_param_value(Parameters.PARAM_TRACE_COST_FUNCTION, parameters, None)
    model_cost_function = exec_utils.get_param_value(Parameters.PARAM_MODEL_COST_FUNCTION, parameters, None)
    sync_cost_function = exec_utils.get_param_value(Parameters.PARAM_SYNC_COST_FUNCTION, parameters, None)
    # Construct Trace Net
    trace_net_constr_function = exec_utils.get_param_value(Parameters.TRACE_NET_CONSTR_FUNCTION, parameters, None)
    trace_net_cost_aware_constr_function = exec_utils.get_param_value(Parameters.TRACE_NET_COST_AWARE_CONSTR_FUNCTION,
                                                                      parameters, construct_trace_net_cost_aware)
    # 1. Configure Log Move Costs (Default: 10000)
    if trace_cost_function is None:
        trace_cost_function = list(
            map(lambda e: utils.STD_MODEL_LOG_MOVE_COST, trace)
        )
        parameters[Parameters.PARAM_TRACE_COST_FUNCTION] = trace_cost_function

        # 2. Configure Model & Sync Costs
        if model_cost_function is None:
            model_cost_function = dict()
            sync_cost_function = dict()
            for t in petri_net.transitions:
                if t.label is not None:
                    model_cost_function[t] = utils.STD_MODEL_LOG_MOVE_COST
                    # We enforce a strictly positive cost (1) for synchronous moves
                    # This ensures min(Sync, Log) > 0, making the heuristic admissible and non-zero
                    sync_cost_function[t] = DEFAULT_UNIT_SYNC_COST
                else:
                    model_cost_function[t] = utils.STD_TAU_COST
            parameters[Parameters.PARAM_MODEL_COST_FUNCTION] = model_cost_function
            parameters[Parameters.PARAM_SYNC_COST_FUNCTION] = sync_cost_function


    if trace_net_constr_function is not None:
        trace_net, trace_im, trace_fm = trace_net_constr_function(trace, activity_key=activity_key)
    else:
        (
            trace_net,
            trace_im,
            trace_fm,
            parameters[Parameters.PARAM_TRACE_NET_COSTS],
        ) = trace_net_cost_aware_constr_function(
            trace, trace_cost_function, activity_key=activity_key
        )

    alignment = apply_trace_net(
        petri_net,
        initial_marking,
        final_marking,
        trace_net,
        trace_im,
        trace_fm,
        parameters,
    )
    return alignment


def apply_from_variant(
    variant, petri_net, initial_marking, final_marking, parameters=None
):
    """
    Apply the alignments from the specification of a single variant

    Parameters
    -------------
    variant
        Variant (as string delimited by the "variant_delimiter" parameter)
    petri_net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    parameters
        Parameters of the algorithm (same as 'apply' method, plus 'variant_delimiter' that is , by default)

    Returns
    ------------
    dictionary: `dict` with keys **alignment**, **cost**, **visited_states**, **queued_states** and **traversed_arcs**
    """
    if parameters is None:
        parameters = {}
    trace = variants_util.variant_to_trace(variant, parameters=parameters)

    return apply(
        trace, petri_net, initial_marking, final_marking, parameters=parameters
    )


def apply_from_variants_dictionary(
    var_dictio, petri_net, initial_marking, final_marking, parameters=None
):
    if parameters is None:
        parameters = {}
    dictio_alignments = {}
    for variant in var_dictio:
        dictio_alignments[variant] = apply_from_variant(
            variant,
            petri_net,
            initial_marking,
            final_marking,
            parameters=parameters,
        )
    return dictio_alignments


def apply_from_variants_list(
    var_list, petri_net, initial_marking, final_marking, parameters=None
):
    """
    Apply the alignments from the specification of a list of variants in the log

    Parameters
    -------------
    var_list
        List of variants (for each item, the first entry is the variant itself, the second entry may be the number of cases)
    petri_net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    parameters
        Parameters of the algorithm (same as 'apply' method, plus 'variant_delimiter' that is , by default)

    Returns
    --------------
    dictio_alignments
        Dictionary that assigns to each variant its alignment
    """
    if parameters is None:
        parameters = {}
    start_time = time.time()
    max_align_time = exec_utils.get_param_value(
        Parameters.PARAM_MAX_ALIGN_TIME, parameters, sys.maxsize
    )
    max_align_time_trace = exec_utils.get_param_value(
        Parameters.PARAM_MAX_ALIGN_TIME_TRACE, parameters, sys.maxsize
    )
    dictio_alignments = {}
    for varitem in var_list:
        this_max_align_time = min(
            max_align_time_trace,
            (max_align_time - (time.time() - start_time)) * 0.5,
        )
        variant = varitem[0]
        parameters[Parameters.PARAM_MAX_ALIGN_TIME_TRACE] = this_max_align_time
        dictio_alignments[variant] = apply_from_variant(
            variant,
            petri_net,
            initial_marking,
            final_marking,
            parameters=parameters,
        )
    return dictio_alignments


def apply_from_variants_list_petri_string(
    var_list, petri_net_string, parameters=None
):
    if parameters is None:
        parameters = {}

    from pm4py.objects.petri_net.importer.variants import (
        pnml as petri_importer,
    )

    petri_net, initial_marking, final_marking = (
        petri_importer.import_petri_from_string(petri_net_string)
    )

    res = apply_from_variants_list(
        var_list,
        petri_net,
        initial_marking,
        final_marking,
        parameters=parameters,
    )
    return res


def apply_from_variants_list_petri_string_mprocessing(
    mp_output, var_list, petri_net_string, parameters=None
):
    if parameters is None:
        parameters = {}

    res = apply_from_variants_list_petri_string(
        var_list, petri_net_string, parameters=parameters
    )
    mp_output.put(res)


def apply_trace_net(
    petri_net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    trace_net,
    trace_im,
    trace_fm,
    parameters=None,
):
    """
        Performs the basic alignment search, given a trace net and a net.

        Parameters
        ----------
        trace: :class:`list` input trace, assumed to be a list of events (i.e. the code will use the activity key
        to get the attributes)
        petri_net: :class:`pm4py.objects.petri.net.PetriNet` the Petri net to use in the alignment
        initial_marking: :class:`pm4py.objects.petri.net.Marking` initial marking in the Petri net
        final_marking: :class:`pm4py.objects.petri.net.Marking` final marking in the Petri net
        parameters: :class:`dict` (optional) dictionary containing one of the following:
            Parameters.PARAM_TRACE_COST_FUNCTION: :class:`list` (parameter) mapping of each index of the trace to a positive cost value
            Parameters.PARAM_MODEL_COST_FUNCTION: :class:`dict` (parameter) mapping of each transition in the model to corresponding
            model cost
            Parameters.PARAM_SYNC_COST_FUNCTION: :class:`dict` (parameter) mapping of each transition in the model to corresponding
            synchronous costs
            Parameters.ACTIVITY_KEY: :class:`str` (parameter) key to use to identify the activity described by the events
            Parameters.PARAM_TRACE_NET_COSTS: :class:`dict` (parameter) mapping between transitions and costs

        Returns
        -------
        dictionary: `dict` with keys **alignment**, **cost**, **visited_states**, **queued_states** and **traversed_arcs**
        """

    if parameters is None:
        parameters = {}

    ret_tuple_as_trans_desc = exec_utils.get_param_value(
        Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE, parameters, False
    )

    trace_cost_function = exec_utils.get_param_value(
        Parameters.PARAM_TRACE_COST_FUNCTION, parameters, None
    )
    model_cost_function = exec_utils.get_param_value(
        Parameters.PARAM_MODEL_COST_FUNCTION, parameters, None
    )
    sync_cost_function = exec_utils.get_param_value(
        Parameters.PARAM_SYNC_COST_FUNCTION, parameters, None
    )
    trace_net_costs = exec_utils.get_param_value(
        Parameters.PARAM_TRACE_NET_COSTS, parameters, None
    )

    # keep identical logic to dijkstra_no_heuristics
    if (
        trace_cost_function is None
        or model_cost_function is None
        or sync_cost_function is None
    ):
        sync_prod, sync_initial_marking, sync_final_marking = construct(
            trace_net,
            trace_im,
            trace_fm,
            petri_net,
            initial_marking,
            final_marking,
            utils.SKIP,
        )
        cost_function = utils.construct_standard_cost_function(
            sync_prod, utils.SKIP
        )
        sync_costs = None
        log_costs = None
    else:
        revised_sync = dict()
        for t_trace in trace_net.transitions:
            for t_model in petri_net.transitions:
                if t_trace.label == t_model.label:
                    revised_sync[(t_trace, t_model)] = sync_cost_function[
                        t_model
                    ]

        sync_prod, sync_initial_marking, sync_final_marking, cost_function = (
            construct_cost_aware(
                trace_net,
                trace_im,
                trace_fm,
                petri_net,
                initial_marking,
                final_marking,
                utils.SKIP,
                trace_net_costs,
                model_cost_function,
                revised_sync,
            )
        )

        # store per\-transition sync/log costs for the heuristic
        sync_costs = sync_cost_function
        log_costs = trace_cost_function

    max_align_time_trace = exec_utils.get_param_value(
        Parameters.PARAM_MAX_ALIGN_TIME_TRACE, parameters, sys.maxsize
    )

    # trace length for remaining\-trace heuristic
    trace_len = len(trace_cost_function) if trace_cost_function is not None else 0

    return apply_sync_prod(
        sync_prod,
        sync_initial_marking,
        sync_final_marking,
        cost_function,
        utils.SKIP,
        ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
        max_align_time_trace=max_align_time_trace,
        trace_len=trace_len,
        log_costs=log_costs,
    )


def apply_sync_prod(
    sync_prod,
    initial_marking,
    final_marking,
    cost_function,
    skip,
    ret_tuple_as_trans_desc=False,
    max_align_time_trace=sys.maxsize,
    trace_len=0,
    log_costs=None,
):
    return __search(
        sync_prod,
        initial_marking,
        final_marking,
        cost_function,
        skip,
        ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
        max_align_time_trace=max_align_time_trace,
        trace_len=trace_len,
        log_costs=log_costs,
    )


def _compute_place_remaining_dist(
        sync_net,
        trace_len: int,
        log_costs: Optional[list] = None,
        sync_costs: Optional[dict] = None,
) -> Dict[Any, float]:
    """
    Precomputes the 'Trace Length' heuristic.

    Logic:
    1. Check provided Log Costs (Default 10000).
    2. Check provided Sync Costs (Default 0).
    3. Take min(Log, Sync) as the weight.

    If Sync Cost is 0 (Standard), weight is 0 -> Admissible (Dijkstra).
    If Sync Cost is 1 (Unit), weight is 1 -> Admissible (Trace Length).
    """

    # 1. Determine Minimum Log Cost
    # If not provided, assume standard high cost (10000)
    min_log = 10000
    if log_costs is not None and len(log_costs) > 0:
        min_log = min(log_costs)

    # 2. Determine Minimum Sync Cost
    # If not provided, we assume 0 (Standard PM4Py behavior) to be safe/admissible.
    min_sync = 0
    if sync_costs is not None and len(sync_costs) > 0:
        # sync_costs is a dict: {transition: cost}
        # We find the cheapest sync move available in the entire model.
        min_sync = min(sync_costs.values())

    # 3. Calculate Admissible Weight
    # We use the minimum of both
    # If min_sync is 0, weight becomes 0
    heuristic_weight = min(min_log, min_sync)

    place_to_remaining_dist: Dict[Any, float] = {}

    place_to_trace_index = utils.__build_place_to_trace_index(
        sync_net,
        trace_len=trace_len,
        infer_source_sink=True,
    )

    # for each place, compute remaining distance (heuristic)
    for place, idx in place_to_trace_index.items():
        # How many events are left from this point?
        remaining_events = max(0, trace_len - idx)

        # Heuristic = (Events Left) * (Cheapest Cost per Event)
        dist = remaining_events * heuristic_weight
        place_to_remaining_dist[place] = dist

    return place_to_remaining_dist

def _heuristic_from_marking(
    marking: Marking,
    place_to_remaining_dist: Dict[Any, float],
) -> float:
    """
    Compute the heuristic value for a given marking as the maximum
    remaining distance over all places in the marking.
    """
    h_vals = [
        place_to_remaining_dist[p]
        for p in marking
        if p in place_to_remaining_dist
    ]
    return max(h_vals) if h_vals else 0.0


def __search(
    sync_net,
    ini,
    fin,
    cost_function,
    skip,
    ret_tuple_as_trans_desc=False,
    max_align_time_trace=sys.maxsize,
    trace_len=0,
    log_costs=None,
):
    start_time = time.time()

    decorate_transitions_prepostset(sync_net)
    decorate_places_preset_trans(sync_net)

    place_to_remaining_dist = _compute_place_remaining_dist(
        sync_net=sync_net,
        trace_len=trace_len,
        log_costs=log_costs,
    )

    def get_heuristic(marking: Marking) -> float:
        return _heuristic_from_marking(marking, place_to_remaining_dist)

    # A* algorithm, identical to Dijkstra but with f = g + h
    closed = set()

    h0 = get_heuristic(ini)
    ini_state = utils.SearchTuple(0 + h0, 0, h0, ini, None, None, None, True)
    open_set = [ini_state]
    heapq.heapify(open_set)

    visited = 0
    queued = 0
    traversed = 0

    trans_empty_preset = set(
        t for t in sync_net.transitions if len(t.in_arcs) == 0
    )

    while open_set:
        if (time.time() - start_time) > max_align_time_trace:
            return None

        curr = heapq.heappop(open_set)
        current_marking = curr.m
        if current_marking in closed:
            continue

        if current_marking == fin:
            return utils.__reconstruct_alignment(
                curr,
                visited,
                queued,
                traversed,
                ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                lp_solved=0,
            )

        closed.add(current_marking)
        visited += 1

        enabled_trans = copy(trans_empty_preset)
        for p in current_marking:
            for t in p.ass_trans:
                if t.sub_marking <= current_marking:
                    enabled_trans.add(t)

        trans_to_visit_with_cost = [
            (t, cost_function[t])
            for t in enabled_trans
            if not (
                t is not None
                and utils.__is_log_move(t, skip)
                and utils.__is_model_move(t, skip)
            )
        ]

        for t, cost in trans_to_visit_with_cost:
            traversed += 1
            new_marking = utils.add_markings(
                current_marking, t.add_marking
            )

            if new_marking in closed:
                continue

            g = curr.g + cost
            h = get_heuristic(new_marking)
            new_f = g + h

            queued += 1
            tp = utils.SearchTuple(new_f, g, h, new_marking, curr, t, None, True)
            heapq.heappush(open_set, tp)

    return None