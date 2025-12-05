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

REACH (Required Activities) Heuristic Alignment Implementation
Based on: Casas-Ramos, J., Mucientes, M., & Lama, M. (2024). REACH: Researching Efficient Alignment-based Conformance Checking.
'''
import heapq
import sys
import time
from copy import copy
from enum import Enum
from typing import Optional, Dict, Any, Union, Set, List

from pm4py.objects.log import obj as log_implementation
from pm4py.objects.log.obj import Trace
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import align_utils as utils
from pm4py.objects.petri_net.utils.align_utils import STD_SYNC_COST, STD_MODEL_LOG_MOVE_COST
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
    ENABLE_OPTIMIZATIONS = "enable_optimizations"  # Toggle for REACH optimizations (SRModel, SRLog, Greedy)


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

    if trace_cost_function is None:
        trace_cost_function = list(
            map(lambda e: utils.STD_MODEL_LOG_MOVE_COST, trace)
        )
        parameters[Parameters.PARAM_TRACE_COST_FUNCTION] = trace_cost_function

    if model_cost_function is None:
        # reset variables value
        model_cost_function = dict()
        sync_cost_function = dict()
        for t in petri_net.transitions:
            if t.label is not None:
                model_cost_function[t] = utils.STD_MODEL_LOG_MOVE_COST
                sync_cost_function[t] = utils.STD_SYNC_COST
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

    # Extract clean list of activity labels for the heuristic
    trace_labels = [e[activity_key] for e in trace]

    # Pre-computation for SRLog (Algorithm 8)
    # This must be done on the Model (petri_net), not the Sync Net.
    enable_optimizations = exec_utils.get_param_value(Parameters.ENABLE_OPTIMIZATIONS, parameters, True)
    if enable_optimizations is None:
        enable_optimizations = True
    alive_activities_map = None

    if enable_optimizations is True or None:
        # We need the model decorated to traverse it
        decorate_transitions_prepostset(petri_net)
        alive_activities_map = get_alive_activities(petri_net, initial_marking)

    alignment = apply_trace_net(
        petri_net,
        initial_marking,
        final_marking,
        trace_net,
        trace_im,
        trace_fm,
        trace_labels,
        parameters,
        alive_activities_map=alive_activities_map
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
    trace_labels,
    parameters=None,
    alive_activities_map=None
):
    """
        Performs the basic alignment search, given a trace net and a net.

        Parameters
        ----------
        trace_labels
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
        sync_costs = sync_cost_function
        log_costs = trace_cost_function

    max_align_time_trace = exec_utils.get_param_value(
        Parameters.PARAM_MAX_ALIGN_TIME_TRACE, parameters, sys.maxsize
    )

    return apply_sync_prod(
        sync_prod,
        sync_initial_marking,
        sync_final_marking,
        cost_function,
        utils.SKIP,
        trace_labels,
        ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
        max_align_time_trace=max_align_time_trace,
        log_costs=log_costs,
        parameters=parameters,
        alive_activities_map=alive_activities_map
    )


def apply_sync_prod(
        sync_prod,
        initial_marking,
        final_marking,
        cost_function,
        skip,
        trace_labels,
        ret_tuple_as_trans_desc=False,
        max_align_time_trace=sys.maxsize,
        log_costs=None,
        parameters=None,
        alive_activities_map=None
):
    return __search(
        sync_prod,
        initial_marking,
        final_marking,
        cost_function,
        skip,
        ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
        max_align_time_trace=max_align_time_trace,
        trace_labels=trace_labels,
        log_costs=log_costs,
        parameters=parameters,
        alive_activities_map=alive_activities_map
    )


# =============================================================================
# REACH Heuristic Logic (Algorithm 3 & 8 in the paper)
# =============================================================================

def get_required_transitions(marking: Marking) -> Set[PetriNet.Transition]:
    """
    Algorithm 3: REQUIRED_TRANSITIONS
    Traverses the process model structurally from the current tokens to find
    transitions that MUST be executed (no alternative paths).
    """
    required_transitions = set()

    # Queue for places to visit (BFS/Traversal)
    places_to_visit = list(marking.keys())
    visited_places = set()

    while places_to_visit:
        place = places_to_visit.pop(0)

        if place in visited_places:
            continue
        visited_places.add(place)

        # Get outgoing transitions
        transitions = [arc.target for arc in place.out_arcs]

        # Algorithm 3 Line 21: if |transitions| != 1 then continue
        # If there is a choice (OR-split) or dead-end, we cannot structurally guarantee a transition
        if len(transitions) != 1:
            continue

        trans = transitions[0]

        # Algorithm 3 Line 23: if not IsSilent(trans)
        if trans.label is not None:
            required_transitions.add(trans)

        # Algorithm 3 Line 25: AddAll(places, transitions[0].out_places)
        for arc in trans.out_arcs:
            p_out = arc.target
            if p_out not in visited_places:
                places_to_visit.append(p_out)

    return required_transitions


def get_heuristic(marking: Marking, remaining_trace_labels: List[str], epsilon=STD_SYNC_COST) -> float:
    """
    Algorithm 3: HEURISTIC
    Calculates the MMR (Model Move Required) heuristic.
    """
    # 1. Get structurally required transitions
    required_trs = get_required_transitions(marking)

    # 2. Extract unique labels
    required_labels = {t.label for t in required_trs if t.label is not None}

    # 3. Unique activities in remaining trace
    remaining_labels = set(remaining_trace_labels)

    # 4. Missing: Required by model but NOT in remaining trace (Algorithm 3 Line 5)
    # These MUST be executed as Model Moves.
    missing = required_labels - remaining_labels

    # 5. MinCostMoves: Required AND in remaining trace (Algorithm 3 Line 6)
    # We optimistically assume these will be Synchronous Moves.
    sync_candidates = required_labels.intersection(remaining_labels)

    # 6. Calculate Cost (Algorithm 3 Line 7)
    # Cost = |missing| * Cost(Model Move) + |sync_candidates| * Cost(Sync Move)
    # Admissibility: We use the cheapest possible cost for sync (0) and standard for model.
    h_val = len(missing) * STD_MODEL_LOG_MOVE_COST + len(sync_candidates) * epsilon

    return h_val


def get_alive_activities(net: PetriNet, initial_marking: Marking) -> Dict[Marking, Set[str]]:
    """
    Algorithm 8: Initialization for LESSSTATESMODEL (SRLog).
    Pre-computes the 'alive activities' for every reachable marking in the model (ignoring the log).
    Returns a map: Marking -> Set[Activity Labels]
    """
    # Phase 1: Discovery (BFS to find all reachable markings)
    rg_nodes = {initial_marking}
    rg_edges = []  # List of (source_m, target_m, label)

    processing_queue = [initial_marking]

    while processing_queue:
        m = processing_queue.pop(0)
        enabled_transitions = utils.semantics.enabled_transitions(net, m)

        for t in enabled_transitions:
            new_m = utils.semantics.execute(t, net, m)
            rg_edges.append((m, new_m, t.label))

            if new_m not in rg_nodes:
                rg_nodes.add(new_m)
                processing_queue.append(new_m)

    # Phase 2: Compute Alive Activities (Backwards propagation)
    alive_map = {m: set() for m in rg_nodes}

    # Initialize with directly reachable labels
    for src, tgt, label in rg_edges:
        if label is not None:
            alive_map[src].add(label)

    # Propagate: If I can reach Tgt, I inherit Tgt's alive activities
    changed = True
    while changed:
        changed = False
        for src, tgt, label in rg_edges:
            new_alive = alive_map[src].union(alive_map[tgt])
            if len(new_alive) > len(alive_map[src]):
                alive_map[src] = new_alive
                changed = True

    return alive_map


# =============================================================================
# Optimization Checks (Algorithm 6 & 7)
# =============================================================================

def check_sr_model(net, marking, remaining_trace_labels):
    """
    Algorithm 6: LESSSTATESLOG (SRModel)
    Returns True if we should SKIP Log Moves (i.e., Force Model Moves).
    Condition: Model has enabled transitions, BUT none of them match any future trace activity.
    """
    enabled = utils.semantics.enabled_transitions(net, marking)
    if not enabled:
        return False  # Deadlock in model, can't force model move

    enabled_labels = {t.label for t in enabled if t.label is not None}
    if not enabled_labels:
        # If only silent transitions are enabled, we rely on standard A* behavior
        # because SRModel compares with trace labels.
        pass

    # Intersection with ANY activity in the remaining trace
    remaining_set = set(remaining_trace_labels)

    if not enabled_labels.isdisjoint(remaining_set):
        return False  # There is a match, so we shouldn't force model move

    return True  # No match possible in future trace, force Model Move now


def check_sr_log(marking, next_trace_label, alive_map):
    """
    Algorithm 7: LESSSTATESMODEL (SRLog)
    Returns True if we should SKIP Model Moves (i.e., Force Log Move).
    Condition: The next trace activity is NOT alive in the model from current state.
    """
    if alive_map is None or marking not in alive_map:
        return False

    alive_activities = alive_map[marking]

    if next_trace_label not in alive_activities:
        # The next activity in trace can NEVER be executed from here.
        return True

    return False


# =============================================================================
# Greedy Upper Bound (Algorithm 9)
# =============================================================================

def run_greedy_search(sync_net, ini, fin, cost_function, skip, trace_labels):
    """
    Algorithm 9: Greedy alignment to find an upper bound cost.
    """
    curr_m = ini
    curr_cost = 0.0
    trace_idx = 0
    trace_len = len(trace_labels)

    while True:
        if curr_m == fin:
            return curr_cost

        enabled_trans = utils.semantics.enabled_transitions(sync_net, curr_m)
        candidates = []

        for t in enabled_trans:
            move_cost = cost_function[t]
            is_log = utils.__is_log_move(t, skip)
            is_model = utils.__is_model_move(t, skip)

            new_idx = trace_idx
            if is_log:
                new_idx += 1
            elif not is_model:
                new_idx += 1  # Sync (assuming non-silent)

            # Simplified heuristic for greedy: Remaining Trace Length * Standard Cost
            # This pushes the greedy search towards finishing the trace
            rem_len = trace_len - new_idx if new_idx < trace_len else 0
            h = rem_len * STD_MODEL_LOG_MOVE_COST

            score = (curr_cost + move_cost) + h
            new_m = utils.semantics.weak_execute(t, sync_net, curr_m)
            candidates.append((score, new_m, move_cost, new_idx))

        if not candidates:
            return float('inf')

        candidates.sort(key=lambda x: x[0])
        best = candidates[0]

        curr_m = best[1]
        curr_cost += best[2]
        trace_idx = best[3]

        if curr_cost > 1000000:  # Safety break
            return float('inf')


def __search(
    sync_net,
    ini,
    fin,
    cost_function,
    skip,
    ret_tuple_as_trans_desc=False,
    max_align_time_trace=sys.maxsize,
    trace_labels=None,
    log_costs=None,
    parameters=None,
    alive_activities_map=None
):
    start_time = time.time()

    decorate_transitions_prepostset(sync_net)
    decorate_places_preset_trans(sync_net)

    if trace_labels is None:
        trace_labels = []

    enable_optimizations = exec_utils.get_param_value(Parameters.ENABLE_OPTIMIZATIONS, parameters, True)
    if enable_optimizations is None:
        enable_optimizations = True

    # Build mapping to identify Model Marking from Sync Marking
    place_to_trace_index = utils.__build_place_to_trace_index(
        sync_net,
        trace_len=len(trace_labels),
        infer_source_sink=True,
    )

    # Helper to extract Model Marking component from Sync Marking
    def split_marking(m):
        model_m = Marking()
        trace_idx = 0
        found_idx = False
        for p, count in m.items():
            if p in place_to_trace_index:
                if not found_idx:
                    trace_idx = place_to_trace_index[p]
                    found_idx = True
                else:
                    trace_idx = max(trace_idx, place_to_trace_index[p])
            else:
                model_m[p] = count
        return model_m, trace_idx

    # Greedy Upper Bound (Algorithm 9)
    upper_bound_cost = float('inf')
    if enable_optimizations:
        try:
            upper_bound_cost = run_greedy_search(sync_net, ini, fin, cost_function, skip, trace_labels)
        except:
            upper_bound_cost = float('inf')

    # Initial State
    ini_model_m, ini_idx = split_marking(ini)
    h0 = get_heuristic(ini_model_m, trace_labels[ini_idx:])

    ini_state = utils.SearchTuple(0 + h0, 0, h0, ini, None, None, None, True)
    open_set = [ini_state]
    heapq.heapify(open_set)

    visited = 0
    queued = 0
    traversed = 0

    closed = set()
    trans_empty_preset = set(
        t for t in sync_net.transitions if len(t.in_arcs) == 0
    )

    while open_set:
        if (time.time() - start_time) > max_align_time_trace:
            return None

        curr = heapq.heappop(open_set)
        current_marking = curr.m

        # Optimization: Upper Bound Pruning
        if enable_optimizations and (curr.g + curr.h > upper_bound_cost):
            continue

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

        # Context for Optimizations
        curr_model_m, curr_trace_idx = split_marking(current_marking)
        rem_trace = trace_labels[curr_trace_idx:]

        # Optimization: SRModel (Alg 6) - Force Model Moves?
        force_model_move = False
        if enable_optimizations:
            # Check on SyncNet but using model logic (enabled transitions)
            # We map Sync Marking -> Enabled Transitions -> Labels
            force_model_move = check_sr_model(sync_net, current_marking, rem_trace)

        # Optimization: SRLog (Alg 7) - Force Log Moves?
        force_log_move = False
        if enable_optimizations and alive_activities_map and rem_trace:
            # Check if next trace activity is alive in current model state
            next_act = rem_trace[0]
            force_log_move = check_sr_log(curr_model_m, next_act, alive_activities_map)

        enabled_trans = copy(trans_empty_preset)
        for p in current_marking:
            for t in p.ass_trans:
                if t.sub_marking <= current_marking:
                    enabled_trans.add(t)

        for t in enabled_trans:
            # Determine move type
            is_log = utils.__is_log_move(t, skip)
            is_model = utils.__is_model_move(t, skip)
            is_sync = not is_log and not is_model

            # Apply SRModel: If forced to move model, skip Log Moves
            if force_model_move and is_log:
                continue

            # Apply SRLog: If forced to move log, skip Model Moves (and Sync, as Sync implies model move valid)
            # SRLog says: Next trace activity is DEAD. So we MUST skip it (Log Move).
            # Model cannot possibly match it now or later.
            if force_log_move:
                # We only allow Log Moves that consume the 'dead' activity?
                # Actually, standard A* generates all neighbors.
                # If we force log move, we should SKIP Model and Sync moves.
                if is_model or is_sync:
                    continue

            cost = cost_function[t]
            traversed += 1
            new_marking = utils.add_markings(
                current_marking, t.add_marking
            )

            if new_marking in closed:
                continue

            g = curr.g + cost

            # Calculate Heuristic
            new_model_m, new_trace_idx = split_marking(new_marking)
            new_rem_trace = trace_labels[new_trace_idx:]

            h = get_heuristic(new_model_m, new_rem_trace)
            new_f = g + h

            # Early Pruning
            if enable_optimizations and (new_f > upper_bound_cost):
                continue

            queued += 1
            tp = utils.SearchTuple(new_f, g, h, new_marking, curr, t, None, True)
            heapq.heappush(open_set, tp)

    return None
