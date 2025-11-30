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
from collections import defaultdict, deque, Counter
from copy import copy
from enum import Enum
from typing import Optional, Dict, Any, Union, Set

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


def _precompute_required_model_labels(sync_net, final_marking) -> Dict[Any, Set[str]]:
    """Backward collection of visible model labels reachable from each place.

    For each place in the synchronous product net, compute a (conservative)
    set of visible model activity labels that may be required to reach the
    final marking from that place.
    """
    place_to_required: Dict[Any, Set[str]] = defaultdict(set)

    # For backward traversal, we need: place -> incoming transitions.
    incoming_transitions: Dict[Any, Set[Any]] = defaultdict(set)
    for t in sync_net.transitions:
        for arc in t.out_arcs:
            incoming_transitions[arc.target].add(t)

    queue = deque(final_marking.keys())
    visited_places: Set[Any] = set(final_marking.keys())

    while queue:
        place = queue.popleft()
        for t in incoming_transitions.get(place, ()):  # transitions producing 'place'
            # identify the model-side label
            if isinstance(t.label, tuple) and len(t.label) == 2:
                model_label = t.label[1]
            else:
                model_label = t.label

            # Add label if visible
            if model_label is not None:
                place_to_required[place].add(model_label)

            # Propagate required labels back to all preset places
            for arc in t.in_arcs:
                p_in = arc.source
                if place_to_required[place] - place_to_required[p_in]:
                    place_to_required[p_in] |= place_to_required[place]
                    if p_in not in visited_places:
                        visited_places.add(p_in)
                        queue.append(p_in)

    return place_to_required


class RequiredActivitiesMode(Enum):
    """Select which required-activities heuristic design to use."""

    SIMPLE = "simple"  # current minimal implementation
    LABEL_REACHABILITY = "label_reachability"  # depth-based
    BITSET = "bitset"  # pattern-based approximation
    STRUCTURAL_SHORTEST_PATH = "structural_shortest_path"  # place-level shortest path


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
    # which heuristic flavour to use inside this variant
    REQUIRED_ACTIVITIES_MODE = "required_activities_mode"

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

    trace_labels = [e[activity_key] for e in trace]

    alignment = apply_trace_net(
        petri_net,
        initial_marking,
        final_marking,
        trace_net,
        trace_im,
        trace_fm,
        trace_labels,
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
    trace_labels,
    parameters=None,
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
    )


# --- Heuristic strategy helpers ------------------------------------------------


def _heuristic_simple_required_activities(
    marking: Marking,
    trace_labels: list,
    trace_len: int,
    place_to_trace_index: Dict[Any, int],
    place_to_required_labels: Dict[Any, Set[str]],
    heuristic_weight: float,
) -> float:
    """Current minimal required-activities heuristic.

    This is the original simple design: compare the set of model-required labels
    for the current marking with the multiset of remaining trace labels and
    derive a lower bound on non-sync moves.
    """
    if trace_len == 0 or not trace_labels:
        return 0.0

    # 1) Infer current trace index from the marking
    idx_candidates = [
        place_to_trace_index[p]
        for p in marking
        if p in place_to_trace_index
    ]
    if idx_candidates:
        trace_idx = max(idx_candidates)
    else:
        # Safe fallback: assume we are at start (worst case: many events remain)
        trace_idx = 0

    if trace_idx >= trace_len:
        return 0.0

    # 2) Remaining trace labels
    remaining_trace = trace_labels[trace_idx:trace_len]
    if not remaining_trace:
        return 0.0

    remaining_counts = Counter(remaining_trace)
    remaining_label_set = set(remaining_counts.keys())

    # 3) Aggregate required model-side labels from all places in the marking
    required_labels: Set[str] = set()
    for p in marking:
        labels = place_to_required_labels.get(p)
        if labels:
            required_labels |= labels

    if not required_labels and not remaining_label_set:
        return 0.0

    # 4) Lower bound on model-only moves: any required label not present
    # in the remaining trace
    lb_model = sum(1 for a in required_labels if a not in remaining_label_set)

    # 5) Lower bound on log-only moves: any remaining trace event whose
    # label is not required by the model
    lb_log = 0
    for lbl, cnt in remaining_counts.items():
        if lbl not in required_labels:
            lb_log += cnt

    # 6) Use the maximum of the per-side lower bounds and scale by weight
    lb_moves = max(lb_model, lb_log)
    return float(lb_moves)


def _precompute_label_depths(sync_net, final_marking) -> Dict[Any, Dict[str, int]]:
    """Backward label-reachability with depth per place/label."""
    incoming_transitions: Dict[Any, Set[Any]] = defaultdict(set)
    for t in sync_net.transitions:
        for arc in t.out_arcs:
            incoming_transitions[arc.target].add(t)

    place_label_depth: Dict[Any, Dict[str, int]] = defaultdict(dict)
    queue = deque(final_marking.keys())
    visited: Set[Any] = set(final_marking.keys())

    # Depth here is the number of visible model steps remaining until final
    while queue:
        place = queue.popleft()
        for t in incoming_transitions.get(place, ()):  # transitions producing 'place'
            # identify model-side label
            if isinstance(t.label, tuple) and len(t.label) == 2:
                model_label = t.label[1]
            else:
                model_label = t.label

            # depth contributed by this transition
            add_cost = 1 if model_label is not None else 0

            # For successors, take their label depths and add cost
            succ_depths = place_label_depth.get(place, {})

            for arc in t.in_arcs:
                p_in = arc.source
                by_label = place_label_depth[p_in]

                # propagate all successor labels
                for lbl, d in succ_depths.items():
                    nd = d + add_cost
                    if lbl not in by_label or nd < by_label[lbl]:
                        by_label[lbl] = nd

                # mark this label itself reachable in one step if visible
                if model_label is not None:
                    if model_label not in by_label or add_cost < by_label[model_label]:
                        by_label[model_label] = add_cost

                if p_in not in visited:
                    visited.add(p_in)
                    queue.append(p_in)

    return place_label_depth


def _precompute_bitset_structures(
    sync_net,
    final_marking,
    trace_labels: list,
) -> (Dict[Any, int], Dict[int, int]):
    """Precompute bitsets for required labels per place and per trace suffix."""
    # 1) Build a label universe from model and trace
    label_to_bit: Dict[str, int] = {}

    def _ensure_bit(lbl: Optional[str]) -> None:
        if lbl is None:
            return
        if lbl not in label_to_bit:
            label_to_bit[lbl] = len(label_to_bit)

    # collect from model (sync net transitions)
    for t in sync_net.transitions:
        if isinstance(t.label, tuple) and len(t.label) == 2:
            _ensure_bit(t.label[1])
        else:
            _ensure_bit(t.label)

    # collect from trace
    for lbl in trace_labels:
        _ensure_bit(lbl)

    # 2) Precompute required label sets per place (like _precompute_required_model_labels)
    place_to_required: Dict[Any, Set[str]] = _precompute_required_model_labels(sync_net, final_marking)

    place_req_mask: Dict[Any, int] = {}
    for p, labels in place_to_required.items():
        mask = 0
        for lbl in labels:
            bit = label_to_bit.get(lbl)
            if bit is not None:
                mask |= 1 << bit
        place_req_mask[p] = mask

    # 3) Precompute suffix masks TraceSuffix(i)
    suffix_masks: Dict[int, int] = {}
    current = 0
    for i in range(len(trace_labels) - 1, -1, -1):
        lbl = trace_labels[i]
        bit = label_to_bit.get(lbl)
        if bit is not None:
            current |= 1 << bit
        suffix_masks[i] = current
    # and for i == len(trace_labels) (empty suffix)
    suffix_masks[len(trace_labels)] = 0

    return place_req_mask, suffix_masks


def _precompute_structural_shortest_path(
    sync_net,
    final_marking,
    cost_function: Dict[Any, float],
) -> Dict[Any, float]:
    """Backward structural shortest-path on places.

    Approximate minimal non-sync cost from each place to final marking by
    iterating Bellman-Ford style relaxations over transitions.
    """
    # initialize distances: 0 for places in final marking, inf otherwise
    dist_place: Dict[Any, float] = {}
    for p in sync_net.places:
        dist_place[p] = float("inf")
    for p in final_marking.keys():
        dist_place[p] = 0.0

    # Build adjacency: for each transition, we need its preset and postset
    transitions = list(sync_net.transitions)

    # perform a bounded number of relaxation rounds (|places| * |transitions| is safe upper bound)
    max_iters = len(sync_net.places) * max(1, len(transitions))
    for _ in range(max_iters):
        improved = False
        for t in transitions:
            # cost of this transition (we consider full cost_function; this
            # remains admissible, just more conservative)
            c_t = cost_function.get(t, 0.0)
            # best distance among postset places
            succ_vals = [dist_place.get(arc.target, float("inf")) for arc in t.out_arcs]
            if not succ_vals:
                continue
            d_out = max(succ_vals)
            new_val = c_t + d_out
            for arc in t.in_arcs:
                p_in = arc.source
                if new_val < dist_place[p_in]:
                    dist_place[p_in] = new_val
                    improved = True
        if not improved:
            break

    return dist_place


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
):
    start_time = time.time()

    decorate_transitions_prepostset(sync_net)
    decorate_places_preset_trans(sync_net)

    if trace_labels is None:
        trace_labels = []
    trace_len = len(trace_labels)

    # decide which heuristic flavour to use
    mode_val = exec_utils.get_param_value(
        Parameters.REQUIRED_ACTIVITIES_MODE, parameters or {}, RequiredActivitiesMode.SIMPLE.value
    )
    try:
        mode = RequiredActivitiesMode(mode_val)
    except ValueError:
        # fallback to SIMPLE if unknown
        mode = RequiredActivitiesMode.SIMPLE

    # per-move weight (for non-sync moves) based on log costs
    if log_costs:
        heuristic_weight = min(log_costs)
    else:
        heuristic_weight = 1.0

    # shared place -> trace index mapping
    place_to_trace_index = utils.__build_place_to_trace_index(
        sync_net,
        trace_len=trace_len,
        infer_source_sink=True,
    )

    # strategy-specific precomputations
    place_to_required_labels: Dict[Any, Set[str]] = {}
    place_label_depth: Dict[Any, Dict[str, int]] = {}
    place_req_mask: Dict[Any, int] = {}
    suffix_masks: Dict[int, int] = {}
    dist_place: Dict[Any, float] = {}

    if mode in {
        RequiredActivitiesMode.SIMPLE,
        RequiredActivitiesMode.BITSET,
    }:
        place_to_required_labels = _precompute_required_model_labels(sync_net, fin)

    if mode == RequiredActivitiesMode.LABEL_REACHABILITY:
        place_label_depth = _precompute_label_depths(sync_net, fin)

    if mode == RequiredActivitiesMode.BITSET:
        place_req_mask, suffix_masks = _precompute_bitset_structures(
            sync_net,
            fin,
            trace_labels,
        )

    if mode == RequiredActivitiesMode.STRUCTURAL_SHORTEST_PATH:
        dist_place = _precompute_structural_shortest_path(
            sync_net,
            fin,
            cost_function,
        )

    def get_heuristic(marking: Marking) -> float:
        if mode == RequiredActivitiesMode.SIMPLE:
            return _heuristic_simple_required_activities(
                marking,
                trace_labels,
                trace_len,
                place_to_trace_index,
                place_to_required_labels,
                heuristic_weight,
            )
        # fallback
        return 0.0

    # A* algorithm
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
