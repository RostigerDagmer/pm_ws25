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
import time
from collections import deque

from pm4py.objects.log import obj as log_implementation
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
from pm4py.objects.petri_net.utils.synchronous_product import (
    construct_cost_aware,
    construct,
)
from pm4py.objects.petri_net.utils.petri_utils import (
    construct_trace_net_cost_aware,
    decorate_places_preset_trans,
    decorate_transitions_prepostset,
)
from pm4py.objects.petri_net.utils import align_utils as utils
from pm4py.util import exec_utils
from copy import copy
from enum import Enum
import sys
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.util import variants_util
from typing import Optional, Dict, Any, Union, List
from pm4py.objects.log.obj import Trace
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.util import typing

from pm4py.pm4py.objects.petri_net import semantics


class Parameters(Enum):
    PARAM_TRACE_COST_FUNCTION = "trace_cost_function"
    PARAM_MODEL_COST_FUNCTION = "model_cost_function"
    PARAM_SYNC_COST_FUNCTION = "sync_cost_function"
    PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE = "ret_tuple_as_trans_desc"
    PARAM_TRACE_NET_COSTS = "trace_net_costs"
    TRACE_NET_CONSTR_FUNCTION = "trace_net_constr_function"
    TRACE_NET_COST_AWARE_CONSTR_FUNCTION = "trace_net_cost_aware_constr_function"
    PARAM_MAX_ALIGN_TIME_TRACE = "max_align_time_trace"
    PARAM_MAX_ALIGN_TIME = "max_align_time"
    PARAMETER_VARIANT_DELIMITER = "variant_delimiter"
    ACTIVITY_KEY = PARAMETER_CONSTANT_ACTIVITY_KEY
    ENABLE_OPTIMIZATIONS = "enable_optimizations"  # Toggle for REACH optimizations (SRModel, SRLog, Greedy)


PARAM_TRACE_COST_FUNCTION = Parameters.PARAM_TRACE_COST_FUNCTION.value
PARAM_MODEL_COST_FUNCTION = Parameters.PARAM_MODEL_COST_FUNCTION.value
PARAM_SYNC_COST_FUNCTION = Parameters.PARAM_SYNC_COST_FUNCTION.value


class ReachHeuristic:
    """
    Implements the REACH 'Model Move Required' heuristic logic.
    1. MMR (Model Move Required) Heuristic (Algorithm 3)
    2. SRModel Optimization (Algorithm 6)
    3. SRLog Optimization (Algorithm 7 & 8)
    """

    def __init__(self, model_net: PetriNet, sync_net: PetriNet, skip_symb=utils.SKIP):
        self.model_net = model_net
        self.sync_net = sync_net
        self.skip_symb = skip_symb

        # 1. Mapping: Sync_Place -> Model_Place
        self.sync_to_model_map = self._build_place_mapping(sync_net, model_net)

        # 2. Cache for MMR Heuristic
        self._cache = {}

        # 3. Optimization Pre-computation (Algorithm 8)
        # Map: FrozenSet(Marking) -> Set[ActivityLabels]
        self.alive_activities_map = self._compute_alive_activities(model_net)

    def _build_place_mapping(self, sync_net: PetriNet, model_net: PetriNet):
        """
        Maps places in the Synchronous Product back to the Original Model.
        Relies on PM4Py's standard naming convention: (SKIP, 'p_name') for model places.
        """
        mapping = {}

        # Index original model places by name for O(1) lookup
        model_places_by_name = {p.name: p for p in model_net.places}

        for sync_p in sync_net.places:
            # Check if this sync place represents a model place
            # PM4Py Sync Product naming: (SKIP, model_place_name)
            if isinstance(sync_p.name, tuple) and len(sync_p.name) == 2:
                left, right = sync_p.name

                # In standard construct(), pn2 (Model) is usually the second argument,
                # so its places are named (SKIP, name).
                if left == self.skip_symb and right in model_places_by_name:
                    mapping[sync_p] = model_places_by_name[right]

        return mapping

    def _project_marking(self, sync_marking: Marking) -> Marking:
        """
        Converts a Synchronous Product Marking into a Model Marking.
        Ignores 'Trace' places (those that don't map to the model).
        """
        model_marking = Marking()
        for sync_p, count in sync_marking.items():
            if sync_p in self.sync_to_model_map:
                model_p = self.sync_to_model_map[sync_p]
                model_marking[model_p] += count
        return model_marking

    def _compute_alive_activities(self, net: PetriNet) -> dict:
        """
        Algorithm 8: Initialization for SRLog.
        Pre-computes the 'alive activities' for every reachable marking in the ORIGINAL model.
        Returns: Dict[FrozenMarking, Set[Labels]]
        """
        # 1. Find Initial Marking of the Model (usually empty input arcs)
        # Note: In standard PM4Py, we might need the IM passed explicitly.
        # If not available, we try to discover it or assume it's standard.
        # For robustness, let's discover it:
        initial_marking = Marking()
        for p in net.places:
            if len(p.in_arcs) == 0:
                initial_marking[p] = 1

        # 2. Build Reachability Graph (Nodes and Edges)
        # We perform a BFS to find all states and transitions between them
        rg_nodes = set()
        rg_edges = []  # List of (source_m_key, target_m_key, label)

        init_key = frozenset(initial_marking.items())
        rg_nodes.add(init_key)

        queue = deque([(initial_marking, init_key)])

        # Map: FrozenMarking -> Set[Labels] (initially just direct outgoing)
        alive_map = {}

        while queue:
            m, m_key = queue.popleft()

            if m_key not in alive_map:
                alive_map[m_key] = set()

            enabled = semantics.enabled_transitions(net, m)
            for t in enabled:
                new_m = semantics.execute(t, net, m)
                new_m_key = frozenset(new_m.items())

                # Add direct alive label
                if t.label is not None:
                    alive_map[m_key].add(t.label)

                # Record Edge for back-propagation
                rg_edges.append((m_key, new_m_key))

                if new_m_key not in rg_nodes:
                    rg_nodes.add(new_m_key)
                    queue.append((new_m, new_m_key))

        # 3. Backwards Propagation (Fixed-Point Iteration)
        # If I can reach State B from State A, then A inherits all alive activities of B.
        changed = True
        while changed:
            changed = False
            for src_key, tgt_key in rg_edges:
                tgt_alive = alive_map.get(tgt_key, set())
                src_alive = alive_map[src_key]

                # If target has something source doesn't, add it
                if not tgt_alive.issubset(src_alive):
                    src_alive.update(tgt_alive)
                    changed = True

        return alive_map

    def compute_required_activities(self, model_marking: Marking) -> set:
        """
        Algorithm 3: REQUIRED_TRANSITIONS
        Performs structural traversal on the PROCESS MODEL to find unavoidable transitions.
        """
        # Cache Check
        marking_key = frozenset(model_marking.items())
        if marking_key in self._cache:
            return self._cache[marking_key]

        required_labels = set()

        # Traverse structually from current tokens
        places_to_visit = list(model_marking.keys())
        visited_places = set()

        while places_to_visit:
            place = places_to_visit.pop(0)

            if place in visited_places:
                continue
            visited_places.add(place)

            # Check if Sink (no out arcs)
            if len(place.out_arcs) == 0:
                continue

            # Get outgoing transitions
            transitions = [arc.target for arc in place.out_arcs]

            # CRITICAL LOGIC: If exactly 1 path exists, it is required.
            if len(transitions) == 1:
                trans = transitions[0]

                # If visible, add to required set
                if trans.label is not None:
                    required_labels.add(trans.label)

                # Continue traversal downstream
                for arc in trans.out_arcs:
                    next_place = arc.target
                    if next_place not in visited_places:
                        places_to_visit.append(next_place)

            # Else: Branch/Choice detected. Stop traversal on this path.
            # We cannot guarantee any specific transition is required beyond this point.

        # Save to cache
        self._cache[marking_key] = required_labels
        return required_labels

    def get_heuristic_value(self, sync_marking: Marking, remaining_trace_labels: list,
                            model_move_cost=utils.STD_MODEL_LOG_MOVE_COST, sync_move_cost=utils.STD_SYNC_COST) -> int:
        """
        Main entry point for the A* algorithm.

        :param sync_marking: The current marking in the A* search (Sync Net)
        :param remaining_trace_labels: List of activity names left in the trace suffix
        """
        # 1. Project Marking (Sync -> Model)
        model_marking = self._project_marking(sync_marking)

        # 2. Get Required Model Activities
        required = self.compute_required_activities(model_marking)

        # 3. Get Available Trace Activities
        remaining = set(remaining_trace_labels)

        # 4. Calculate Costs
        # Missing: Required by Model but NOT in Remaining Trace -> Must be Model Move
        missing = required - remaining

        # Sync Candidates = Required AND Remaining (Optimistically Sync)
        sync_candidates = required.intersection(remaining)

        h_val = (len(missing) * model_move_cost) + (len(sync_candidates) * sync_move_cost)

        return h_val


    def check_sr_model(self, sync_marking: Marking, remaining_trace_labels: list) -> bool:
        """
        Algorithm 6: SRModel (LessStatesLog)
        Checks if the current model state can EVER produce any of the remaining trace activities.
        If disjoint, we must force Model Moves (skip Log/Sync moves for this step).
        """
        model_marking = self._project_marking(sync_marking)

        # Get all currently enabled activities in the model
        enabled_transitions = semantics.enabled_transitions(self.model_net, model_marking)
        enabled_labels = {t.label for t in enabled_transitions if t.label is not None}

        # If no visible transitions are enabled, we can't really "match" anyway.
        if not enabled_labels:
            return False

        remaining_set = set(remaining_trace_labels)

        # If the intersection is empty, the model cannot match the trace here.
        # Implies we must process the model to reach a state where it CAN match.
        # Return True -> Force Model Move
        return enabled_labels.isdisjoint(remaining_set)

    def check_sr_log(self, sync_marking: Marking, next_trace_label: str) -> bool:
        """
        Algorithm 7: SRLog (LessStatesModel)
        Checks if the NEXT trace activity is "Alive" (reachable) from the current model state.
        If not alive, we must force a Log Move (skip Model/Sync moves).
        """
        model_marking = self._project_marking(sync_marking)
        m_key = frozenset(model_marking.items())

        if m_key not in self.alive_activities_map:
            # Should not happen if map is complete, but safe fallback
            return False

        alive_activities = self.alive_activities_map[m_key]

        # If the next trace activity is NOT in the set of reachable activities,
        # we cannot possibly sync it now or in the future.
        # Return True -> Force Log Move
        return next_trace_label not in alive_activities


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


def apply(
    trace: Trace,
    petri_net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
) -> typing.AlignmentResult:
    """
       Performs the basic alignment search, given a trace and a net.

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

       Returns
       -------
       dictionary: `dict` with keys **alignment**, **cost**, **visited_states**, **queued_states** and **traversed_arcs**
       """
    if parameters is None:
        parameters = {}

    activity_key = exec_utils.get_param_value(
        Parameters.ACTIVITY_KEY, parameters, DEFAULT_NAME_KEY
    )
    trace_cost_function = exec_utils.get_param_value(
        Parameters.PARAM_TRACE_COST_FUNCTION, parameters, None
    )
    model_cost_function = exec_utils.get_param_value(
        Parameters.PARAM_MODEL_COST_FUNCTION, parameters, None
    )
    trace_net_constr_function = exec_utils.get_param_value(
        Parameters.TRACE_NET_CONSTR_FUNCTION, parameters, None
    )
    trace_net_cost_aware_constr_function = exec_utils.get_param_value(
        Parameters.TRACE_NET_COST_AWARE_CONSTR_FUNCTION,
        parameters,
        construct_trace_net_cost_aware,
    )

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
    trace_net: PetriNet,
    trace_im: Marking,
    trace_fm: Marking,
    trace_labels: List[str],
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

    max_align_time_trace = exec_utils.get_param_value(
        Parameters.PARAM_MAX_ALIGN_TIME_TRACE, parameters, sys.maxsize
    )
    # We scan the cost function to find the minimum weights for Model and Sync moves
    h_model_cost = utils.STD_MODEL_LOG_MOVE_COST
    h_sync_cost = utils.STD_SYNC_COST
    model_moves = [c for t, c in cost_function.items() if utils.__is_model_move(t, utils.SKIP)]
    if model_moves: h_model_cost = min(model_moves)
    sync_moves = [c for t, c in cost_function.items() if
                  not utils.__is_model_move(t, utils.SKIP) and not utils.__is_log_move(t, utils.SKIP)]
    if sync_moves: h_sync_cost = min(sync_moves)

    reach_heuristic = ReachHeuristic(petri_net, sync_prod, utils.SKIP)

    return apply_sync_prod(
        sync_prod,
        sync_initial_marking,
        sync_final_marking,
        cost_function,
        utils.SKIP,
        trace_labels,
        reach_heuristic,
        h_model_cost,
        h_sync_cost,
        ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
        max_align_time_trace=max_align_time_trace,
        parameters=parameters,
    )


def apply_sync_prod(
        sync_prod,
        initial_marking,
        final_marking,
        cost_function,
        skip,
        trace_labels,
        reach_heuristic,
        h_model_cost,
        h_sync_cost,
        ret_tuple_as_trans_desc=False,
        max_align_time_trace=sys.maxsize,
        parameters=None,
):
    return __search(
        sync_prod,
        initial_marking,
        final_marking,
        cost_function,
        skip,
        heuristic_oracle=reach_heuristic,
        h_model_cost=h_model_cost,
        h_sync_cost=h_sync_cost,
        trace_labels=trace_labels,
        ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
        max_align_time_trace=max_align_time_trace,
        parameters=parameters,
    )


# =============================================================================
# Greedy Upper Bound (Algorithm 9)
# =============================================================================

def run_greedy_search(sync_net, ini, fin, cost_function, skip, trace_labels):
    """
    Fast greedy search to find an Upper Bound cost.
    Prioritizes making progress in the trace to reach the end quickly.
    """
    curr_m = ini
    curr_cost = 0.0
    trace_idx = 0
    trace_len = len(trace_labels)

    # Safety limit to prevent infinite loops in cyclic models
    steps = 0
    max_steps = len(sync_net.transitions) * 2 + trace_len * 2

    while steps < max_steps:
        if curr_m == fin:
            return curr_cost

        enabled = semantics.enabled_transitions(sync_net, curr_m)
        if not enabled: return float('inf')  # Deadlock

        best_t = None
        best_score = float('inf')
        best_new_idx = trace_idx

        for t in enabled:
            cost = cost_function[t]
            is_log = utils.__is_log_move(t, skip)
            is_model = utils.__is_model_move(t, skip)

            new_idx = trace_idx
            if is_log:
                new_idx += 1
            elif not is_model:
                new_idx += 1  # Sync

            # Greedy Heuristic: Local Cost + Remaining Trace Penalty
            # We penalize remaining trace heavily to encourage finishing the event log
            rem_penalty = (trace_len - new_idx) * 1000
            score = curr_cost + cost + rem_penalty

            if score < best_score:
                best_score = score
                best_t = t
                best_new_idx = new_idx

        if best_t:
            curr_m = semantics.weak_execute(best_t, sync_net, curr_m)
            curr_cost += cost_function[best_t]
            trace_idx = best_new_idx
            steps += 1
        else:
            return float('inf')

    return float('inf')

def __search(
    sync_net,
    ini,
    fin,
    cost_function,
    skip,
    heuristic_oracle: ReachHeuristic,
    h_model_cost,
    h_sync_cost,
    trace_labels=None,
    ret_tuple_as_trans_desc=False,
    max_align_time_trace=sys.maxsize,
    parameters=None,
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

    def get_trace_suffix(m):
        """ Helper: Extract remaining trace suffix from current marking """
        idx = 0
        found = False
        for p in m:
            if p in place_to_trace_index:
                if not found:
                    idx = place_to_trace_index[p]
                    found = True
                else:
                    idx = max(idx, place_to_trace_index[p])
        return trace_labels[idx:]


    # Greedy Upper Bound (Algorithm 9)
    upper_bound = float('inf')
    if enable_optimizations:
        try:
            upper_bound = run_greedy_search(sync_net, ini, fin, cost_function, skip, trace_labels)
        except:
            upper_bound = float('inf')

    # Initial State
    initial_suffix = trace_labels  # Start with full trace
    h0 = heuristic_oracle.get_heuristic_value(ini, initial_suffix, h_model_cost, h_sync_cost)

    ini_state = utils.SearchTuple(0 + h0, 0, h0, ini, None, None, None, True)
    open_set = [ini_state]
    heapq.heapify(open_set)

    closed = set()
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

        # Pruning by Upper Bound
        if enable_optimizations and (curr.g + curr.h > upper_bound):
            continue

        if current_marking in closed: continue
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

        # Optimizations (Phase 2)
        rem_trace = get_trace_suffix(current_marking)

        force_model = False
        force_log = False

        if enable_optimizations:
            force_model = heuristic_oracle.check_sr_model(current_marking, rem_trace)
            if rem_trace:
                force_log = heuristic_oracle.check_sr_log(current_marking, rem_trace[0])

        # Conflict Resolution: If heuristics contradict, disable both to be safe
        if force_model and force_log:
            force_model = False
            force_log = False

        # Generate Neighbors
        enabled_trans = copy(trans_empty_preset)
        for p in current_marking:
            for t in p.ass_trans:
                if t.sub_marking <= current_marking:
                    enabled_trans.add(t)

        for t in enabled_trans:
            # Determine Move Type
            is_log = utils.__is_log_move(t, skip)
            is_model = utils.__is_model_move(t, skip)
            is_sync = not is_log and not is_model

            # Apply Optimization Filters
            if force_model and is_log: continue  # Skip Log if forced to Model
            if force_log and (is_model or is_sync): continue  # Skip Model/Sync if forced to Log

            cost = cost_function[t]
            traversed += 1
            new_marking = utils.add_markings(current_marking, t.add_marking)

            if new_marking in closed: continue

            g = curr.g + cost

            # Calculate Heuristic for Neighbor
            new_rem_trace = get_trace_suffix(new_marking)
            h = heuristic_oracle.get_heuristic_value(new_marking, new_rem_trace, h_model_cost, h_sync_cost)

            new_f = g + h

            # Early Pruning check
            if enable_optimizations and (new_f > upper_bound):
                continue

            queued += 1
            tp = utils.SearchTuple(new_f, g, h, new_marking, curr, t, None, True)
            heapq.heappush(open_set, tp)

    return None