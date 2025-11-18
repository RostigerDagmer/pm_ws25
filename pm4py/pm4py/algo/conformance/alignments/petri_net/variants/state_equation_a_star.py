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
'''
"""
This module contains code that allows us to compute alignments on the basis of a regular A* search on the state-space
of the synchronous product net of a trace and a Petri net.
The main algorithm follows [1]_.
When running the log-based variant, the code is running in parallel on a trace based level.
Furthermore, by default, the code applies heuristic estimation, and prefers those states that have the smallest h-value
in case the f-value of two states is equal.

References
----------
.. [1] Sebastiaan J. van Zelst et al., "Tuning Alignment Computation: An Experimental Evaluation",
      ATAED@Petri Nets/ACSD 2017: 6-20. `http://ceur-ws.org/Vol-1847/paper01.pdf`_.

"""

import heapq
import sys
import time
from copy import copy
from enum import Enum

import numpy as np
# Import cvxopt components directly for lean LP solving
from cvxopt import matrix, glpk

from pm4py.objects.log import obj as log_implementation
from pm4py.objects.petri_net.utils import align_utils as utils
from pm4py.objects.petri_net.utils.incidence_matrix import (
    construct as inc_mat_construct,
)
from pm4py.objects.petri_net.utils.synchronous_product import (
    construct_cost_aware,
    construct,
)
from pm4py.objects.petri_net.utils.petri_utils import (
    construct_trace_net_cost_aware,
    decorate_places_preset_trans,
    decorate_transitions_prepostset,
)
from pm4py.util import exec_utils
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
from pm4py.util import variants_util
from typing import Optional, Dict, Any, Union
from pm4py.objects.log.obj import Trace
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.util import typing


class Parameters(Enum):
    PARAM_TRACE_COST_FUNCTION = "trace_cost_function"
    PARAM_MODEL_COST_FUNCTION = "model_cost_function"
    PARAM_SYNC_COST_FUNCTION = "sync_cost_function"
    PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE = "ret_tuple_as_trans_desc"
    PARAM_TRACE_NET_COSTS = "trace_net_costs"
    TRACE_NET_CONSTR_FUNCTION = "trace_net_constr_function"
    TRACE_NET_COST_AWARE_CONSTR_FUNCTION = (
        "trace_net_cost_aware_constr_function"
    )
    PARAM_MAX_ALIGN_TIME_TRACE = "max_align_time_trace"
    PARAM_MAX_ALIGN_TIME = "max_align_time"
    PARAMETER_VARIANT_DELIMITER = "variant_delimiter"
    ACTIVITY_KEY = PARAMETER_CONSTANT_ACTIVITY_KEY
    VARIANTS_IDX = "variants_idx"
    RETURN_SYNC_COST_FUNCTION = "return_sync_cost_function"


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
        # keep the possibility to pass TRACE_NET_CONSTR_FUNCTION in this old
        # version
        trace_net, trace_im, trace_fm = trace_net_constr_function(
            trace, activity_key=activity_key
        )
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
    """
    Apply the alignments from the specification of a variants dictionary

    Parameters
    -------------
    var_dictio
        Dictionary of variants (along possibly with their count, or the list of indexes, or the list of involved cases)
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
    """
    Apply the alignments from the specification of a list of variants in the log

    Parameters
    -------------
    var_list
        List of variants (for each item, the first entry is the variant itself, the second entry may be the number of cases)
    petri_net_string
        String representing the accepting Petri net

    Returns
    --------------
    dictio_alignments
        Dictionary that assigns to each variant its alignment
    """
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
    """
    Apply the alignments from the specification of a list of variants in the log

    Parameters
    -------------
    mp_output
        Multiprocessing output
    var_list
        List of variants (for each item, the first entry is the variant itself, the second entry may be the number of cases)
    petri_net_string
        String representing the accepting Petri net

    Returns
    --------------
    dictio_alignments
        Dictionary that assigns to each variant its alignment
    """
    if parameters is None:
        parameters = {}

    res = apply_from_variants_list_petri_string(
        var_list, petri_net_string, parameters=parameters
    )
    mp_output.put(res)


def apply_trace_net(
    petri_net,
    initial_marking,
    final_marking,
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

    alignment = apply_sync_prod(
        sync_prod,
        sync_initial_marking,
        sync_final_marking,
        cost_function,
        utils.SKIP,
        ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
        max_align_time_trace=max_align_time_trace,
    )

    return_sync_cost = exec_utils.get_param_value(
        Parameters.RETURN_SYNC_COST_FUNCTION, parameters, False
    )
    if return_sync_cost:
        # needed for the decomposed alignments (switching them from
        # state_equation_less_memory)
        return alignment, cost_function

    return alignment


def apply_sync_prod(
    sync_prod,
    initial_marking,
    final_marking,
    cost_function,
    skip,
    ret_tuple_as_trans_desc=False,
    max_align_time_trace=sys.maxsize,
):
    """
    Performs the basic alignment search on top of the synchronous product net, given a cost function and skip-symbol

    Parameters
    ----------
    sync_prod: :class:`pm4py.objects.petri.net.PetriNet` synchronous product net
    initial_marking: :class:`pm4py.objects.petri.net.Marking` initial marking in the synchronous product net
    final_marking: :class:`pm4py.objects.petri.net.Marking` final marking in the synchronous product net
    cost_function: :class:`dict` cost function mapping transitions to the synchronous product net
    skip: :class:`Any` symbol to use for skips in the alignment

    Returns
    -------
    dictionary : :class:`dict` with keys **alignment**, **cost**, **visited_states**, **queued_states**
    and **traversed_arcs**
    """
    return __search(
        sync_prod,
        initial_marking,
        final_marking,
        cost_function,
        skip,
        ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
        max_align_time_trace=max_align_time_trace,
    )


def _to_cvxopt_mat(x):
    arr = np.asarray(x, dtype=float)
    return matrix(arr)


def __compute_heuristic_lean_lp(
    sync_net,
    a_matrix,
    h_cvx,
    g_matrix,
    cost_vec,
    incidence_matrix,
    marking,
    fin_vec
):
    """
    Computes the heuristic using a LEAN LP formulation.
    This skips Python overhead by calling GLPK LP solver directly via CVXOPT.
    """
    # 1. Calculate b_term (target marking vector: m_f - m)
    m_vec = incidence_matrix.encode_marking(marking)
    b_term = [i - j for i, j in zip(fin_vec, m_vec)]
    b_term = np.matrix([x * 1.0 for x in b_term]).transpose()

    # 2. Convert to CVXOPT matrices (required by raw interface)
    try:
        cost_c = _to_cvxopt_mat(cost_vec)
        g_c = _to_cvxopt_mat(g_matrix)
        h_c = _to_cvxopt_mat(h_cvx)
        a_c = _to_cvxopt_mat(a_matrix)
        b_c = _to_cvxopt_mat(b_term)
    except Exception:
        # Fallback if conversion fails
        return sys.maxsize, [0.0] * len(sync_net.transitions)

    # 3. Configure GLPK options (Silent)
    glpk.options["msg_lev"] = "GLP_MSG_OFF"

    # 4. Call GLPK LP directly
    try:
        status, x, y, z = glpk.lp(cost_c, g_c, h_c, a_c, b_c)
    except Exception:
        status = None
        x = None

    # 5. Extract result
    prim_obj = None
    points = None

    if status == 'optimal' and x is not None:
        # Calculate objective: c^T * x
        from cvxopt import blas
        prim_obj = blas.dot(cost_c, x)
        points = list(x)

    prim_obj = prim_obj if prim_obj is not None else sys.maxsize
    points = points if points is not None else [0.0] * len(sync_net.transitions)

    return prim_obj, points


def __search(
    sync_net,
    ini,
    fin,
    cost_function,
    skip,
    ret_tuple_as_trans_desc=False,
    max_align_time_trace=sys.maxsize,
):
    start_time = time.time()

    decorate_transitions_prepostset(sync_net)
    decorate_places_preset_trans(sync_net)

    incidence_matrix = inc_mat_construct(sync_net)
    ini_vec, fin_vec, cost_vec = utils.__vectorize_initial_final_cost(
        incidence_matrix, ini, fin, cost_function
    )

    closed = set()

    # Prepare static matrices for LP
    # G and h for non-negativity constraints: -I * x <= 0  =>  x >= 0
    a_matrix = np.asmatrix(incidence_matrix.a_matrix).astype(np.float64)
    g_matrix = -np.eye(len(sync_net.transitions))
    h_cvx = np.matrix(np.zeros(len(sync_net.transitions))).transpose()
    cost_vec = [x * 1.0 for x in cost_vec]

    # Compute initial heuristic using LEAN LP
    h, x = __compute_heuristic_lean_lp(
        sync_net,
        a_matrix,
        h_cvx,
        g_matrix,
        cost_vec,
        incidence_matrix,
        ini,
        fin_vec
    )

    ini_state = utils.SearchTuple(0 + h, 0, h, ini, None, None, x, True)
    open_set = [ini_state]
    heapq.heapify(open_set)
    visited = 0
    queued = 0
    traversed = 0
    lp_solved = 1

    trans_empty_preset = set(
        t for t in sync_net.transitions if len(t.in_arcs) == 0
    )

    while not len(open_set) == 0:
        if (time.time() - start_time) > max_align_time_trace:
            return None

        curr = heapq.heappop(open_set)

        current_marking = curr.m

        while not curr.trust:
            if (time.time() - start_time) > max_align_time_trace:
                return None

            already_closed = current_marking in closed
            if already_closed:
                curr = heapq.heappop(open_set)
                current_marking = curr.m
                continue

            # Recalculate heuristic using LEAN LP for untrusted states
            h, x = __compute_heuristic_lean_lp(
                sync_net,
                a_matrix,
                h_cvx,
                g_matrix,
                cost_vec,
                incidence_matrix,
                curr.m,
                fin_vec
            )
            lp_solved += 1

            tp = utils.SearchTuple(
                curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True
            )
            curr = heapq.heappushpop(open_set, tp)
            current_marking = curr.m

        if curr.h > 10**15: # MAX_ALLOWED_HEURISTICS equivalent
            continue

        already_closed = current_marking in closed
        if already_closed:
            continue

        if curr.h < 0.01:
            if current_marking == fin:
                return utils.__reconstruct_alignment(
                    curr,
                    visited,
                    queued,
                    traversed,
                    ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                    lp_solved=lp_solved,
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
            new_marking = utils.add_markings(current_marking, t.add_marking)

            if new_marking in closed:
                continue
            g = curr.g + cost

            queued += 1
            h, x = utils.__derive_heuristic(
                incidence_matrix, cost_vec, curr.x, t, curr.h
            )
            trustable = utils.__trust_solution(x)
            new_f = g + h

            tp = utils.SearchTuple(
                new_f, g, h, new_marking, curr, t, x, trustable
            )
            heapq.heappush(open_set, tp)