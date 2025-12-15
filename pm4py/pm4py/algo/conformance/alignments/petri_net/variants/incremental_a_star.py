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
import logging
from copy import copy
from enum import Enum
from typing import Optional, Dict, Any, Union, Tuple

import numpy as np
from cvxopt import matrix, glpk
# Import the Extended Marking Equation builder/variants directly
from pm4py.algo.analysis.extended_marking_equation.variants.classic import (
    build as eme_build,
    Parameters as EME_Params,
)
from pm4py.objects.log import obj as log_implementation
from pm4py.objects.log.obj import Trace
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import align_utils as utils
from pm4py.objects.petri_net.utils.incidence_matrix import (
    construct as inc_mat_construct,
)
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
from pm4py.util import typing
from pm4py.util import variants_util
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.util.lp import solver as lp_solver
from pm4py.util.xes_constants import DEFAULT_NAME_KEY

logger = logging.getLogger(__name__)


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
    # additional knobs forwarded to the extended marking equation solver
    EXT_ME_MAX_K_VALUE = "max_k_value"
    EXT_ME_SPLIT_IDX = "split_idx"
    EXT_ME_FULL_BOOTSTRAP_REQUIRED = "full_bootstrap_required"
    # toggle and solver selection for extended-ME heuristic
    EXT_HEUR_USE_ILP = "ext_heur_use_ilp"
    EXT_HEUR_SOLVER_VARIANT = "ext_heur_solver_variant"


PARAM_TRACE_COST_FUNCTION = Parameters.PARAM_TRACE_COST_FUNCTION.value
PARAM_MODEL_COST_FUNCTION = Parameters.PARAM_MODEL_COST_FUNCTION.value
PARAM_SYNC_COST_FUNCTION = Parameters.PARAM_SYNC_COST_FUNCTION.value


class RestartException(Exception):
    """
    Internal exception to trigger a restart of the A* search
    when a new split point is added.
    """

    def __init__(self, new_split_index):
        self.new_split_index = new_split_index


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
        # pass original trace along for the extended marking equation heuristic
        original_trace=trace,
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
        original_trace: Optional[Trace] = None,
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

    # Prepare parameters for EME (Extended Marking Equation)
    # These will be passed to __search, which will handle the EME builder setup internally
    ext_me_parameters = {
        # Costs are mandatory for EME, use the same cost function as the alignment
        EME_Params.COSTS: cost_function,
        # Default parameter values
        EME_Params.MAX_K_VALUE: exec_utils.get_param_value(
            Parameters.EXT_ME_MAX_K_VALUE, parameters, min(10, max(3, len(original_trace) // 5))

        ),
        EME_Params.SPLIT_IDX: exec_utils.get_param_value(
            Parameters.EXT_ME_SPLIT_IDX, parameters, None
        ),
        EME_Params.FULL_BOOTSTRAP_REQUIRED: exec_utils.get_param_value(
            Parameters.EXT_ME_FULL_BOOTSTRAP_REQUIRED, parameters, True
        ),
    }

    use_ilp = exec_utils.get_param_value(
        Parameters.EXT_HEUR_USE_ILP, parameters, False
    )

    alignment = apply_sync_prod(
        sync_prod,
        sync_initial_marking,
        sync_final_marking,
        cost_function,
        utils.SKIP,
        ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
        max_align_time_trace=max_align_time_trace,
        original_trace=original_trace,
        ext_me_parameters=ext_me_parameters,
        use_ilp=use_ilp
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
        original_trace: Optional[Trace] = None,
        ext_me_parameters: Optional[Dict[Any, Any]] = None,
        use_ilp: bool = False,
        solver_variant: Optional[str] = None,
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
        original_trace=original_trace,
        ext_me_parameters=ext_me_parameters,
        use_ilp=use_ilp,
        solver_variant=solver_variant,
    )


def to_mat_cvtopt(m):
    """
    Helper to convert numpy arrays/matrices to CVXOPT matrix.
    """
    if isinstance(m, (np.ndarray, np.matrix)):
        if m.size == 0:
            # Handle empty matrices explicitly for safety
            return matrix([], (m.shape[0], m.shape[1] if len(m.shape) > 1 else 1))
        return matrix(np.asarray(m, dtype=np.float64))
    return matrix(m)


def _eme_solve_inner(
        eme_solver,
        current_marking: Marking,
        use_ilp: bool,
) -> Tuple[float, np.ndarray, bool, str]:
    """
    Internal function to solve the EME problem for a given marking using LEAN LP.
    Uses cvxopt.glpk.lp directly for performance.

    Returns:
        (h_val, x_array, is_feasible, status)
        status: 'optimal' | 'primal_infeasible' | 'dual_infeasible' | 'unknown' | 'error'
    """
    try:
        # 1. Update initial marking vector in the solver
        # This recalculates vectors based on the new current marking
        eme_solver.change_ini_vec(current_marking)

        # 2. Get components (c, Aub, bub, Aeq, beq) from the solver
        c, Aub, bub, Aeq, beq = eme_solver.get_components()

        # 3. Convert to CVXOPT matrices (required for glpk.lp)
        c_cvx = to_mat_cvtopt(c)
        Aub_cvx = to_mat_cvtopt(Aub)
        bub_cvx = to_mat_cvtopt(bub)
        Aeq_cvx = to_mat_cvtopt(Aeq)
        beq_cvx = to_mat_cvtopt(beq)

        # 4. Configure GLPK options (Silent)
        glpk.options["msg_lev"] = "GLP_MSG_OFF"

        # 5. Solve directly using GLPK (Lean LP)
        if use_ilp:
            logger.debug("Using ILP solver for EME heuristic computation.")
            # Identify integer variable indices
            size = Aub.size[1]
            I = set(range(size))
            status, x = glpk.ilp(c_cvx, Aub_cvx, bub_cvx, Aeq_cvx, beq_cvx, I=I)
        else:
            logger.debug("Using LP solver for EME heuristic computation.")
            status, x, y, z = glpk.lp(c_cvx, Aub_cvx, bub_cvx, Aeq_cvx, beq_cvx)

        # 6. Map statuses to clearer identifiers
        # cvxopt.glpk uses: 'optimal', 'primal infeasible', 'dual infeasible', 'unknown'
        if isinstance(status, str):
            stat = status.lower().strip()
        else:
            stat = str(status).lower().strip()

        if stat == "optimal" and x is not None:
            # Use EME solver to compute robust h and x-vector mapping
            x_list = list(x)
            x_vec_list = eme_solver.get_x_vector(x_list)
            h_val = float(eme_solver.get_h(x_list))
            return h_val, np.array(x_vec_list), True, "optimal"
        elif "primal infeasible" in stat:
            return float(sys.maxsize), np.array([]), False, "primal_infeasible"
        elif "dual infeasible" in stat:
            # treat as infeasible for our purposes
            return float(sys.maxsize), np.array([]), False, "dual_infeasible"
        elif "unknown" in stat:
            # numerical trouble; we treat as dead-end
            return float(sys.maxsize), np.array([]), False, "unknown"
        else:
            # Fallback to 'error' for any other unexpected status
            return float(sys.maxsize), np.array([]), False, "error"

    except Exception:
        # Any exception during LP -> report as 'error'
        return float(sys.maxsize), np.array([]), False, "error"


def get_trace_index(marking: Marking, place_to_trace_index: Dict[Any, int], trace_len: int) -> int:
    """
    Infers the current trace index from a synchronous product marking.
    Iterates through the marked places and checks if they correspond to a known trace net place.
    """
    for p in marking:
        if p in place_to_trace_index:
            return place_to_trace_index[p]

    # Fallback: Check for common final place names or assume bounds
    for p in marking:
        if p.name == "sink" or p.name == "p_sink" or p.name == "end":
            return trace_len

    # Default to trace_len if we are at the end (no trace places marked usually means end in some constructions)
    # However, standard trace net has a token in 'sink' at the end.
    # If no mapped place is found, and we are not empty, return a safe fallback (e.g. 0 or trace_len)
    # For A*, returning trace_len usually forces a specific heuristic check.
    return trace_len


def __search(
        sync_net,
        ini,
        fin,
        cost_function,
        skip,
        ret_tuple_as_trans_desc=False,
        max_align_time_trace=sys.maxsize,
        original_trace: Optional[Trace] = None,
        ext_me_parameters: Optional[Dict[Any, Any]] = None,
        use_ilp: bool = False,
        solver_variant: Optional[str] = None,
):
    start_time = time.time()

    # 1. Setup Petri Net Decorators and Incidence Matrix (Standard A* Setup)
    decorate_transitions_prepostset(sync_net)
    decorate_places_preset_trans(sync_net)

    incidence_matrix = inc_mat_construct(sync_net)
    ini_vec, fin_vec, cost_vec = utils.__vectorize_initial_final_cost(
        incidence_matrix, ini, fin, cost_function
    )
    cost_vec_derivation = [x * 1.0 for x in cost_vec]

    # 2. Build Mapping from SyncNet Places -> Trace Index
    trace_len = len(original_trace) if original_trace else 0
    place_to_trace_index = utils.__build_place_to_trace_index(sync_net)

    # 3. Initialize Split Points
    # "Incremental" means we start with minimum split points (Start and End)
    split_points = {0, trace_len}
    restarts = 0

    # Track the maximum "explained" trace index seen among LP solutions
    max_explained_idx = 0

    if ext_me_parameters is None:
        ext_me_parameters = {}
    if EME_Params.COSTS not in ext_me_parameters:
        ext_me_parameters[EME_Params.COSTS] = cost_function

    # === MAIN RESTART LOOP ===
    while True:
        try:
            # A. Update EME Solver Parameters
            valid_split_idx = sorted([i for i in split_points if i < trace_len])

            ext_me_parameters[EME_Params.SPLIT_IDX] = valid_split_idx
            # Keep a margin on MAX_K_VALUE so chosen splits are not trimmed away by the builder
            ext_me_parameters[EME_Params.MAX_K_VALUE] = len(valid_split_idx) + 5  # Ensure no trimming

            # B. Build EME Solver
            eme_solver = eme_build(original_trace, sync_net, ini, fin, parameters=ext_me_parameters)

            # given x (list/ndarray), update max_explained_idx using the EME solver
            def _maybe_update_max_explained(x_vec):
                nonlocal max_explained_idx
                try:
                    if x_vec is None or len(x_vec) == 0:
                        return
                    # ensure integer list for get_firing_sequence
                    x_list = [int(v) for v in (list(x_vec) if not isinstance(x_vec, list) else x_vec)]
                    _, _, explained_events = eme_solver.get_firing_sequence(x_list)
                    if explained_events is not None:
                        max_explained_idx = max(max_explained_idx, explained_events)
                except Exception:
                    pass

            # C. A* Initialization
            closed = set()
            visited = 0
            queued = 0
            traversed = 0
            lp_solved = 0

            h, x, is_feasible, status = _eme_solve_inner(eme_solver, ini, use_ilp=use_ilp)
            if status != "optimal":
                logger.debug(f"Root LP failed with status '{status}'. Falling back to h=0.")
                # root LP must be solvable: otherwise no alignment via this decomposition
                return None
            _maybe_update_max_explained(x)
            lp_solved += 1
            ini_state = utils.SearchTuple(0 + h, 0, h, ini, None, None, x, True)

            open_set = [ini_state]
            heapq.heapify(open_set)

            trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)

            # D. Inner A* Loop
            while open_set:
                # time guard
                if (time.time() - start_time) > max_align_time_trace:
                    return None

                curr = heapq.heappop(open_set)
                current_marking = curr.m

                # --- INCREMENTAL LOGIC START ---
                # If a state is "Trusted" (heuristic derived from parent), we skip LP.
                # If "Untrusted", we must solve LP for that marking. If it's not at a split point,
                # we add a new split (selected at max_explained_idx as per the practical paper) and restart the whole search.

                while not curr.trust:
                    if (time.time() - start_time) > max_align_time_trace:
                        return None

                    if current_marking in closed:
                        if open_set:
                            curr = heapq.heappop(open_set)
                            current_marking = curr.m
                            continue
                        else:
                            return None

                    # Determine trace index
                    curr_trace_idx = get_trace_index(current_marking, place_to_trace_index, trace_len)

                    # If we need an exact calculation but are not at a split point -> RESTART
                    if curr_trace_idx not in split_points:
                        # choose split at the maximum number of events explained so far
                        new_split_index = max_explained_idx if (
                                    max_explained_idx and (max_explained_idx not in split_points)) else curr_trace_idx
                        # Safety bounds
                        if new_split_index in split_points or new_split_index <= 0 or new_split_index >= trace_len:
                            new_split_index = curr_trace_idx
                        raise RestartException(new_split_index) # trigger restart of the outer loop

                    # We are at a split point -> solve LP for curr.m
                    h, x, is_feasible, status = _eme_solve_inner(eme_solver, curr.m, use_ilp=False)
                    logger.debug(f"Solved LP at marking {current_marking} (trace idx {curr_trace_idx}) with status '{status}' and h={h}.")
                    # Only update max_explained if we received an actual x
                    _maybe_update_max_explained(x)

                    # as described in paper: handle LP statuses:
                    # - 'optimal' -> proceed
                    # - 'primal_infeasible' or 'dual_infeasible' or 'unknown' or 'error' -> treat as dead-end
                    if status != "optimal":
                        # dead path according to LP
                        closed.add(current_marking)
                        # get next candidate from open set
                        if open_set:
                            curr = heapq.heappop(open_set)
                            current_marking = curr.m
                            continue
                        else:
                            return None

                    # LP successful
                    lp_solved += 1
                    tp = utils.SearchTuple(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True)
                    curr = heapq.heappushpop(open_set, tp)
                    current_marking = curr.m
                # --- INCREMENTAL LOGIC END ---

                # Standard A* checks
                if curr.h > lp_solver.MAX_ALLOWED_HEURISTICS:
                    continue
                if current_marking in closed:
                    continue

                # Goal Reached?
                if curr.h < 0.01 and current_marking == fin:
                    return utils.__reconstruct_alignment(
                        curr, visited, queued, traversed,
                        ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                        lp_solved=lp_solved
                    )

                closed.add(current_marking)
                visited += 1

                # Expand transitions
                enabled_trans = copy(trans_empty_preset)
                for p in current_marking:
                    for t in p.ass_trans:
                        if t.sub_marking <= current_marking:
                            enabled_trans.add(t)

                trans_to_visit_with_cost = [
                    (t, cost_function[t]) for t in enabled_trans
                    if not (t is not None and utils.__is_log_move(t, skip) and utils.__is_model_move(t, skip))
                ]

                for t, cost in trans_to_visit_with_cost:
                    traversed += 1
                    new_marking = utils.add_markings(current_marking, t.add_marking)
                    if new_marking in closed:
                        continue

                    g = curr.g + cost

                    # Try to derive heuristic (cheaply - no lp computation)
                    h_derived, x_derived = utils.__derive_heuristic(
                        incidence_matrix, cost_vec_derivation, curr.x, t, curr.h
                    )

                    # If derivation is valid, trust=True. If invalid, trust=False.
                    trustable = utils.__trust_solution(x_derived)

                    queued += 1
                    new_f = g + h_derived

                    # Add to queue. If trustable=False, it will trigger the Incremental Logic check when popped.
                    tp = utils.SearchTuple(new_f, g, h_derived, new_marking, curr, t, x_derived, trustable)
                    heapq.heappush(open_set, tp)

            # no alignment found for this decomposition
            return None

        except RestartException as e:
            # Add the new split point suggested by the incremental logic and restart
            split_points.add(e.new_split_index)
            restarts += 1
            # Loop continues, rebuilding eme_solver with new splits
            continue

    return None
