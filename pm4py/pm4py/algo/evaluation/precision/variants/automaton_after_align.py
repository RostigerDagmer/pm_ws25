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
from copy import copy
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from pm4py.algo.conformance.alignments.petri_net import algorithm as petri_alignments
from pm4py.objects import log as log_lib
from pm4py.objects.log.obj import EventLog, EventStream
from pm4py.objects.petri_net import semantics
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import (
    align_utils as pn_align_utils,
    check_soundness,
)
from pm4py.statistics.start_activities.log.get import get_start_activities
from pm4py.util import constants, exec_utils, variants_util


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY
    TOKEN_REPLAY_VARIANT = "token_replay_variant"
    CLEANING_TOKEN_FLOOD = "cleaning_token_flood"
    SHOW_PROGRESS_BAR = "show_progress_bar"
    MULTIPROCESSING = "multiprocessing"
    CORES = "cores"


# -----------------------------------------------------------------------------#
# Helper functions
# -----------------------------------------------------------------------------#
def _extract_model_sequence(
    alignment: List[Tuple[Tuple[str, str], Tuple[str, str]]]
) -> List[str]:
    """
    Given an alignment (returned by PM4Py with
    ``ret_tuple_as_trans_desc = True``)
    extract the sequence of *visible* model moves
    (i.e. the bottom row in the alignment table).

    Invisible moves are ignored because they do not constitute
    log activities/prefixes.
    """
    seq: List[str] = []
    for move in alignment:
        # move[0] – model side, move[1] – log side
        # move[0][0] == '>>'  ⇒  move in log only
        if move[0][0] != ">>":  # model fired a transition
            label = move[1][0]  # log label (same as transition label for sync moves)
            if label is not None and label != ">>":
                seq.append(label)
    return seq


def _update_prefix_stats(
    seq: List[str],
    weight: int,
    prefixes: Dict[str, Set[str]],
    prefix_count: Dict[str, int],
) -> None:
    """
    From a *model* sequence build/extend:
      * ``prefixes``      – prefix  →  set(next visible activity labels)
      * ``prefix_count``  – prefix  →  aggregated frequency

    The very last activity in a trace is *not* considered,
    exactly like in the original ET‑Conformance definition:
    there is no “next” move after the last event.
    """
    if not seq:
        return

    current_prefix = None
    for i, activity in enumerate(seq[:-1]):  # stop before the last element
        current_prefix = activity if current_prefix is None else f"{current_prefix},{activity}"

        next_act = seq[i + 1]

        # update set of next activities
        prefixes.setdefault(current_prefix, set()).add(next_act)
        # update multiplicity
        prefix_count[current_prefix] = prefix_count.get(current_prefix, 0) + weight


# -----------------------------------------------------------------------------#
# Main algorithm
# -----------------------------------------------------------------------------#
def apply(
    log: Union[EventLog, EventStream, pd.DataFrame],
    net: PetriNet,
    im: Marking,
    fm: Marking,
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
) -> float:
    """
    Compute the Align‑ETConformance *precision* where the prefix‑automaton
    is generated from **aligned traces** (model projections).

    Implements: Adriansyah, Arya, et al. "Measuring precision of modeled behavior." Information systems and e-Business Management 13.1 (2015): 37-67.

    The only difference with the reference implementation contained in PM4Py
    is that the prefix automaton is built *after* aligning every
    (variant of the) trace with the model; hence each prefix is guaranteed to end
    in a reachable marking and the set of enabled visible transitions can be
    computed without risk of being undefined.

    Parameters
    ----------
    log
        The event log / event stream / dataframe.
    net
        Petri net.
    im
        Initial marking.
    fm
        Final marking.
    parameters
        Same dictionary accepted by the original function.
    """
    if parameters is None:
        parameters = {}

    # ------------------------------------------------------------------#
    # Basic checks and parameter extraction
    # ------------------------------------------------------------------#
    if not check_soundness.check_easy_soundness_net_in_fin_marking(net, im, fm):
        raise ValueError(
            "Align‑ET precision can only be applied on a Petri net that is "
            "a *sound* WF‑net (easy‑sound)."
        )

    activity_key = exec_utils.get_param_value(
        Parameters.ACTIVITY_KEY, parameters, log_lib.util.xes.DEFAULT_NAME_KEY
    )
    case_id_key = exec_utils.get_param_value(
        Parameters.CASE_ID_KEY, parameters, constants.CASE_CONCEPT_NAME
    )

    debug_level = parameters.get("debug_level", 0)

    # ------------------------------------------------------------------#
    # 1.  Group traces into variants  →  drastically fewer alignments
    # ------------------------------------------------------------------#
    import pm4py

    log = pm4py.convert_to_dataframe(log)

    variants = pm4py.get_variants(log, activity_key)
    variant_keys = list(variants.keys())

    # reduced log (one trace per variant) – alignment cost dominates
    red_log = EventLog()
    for var in variant_keys:
        red_log.append(variants_util.variant_to_trace(var, parameters=parameters))

    # ------------------------------------------------------------------#
    # 2.  Align every variant trace with the model
    # ------------------------------------------------------------------#
    align_params = copy(parameters)
    align_params["ret_tuple_as_trans_desc"] = True

    aligned_traces = petri_alignments.apply(red_log, net, im, fm, parameters=align_params)

    # Map from transition *name* → Transition object (faster look‑up later on)
    trans_by_name = {t.name: t for t in net.transitions}

    # ------------------------------------------------------------------#
    # 3.  Build prefix statistics from the *aligned* sequences
    # ------------------------------------------------------------------#
    prefixes: Dict[str, Set[str]] = {}
    prefix_count: Dict[str, int] = {}

    for variant_idx, aligned in enumerate(aligned_traces):
        alignment = aligned["alignment"]
        seq = _extract_model_sequence(alignment)

        # frequency of the variant (= number of original traces with this exact sequence)
        freq = variants[variant_keys[variant_idx]]
        _update_prefix_stats(seq, freq, prefixes, prefix_count)

    # ------------------------------------------------------------------#
    # 4.  Precision calculation – identical to the reference implementation,
    #     but using the new `prefixes` / `prefix_count`.
    # ------------------------------------------------------------------#
    precision = 1.0               # default when no AT found
    sum_ee = 0                    # escaping edges   (numerator)
    sum_at = 0                    # activated trans. (denominator)

    visited_markings: Dict[Marking, Set[str]] = {}
    visited_prefixes: Set[str] = set()

    for variant_idx, aligned in enumerate(aligned_traces):
        alignment = aligned["alignment"]
        freq = variants[variant_keys[variant_idx]]

        marking = copy(im)
        prefix = None

        # last index referring to a *log* move (to emulate original behaviour)
        idxs = [i for i, m in enumerate(alignment) if m[0][0] != ">>"]
        if not idxs:
            continue
        last_log_idx = idxs[-1]

        for i in range(last_log_idx):
            move = alignment[i]

            # execute the transition on the net (if any)
            if move[0][1] != ">>":
                transition = trans_by_name[move[0][1]]
                marking = semantics.execute(transition, net, marking)

            # update prefix when the move corresponds to a visible activity
            if move[0][0] != ">>":
                activity = move[1][0]
                prefix = activity if prefix is None else f"{prefix},{activity}"

                if prefix not in visited_prefixes:
                    # cache enabled set per marking
                    if marking in visited_markings:
                        enabled_vis = visited_markings[marking]
                    else:
                        enabled_vis = {
                            t.label
                            for t in pn_align_utils.get_visible_transitions_eventually_enabled_by_marking(
                                net, marking
                            )
                            if t.label is not None
                        }
                        visited_markings[marking] = enabled_vis

                    # transitions that *actually* happened (taken from the log) …
                    log_transitions = prefixes.get(prefix, set())
                    # … vs those made possible by the model
                    escaping = enabled_vis.difference(log_transitions)

                    multiplicity = prefix_count.get(prefix, 0)

                    sum_at += len(enabled_vis) * multiplicity
                    sum_ee += len(escaping) * multiplicity

                    visited_prefixes.add(prefix)

    # ------------------------------------------------------------------#
    # 5.  Empty prefix (⊥) handling – 100 % identical to PM4Py code
    # ------------------------------------------------------------------#
    start_acts = set(get_start_activities(log, parameters=parameters))
    enabled_ini = {
        t.label
        for t in pn_align_utils.get_visible_transitions_eventually_enabled_by_marking(
            net, im
        )
        if t.label is not None
    }
    diff_ini = enabled_ini.difference(start_acts)

    n_traces = len(log) if isinstance(log, EventLog) else log[case_id_key].nunique()
    sum_at += len(enabled_ini) * n_traces
    sum_ee += len(diff_ini) * n_traces

    # ------------------------------------------------------------------#
    # 6.  Final ratio
    # ------------------------------------------------------------------#
    if sum_at > 0:
        precision = 1.0 - float(sum_ee) / float(sum_at)

    if debug_level > 0:
        print(
            f"[Align‑ETPrecision‑Aligned]  AT={sum_at}  EE={sum_ee}  precision={precision:.5f}"
        )

    return precision
