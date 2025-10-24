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
from collections import Counter
from math import log2
from typing import Dict, Union, Any, Optional, List
from pm4py.objects.log.obj import EventLog
import pandas as pd
from enum import Enum
from pm4py.util import constants, exec_utils, xes_constants


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    ALPHA = "alpha"


def apply(log: Union[pd.DataFrame, EventLog], parameters: Optional[Dict[Any, Any]] = None) -> List[Dict[str, Any]]:
    """
    Compute information–theoretic metrics used to detect *chaotic activities* in an event log, as defined in

    Tax, Niek, Natalia Sidorova, and Wil MP van der Aalst. "Discovering more precise process models from event logs by filtering out chaotic activities." Journal of Intelligent Information Systems 52.1 (2019): 107-139.

    The result maps each activity to:
    * freq                – absolute frequency #(a,L)
    * entropy             – H(a,L)  (direct entropy)
    * entropy_smooth      – Hₛ(a,L) (Laplace‑smoothed entropy)
    * entropy_gain        – ΔH      (drop in total log‑entropy if *a* is removed)
    * chaotic_score       – simple aggregate = (entropy_smooth+entropy_gain)/2

    Parameters
    -----------------
    log
        Event log or Pandas dataframe
    parameters
        Variant-specific parameters, including:
        - Parameters.ALPHA => Laplace/Lidstone smoothing parameter α. *None* reproduces the raw entropy H(a,L); a typical choice following the paper is α = 1/|A|.
        - Parameters.ACTIVITY_KEY => the attribute to be used as activity. Default: "concept:name"

    Returns
    -----------------
    chaotic_activities
        List of dictionaries, each representing an activity, sorted decreasingly based on the chaotic score (less is better).
    """
    if parameters is None:
        parameters = {}

    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes_constants.DEFAULT_NAME_KEY)
    alpha = exec_utils.get_param_value(Parameters.ALPHA, parameters, None)

    import pm4py

    traces = pm4py.project_on_event_attribute(log, activity_key)

    return chaotic_metrics(traces, alpha=alpha)


def chaotic_metrics(traces, alpha=None):
    """
    Parameters
    ----------
    traces   : list[list[str]]
        The event log where each inner list is a trace (ordered events).
    alpha : float | None
        Laplace/Lidstone smoothing parameter α.
        *None* reproduces the raw entropy H(a,L);
        a typical choice following the paper is α = 1/|A|.

    Returns
    -------
    dict[str, dict]   (activity  →  metrics)
    """
    # ------------------------------------------------------------------ pre‑compute counts
    act_count, pair_count = _pair_counts(traces)
    activities = set(act_count)
    if alpha is None:
        # default α = 1/|A|    (Sec. 3.2, last paragraph)
        alpha = 1.0 / len(activities)

    # ------------------------------------------------------------------ baseline totals
    H_total = total_entropy(traces, alpha=None)  # raw total entropy

    results = {}
    for a in sorted(activities):
        # ------------ H(a,L) and Hₛ(a,L)
        dfr_raw, dpr_raw = _entropy_vectors(a, activities,
                                            act_count, pair_count,
                                            alpha=None)
        dfr_sm, dpr_sm = _entropy_vectors(a, activities,
                                          act_count, pair_count,
                                          alpha=alpha)

        H_a = _entropy(dfr_raw) + _entropy(dpr_raw)
        Hs_a = _entropy(dfr_sm) + _entropy(dpr_sm)

        # ------------ indirect metric: ΔH = H(L) − H(L\{a})
        # Build log without *a*
        log_wo_a = [[x for x in t if x != a] for t in traces]
        # remove empty traces to avoid zero‑division
        log_wo_a = [t for t in log_wo_a if t]
        H_total_wo = total_entropy(log_wo_a, alpha=None)
        entropy_gain = H_total - H_total_wo  # larger ⇒ bigger reduction

        chaotic_score = (Hs_a + entropy_gain) / 2.0

        results[a] = {
            "freq": act_count[a],
            "entropy": H_a,
            "entropy_smooth": Hs_a,
            "entropy_gain": entropy_gain,
            "chaotic_score": chaotic_score
        }

    results = [{'activity': a, **data} for a, data in results.items()]
    results.sort(key=lambda x: (-x["chaotic_score"], -x["freq"], x["activity"]))

    return results


def _entropy(probabilities):
    """Shannon entropy of a discrete distribution (list/iterable of pᵢ)."""
    return -sum(p * log2(p) for p in probabilities if p > 0.0)


def _pair_counts(traces):
    """
    Return:
        act_count : Counter of single‑activity frequencies  #(a,L)
        pair_count: Counter of directly‑follows pairs       #( (a,b), L )
    """
    act_count = Counter()
    pair_count = Counter()

    for trace in traces:
        prev = constants.DEFAULT_ARTIFICIAL_START_ACTIVITY
        for act in trace:
            act_count[act] += 1
            pair_count[(prev, act)] += 1
            prev = act
        pair_count[(prev, constants.DEFAULT_ARTIFICIAL_END_ACTIVITY)] += 1
    return act_count, pair_count


def _entropy_vectors(act, activities, act_count, pair_count, alpha=None):
    """
    Build the directly‑precedes (dpr) and directly‑follows (dfr) probability
    vectors for *act*.  If alpha is given (≥0) Laplace/Lidstone smoothing
    is applied as in Eq. (1) of the paper.
    """
    k = len(activities) + 1  # +1 for END / START sentinel
    denom = act_count[act]
    if alpha is not None:  # Laplace denominator
        denom += alpha * k

    # --- directly‑follows -------------------------------------------------
    dfr = []
    for b in activities | {constants.DEFAULT_ARTIFICIAL_END_ACTIVITY}:
        num = pair_count.get((act, b), 0)
        p = (alpha + num) / denom if alpha is not None else num / denom
        dfr.append(p)

    # --- directly‑precedes ------------------------------------------------
    dpr = []
    for b in activities | {constants.DEFAULT_ARTIFICIAL_START_ACTIVITY}:
        num = pair_count.get((b, act), 0)
        # note: for precedes we smooth with same alpha and k
        p = (alpha + num) / denom if alpha is not None else num / denom
        dpr.append(p)

    return dfr, dpr


def total_entropy(traces, alpha=None):
    """Return Σₐ H(a,L) or Σₐ Hₛ(a,L)."""
    act_count, pair_count = _pair_counts(traces)
    activities = set(act_count)
    total = 0.0
    for a in activities:
        dfr, dpr = _entropy_vectors(a, activities, act_count, pair_count, alpha)
        total += _entropy(dfr) + _entropy(dpr)
    return total
