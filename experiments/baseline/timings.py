# %%
import time
import pandas as pd
from pm4py.algo.conformance.alignments.petri_net import algorithm as align_alg
from pm4py.algo.conformance.alignments.petri_net import variants as variants
import pm4py.objects.petri_net.utils.align_utils as align_utils
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import Marking, PetriNet


def measure_alignment(
    pn: PetriNet,
    im: Marking,
    fm: Marking,
    log: EventLog,
    variant=align_alg.Variants.VERSION_STATE_EQUATION_A_STAR,
    cost=None,
) -> dict[str, float | int]:
    params = {
        align_alg.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True,
    }
    if cost is not None:
        params[align_alg.Parameters.PARAM_COST_FUNCTION] = cost

    start = time.time()
    aligned_traces = align_alg.apply_log(
        log, pn, im, fm, parameters=params, variant=variant
    )
    end = time.time()
    metrics = {
        "runtime_sec": end - start,
        "num_traces": len(aligned_traces),
        "total_cost": sum(a["cost"] for a in aligned_traces),
        # optional: add custom instrumentation from pm4py.algo.conformance.alignments.utils
    }
    return metrics


if __name__ == "__main__":
    # Example usage
    # pn, im, fm = pnml_importer.apply("model.pnml")
    # log = xes_importer.apply("log.xes")

    from experiments.simulation.driver import generate_dataset
    from pm4py.util.lp import solver as lp_solver
    import numpy as np
    import sys

    def my_derive_heuristic(incidence_matrix, cost_vec, x, t, h):
        x_prime = x.copy()
        x_prime[incidence_matrix.transitions[t]] -= 1
        print("x_prime: ", x_prime)
        return max(0, h - cost_vec[incidence_matrix.transitions[t]]), x_prime

    def exact_heuristic(
        sync_net,
        a_matrix,
        h_cvx,
        g_matrix,
        cost_vec,
        incidence_matrix,
        marking,
        fin_vec,
        variant,
        use_cvxopt=False,
        strict=True,
    ):
        # Exact copy from align_utils just to test.
        print("helo")
        m_vec = incidence_matrix.encode_marking(marking)
        b_term = [i - j for i, j in zip(fin_vec, m_vec)]
        b_term = np.matrix([x * 1.0 for x in b_term]).transpose()

        if not strict:
            g_matrix = np.vstack([g_matrix, a_matrix])
            h_cvx = np.vstack([h_cvx, b_term])
            a_matrix = np.zeros((0, a_matrix.shape[1]))
            b_term = np.zeros((0, b_term.shape[1]))

        if use_cvxopt:
            # not available in the latest version of PM4Py
            from cvxopt import matrix

            b_term = matrix(b_term)

        parameters_solving = {"solver": "glpk"}

        sol = lp_solver.apply(
            cost_vec,
            g_matrix,
            h_cvx,
            a_matrix,
            b_term,
            parameters=parameters_solving,
            variant=variant,
        )
        prim_obj = lp_solver.get_prim_obj_from_sol(sol, variant=variant)
        points = lp_solver.get_points_from_sol(sol, variant=variant)

        prim_obj = prim_obj if prim_obj is not None else sys.maxsize
        points = (
            points if points is not None else [0.0] * len(sync_net.transitions)
        )

        return prim_obj, points

    # monkey-patch the function globally
    align_utils.__derive_heuristic = my_derive_heuristic
    align_utils.__compute_exact_heuristic_new_version = exact_heuristic

    # simulate a log on a generated petrinet
    pn, im, fm, log = next(
        generate_dataset(n_models=1, parameters={"no_traces": 1000})
    )

    results = []
    for variant in [
        align_alg.Variants.VERSION_STATE_EQUATION_A_STAR,
        align_alg.Variants.VERSION_DISCOUNTED_A_STAR,
    ]:
        m = measure_alignment(pn, im, fm, log, variant=variant)
        m["variant"] = variant
        results.append(m)

    print(pd.DataFrame(results))
# %%
