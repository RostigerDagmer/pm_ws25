Designing the Required Activities Heuristic

Goal: Required Activities Heuristic: Estimate the remaining cost by looking at:

  - Activities that are required in the model to reach the final marking from the current marking, and
  - Remaining events in the trace that still need to be matched.


Conceptually:
1. From the model side, compute a set / multiset of required activities that must still occur (at least once) to reach the final marking.
2. From the trace side, consider the remaining segment of the trace from the current index onward.
3. Compare these two:
    - Some required activities will be matched by remaining trace events via sync moves (cost 0 or low).
    - Some required activities may need model moves only (if they do not appear in the remainder of the trace).
    - Some remaining trace events may require log moves only (if they cannot be matched to visible model transitions still reachable from here).
4. Count the minimum number of moves implied by these mismatches and unmatched elements; multiply by suitable per-move costs.
This should be more informed than “remaining events only” but must still be admissible.
Below is a concrete, simple but admissible design that is implementable and still efficient.

we can try to implement this by following these steps:
1. precomputation step (once per syn product net)
    - compute idx for each place (for each place we know the position in trace) -> helps with idetifying which suffix of the trace is still relevant
    - compute a reachability analysis over all place in spn, a set of reachable visible labels (activities) in the model from current place to the final marking
      - for each place (p) in sspn:
        - look for outgoing transitions
        - for each transition, extract the model label
        - propgate backwards from the final marking to all reachable places:
            - if a visible model transition label <a> appears in some path from p to the final marking, add a to required_model_labels[p]
    - we can be conservative: mark all visible labels reachable in any path (overapproximation keeps the admissability)
    
2. Heuristic for a marking
         
    
