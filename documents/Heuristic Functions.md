pm4py/algo/conformance/alignments/petri_net/variants/discounted_a_star.py
pm4py/algo/conformance/alignments/petri_net/variants/state_equation_a_star.py

To systematically search for all possible alignments, one constructs a **product Petri net (synchronous product)** of:

- the Petri net representing the process model, and
    
- a linear “trace net” representing the recorded trace.
  

This product net encodes _both_ the behavior of the model and the observed trace, with transitions corresponding to synchronized steps or deviations.

The search space of possible alignments is combinatorially huge, so we frame it as a **shortest-path problem**:

- Each state is a pair (marking in model, position in trace).
    
- Each transition (a move) incurs a _cost_ — e.g.,
    
    - 0 for synchronous moves (both model and log agree),
        
    - positive cost for deviations (model-only or log-only moves).
    
- The goal is to reach the _final marking_ of the model and the _end_ of the trace with _minimal total cost_.

#### Existing

pm4py/objects/petri_net/utils/align_utils.py
- [Levensthein](https://en.wikipedia.org/wiki/Levenshtein_distance)
- Discounted [Edit Distance](https://en.wikipedia.org/wiki/Edit_distance)

Anything else?
