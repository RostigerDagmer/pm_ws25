```
pm4py/algo/conformance/alignments/petri_net/variants/discounted_a_star.py
pm4py/algo/conformance/alignments/petri_net/variants/state_equation_a_star.py
```

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

---
## Existing

```
pm4py/objects/petri_net/utils/align_utils.py
```

- [Levensthein](https://en.wikipedia.org/wiki/Levenshtein_distance)
- Discounted [Edit Distance](https://en.wikipedia.org/wiki/Edit_distance)

_Anything else?_

---
##### **Label-based matching heuristics (Levenshtein-type lower bounds)**

Compute:

$$h(M,i) = \text{edit\_LB}(\sigma[i:], L(M \Rightarrow M_f))$$

where $L(M \Rightarrow M_f)$ is any _over-approximation_ of all model label sequences reachable from M and $\sigma[i:]$ is the current remaining trace suffix.

If you relax concurrency, the lower bound on edit distance is admissible.

Implementation trick:

- Approximate $L(M \Rightarrow M_f)$ by the **set of enabled labels** or by an **automaton abstraction** of the model.
    
- Compute minimal unmatched symbols.

Examples:

- $h(M,i) = |\text{remaining labels in trace not in enabled set}|$
    
- $h(M,i) = \text{edit-distance}(\text{trace suffix}, \text{any topological label sequence})$ truncated to an underestimate.

---
##### **Relaxed model heuristics** (the most principled class)

Idea: use the **relaxation of the Petri net** where synchronizations are easier — any cost in the relaxation ≤ true cost → admissible.
###### **Relaxed reachability heuristic**

Ignore token consumption (treat model as fully concurrent), so any transition whose preplaces _might eventually_ be marked is considered enabled.

Compute minimal \#mismatched transitions to reach a labeling compatible with the trace suffix.

Used in van der Aalst’s “cost-based alignment with admissible heuristics” papers.
###### **State-space relaxation (delete relaxation)**

Analogous to classical planning: remove some constraints so planning becomes easier.

In practice:

- compute a **shortest path in the relaxed model** (no place capacities, all transitions fireable).
    
- cost of that path → lower bound for true cost.
###### **Place relaxation** (abstract the Petri net)

Precompute the cost-to-go in small _subnets_ (pattern databases). See next.

---
##### **Pattern database (PDB) heuristics**

This is directly borrowed from heuristic search in planning.

Pick a subset of transitions/places $P’$, project the Petri net onto that subset (ignore others).

Compute exact minimal cost-to-go in this _smaller_ net, store it as $h_{P’}(M)$.

For multiple disjoint projections, you can sum them for a stronger heuristic:

$$h(M,i) = \sum_{P’ \in \mathcal{P}} h_{P’}(M|_{P’})$$

Admissible if projections are disjoint.
These are often the _strongest admissible_ heuristics used in practice.

---
##### **Precomputed marking-distance heuristics**

Compute, once for the model alone, the **graph distance between each reachable marking and the final marking**, where
*cost = minimal \#visible transitions*.

Store this as $d_M(M)$.

Then:

$$h(M,i) = \max(0, d_M(M) - (|\sigma| - i))$$

Cheap and admissible under standard costs.

---
##### **Compositional / hybrid heuristics**

Combine lower bounds safely:

If $h_1, h_2, …, h_k$ are admissible:

$$h(M,i) = \max_j h_j(M,i)$$

is also admissible (best-of).

This lets you mix label-based and structure-based heuristics without losing guarantees.

---

| **Class**                         | **Admissible?** | **Depends on model?** | **Depends on trace?** | **Typical strength**       |
| --------------------------------- | --------------- | --------------------- | --------------------- | -------------------------- |
| Zero (-> Dijkstra's)              | ✅               | ✗                     | ✗                     | Weak                       |
| Remaining trace length            | ✅               | ✗                     | ✅                     | Weak                       |
| Model distance                    | ✅               | ✅                     | ✗                     | Weak–Medium                |
| Label-based lower bounds          | ✅ (if relaxed)  | ✅                     | ✅                     | Medium                     |
| Relaxed model (delete relaxation) | ✅               | ✅                     | ✅                     | Strong                     |
| Pattern DB                        | ✅               | ✅                     | ✗                     | Strong                     |
| Hybrid (max of others)            | ✅               | ✅                     | ✅                     | Medium–Strong              |
| Learned (unconstrained)           | ❌ (usually)     | ✅                     | ✅                     | Potentially strong, unsafe |




# Integration

All heuristic behavior is controlled by functions in:  
`pm4py/objects/petri_net/utils/align_utils.py`

Look there for:

```python
__compute_exact_heuristic_new_version(...)
__derive_heuristic(...)
__trust_solution(...)
```

That’s the layer you’d edit or extend if you want to compare “different heuristics.”

For example, you might create:

```python
__compute_simple_heuristic_remaining_tokens(...)
```

or

```python
__compute_pessimistic_heuristic(...)
```

Then expose a switch in parameters:

```python
heuristic_variant = exec_utils.get_param_value("heuristic_variant", parameters, "state_equation")
if heuristic_variant == "tokens":
    h = compute_token_distance(sync_net, current_marking, fin)
elif heuristic_variant == "trace_length":
    h = (remaining_trace_len * min_cost)
else:
    h, x = utils.__compute_exact_heuristic_new_version(...)
```

That’s the clean insertion point.

**Alternatively**: Monkey-patch (current)

```python
import pm4py.objects.petri_net.utils.align_utils as align_utils

def my_derive_heuristic(...) -> ...
def exact_heuristic(...) -> ...

align_utils.__derive_heuristic = my_derive_heuristic
align_utils.__compute_exact_heuristic_new_version = exact_heuristic
```