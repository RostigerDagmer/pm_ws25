experiments/simulate/simulator.py

---

For a safe Petri net
$N=(P,T,F,W,M0)$
$P$: set of places, size = n
$T$: set of transitions, size = m
$W_{t,p}^-$: weight of arc p → t (consumption)
$W_{t,p}^+$: weight of arc t → p (production)
$M_t ∈ N|P|$: current marking (token vector)

A transition $t$ is enabled if
$$M_t \geq W_t^-$$

(element-wise comparison).
When it fires, the new marking is
$$M_{t+1}=M_t - W_t^- + W_t^+$$

The entire token game is essentially two lines of vector math.

#### Sampling

We take a parametrizable but fixed number of steps using the rule above.
This is parallel in a batch dimension and consists of simple matrix vector operations.

Using torch we can get an accelerator-agnostic version of simulate in ~40 lines of code.
By not materializing transition names as strings until the very end we avoid device/host synch and excessive memory use until we materialize.

By doing so we achieve execution times ~300x faster on sufficiently large nets at sufficiently large event log sizes.

