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


