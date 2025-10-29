# Efficient Alignments with the Extended Marking Equation

This document explains an advanced technique for computing optimal alignments. It builds directly on the A\*/State Equation method, addressing its primary weakness to dramatically improve performance.

## 1. Recap: The Problem with the Standard Heuristic

The foundational approach uses the A\* algorithm with a heuristic $h(n)$ derived from the standard **state equation**:

$\vec{m}_{f} = \vec{m} + C \cdot \vec{x}$

This heuristic is weak because it **ignores the enabling condition** of Petri nets. It only ensures that, *in total*, the token counts balance out. It has no concept of *ordering* or whether a transition is *actually* fireable.

This leads to A\* exploring huge state spaces, especially in models with parallelism and swapped events, because the heuristic provides an overly optimistic (and inaccurate) underestimate.

## 2. Core Idea 1: The Extended Marking Equation

To improve the heuristic, we must add constraints that model the **enabling condition**. The key is to force the LP solver to recognize that *tokens must be present in a place **before** a transition can consume them*.

This is achieved using two new concepts:

1.  **Consumption Matrix ($C^-$):** This matrix describes *only* the tokens *consumed* by transitions (the input arcs). This is different from the incidence matrix $C$, which describes the *net change* (output - input).
2.  **Extended Marking Equation:** We can now state the enabling condition as an equation. For a firing sequence $m_1 \xrightarrow{\theta_1} m_2 \xrightarrow{t} \dots$, the transition $t$ can only fire at marking $m_2$ if $m_2$ has enough tokens for $t$ to consume.
    * Expressed mathematically: $\vec{m_2} + C^- \cdot \vec{1}_t \ge \vec{0}$
    * We know $\vec{m_2} = \vec{m_1} + C \cdot \vec{\theta_1}$ (from the standard equation).
    * Substituting this in gives the **Extended Marking Equation**:
        **$\vec{m_1} + C \cdot \vec{\theta_1} + C^- \cdot \vec{1}_t \ge \vec{0}$**

This new constraint is powerful. It links the firing sequence $\theta_1$ to the enabling of the *next* transition $t$. This re-introduces the concept of *ordering* that the standard equation was missing.

## 3. Core Idea 2: Splitting the Trace

How do we use this new equation? We need to "guess" the split points (like $t$ in the example above). The **event log trace provides a perfect, natural way to split the problem**.

We know the alignment must, at minimum, explain every event in the remaining trace. We can split the remaining trace $\sigma$ into $k$ sub-traces:

$\sigma = \sigma_1 \circ \sigma_2 \circ \dots \circ \sigma_k$

We then model the *full* remaining firing sequence $\gamma$ as $k$ parts, where each part $\gamma_a$ *must* start with a transition $\vec{y_a}$ that corresponds to the first event of the sub-trace $\sigma_a$. The rest of the firings in that part are $\vec{x_a}$.

## 4. The New, Tighter Heuristic ($h^{LP, k}$)

We can now build a new, much stronger LP heuristic ($h^{LP, k}$) by combining these ideas:

* **Minimize:** $\vec{c}^T \cdot (\sum \vec{x_a} + \sum \vec{y_a})$ (the total cost)
* **Subject to:**
    1.  **Standard Equation:** $\vec{m} + C \cdot (\sum \vec{x_a} + \sum \vec{y_a}) = \vec{m_f}$
        *(The total firings must still reach the end.)*
    2.  **Extended Equation (The new constraint!):** For *each* split point $a$:
        $\vec{m} + C \cdot (\sum_{b<a} (\vec{x_b}+\vec{y_b})) + C^- \cdot \vec{y_a} \ge \vec{0}$
        *(The firings of all previous parts must provide enough tokens to enable the *first* transition $\vec{y_a}$ of the current part.)*
    3.  **Trace Splitting Rules:** Additional constraints to ensure each $\vec{y_a}$ is exactly one transition corresponding to the start of $\sigma_a$.

This new heuristic $h^{LP, k}$ is **much more accurate**. It provides a "tighter" underestimate, meaning $h^{LP, k} \ge h^{LP}$ (the old heuristic). This new value is much closer to the *true* remaining cost, so it guides the A\* search far more effectively, drastically reducing the number of states it needs to explore.


## 5. The Algorithm: Incremental A\*

There's a new problem: this $h^{LP, k}$ heuristic is **very slow to compute**. The LP is $k$ times larger and more complex. Re-computing it at every state is infeasible, and the simple "re-use" optimization from the first approach no longer works in the same way.

The solution is an **Incremental A\*** algorithm that gets the best of both worlds:

1.  **Start Fast:** Begin the A\* search using the *old, fast* heuristic (effectively $k=1$). Compute the solution vector $\vec{z}$ for the *entire* remaining trace.
2.  **Run with Re-use:** Run the A\* search, using the fast re-use optimization ($h(m') = h(m) - c(t)$) as long as the search "follows" the plan in $\vec{z}$.
3.  **Detect Failure:** The search will eventually get "stuck." It will reach a state $m'$ where it must explore a transition $t$, but the original plan $\vec{z}$ didn't account for this (i.e., $\vec{z}(t) < 1$). **This state $m'$ proves that the $k=1$ heuristic's plan was unrealizable.**
4.  **Increment and Restart:** The algorithm now *stops* and **restarts the entire A\* search from the beginning**.
5.  **Add a Split:** This time, it computes the heuristic using $k=2$. It adds a *new split point* at the event in the trace where the previous search got stuck.
6.  **Run Again:** The A\* search now runs with the $k=2$ heuristic. This heuristic is slower to compute *once*, but it's more accurate, so the *overall* search explores fewer states.

This process repeats. If the $k=2$ search gets stuck, it restarts with $k=3$, and so on.

This incremental approach combines the **raw speed** of the simple heuristic (for the "easy" parts of the process) with the **pinpoint accuracy** of the extended marking equation (which is only applied *exactly* where the simple heuristic fails). This results in a massive speedup for computing optimal alignments.
