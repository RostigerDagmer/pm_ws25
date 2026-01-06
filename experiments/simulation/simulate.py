"""
Replacement for pm4py.algo.simulation.playout.petri_net.algorithm (extremely slow)

For a safe Petri net
$N=(P,T,F,W,M0)$
$P$: set of places, size = n
$T$: set of transitions, size = m
$W_{t,p}^-$: weight of arc p → t (consumption)
$W_{t,p}^+$: weight of arc t → p (production)
$M_t ∈ N|P|$: current marking (token vector)

A transition $t$ is enabled if
$$M_t \\geq W_t^-$$

(element-wise comparison).
When it fires, the new marking is
$$M_{t+1}=M_t - W_t^- + W_t^+$$

The entire token game is essentially two lines of vector math.
"""

# %%
import torch
from typing import Optional
from pm4py.objects.log.obj import EventLog, Trace, Event


def simulate(
    net_tensors,
    M0,
    Mf,
    labels,
    weights: Optional[torch.Tensor] = None,
    steps=100,
    device="cpu",
):
    pre, post = net_tensors
    n_trans, n_places = pre.shape
    if weights is None:
        weights = torch.ones(n_trans, device=device, dtype=torch.float)
    M = M0.clone()
    log = []
    for _ in range(steps):
        enabled = (M >= pre).all(dim=1)
        idx = enabled.nonzero(as_tuple=False).flatten()
        print(idx)
        if len(idx) == 0:
            break
        probs = weights[idx].clone()
        probs = probs / probs.sum()
        t = idx[torch.multinomial(probs, 1)]
        M = M - pre[t] + post[t]
        label = labels[t]
        if label != "":
            log.append(label)
        if torch.equal(M, Mf):
            break
    return log


def simulate_batch(
    net_tensors,
    M0,
    Mf,
    labels,
    weights: Optional[torch.Tensor] = None,
    steps: int = 100,
    batch_size: int = 128,
    compact: bool = True,
    record_enabled_history: bool = False,
    generator: Optional[torch.Generator] = None,
):
    pre, post = net_tensors
    n_trans, n_places = pre.shape
    device = pre.device

    silent_mask = torch.tensor(
        [label == "" for label in labels], dtype=torch.bool, device=device
    )

    # broadcast initial marking
    M = M0.expand(batch_size, n_places).clone()

    # default uniform weights per transition
    if weights is None:
        weights = torch.ones(n_trans, device=device, dtype=torch.float)

    transitions = -torch.ones(
        (batch_size, steps), dtype=torch.long, device=device
    )
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    enabled_hist = None
    if record_enabled_history:
        enabled_hist = torch.zeros(
            (batch_size, steps, n_trans), dtype=torch.bool, device=device
        )
    visible_hist = torch.zeros(
        (batch_size, steps), dtype=torch.bool, device=device
    )

    for step in range(steps):
        # enabled[b,t] = (M[b] >= pre[t]).all(p)
        enabled = (M.unsqueeze(1) >= pre).all(dim=2)  # [B, T]

        if record_enabled_history:
            enabled_hist[:, step, :] = enabled

        # zero out probs where not enabled
        probs = enabled.float() * weights  # [B, T]

        # Identify active rows (at least one enabled transition)
        active = enabled.any(dim=1) & (~done)  # [B]

        # Handle all-zero probs to avoid multinomial error
        # We set a dummy probability for deadlocked rows, but we won't use the result
        probs[~active, 0] = 1.0

        probs_sum = probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
        probs = probs / probs_sum
        # sample one transition per batch row
        # torch.multinomial expects non-negative rows that sum to 1
        t_idx = torch.multinomial(probs, 1, generator=generator).squeeze(
            1
        )  # [B]

        # build delta = post - pre
        # build delta = post - pre
        delta = post[t_idx] - pre[t_idx]

        # Only update M for active rows
        M[active] = M[active] + delta[active]

        # record label if visible
        visible = (~silent_mask[t_idx]) & active
        visible_hist[:, step] = visible

        transitions[visible, step] = t_idx[visible]

        # check completion
        done |= torch.all(M == Mf, dim=1)
        if done.all():
            break

    if compact:
        # logs: [B, steps] tensor of label IDs, 0 for silent, pad_id for beyond max_steps
        compacted = -torch.ones_like(transitions)
        compacted_enabled = None
        if record_enabled_history:
            compacted_enabled = torch.zeros_like(
                enabled_hist
            )  # bool padded with False
        lengths = visible_hist.sum(dim=1)

        for b in range(transitions.size(0)):
            idx = visible_hist[b].nonzero(as_tuple=False).squeeze(1)
            L = idx.numel()
            compacted[b, :L] = transitions[b, idx]
            if record_enabled_history:
                compacted_enabled[b, :L] = enabled_hist[b, idx]

        return compacted, compacted_enabled, lengths

    return transitions, enabled_hist, visible_hist


def apply_labels(log: torch.Tensor, labels: list[str]) -> EventLog:
    """
    Converts a tensor of token indices into an EventLog of Traces of Events.
    Filters out silent transitions (indices < 0).
    """
    ret = EventLog()
    for trace_tensor in log:
        events = []
        for tok in trace_tensor:
            tok_idx = tok.item()
            if tok_idx >= 0:
                # Create an Event object (dict-like)
                events.append(Event({"concept:name": labels[tok_idx]}))
        ret.append(Trace(events))
    return ret


if __name__ == "__main__":
    from util.distributions import (
        BernoulliDepthLinearSpec,
        CategoricalSpec,
        PoissonSpec,
    )
    from experiments.simulation.models import sample_net
    from util.rng import RNG
    import matplotlib.pyplot as plt

    RNG.initialize(4)

    dist_params = {
        "op": CategoricalSpec([0.1, 0.5, 0.3, 0.1]),
        "seq_len": PoissonSpec(4),
        "p_stop": BernoulliDepthLinearSpec(base=0.15, slope=0.1),
        "width": PoissonSpec(10),
    }
    stnet = sample_net(
        dist_params, max_depth=3, generator=RNG.torch_generator()
    )

    tnet = stnet.to_tensor()

    log, hist, vis = simulate_batch(
        (tnet.pre, tnet.post),
        tnet.M0,
        tnet.Mf,
        tnet.labels,
        steps=10,
        batch_size=4,
        compact=True,
        record_enabled_history=True,
    )
    print(hist)
    print(hist.shape)
    plt.imshow(hist.cpu().float()[0], aspect="auto")
    plt.show()
    print(vis)
    print(log)
