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
from pm4py.pm4py.objects.log.obj import EventLog, Trace


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
):
    pre, post = net_tensors
    n_trans, n_places = pre.shape
    device = pre.device

    labels_t = torch.tensor(list(range(len(labels))), device=device)
    print(f"labels_t: {labels_t}")

    silent_mask = torch.tensor(
        [label == "" for label in labels], dtype=torch.bool, device=device
    )
    print(f"silent_mask: {silent_mask}")

    # broadcast initial marking
    M = M0.expand(batch_size, n_places).clone()

    # default uniform weights per transition
    if weights is None:
        weights = torch.ones(n_trans, device=device, dtype=torch.float)

    logs = -torch.ones((batch_size, steps), dtype=torch.long, device=device)
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for step in range(steps):
        # enabled[b,t] = (M[b] >= pre[t]).all(p)
        enabled = (M.unsqueeze(1) >= pre).all(dim=2)  # [B, T]

        # zero out probs where not enabled
        probs = enabled.float() * weights  # [B, T]
        probs = torch.nn.functional.normalize(probs, dim=1)
        # sample one transition per batch row
        # torch.multinomial expects non-negative rows that sum to 1
        t_idx = torch.multinomial(probs, 1).squeeze(1)  # [B]

        # build delta = post - pre
        delta = post[t_idx] - pre[t_idx]
        M = M + delta  # broadcast add per row

        # record label if visible
        visible = ~silent_mask[t_idx]
        logs[torch.arange(batch_size, device=device), step] = (
            labels_t[t_idx] + 1
        ) * visible.long() - 1

        # check completion
        done |= torch.all(M == Mf, dim=1)
        if done.all():
            break

    if compact:
        # logs: [B, steps] tensor of label IDs, 0 for silent, pad_id for beyond max_steps
        visible = logs != -1  # mask non-silent transitions
        compacted = -torch.ones_like(logs)
        lengths = visible.sum(dim=1)

        for b in range(logs.size(0)):
            compacted[b, : lengths[b]] = logs[b, visible[b]]

        return compacted
    return logs


def apply_labels(log: torch.Tensor, labels: list[str]) -> EventLog:
    def apply_label(tok: torch.Tensor) -> str:
        tok = tok.item()
        if tok >= 0:
            return labels[tok]
        else:
            return "τ"

    return EventLog([Trace(list(map(apply_label, trace))) for trace in log])
