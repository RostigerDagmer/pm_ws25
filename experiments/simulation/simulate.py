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


def simulate(net_tensors, M0, Mf, labels, steps=100):
    pre, post = net_tensors
    M = M0.clone()
    log = []
    for _ in range(steps):
        enabled = (M >= pre).all(dim=1)
        idx = enabled.nonzero(as_tuple=False).flatten()
        if len(idx) == 0:
            break
        t = idx[torch.randint(len(idx), (1,))]
        M = M - pre[t] + post[t]
        label = labels[t]
        if label != "":
            log.append(label)
        if torch.equal(M, Mf):
            break
    return log


# # if __name__ == "__main__":
# # from experiments.simulation.models import sample_net
# import sys

# sys.path.append('../..')
# from models import sample_net

# dist_params = {
#     "op": lambda: torch.distributions.Categorical(
#         torch.tensor([0.3, 0.3, 0.3, 0.1])
#     ).sample(),
#     "seq_len": lambda: torch.distributions.Poisson(4).sample().int(),
#     "p_stop": lambda d: torch.distributions.Bernoulli(
#         0.2 + 0.1 * d
#     ).sample(),  # deeper → likelier to stop
# }
# stnet = sample_net(dist_params)
# print(stnet)
# N = stnet.to_tensor()
# print(N)
# from pm4py.vis import view_petri_net

# view_petri_net(stnet.net, stnet.im, stnet.fm)

# # %%
# N["M0"].shape

# # %%
# print(N["pre"].shape)
# print(N["post"].shape)

# # %%
# import matplotlib.pyplot as plt

# plt.imshow(N["pre"].float())
# plt.show()
# plt.imshow(N["post"].float())
# plt.show()
# # %%
# simulate((N['pre'], N['post']), N['M0'], N['Mf'], N['labels'])
# # %%
# simulate((N['pre'], N['post']), N['M0'], N['Mf'], N['labels'])
# # %%

# # %%

# %%
