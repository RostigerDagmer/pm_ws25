# %%
import random
from pm4py.vis import view_petri_net
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from enum import Enum
from experiments.simulation.structured_net import StructuredNet
import logging
import uuid
import torch
from util.distributions import make_distribution
import hashlib
import json


def seq(name, labels):
    if not labels:
        raise ValueError(
            "labels must be a non-empty list of transition labels"
        )
    net = PetriNet(name)

    # create places
    places = [PetriNet.Place(f"p_{l}") for l in labels + [f"{labels[-1]}_end"]]
    for p in places:
        net.places.add(p)

    # create transitions and arcs
    for i, lbl in enumerate(labels):
        t = PetriNet.Transition(f"t_{lbl}", lbl)
        net.transitions.add(t)
        petri_utils.add_arc_from_to(places[i], t, net)
        petri_utils.add_arc_from_to(t, places[i + 1], net)

    # define initial/final markings
    im = Marking({places[0]: 1})
    fm = Marking({places[-1]: 1})

    name = hashlib.sha1(json.dumps(labels).encode("utf-8")).hexdigest()

    return StructuredNet(f"block_{name}", net, im, fm)


class Composition(Enum):
    XOR = 0
    AND = 1
    LOOP = 2
    SEQ = 3

    def compose(
        self, left: StructuredNet, right: StructuredNet
    ) -> StructuredNet:
        match self:
            case Composition.XOR:
                logging.debug(f"Composing:\n {left} ^ {right}")
                return left ^ right

            case Composition.AND:
                logging.debug(f"Composing:\n {left} & {right}")
                return left & right

            case Composition.LOOP:
                logging.debug(f"Looping:\n {left} @ {right}")
                return left @ right

            case Composition.SEQ:
                logging.debug(f"Composing:\n {left} >> {right}")
                return left >> right


def random_block_structured(
    num_blocks=3,
    xor_prob=0.3,
    and_prob=0.3,
    loop_prob=0.1,
    seq_prob=0.1,
    p_depth=0.1,
    max_depth=3,
) -> StructuredNet:

    if max_depth <= 0 or random.random() < p_depth:
        labels = [f"{uuid.uuid4().hex}" for i in range(num_blocks)]
        return seq("seq_model", labels)

    comp_op = random.choices(
        [e.value for e in Composition],
        [xor_prob, and_prob, loop_prob, seq_prob],
        k=1,
    )[0]
    logging.debug(f"comp_op: {comp_op}")

    left = random_block_structured(
        num_blocks,
        xor_prob,
        and_prob,
        loop_prob,
        seq_prob,
        p_depth,
        max_depth - 1,
    )
    right = random_block_structured(
        num_blocks,
        xor_prob,
        and_prob,
        loop_prob,
        seq_prob,
        p_depth,
        max_depth - 1,
    )
    logging.debug(f"left: {left}")
    logging.debug(f"right: {right}")
    comp = Composition(comp_op)
    logging.debug(f"comp: {comp}")

    prod = Composition(comp_op).compose(left, right)
    logging.debug(f"prod: {prod}")
    return prod


def sample_net(
    dist_params, depth=0, min_depth=None, max_depth=None, generator=None
) -> StructuredNet:
    # depth termination
    dists = {
        k: make_distribution(v, depth=depth) for k, v in dist_params.items()
    }
    stop = bool(dists["p_stop"].sample(generator=generator).item()) and (
        min_depth is None or depth >= min_depth
    )
    if stop or (max_depth and depth >= max_depth):
        seq_len = int(dists["seq_len"].sample(generator=generator).item())
        if seq_len == 0:
            name = f"tau_{torch.randint(0, 2**32, (1,), dtype=torch.int64, generator=generator).item():x}"
            return StructuredNet.tau(name)
        random_ints = torch.randint(
            0, 2**32, (seq_len,), dtype=torch.int64, generator=generator
        )
        labels = [f"{x.item():x}" for x in random_ints]
        return seq("seq", labels)

    op = dists["op"].sample(generator=generator).item()
    comp = Composition(op)

    if comp in [Composition.XOR, Composition.AND]:
        width = int(dists["width"].sample(generator=generator).item())
        width = max(2, width)  # Ensure at least binary
        subnets = [
            sample_net(
                dist_params,
                depth + 1,
                min_depth,
                max_depth,
                generator=generator,
            )
            for _ in range(width)
        ]

        if comp == Composition.XOR:
            return StructuredNet.n_xor(subnets)
        else:  # AND
            return StructuredNet.n_and(subnets)

    # Binary ops (LOOP, SEQ)
    left = sample_net(
        dist_params, depth + 1, min_depth, max_depth, generator=generator
    )
    right = sample_net(
        dist_params, depth + 1, min_depth, max_depth, generator=generator
    )

    return comp.compose(left, right)


if __name__ == "__main__":
    from util.distributions import (
        BernoulliDepthLinearSpec,
        CategoricalSpec,
        PoissonSpec,
    )
    from util.rng import RNG

    logging.basicConfig(level=logging.DEBUG)
    RNG.initialize(4)

    dist_params = {
        "op": CategoricalSpec([0.1, 0.5, 0.3, 0.1]),
        "seq_len": PoissonSpec(4),
        "p_stop": BernoulliDepthLinearSpec(base=0.15, slope=0.1),
        "width": PoissonSpec(10),
    }
    stnet = sample_net(
        dist_params, max_depth=5, generator=RNG.torch_generator()
    )
    view_petri_net(stnet.net, stnet.im, stnet.fm)

    # perform some random sampling
    t = torch.randint(0, 2**32, (1000,), dtype=torch.int64)

    stnet_resample = sample_net(
        dist_params, max_depth=5, generator=RNG.torch_generator()
    )
    view_petri_net(stnet_resample.net, stnet_resample.im, stnet_resample.fm)


# %%
