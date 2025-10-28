# %%
import random
from pm4py.vis import view_petri_net
from pm4py.pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.pm4py.objects.petri_net.utils import petri_utils
from typing import TypeAlias
from enum import Enum
from experiments.simulation.structured_net import StructuredNet
import logging
import uuid

logging.getLogger(None)
logging.basicConfig(level=logging.DEBUG)


def make_sequence(name, labels):
    if not labels:
        raise ValueError(
            "labels must be a non-empty list of transition labels"
        )
    net = PetriNet(name)

    # create places
    places = [PetriNet.Place(f"p{i}") for i in range(len(labels) + 1)]
    for p in places:
        net.places.add(p)

    # create transitions and arcs
    for i, lbl in enumerate(labels):
        t = PetriNet.Transition(f"t{i}", lbl)
        net.transitions.add(t)
        petri_utils.add_arc_from_to(places[i], t, net)
        petri_utils.add_arc_from_to(t, places[i + 1], net)

    # define initial/final markings
    im = Marking({places[0]: 1})
    fm = Marking({places[-1]: 1})
    return StructuredNet("Seqnet", net, im, fm)


class Composition(Enum):
    XOR = 0
    AND = 1
    LOOP = 2

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
                logging.debug(f"Composing:\n ~{left} >> {right}")
                return ~left >> right


def random_block_structured(
    num_blocks=3,
    xor_prob=0.3,
    and_prob=0.3,
    loop_prob=0.1,
    p_depth=0.1,
    max_depth=3,
) -> StructuredNet:

    if max_depth <= 0 or random.random() < p_depth:
        labels = [f"{uuid.uuid4().hex}{i}" for i in range(num_blocks)]
        return make_sequence("seq_model", labels)

    comp_op = random.choices(
        [e.value for e in Composition], [xor_prob, and_prob, loop_prob], k=1
    )[0]
    logging.debug(f"comp_op: {comp_op}")

    left = random_block_structured(
        num_blocks, xor_prob, and_prob, loop_prob, p_depth, max_depth - 1
    )
    right = random_block_structured(
        num_blocks, xor_prob, and_prob, loop_prob, p_depth, max_depth - 1
    )
    logging.debug(f"left: {left}")
    logging.debug(f"right: {right}")
    comp = Composition(comp_op)
    logging.debug(f"comp: {comp}")

    prod = Composition(comp_op).compose(left, right)
    logging.debug(f"prod: {prod}")
    return prod


if __name__ == "__main__":

    stnet = random_block_structured(
        num_blocks=3, xor_prob=0.3, and_prob=0.3, loop_prob=0.3, max_depth=4
    )
    view_petri_net(stnet.net, stnet.im, stnet.fm)
# %%
