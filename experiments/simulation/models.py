# %%
import random
from pm4py.pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.pm4py.objects.petri_net.utils import petri_utils
from typing import TypeAlias
from enum import Enum


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
    return net, im, fm


Net: TypeAlias = tuple[PetriNet, Marking, Marking]


class Composition(Enum):
    XOR = 0
    AND = 1
    LOOP = 2

    def compose(self, left: Net, right: Net) -> Net:
        net = PetriNet(f"{left[0].name + right[0].name}_{self.value}")
        match self.value:
            case Composition.XOR:
                places = [PetriNet.Place("p_xor")]

            case Composition.AND:
                places = [PetriNet.Place("p_and")]

            case Composition.LOOP:
                places = [PetriNet.Place("p_loop")]


def random_block_structured(
    num_blocks=3, xor_prob=0.3, and_prob=0.3, loop_prob=0.1, max_depth=3
):
    # TODO: Compose blocks recursively
    # Start with simple sequential composition for baseline
    if max_depth == 0:
        labels = [f"a{i}" for i in range(num_blocks)]
        return make_sequence("seq_model", labels)
    comp_op = random.choice(
        [e.value for e in Composition], [xor_prob, and_prob, loop_prob]
    )

    left = random_block_structured(
        num_blocks, xor_prob, and_prob, loop_prob, max_depth - 1
    )
    right = random_block_structured(
        num_blocks, xor_prob, and_prob, loop_prob, max_depth - 1
    )
    return Composition(comp_op).compose(left, right)


if __name__ == "__main__":
    import pm4py

    pn, im, fm = random_block_structured(num_blocks=5, max_depth=2)
    pm4py.view_petri_net(pn, im, fm)
# %%
