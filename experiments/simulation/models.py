# %%
import random
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils


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


def random_block_structured(
    num_blocks=3, xor_prob=0.3, and_prob=0.3, loop_prob=0.1
):
    # TODO: Compose blocks recursively
    # Start with simple sequential composition for baseline
    labels = [f"a{i}" for i in range(num_blocks)]
    return make_sequence("seq_model", labels)


if __name__ == "__main__":
    import pm4py

    pn, im, fm = random_block_structured(num_blocks=5)
    pm4py.view_petri_net(pn, im, fm)
# %%
