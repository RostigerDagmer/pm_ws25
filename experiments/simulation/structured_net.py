from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
import torch
from dataclasses import dataclass


@dataclass
class TensorNet:
    pre: torch.Tensor  # [T, P]
    post: torch.Tensor  # [T, P]
    labels: list[str]  # len=T, "" = tau
    init: int  # index of initial place
    final: int  # index of final place

    @property
    def M0(self) -> torch.Tensor:
        m = torch.zeros(
            self.pre.shape[1], dtype=torch.int, device=self.pre.device
        )
        m[self.init] = 1
        return m

    @property
    def Mf(self) -> torch.Tensor:
        m = torch.zeros(
            self.pre.shape[1], dtype=torch.int, device=self.pre.device
        )
        m[self.final] = 1
        return m


class StructuredNet:
    def __init__(self, name: str, net: PetriNet, im: Marking, fm: Marking):
        self.name = name
        self.net = net
        self.im = im
        self.fm = fm

    def __rshift__(self, other: "StructuredNet"):
        # sequence
        net = PetriNet(f"{self.name}_seq_{other.name}")
        petri_utils.merge(net, [self.net, other.net])

        t_seq = PetriNet.Transition("t_seq", None)
        net.transitions.update({t_seq})

        pA_end = list(self.fm.keys())[0]
        pB_start = list(other.im.keys())[0]

        # connect exit of A → entry of B
        petri_utils.add_arc_from_to(pA_end, t_seq, net)
        petri_utils.add_arc_from_to(t_seq, pB_start, net)

        im = Marking({list(self.im.keys())[0]: 1})
        fm = Marking({list(other.fm.keys())[0]: 1})
        return StructuredNet(net.name, net, im, fm)

    def __xor__(self, other: "StructuredNet") -> "StructuredNet":
        return StructuredNet.n_xor([self, other])

    @staticmethod
    def n_xor(nets: list["StructuredNet"]) -> "StructuredNet":
        if not nets:
            raise ValueError("n_xor requires at least one net")
        if len(nets) == 1:
            return nets[0]

        # Create a new name combining all names
        name = "_xor_".join(n.name for n in nets)
        net = PetriNet(name)

        # Merge all subnets
        petri_utils.merge(net, [n.net for n in nets])

        p_in = PetriNet.Place("p_xor_in")
        p_out = PetriNet.Place("p_xor_out")
        net.places.update({p_in, p_out})

        # Create split and join transitions for each subnet
        for i, subnet in enumerate(nets):
            t_split = PetriNet.Transition(
                f"t_xor_split_{subnet.name}_{i}", None
            )
            t_join = PetriNet.Transition(f"t_xor_join_{subnet.name}_{i}", None)
            net.transitions.update({t_split, t_join})

            p_start = list(subnet.im.keys())[0]
            p_end = list(subnet.fm.keys())[0]

            # Connect split: p_in -> t_split -> p_start
            petri_utils.add_arc_from_to(p_in, t_split, net)
            petri_utils.add_arc_from_to(t_split, p_start, net)

            # Connect join: p_end -> t_join -> p_out
            petri_utils.add_arc_from_to(p_end, t_join, net)
            petri_utils.add_arc_from_to(t_join, p_out, net)

        im = Marking({p_in: 1})
        fm = Marking({p_out: 1})
        return StructuredNet(net.name, net, im, fm)

    def __and__(self, other: "StructuredNet"):
        return StructuredNet.n_and([self, other])

    @staticmethod
    def n_and(nets: list["StructuredNet"]) -> "StructuredNet":
        if not nets:
            raise ValueError("n_and requires at least one net")
        if len(nets) == 1:
            return nets[0]

        name = "_and_".join(n.name for n in nets)
        net = PetriNet(name)
        petri_utils.merge(net, [n.net for n in nets])

        p_in = PetriNet.Place("p_and_in")
        p_out = PetriNet.Place("p_and_out")
        t_split = PetriNet.Transition("t_split", None)
        t_join = PetriNet.Transition("t_join", None)

        net.places.update({p_in, p_out})
        net.transitions.update({t_split, t_join})

        # split to all
        petri_utils.add_arc_from_to(p_in, t_split, net)
        for subnet in nets:
            p_start = list(subnet.im.keys())[0]
            petri_utils.add_arc_from_to(t_split, p_start, net)

        # join from all
        for subnet in nets:
            p_end = list(subnet.fm.keys())[0]
            petri_utils.add_arc_from_to(p_end, t_join, net)
        petri_utils.add_arc_from_to(t_join, p_out, net)

        im = Marking({p_in: 1})
        fm = Marking({p_out: 1})
        return StructuredNet(net.name, net, im, fm)

    # loop op A @ B reads as "loop A with exit B"
    def __matmul__(self, exit: "StructuredNet"):
        net = PetriNet(f"{self.name}_loop_{exit.name}")
        petri_utils.merge(net, [self.net, exit.net])

        p_in = PetriNet.Place("p_loop_in")
        p_out = PetriNet.Place("p_loop_out")
        t_split = PetriNet.Transition("t_loop_split", None)
        t_link = PetriNet.Transition("t_loop_link", None)
        t_exit = PetriNet.Transition(f"t_exit_{self.name}_{exit.name}", None)
        t_repeat = PetriNet.Transition(
            f"t_repeat_{self.name}_{exit.name}", None
        )
        net.places.update({p_in, p_out})
        net.transitions.update({t_split, t_link, t_exit, t_repeat})

        p_body_start = list(self.im.keys())[0]
        p_body_end = list(self.fm.keys())[0]
        p_exit_start = list(exit.im.keys())[0]
        p_exit_end = list(exit.fm.keys())[0]

        # enter body
        petri_utils.add_arc_from_to(p_in, t_split, net)
        petri_utils.add_arc_from_to(t_split, p_body_start, net)

        # connect body → exit
        petri_utils.add_arc_from_to(p_body_end, t_link, net)
        petri_utils.add_arc_from_to(t_link, p_exit_start, net)

        # after exit, decide repeat or exit
        petri_utils.add_arc_from_to(p_exit_end, t_exit, net)
        petri_utils.add_arc_from_to(p_exit_end, t_repeat, net)
        petri_utils.add_arc_from_to(t_repeat, p_body_start, net)  # repeat
        petri_utils.add_arc_from_to(t_exit, p_out, net)  # exit

        im = Marking({p_in: 1})
        fm = Marking({p_out: 1})
        return StructuredNet(net.name, net, im, fm)

    # the silent transition
    @staticmethod
    def tau(name: str = "tau") -> "StructuredNet":
        net = PetriNet(name)
        p_in = PetriNet.Place("p_in")
        p_out = PetriNet.Place("p_out")
        t = PetriNet.Transition("t_tau", None)  # None == silent
        net.places.update({p_in, p_out})
        net.transitions.add(t)
        petri_utils.add_arc_from_to(p_in, t, net)
        petri_utils.add_arc_from_to(t, p_out, net)

        im = Marking({p_in: 1})
        fm = Marking({p_out: 1})
        return StructuredNet(name, net, im, fm)

    def __repr__(self):
        if self.name == "tau":
            return "τ"
        return f"StructuredNet: {self.net}\nim: {self.im}\nfm: {self.fm}"

    def into_tuple(self) -> tuple[PetriNet, Marking, Marking]:
        return (self.net, self.im, self.fm)

    @staticmethod
    def from_tuple(t: tuple[PetriNet, Marking, Marking]) -> "StructuredNet":
        return StructuredNet(t[0].name, t[0], t[1], t[2])

    def to_tensor(self, device=None) -> TensorNet:
        places = list(self.net.places)
        transitions = list(self.net.transitions)

        P = len(places)
        T = len(transitions)

        place_index = {p: i for i, p in enumerate(places)}
        trans_index = {t: j for j, t in enumerate(transitions)}

        pre = torch.zeros((T, P), dtype=torch.int, device=device)
        post = torch.zeros((T, P), dtype=torch.int, device=device)
        labels = [""] * T

        for t in transitions:
            j = trans_index[t]
            labels[j] = t.label or ""
            for arc in t.in_arcs:
                pre[j, place_index[arc.source]] += 1
            for arc in t.out_arcs:
                post[j, place_index[arc.target]] += 1

        init = place_index[next(iter(self.im.keys()))]
        final = place_index[next(iter(self.fm.keys()))]

        return TensorNet(pre, post, labels, init, final)

    @staticmethod
    def from_tensor(tn: TensorNet) -> "StructuredNet":
        P = tn.pre.shape[1]
        T = tn.pre.shape[0]

        net = PetriNet("Reconstructed")

        # rebuild fresh places and transitions deterministically
        places = [PetriNet.Place(f"p{i}") for i in range(P)]
        trans = [
            PetriNet.Transition(f"t{j}", tn.labels[j] or None)
            for j in range(T)
        ]

        net.places.update(places)
        net.transitions.update(trans)

        # arcs from tensors
        for j in range(T):
            for i in range(P):
                if tn.pre[j, i] > 0:
                    petri_utils.add_arc_from_to(places[i], trans[j], net)
                if tn.post[j, i] > 0:
                    petri_utils.add_arc_from_to(trans[j], places[i], net)

        im = Marking({places[tn.init]: 1})
        fm = Marking({places[tn.final]: 1})

        return StructuredNet("Reconstructed", net, im, fm)


def test_tensor_conversion():
    from pm4py.vis import view_petri_net
    from experiments.simulation.models import (
        CategoricalSpec,
        PoissonSpec,
        BernoulliDepthLinearSpec,
        sample_net,
    )
    from util.rng import RNG

    rng = RNG()
    rng.initialize(3)

    dist_params = {
        "op": CategoricalSpec([0.3, 0.3, 0.3, 0.1]),
        "seq_len": PoissonSpec(4),
        "p_stop": BernoulliDepthLinearSpec(base=0.1, slope=0.1),
    }

    # Create a random StructuredNet for testing
    sample_stnet = sample_net(
        dist_params, max_depth=3
    )  # Assuming sample_net() provides a random StructuredNet

    # Convert the StructuredNet to TensorNet
    tensor_net = sample_stnet.to_tensor()

    # Convert the TensorNet back to StructuredNet
    reconstructed_stnet = StructuredNet.from_tensor(tensor_net)

    # Check if the original StructuredNet and reconstructed StructuredNet are the same
    original_petri_net = sample_stnet.net
    reconstructed_petri_net = reconstructed_stnet.net

    # View the nets for inspection
    print("Original Petri Net:")
    view_petri_net(original_petri_net, sample_stnet.im, sample_stnet.fm)

    print("\nReconstructed Petri Net:")

    view_petri_net(
        reconstructed_petri_net, reconstructed_stnet.im, reconstructed_stnet.fm
    )

    # Optionally, you can add a more formal check here based on your requirements
    # For example, comparing places, transitions, and arcs in both nets


if __name__ == "__main__":
    test_tensor_conversion()
