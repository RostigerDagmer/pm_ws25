from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
import torch


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
        net = PetriNet(f"{self.name}_xor_{other.name}")
        petri_utils.merge(net, [self.net, other.net])

        p_in = PetriNet.Place("p_xor_in")
        p_out = PetriNet.Place("p_xor_out")
        net.places.update({p_in, p_out})

        # Split transitions (competing for same token)
        t_split_a = PetriNet.Transition(f"t_xor_split_{self.name}", None)
        t_split_b = PetriNet.Transition(f"t_xor_split_{other.name}", None)
        net.transitions.update({t_split_a, t_split_b})

        # Join transitions (only one will fire)
        t_join_a = PetriNet.Transition(f"t_xor_join_{self.name}", None)
        t_join_b = PetriNet.Transition(f"t_xor_join_{other.name}", None)
        net.transitions.update({t_join_a, t_join_b})

        p_a_start = list(self.im.keys())[0]
        p_a_end = list(self.fm.keys())[0]
        p_b_start = list(other.im.keys())[0]
        p_b_end = list(other.fm.keys())[0]

        # XOR-split: two competing transitions from the same input place
        petri_utils.add_arc_from_to(p_in, t_split_a, net)
        petri_utils.add_arc_from_to(p_in, t_split_b, net)
        petri_utils.add_arc_from_to(t_split_a, p_a_start, net)
        petri_utils.add_arc_from_to(t_split_b, p_b_start, net)

        # XOR-join: two competing transitions into the same output place
        petri_utils.add_arc_from_to(p_a_end, t_join_a, net)
        petri_utils.add_arc_from_to(p_b_end, t_join_b, net)
        petri_utils.add_arc_from_to(t_join_a, p_out, net)
        petri_utils.add_arc_from_to(t_join_b, p_out, net)

        im = Marking({p_in: 1})
        fm = Marking({p_out: 1})
        return StructuredNet(net.name, net, im, fm)

    def __and__(self, other: "StructuredNet"):
        # and
        net = PetriNet(f"{self.name}_and_{other.name}")
        petri_utils.merge(net, [self.net, other.net])

        p_in = PetriNet.Place("p_and_in")
        p_out = PetriNet.Place("p_and_out")
        t_split = PetriNet.Transition("t_split", None)
        t_join = PetriNet.Transition("t_join", None)

        net.places.update({p_in, p_out})
        net.transitions.update({t_split, t_join})

        # split to both
        petri_utils.add_arc_from_to(p_in, t_split, net)
        for p_start in [list(self.im.keys())[0], list(other.im.keys())[0]]:
            petri_utils.add_arc_from_to(t_split, p_start, net)

        # join from both
        for p_end in [list(self.fm.keys())[0], list(other.fm.keys())[0]]:
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
        t_join = PetriNet.Transition("t_loop_join", None)
        t_link = PetriNet.Transition(f"t_link_{self.name}_{exit.name}", None)
        net.places.update({p_in, p_out})
        net.transitions.update({t_split, t_join, t_link})

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
        petri_utils.add_arc_from_to(p_exit_end, t_join, net)
        petri_utils.add_arc_from_to(t_join, p_body_start, net)  # repeat
        petri_utils.add_arc_from_to(t_join, p_out, net)  # exit

        im = Marking({p_in: 1})
        fm = Marking({p_out: 1})
        return StructuredNet(net.name, net, im, fm)

    # the silent transition
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

    def from_tuple(t: tuple[PetriNet, Marking, Marking]) -> "StructuredNet":
        return StructuredNet(t[0].name, t[0], t[1], t[2])

    def to_tensor(self, device=None):
        """Convert StructuredNet into tensor form for vectorized simulation."""
        places = list(self.net.places)
        transitions = list(self.net.transitions)

        num_places = len(places)
        num_trans = len(transitions)

        place_index = {p: i for i, p in enumerate(places)}
        trans_index = {t: j for j, t in enumerate(transitions)}

        pre = torch.zeros(
            (num_trans, num_places), dtype=torch.int, device=device
        )
        post = torch.zeros(
            (num_trans, num_places), dtype=torch.int, device=device
        )
        labels = []

        for t in transitions:
            j = trans_index[t]
            labels.append(t.label or "")  # empty string for τ
            for arc in t.in_arcs:
                i = place_index[arc.source]
                pre[j, i] += 1
            for arc in t.out_arcs:
                i = place_index[arc.target]
                post[j, i] += 1

        # markings
        M0 = torch.zeros(num_places, dtype=torch.int, device=device)
        Mf = torch.zeros(num_places, dtype=torch.int, device=device)
        for p, w in self.im.items():
            M0[place_index[p]] = w
        for p, w in self.fm.items():
            Mf[place_index[p]] = w

        return {
            "pre": pre,
            "post": post,
            "labels": labels,
            "M0": M0,
            "Mf": Mf,
            "place_index": place_index,
            "trans_index": trans_index,
        }
