from pm4py.pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.pm4py.objects.petri_net.utils import petri_utils


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

    # loop op ~
    def __invert__(self):
        net = PetriNet(f"{self.name}_loop")
        petri_utils.merge(net, [self.net])

        # control structure
        p_in = PetriNet.Place("p_loop_in")
        p_out = PetriNet.Place("p_loop_out")
        t_enter = PetriNet.Transition("t_loop_enter", None)
        t_decide = PetriNet.Transition("t_loop_decide", None)

        net.places.update({p_in, p_out})
        net.transitions.update({t_enter, t_decide})

        p_body_start = list(self.im.keys())[0]
        p_body_end = list(self.fm.keys())[0]

        # 1. entry to body (via t_enter)
        petri_utils.add_arc_from_to(p_in, t_enter, net)
        petri_utils.add_arc_from_to(t_enter, p_body_start, net)

        # 2. after body, decision to loop again or exit
        petri_utils.add_arc_from_to(p_body_end, t_decide, net)
        petri_utils.add_arc_from_to(t_decide, p_in, net)  # loop back
        petri_utils.add_arc_from_to(t_decide, p_out, net)  # exit

        im = Marking({p_in: 1})
        fm = Marking({p_out: 1})
        return StructuredNet(net.name, net, im, fm)

    def __repr__(self):
        return f"StructuredNet: {self.net}\nim: {self.im}\nfm: {self.fm}"
