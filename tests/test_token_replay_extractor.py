import unittest
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.log.obj import Trace, Event

from features.token_replay_extractor import TokenReplayFitnessExtractor


def create_simple_model() -> tuple[PetriNet, Marking, Marking]:
    """
    Create a simple sequential Petri net: A -> B -> C
    """
    net = PetriNet("simple_model")

    # Places
    p_start = PetriNet.Place("p_start")
    p1 = PetriNet.Place("p1")
    p2 = PetriNet.Place("p2")
    p_end = PetriNet.Place("p_end")
    net.places.add(p_start)
    net.places.add(p1)
    net.places.add(p2)
    net.places.add(p_end)

    # Transitions
    t_a = PetriNet.Transition("t_a", "A")
    t_b = PetriNet.Transition("t_b", "B")
    t_c = PetriNet.Transition("t_c", "C")
    net.transitions.add(t_a)
    net.transitions.add(t_b)
    net.transitions.add(t_c)

    # Arcs
    petri_utils.add_arc_from_to(p_start, t_a, net)
    petri_utils.add_arc_from_to(t_a, p1, net)
    petri_utils.add_arc_from_to(p1, t_b, net)
    petri_utils.add_arc_from_to(t_b, p2, net)
    petri_utils.add_arc_from_to(p2, t_c, net)
    petri_utils.add_arc_from_to(t_c, p_end, net)

    im = Marking({p_start: 1})
    fm = Marking({p_end: 1})

    return net, im, fm


def trace_to_net(trace: Trace) -> tuple[PetriNet, Marking, Marking]:
    """Convert a Trace to a linear Petri Net."""
    net = PetriNet("trace_net")

    p_start = PetriNet.Place("p_start")
    net.places.add(p_start)

    prev_place = p_start

    for i, event in enumerate(trace):
        label = event["concept:name"]
        t = PetriNet.Transition(f"t_{i}", label)
        net.transitions.add(t)

        petri_utils.add_arc_from_to(prev_place, t, net)

        p_next = PetriNet.Place(f"p_{i + 1}")
        net.places.add(p_next)
        petri_utils.add_arc_from_to(t, p_next, net)

        prev_place = p_next

    im = Marking({p_start: 1})
    fm = Marking({prev_place: 1})

    return net, im, fm


class TestTokenReplayFitnessExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = TokenReplayFitnessExtractor(use_cache=False)

    def test_perfect_fit_trace(self):
        """Test with a trace that perfectly fits the model."""
        # Model: A -> B -> C
        model_net, model_im, model_fm = create_simple_model()

        # Trace: A -> B -> C (perfect fit)
        trace = Trace()
        trace.append(Event({"concept:name": "A"}))
        trace.append(Event({"concept:name": "B"}))
        trace.append(Event({"concept:name": "C"}))
        trace_net, trace_im, trace_fm = trace_to_net(trace)

        # Extract features
        features = self.extractor.extract(
            model_net, model_im, model_fm,
            trace_net, trace_im, trace_fm,
            return_as_dict=True
        )

        # Assertions
        self.assertEqual(features['token_replay_trace_is_fit'], 1.0)
        self.assertEqual(features['token_replay_trace_fitness'], 1.0)
        self.assertEqual(features['token_replay_missing_tokens'], 0.0)
        self.assertEqual(features['token_replay_remaining_tokens'], 0.0)

    def test_partial_fit_trace(self):
        """Test with a trace that partially fits the model."""
        # Model: A -> B -> C
        model_net, model_im, model_fm = create_simple_model()

        # Trace: A -> D -> C (D is not in the model)
        trace = Trace()
        trace.append(Event({"concept:name": "A"}))
        trace.append(Event({"concept:name": "D"}))
        trace.append(Event({"concept:name": "C"}))
        trace_net, trace_im, trace_fm = trace_to_net(trace)

        # Extract features
        features = self.extractor.extract(
            model_net, model_im, model_fm,
            trace_net, trace_im, trace_fm,
            return_as_dict=True
        )

        # Assertions - trace should not be perfectly fit
        self.assertEqual(features['token_replay_trace_is_fit'], 0.0)
        self.assertLess(features['token_replay_trace_fitness'], 1.0)
        self.assertGreater(features['token_replay_missing_tokens'], 0.0)

    def test_feature_names(self):
        """Test that feature names are correctly defined."""
        expected_names = [
            'token_replay_trace_is_fit',
            'token_replay_trace_fitness',
            'token_replay_missing_tokens',
            'token_replay_consumed_tokens',
            'token_replay_remaining_tokens',
            'token_replay_produced_tokens',
        ]
        self.assertEqual(self.extractor.feature_names, expected_names)

    def test_returns_numpy_array(self):
        """Test that extract returns a numpy array by default."""
        import numpy as np

        model_net, model_im, model_fm = create_simple_model()

        trace = Trace()
        trace.append(Event({"concept:name": "A"}))
        trace.append(Event({"concept:name": "B"}))
        trace.append(Event({"concept:name": "C"}))
        trace_net, trace_im, trace_fm = trace_to_net(trace)

        features = self.extractor.extract(
            model_net, model_im, model_fm,
            trace_net, trace_im, trace_fm
        )

        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), len(self.extractor.feature_names))


if __name__ == '__main__':
    unittest.main()
