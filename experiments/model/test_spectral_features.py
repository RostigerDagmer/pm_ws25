import torch
import logging
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.log.obj import Trace, Event

from util.distributions import (
    BernoulliDepthLinearSpec,
    CategoricalSpec,
    PoissonSpec,
)
from experiments.simulation.models import sample_net
from experiments.simulation.simulate import simulate_batch, apply_labels
from features.spectral_extractor import SpectralFeatureExtractor

logging.basicConfig(level=logging.INFO)


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


def test_spectral_extractor():
    # 1. Generate Random Model
    print("Generating random model...")
    dist_params = {
        "op": CategoricalSpec([0.3, 0.3, 0.3, 0.1]),
        "seq_len": PoissonSpec(4),
        "p_stop": BernoulliDepthLinearSpec(base=0.15, slope=0.1),
        "width": PoissonSpec(3),
    }
    stnet = sample_net(dist_params, max_depth=5)
    net, im, fm = stnet.net, stnet.im, stnet.fm

    print(
        f"Model generated: {len(net.transitions)} transitions, {len(net.places)} places"
    )

    # 2. Simulate Traces
    print("Simulating traces...")
    net_tensor = stnet.to_tensor()
    # Need to move to device if using GPU, but CPU is fine for test
    logs_tensor, _, _ = simulate_batch(
        (net_tensor.pre, net_tensor.post),
        net_tensor.M0,
        net_tensor.Mf,
        net_tensor.labels,
        steps=50,
        batch_size=5,
    )

    event_log = apply_labels(logs_tensor, net_tensor.labels)
    trace = event_log[0]
    print(f"Sample trace: {[e['concept:name'] for e in trace]}")

    # 3. Create Trace with Unknown Activity
    print("Creating trace with unknown activity...")
    unknown_trace = Trace()
    for e in trace:
        unknown_trace.append(e)
    unknown_trace.append(Event({"concept:name": "UNKNOWN_ACTIVITY_XYZ"}))
    print(f"Unknown trace: {[e['concept:name'] for e in unknown_trace]}")

    # 4. Extract Features
    print("Extracting features...")
    d_model = 64
    n_coeffs = 8
    extractor = SpectralFeatureExtractor(d_model=d_model, n_coeffs=n_coeffs)

    # Convert traces to nets
    trace_net, trace_im, trace_fm = trace_to_net(trace)
    unknown_trace_net, unknown_trace_im, unknown_trace_fm = trace_to_net(
        unknown_trace
    )

    features_normal = extractor.extract(
        net, im, fm, trace_net, trace_im, trace_fm, return_as_dict=False
    )

    features_unknown = extractor.extract(
        net,
        im,
        fm,
        unknown_trace_net,
        unknown_trace_im,
        unknown_trace_fm,
        return_as_dict=False,
    )
    print(features_normal)
    print(features_unknown)

    import matplotlib.pyplot as plt

    # set backend to Agg
    plt.switch_backend('qtagg')

    plt.plot(features_normal)
    plt.plot(features_unknown)
    plt.show()

    features_normal = extractor.extract(
        net, im, fm, trace_net, trace_im, trace_fm, return_as_dict=True
    )

    features_unknown = extractor.extract(
        net,
        im,
        fm,
        unknown_trace_net,
        unknown_trace_im,
        unknown_trace_fm,
        return_as_dict=True,
    )

    # 5. Verify
    print("\n--- Verification ---")
    expected_dims = n_coeffs * d_model
    print(f"Expected feature count: {expected_dims}")
    print(f"Normal features count: {len(features_normal)}")
    print(f"Unknown features count: {len(features_unknown)}")

    assert len(features_normal) == expected_dims
    assert len(features_unknown) == expected_dims

    # Check if unknown dimension is active in unknown trace
    # The unknown dimension is the last one (index d_model)
    # We check coefficients for that dimension

    unknown_activity_energy = 0.0
    for c in range(n_coeffs):
        key = f"spectral_dct_c{c}_d{d_model - 1}"
        val = features_unknown[key]
        unknown_activity_energy += abs(val)
        print(f"{key}: {val:.4f}")

    print(f"Total energy in unknown dimension: {unknown_activity_energy:.4f}")

    if unknown_activity_energy > 0.001:
        print("SUCCESS: Unknown activity detected in spectral features.")
    else:
        print("FAILURE: Unknown activity NOT detected (energy too low).")

    # Check normal trace has 0 energy in unknown dimension
    normal_unknown_energy = 0.0
    for c in range(n_coeffs):
        key = f"spectral_dct_c{c}_d{d_model - 1}"
        normal_unknown_energy += abs(features_normal[key])

    print(
        f"Normal trace energy in unknown dimension: {normal_unknown_energy:.4f}"
    )
    assert (
        normal_unknown_energy < 1e-6
    ), "Normal trace should have 0 energy in unknown dimension"

    print("\nTest Passed!")


def test_fast_path():
    import time

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("\n--- Testing Fast Path ---")
    # 1. Generate Random Model
    dist_params = {
        "op": CategoricalSpec([0.3, 0.3, 0.3, 0.1]),
        "seq_len": PoissonSpec(4),
        "p_stop": BernoulliDepthLinearSpec(base=0.15, slope=0.1),
        "width": PoissonSpec(3),
    }
    stnet = sample_net(dist_params, max_depth=5)
    net, im, fm = stnet.net, stnet.im, stnet.fm

    # 2. Simulate Traces
    net_tensor = stnet.to_tensor(device=device)
    logs_tensor, _, _ = simulate_batch(
        (net_tensor.pre, net_tensor.post),
        net_tensor.M0,
        net_tensor.Mf,
        net_tensor.labels,
        steps=50,
        batch_size=128,
    )

    # 3. Extract Features (Fast Path)
    d_model = 64
    n_coeffs = 8
    extractor = SpectralFeatureExtractor(d_model=d_model, n_coeffs=n_coeffs)

    start = time.time()
    tensors = extractor.extract_batch_tensors(
        (net_tensor.pre, net_tensor.post), net_tensor.labels, logs_tensor
    )
    end = time.time()
    feature_time = end - start
    print(f"Extracted in {feature_time:.4f} seconds")
    print(
        f"per sample time: {feature_time / logs_tensor.shape[0]:.4f} seconds"
    )

    model_basis = tensors["model_basis"]
    trace_embeddings = tensors["trace_embedding"]

    print(f"Model Basis Shape: {model_basis.shape}")
    print(f"Trace Embeddings Shape: {trace_embeddings.shape}")

    # Expected Shapes
    # model_basis: [1, T_unique, d_model]
    # trace_embeddings: [B, d_trace]

    B = logs_tensor.shape[0]
    d_trace = n_coeffs * (extractor.d_model + 1)

    assert model_basis.dim() == 3
    assert model_basis.shape[0] == 1
    assert model_basis.shape[2] == extractor.d_model

    assert trace_embeddings.shape == (B, d_trace)

    print("Fast Path Test Passed!")


if __name__ == "__main__":
    test_spectral_extractor()
    test_fast_path()
