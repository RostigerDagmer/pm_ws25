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
from features.base_extractor import SpectralFeatureExtractor
from experiments.model.spectral_model import SpectralModel

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


def test_spectral_model():
    import time

    print("1. Generating random model...")
    dist_params = {
        "op": CategoricalSpec([0.3, 0.3, 0.3, 0.1]),
        "seq_len": PoissonSpec(4),
        "p_stop": BernoulliDepthLinearSpec(base=0.15, slope=0.1),
        "width": PoissonSpec(3),
    }
    stnet = sample_net(dist_params, max_depth=5)
    net, im, fm = stnet.net, stnet.im, stnet.fm
    print(f"Model: {len(net.transitions)} transitions")

    print("2. Simulating trace...")
    net_tensor = stnet.to_tensor()
    logs_tensor = simulate_batch(
        (net_tensor.pre, net_tensor.post),
        net_tensor.M0,
        net_tensor.Mf,
        net_tensor.labels,
        steps=50,
        batch_size=1024,
    )
    # event_log = apply_labels(logs_tensor, net_tensor.labels)
    # trace = event_log[0]
    # trace_net, trace_im, trace_fm = trace_to_net(trace)

    print("3. Extracting tensors...")
    d_model = 64
    n_coeffs = 8
    extractor = SpectralFeatureExtractor(d_model=d_model, n_coeffs=n_coeffs)

    # tensors = extractor.extract_tensors(
    #     net, im, fm, trace_net, trace_im, trace_fm
    # )
    tensors = extractor.extract_batch_tensors(
        (net_tensor.pre, net_tensor.post), net_tensor.labels, logs_tensor
    )

    model_basis = tensors["model_basis"]
    trace_embedding = tensors["trace_embedding"]

    print(f"Model Basis Shape: {model_basis.shape}")  # Expected: [T, d_model]
    print(
        f"Trace Embedding Shape: {trace_embedding.shape}"
    )  # Expected: [d_trace]

    # Batchify
    model_basis = model_basis.repeat(
        trace_embedding.shape[0], 1, 1
    )  # [B, T, d_model]
    trace_embedding = trace_embedding  # [B, d_trace]

    print("4. Running SpectralModel...")
    d_trace = trace_embedding.shape[1]
    hidden_dim = 256
    n_classes = 5

    model = SpectralModel(
        d_model=extractor.d_model,
        d_trace=d_trace,
        hidden_dim=hidden_dim,
        mlp_hidden_dim=hidden_dim * 2,
        n_classes=n_classes,
        num_heads=4,
        n_layers=4,
        dropout=0.1,
    )

    print(f"num_params: {sum(p.numel() for p in model.parameters())}")
    logits = model(model_basis, trace_embedding)
    print(f"Logits Shape: {logits.shape}")  # Expected: [B, n_classes]

    assert logits.shape == (trace_embedding.shape[0], n_classes)

    print("5. Backward pass...")
    loss = logits.sum()
    loss.backward()
    print("Backward pass successful.")

    print("6. Forward pass in eval with timing...")
    model.eval()
    start = time.perf_counter()
    logits = model(model_basis, trace_embedding)
    end = time.perf_counter()
    forward_time = end - start
    print(f"Forward pass in eval with timing: {forward_time:.6f} seconds")
    print(
        f"per sample time: {forward_time / logs_tensor.shape[0]:.6f} seconds"
    )
    # for size 128:
    # Forward pass in eval with timing: 0.041672 seconds
    # per sample time: 0.000326 seconds

    # for size 1024:
    # Forward pass in eval with timing: 0.201492 seconds
    # per sample time: 0.000197 seconds
    print("\nTest Passed!")


if __name__ == "__main__":
    test_spectral_model()
