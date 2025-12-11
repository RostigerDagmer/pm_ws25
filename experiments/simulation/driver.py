
import torch
from experiments.simulation.models import sample_net
from pm4py.vis import view_petri_net
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.algo.simulation.playout.petri_net import algorithm as pn_sim
from experiments.simulation.noise import inject_noise
from typing import Any, Dict, Generator, Optional
from util.distributions import (
    CategoricalSpec,
    PoissonSpec,
    BernoulliDepthLinearSpec,
)
from util.rng import RNG


def generate_dataset(
    n_models: int = 1,
    parameters: Optional[Dict[Any, Any]] = None,
) -> Generator[Any, tuple[PetriNet, Marking, Marking, EventLog], None]:
    for _ in range(n_models):
        dist_params = {
            "op": CategoricalSpec([0.3, 0.3, 0.3, 0.1]),
            "seq_len": PoissonSpec(3),
            "p_stop": BernoulliDepthLinearSpec(base=0.2, slope=0.1),
            "width": PoissonSpec(3),
        }
        stnet = sample_net(dist_params, generator=RNG.torch_generator())
        pn, im, fm = stnet.net, stnet.im, stnet.fm
        log = pn_sim.apply(
            pn,
            im,
            fm,
            parameters=parameters,
            variant=pn_sim.Variants.BASIC_PLAYOUT,
        )
        labels = [t.label for t in pn.transitions if t.label]
        noisy_log = inject_noise(
            log, p_insert=0.1, p_delete=0.05, p_swap=0.05, labels=labels
        )
        yield (pn, im, fm, noisy_log)


if __name__ == "__main__":
    RNG.initialize(42)  # Initialize RNG for reproducibility
    i = 0
    for item in generate_dataset():
        print("event_log: ", item[-1])
        view_petri_net(*item[:-1])
        i += 1
        if i > 10:
            break
