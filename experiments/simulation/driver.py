# %%
from experiments.simulation.models import random_block_structured
from pm4py.vis import view_petri_net
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.algo.simulation.playout.petri_net import algorithm as pn_sim
from experiments.simulation.noise import inject_noise
from typing import Any, Dict, Generator, Optional


def generate_dataset(
    n_models: int = 3,
    num_blocks: int = 5,
    parameters: Optional[Dict[Any, Any]] = None,
) -> Generator[Any, tuple[PetriNet, Marking, Marking, EventLog], None]:
    for _ in range(n_models):
        pn, im, fm = random_block_structured(num_blocks=num_blocks)
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
    i = 0
    for item in generate_dataset():
        print("event_log: ", item[-1])
        view_petri_net(*item[:-1])
        i += 1
        if i > 10:
            break
# %%
