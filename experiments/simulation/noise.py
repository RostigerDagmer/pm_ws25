import random
from copy import deepcopy
from typing import Optional
from pm4py.objects.log.obj import EventLog, Trace


def inject_noise(
    log: EventLog,
    p_insert: float = 0.05,
    p_delete: float = 0.05,
    p_swap: float = 0.02,
    labels: Optional[str] = None,
    activity_key: str = "concept:name",
) -> EventLog:
    """
    Returns a *deepcopy* of the log with noise injected.
    Each trace is modified independently.
    """
    noisy_log = EventLog(attributes=getattr(log, "attributes", {}))

    for trace in log:
        # rewrap into a PM4Py Trace object
        new_trace = inject_noise_trace(
            trace, p_insert, p_delete, p_swap, labels, activity_key
        )
        noisy_log.append(new_trace)

    return noisy_log


def inject_noise_trace(
    trace: Trace,
    p_insert: float = 0.05,
    p_delete: float = 0.05,
    p_swap: float = 0.02,
    labels: Optional[str] = None,
    activity_key: str = "concept:name",
) -> Trace:
    new_events = list(deepcopy(trace))  # mutable copy of events list
    i = 0
    while i < len(new_events):
        r = random.random()
        if r < p_delete:
            # delete this event
            new_events.pop(i)
            continue
        elif r < p_delete + p_insert and labels:
            # insert a new event copy with random label
            new_event = deepcopy(new_events[i]) if new_events else {}
            new_event[activity_key] = random.choice(labels)
            new_events.insert(i, new_event)
        elif r < p_delete + p_insert + p_swap and i < len(new_events) - 1:
            # swap two adjacent events
            new_events[i], new_events[i + 1] = (
                new_events[i + 1],
                new_events[i],
            )
        i += 1

    # rewrap into a PM4Py Trace object
    return Trace(new_events, attributes=getattr(trace, "attributes", {}))


"""
TODO:
- Structured noise (burst loops, reorder concurrent segments)
- Label confusion models (replace labels with near-synonyms)
"""
