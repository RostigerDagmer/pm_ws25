import random
from copy import deepcopy
from pm4py.objects.log.obj import EventLog, Trace


def inject_noise(
    log: EventLog,
    p_insert=0.05,
    p_delete=0.05,
    p_swap=0.02,
    labels=None,
    activity_key="concept:name",
) -> EventLog:
    """
    Returns a *deepcopy* of the log with noise injected.
    Each trace is modified independently.
    """
    noisy_log = EventLog(attributes=getattr(log, "attributes", {}))

    for trace in log:
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
        new_trace = Trace(
            new_events, attributes=getattr(trace, "attributes", {})
        )
        noisy_log.append(new_trace)

    return noisy_log


"""
TODO:
- Structured noise (burst loops, reorder concurrent segments)
- Label confusion models (replace labels with near-synonyms)
"""
