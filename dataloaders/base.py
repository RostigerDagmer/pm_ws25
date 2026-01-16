from typing import Callable
import torch
from torch.utils.data import Dataset
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.obj import EventLog, Trace, EventStream
from collections.abc import Sequence
import pandas as pd


def _normalize_log_input(subset) -> pd.DataFrame | EventLog | Trace:
    """
    Normalize 'subset' to a pm4py-compatible pandas DataFrame.
    Supports:
      - pandas.DataFrame
      - pm4py EventLog / EventStream
      - any Sequence (e.g., list, tuple, TraceSubset) of pm4py Traces
      - any Sequence of event dicts
    """
    # 1) Already a DataFrame
    if isinstance(subset, pd.DataFrame):
        return subset

    # 2) pm4py log types
    if isinstance(subset, (EventLog, EventStream)):
        return log_converter.apply(
            subset, variant=log_converter.Variants.TO_DATA_FRAME
        )

    # 3) Generic sequences (includes your TraceSubset), excluding text
    if isinstance(subset, Sequence) and not isinstance(subset, (str, bytes)):
        seq = list(subset)

        if len(seq) == 0:
            raise ValueError(
                "Empty subset: cannot discover a model from zero events/traces."
            )

        # 3a) Sequence of pm4py Traces -> wrap into EventLog then to DataFrame
        if all(isinstance(t, Trace) for t in seq):
            evlog = EventLog(seq)
            return log_converter.apply(
                evlog, variant=log_converter.Variants.TO_DATA_FRAME
            )

        # 3b) Sequence of event dicts -> directly to DataFrame
        if all(isinstance(e, dict) for e in seq):
            return pd.DataFrame(seq)

        # Mixed or unsupported element type
        raise TypeError(
            "Unsupported sequence element types for subset: expected all pm4py Traces or all event dicts; "
            f"got examples like {type(seq[0])!r}."
        )

    # 4) Anything else -> unsupported
    raise TypeError(f"Unsupported subset type: {type(subset)}")


class BaseEventLogDataset(Dataset):
    """
    Base class for event log datasets from pm4py EventLog objects.

    The user provides a feature extraction function that maps traces/events to tensors.
    """

    def __init__(
        self,
        source_path,
        **kwargs,
    ):
        """
        Args:
            source_path (str): Path to event log file (format depends on subclass).
            **kwargs: Passed to subclass loader.
        """
        self.source_path = source_path

        # Let subclass load the pm4py log
        self.log = self._load_log(source_path, **kwargs)
        self.log: EventLog = log_converter.apply(
            self.log, variant=log_converter.Variants.TO_EVENT_LOG
        )

    @property
    def log_uuid(self) -> str:
        return self.source_path.split("/")[-2]

    def _load_log(self, source_path, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return len(self.log)

    def __getitem__(self, idx: int) -> Trace:
        return self.log[idx]


""" ============= Example Usage ============= """

if __name__ == "__main__":
    from dataloaders.xes_log import XESEventLogDataset

    path = "data/c2c3b154-ab26-4b31-a0e8-8f2350ddac11/BPI_Challenge_2013_closed_problems.xes"

    dataset = XESEventLogDataset(path)

    for item in dataset:
        print(item)  # [B, T, feature_dim]
        break
