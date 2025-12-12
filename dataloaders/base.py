from typing import Callable
import torch
from torch.utils.data import Dataset
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.obj import EventLog, Trace


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
