import torch
from torch.utils.data import Dataset
from pm4py.objects.conversion.log import converter as log_converter


def _build_vocabs(log, attributes=None):
    """
    Builds vocabularies for the given attributes.
    If attributes=None, builds for all string-valued keys in the log.
    Returns: dict of {attribute_name: {value: id}}
    """
    value_sets = {}

    for trace in log:
        for event in trace:
            for key, val in event.items():
                # Only track specified attributes, or all if None
                if attributes is not None and key not in attributes:
                    continue
                # Skip non-string-like / non-discrete values (e.g. timestamps)
                if not isinstance(val, (str, int, float)):
                    continue
                value_sets.setdefault(key, set()).add(str(val))

    vocabs = {}
    for key, values in value_sets.items():
        vocabs[key] = {v: i + 1 for i, v in enumerate(sorted(values))}
    return vocabs


def _build_unified_vocab(log):
    values = set()
    for trace in log:
        for event in trace:
            for key, val in event.items():
                if isinstance(val, (str, int, float)):
                    values.add(f"{key}={val}")
    return {v: i + 1 for i, v in enumerate(sorted(values))}


class BaseEventLogDataset(Dataset):
    """
    Base class for event log datasets from pm4py EventLog objects.

    The user provides a feature extraction function that maps traces/events to tensors.
    """

    def __init__(
        self,
        source_path,
        feature_fn,
        vocab_fn=_build_vocabs,
        max_len=None,
        padding_value=0,
        **kwargs,
    ):
        """
        Args:
            source_path (str): Path to event log file (format depends on subclass).
            feature_fn (Callable -> Callable): Function that takes a vocabulary dict
                                and returns a function that takes a trace (list of events) or an event
                                and returns a tensor or list of tensors.
                                Signature options:
                                    - f(trace: List[dict]) -> Tensor
                                    - f(event: dict) -> Tensor (if event-level granularity)
            vocab_fn (Callable): Function that takes a log and returns a vocabulary dictionary.
            max_len (int): Optional maximum sequence length.
            padding_value (int): Value for sequence padding.
            **kwargs: Passed to subclass loader.
        """
        self.max_len = max_len
        self.padding_value = padding_value
        self.feature_fn_fac = feature_fn

        # Let subclass load the pm4py log
        self.log = self._load_log(source_path, **kwargs)
        self.log = log_converter.apply(
            self.log, variant=log_converter.Variants.TO_EVENT_LOG
        )

        self.vocab_fn = vocab_fn
        self.vocab = self.vocab_fn(self.log)
        self.feature_fn = self.feature_fn_fac(self.vocab)

        # Encode traces via user-supplied function
        self.encoded_traces = [self._encode_trace(trace) for trace in self.log]

    def _load_log(self, source_path, **kwargs):
        raise NotImplementedError

    def _encode_trace(self, trace):
        """Use the user-provided feature_fn to encode a trace."""
        result = self.feature_fn(trace)

        if isinstance(result, list):
            # If user returns a list (e.g. per-event vectors)
            result = torch.stack(result)

        if self.max_len is not None:
            result = result[: self.max_len]

        return result

    def __len__(self):
        return len(self.encoded_traces)

    def __getitem__(self, idx):
        return self.encoded_traces[idx]

    def collate_fn(self, batch):
        return torch.nn.utils.rnn.pad_sequence(
            batch, batch_first=True, padding_value=self.padding_value
        )


""" ============= Example Usage ============= """


# Example feature function
def make_feature_fn(vocab):
    def feature_fn(trace):
        start_time = trace[0]["time:timestamp"]
        vectors = []
        for e in trace:
            activity_id = vocab["concept:name"][e["concept:name"]]
            delta_h = (
                e["time:timestamp"] - start_time
            ).total_seconds() / 3600.0
            vectors.append(
                torch.tensor([activity_id, delta_h], dtype=torch.float)
            )
        return vectors  # list of tensors

    return feature_fn


if __name__ == "__main__":
    from dataloaders.xes import XESEventLogDataset

    path = "data/dx.doi.org_10.4121_uuid_c2c3b154-ab26-4b31-a0e8-8f2350ddac11/BPI_Challenge_2013_closed_problems.xes"

    dataset = XESEventLogDataset(path, feature_fn=make_feature_fn, max_len=50)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, collate_fn=dataset.collate_fn
    )

    for batch in loader:
        print(batch.shape)  # [B, T, feature_dim]
        break
