import torch
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.utils import format_dataframe
import pandas as pd
from dataloaders.base import BaseEventLogDataset, make_feature_fn
from charset_normalizer import from_path


class CSVEventLogDataset(BaseEventLogDataset):
    def _load_log(
        self,
        source_path,
        case_id_col="case:concept:name",
        activity_col="concept:name",
        timestamp_col="time:timestamp",
        sep=",",
        **_,
    ):
        # determine encoding
        result = from_path(source_path).best()
        encoding = result.encoding or 'utf-8-sig'
        # read CSV
        with open(source_path, encoding=encoding, errors='replace') as f:
            df = pd.read_csv(f, sep=sep, index_col=False)
        # standardise columns via format_dataframe (or manually rename)
        df = format_dataframe(
            df,
            case_id=case_id_col,
            activity_key=activity_col,
            timestamp_key=timestamp_col,
        )
        self.case_id_col = case_id_col
        self.activity_col = activity_col
        self.timestamp_col = timestamp_col
        # convert timestamp column to datetime (if not already)
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        # convert to EventLog
        event_log = log_converter.apply(
            df, variant=log_converter.Variants.TO_EVENT_LOG
        )
        return event_log


# Example usage
if __name__ == "__main__":
    path = "data/c3f3ba2d-e81e-4274-87c7-882fa1dbab0d/BPI2016_Werkmap_Messages.csv"
    path = "data/e30ba0c8-0039-4835-a493-6e3aa2301d3f/BPI2016_Complaints.csv"
    # path = "data/9b99a146-51b5-48df-aa70-288a76c82ec4/BPI2016_Clicks_NOT_Logged_In.csv"

    from dataloaders.util import CONSTRUCTION_PARAMS

    params = CONSTRUCTION_PARAMS["e30ba0c8-0039-4835-a493-6e3aa2301d3f"]

    dataset = CSVEventLogDataset(
        path,
        feature_fn=make_feature_fn,
        **params,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, collate_fn=dataset.collate_fn
    )

    for batch in dataloader:
        print(batch)
        break
