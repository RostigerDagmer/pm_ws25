import torch
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.utils import format_dataframe
import pandas as pd
from dataloaders.base import BaseEventLogDataset, make_feature_fn


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
        # read CSV
        df = pd.read_csv(source_path, sep=sep, index_col=False)
        # standardise columns via format_dataframe (or manually rename)
        df = format_dataframe(
            df,
            case_id=case_id_col,
            activity_key=activity_col,
            timestamp_key=timestamp_col,
        )
        # convert timestamp column to datetime (if not already)
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        # convert to EventLog
        event_log = log_converter.apply(
            df, variant=log_converter.Variants.TO_EVENT_LOG
        )
        return event_log


# Example usage
if __name__ == "__main__":
    path = "data/dx.doi.org_10.4121_uuid_c3f3ba2d-e81e-4274-87c7-882fa1dbab0d/BPI2016_Werkmap_Messages.csv"
    dataset = CSVEventLogDataset(
        path,
        # attribute="EventType",
        case_id_col="CustomerID",
        timestamp_col="EventDateTime",
        activity_col="HandlingChannelID",
        sep=";",
        feature_fn=make_feature_fn,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, collate_fn=dataset.collate_fn
    )

    for batch in dataloader:
        print(batch)
        break
