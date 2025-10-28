import torch
from pm4py.objects.log.importer.xes import importer as xes_importer
from dataloaders.base import BaseEventLogDataset, make_feature_fn


class XESEventLogDataset(BaseEventLogDataset):
    """Dataset for XES files."""

    def _load_log(self, source_path, **_):
        return xes_importer.apply(source_path)


# Example usage
if __name__ == "__main__":
    path = "data/d9769f3d-0ab0-4fb8-803b-0d1120ffcf54/Hospital_log.xes"
    dataset = XESEventLogDataset(
        path, attribute="concept:name", feature_fn=make_feature_fn
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, collate_fn=dataset.collate_fn
    )

    for batch in dataloader:
        print(batch)
        break
