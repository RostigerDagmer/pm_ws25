from pm4py.objects.log.importer.xes import importer as xes_importer
from dataloaders.base import BaseEventLogDataset


class XESEventLogDataset(BaseEventLogDataset):
    """Dataset for XES files."""

    def _load_log(self, source_path, **_):
        return xes_importer.apply(source_path)


# Example usage
if __name__ == "__main__":
    path = "data/d9769f3d-0ab0-4fb8-803b-0d1120ffcf54/Hospital_log.xes"
    dataset = XESEventLogDataset(
        path,
        attribute="concept:name",
    )

    for item in dataset:
        print(item)
        break
