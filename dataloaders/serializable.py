from collections.abc import Iterable
from typing import Protocol, TypeVar, Generic, Callable, Generator
from torch.utils.data import Dataset

S = TypeVar("S", covariant=True)
T = TypeVar("T", covariant=True)


class Serializable(Protocol[S]):
    def serialize(self) -> S: ...


class Deserializable(Protocol[T]):
    def deserialize(self) -> T: ...


class WithSerializedView(Generic[T, S], Iterable[T]):
    type ItemType = T
    type SerializedItemType = S
    type SerializedView = "SerializedView[S]"

    def _get_serialized(self, idx: int) -> S: ...

    @property
    def serialized(self) -> "SerializedView[S]":
        return SerializedView(self, self._get_serialized)


class SerializedView(Dataset[S]):
    type ItemType = S

    def __init__(
        self, dataset: Iterable[T], get_serialized_fn: Callable[[int], S]
    ) -> None:
        """
        Args:
            dataset: The dataset to create a view for
            get_serialized_fn: Function that takes an index and returns SerializedView.ItemType
        """
        self.dataset = dataset
        self.get_serialized_fn = get_serialized_fn

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> S:
        return self.get_serialized_fn(idx)

    def __iter__(self) -> Generator[S, None, None]:
        for i in range(len(self)):
            yield self[i]
