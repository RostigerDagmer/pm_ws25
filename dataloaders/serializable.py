from collections.abc import Iterable
from typing import Protocol, TypeVar, Generic, Callable, Generator
from torch.utils.data import Dataset

S = TypeVar("S", covariant=True)
T = TypeVar("T", covariant=True)


class Serializable(Protocol[S]):
    def serialize(self) -> S: ...  # noqa: E704


class Deserializable(Protocol[T]):
    def deserialize(self) -> T: ...  # noqa: E704


class Hashable(Protocol):
    def hash(self) -> str: ...  # noqa: E704


class WithSerializedView(Generic[T, S], Iterable[T]):
    # Type aliases for Python 3.10 compatibility
    # ItemType = T
    # SerializedItemType = S
    # SerializedView = "SerializedView[S]"

    def _get_serialized(self, idx: int) -> S: ...  # noqa: E704

    @property
    def serialized(self) -> "SerializedView[S]":
        return SerializedView(self, self._get_serialized)


class SerializedView(Dataset[S]):
    # Type alias for Python 3.10 compatibility
    # ItemType = S

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
