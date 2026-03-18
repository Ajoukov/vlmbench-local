from abc import ABC, abstractmethod


class Dataset(ABC):
    """A dataset is a collection of entries that can be used for benchmarking."""

    def __init__(self, address: str):
        """Initialize the dataset with an address (e.g. a file path or URL)."""

        self._addr = address

    def address(self) -> str:
        """Return the address of the dataset."""

        return self._addr

    @abstractmethod
    def count(self) -> int:
        """Total number of entries."""

        pass

    @abstractmethod
    def next(self):
        """Return the next entry, or raise StopIteration."""

        pass
