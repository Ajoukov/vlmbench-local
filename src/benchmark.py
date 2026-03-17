from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from .dataset import Dataset
from .task import Task


class Benchmark(ABC):
    """A benchmark is a combination of a dataset and a task.

    Benchmarks are responsible for converting dataset entries into task inputs,
    and then running the task on those inputs.
    """

    def __init__(self, dataset: Dataset, task: Task, limit: int = -1):
        """Initialize the benchmark with a dataset and a task."""

        self.dataset = dataset
        self.task = task
        self.limit = limit

    @abstractmethod
    def build_input(self, entry: Any) -> Tuple[str, Dict]:
        """Given one dataset entry, return (prompt_str, task_opts)."""

        pass

    def run_one(self):
        """Run benchmark on a single dataset entry."""

        entry = self.dataset.next()
        if entry is None:
            return "", {}

        prompt, opts = self.build_input(entry)
        return self.task.payload(prompt, opts)

    def run(self):
        """Generator over the full dataset."""

        count = 0
        while True:
            if self.limit > 0 and count >= self.limit:
                break

            count += 1

            try:
                yield self.run_one()
            except StopIteration:
                break
