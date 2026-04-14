"""Alpaca instruction-following benchmark."""

from dataloaders import LocalDataset
from dataloaders.hf_dataset import HFDataset
from src.benchmark import Benchmark
from tasks.completion import Completion


def _to_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


class AlpacaBenchmark(Benchmark):
    """Alpaca instruction-following: generates responses to instructions."""

    def build_input(self, entry):
        inst = _to_text(entry.get("instruction"))
        inp = _to_text(entry.get("input"))
        prompt = inst if not inp else f"{inst}\n\nInput: {inp}"
        opts = {"temperature": 0.0, "max_tokens": 512}
        return prompt, opts

    @classmethod
    def create(cls, model: str, cache_dir: str) -> "AlpacaBenchmark":
        dataset = HFDataset("yahma/alpaca-cleaned", None, "train", cache_dir, limit=3)
        task = Completion(model=model)
        return cls(dataset, task)


class LocalAlpacaBenchmark(Benchmark):
    """Alpaca instruction-following: generates responses to instructions."""

    def build_input(self, entry):
        inst = _to_text(entry.get("instruction"))
        inp = _to_text(entry.get("input"))
        prompt = inst if not inp else f"{inst}\n\nInput: {inp}"
        opts = {"temperature": 0.0, "max_tokens": 1024}
        return prompt, opts

    @classmethod
    def create(cls, model: str, cache_dir: str) -> "LocalAlpacaBenchmark":
        dataset = LocalDataset("alpaca.csv", cache_dir)
        task = Completion(model=model)
        return cls(dataset, task)
