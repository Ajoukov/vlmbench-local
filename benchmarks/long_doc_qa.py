"""Long-doc QA rotating-document workload."""

from dataloaders.long_doc_qa_dataset import LongDocQADataset
from src.benchmark import Benchmark
from tasks.completion import Completion


class LongDocQABenchmark(Benchmark):
    GEN_TOKENS = 200

    def build_input(self, entry):
        prompt = entry.get("prompt", "")
        opts = {"temperature": 0.0, "max_tokens": self.GEN_TOKENS}
        return prompt, opts

    @classmethod
    def create(cls, model: str, cache_dir: str) -> "LongDocQABenchmark":
        dataset = LongDocQADataset(cache_dir=cache_dir)
        task = Completion(model=model)
        return cls(dataset, task)
