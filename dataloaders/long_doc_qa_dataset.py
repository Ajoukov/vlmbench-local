"""Long-doc QA synthetic dataset — generates a rotating-document sequence in-process.

N distinct wikitext passages are built at init, each replicated REPEAT_COUNT
times in the configured order. This working-set shape exceeds vLLM's in-GPU
prefix-cache capacity, forcing evicted document blocks to be recovered from an
external KV tier (e.g. LMCache) on repeats — which is what makes the storage
layer measurable. Mirrors the shape of LMCache's Long Doc QA benchmark:
https://docs.lmcache.ai/getting_started/benchmarking.html
"""

import random
from typing import Optional

from plugins.simulator.text_sources import WikitextSource
from src.dataset import Dataset


class LongDocQADataset(Dataset):
    NUM_DOCUMENTS = 20
    DOCUMENT_TOKENS = 4000
    REPEAT_COUNT = 3
    REPEAT_MODE = "random"  # "random" | "tile" | "interleave"
    SEED = 42
    CHARS_PER_TOKEN = 4
    INSTRUCTION = "Summarize the above passage in three to five sentences."

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(address="long_doc_qa://synthetic")

        rng = random.Random(self.SEED)
        source = WikitextSource(cache_dir=cache_dir, seed=self.SEED)

        target_chars = self.DOCUMENT_TOKENS * self.CHARS_PER_TOKEN
        docs = [
            source.fetch_passage(min_chars=target_chars // 2, max_chars=target_chars)
            for _ in range(self.NUM_DOCUMENTS)
        ]
        base_prompts = [f"{doc}\n\n{self.INSTRUCTION}" for doc in docs]

        if self.REPEAT_MODE == "random":
            seq = list(base_prompts) * self.REPEAT_COUNT
            rng.shuffle(seq)
        elif self.REPEAT_MODE == "tile":
            seq = list(base_prompts) * self.REPEAT_COUNT
        elif self.REPEAT_MODE == "interleave":
            seq = [p for p in base_prompts for _ in range(self.REPEAT_COUNT)]
        else:
            raise ValueError(f"unknown REPEAT_MODE: {self.REPEAT_MODE}")

        self._entries = seq
        self._idx = 0

    def count(self) -> int:
        return len(self._entries)

    def next(self):
        if self._idx >= len(self._entries):
            raise StopIteration
        entry = {"prompt": self._entries[self._idx]}
        self._idx += 1
        return entry

    def reset(self):
        self._idx = 0
