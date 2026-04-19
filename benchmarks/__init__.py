"""Benchmark registry."""

from typing import Dict

from src.benchmark import Benchmark

from .alpaca import LocalAlpacaBenchmark
from .long_doc_qa import LongDocQABenchmark
from .longbench_qmsum import LocalLongBenchQMSumBenchmark
from .narrativeqa import LocalNarrativeQABenchmark
from .sharegpt import LocalShareGPTBenchmark
from .wmt16 import WMT16Benchmark

# from .alpaca import AlpacaBenchmark
# from .longbench_qmsum import LongBenchQMSumBenchmark
# from .narrativeqa import NarrativeQABenchmark
# from .sharegpt import ShareGPTBenchmark
# from .loogle import LooGLEBenchmark
# from .humaneval import HumanEvalBenchmark
# from .kvprobe import KVProbeBenchmark
# from .leval import LEvalBenchmark
# from .longbench_gov import LongBenchGovBenchmark
# from .triviaqa import TriviaQABenchmark
# from .wikitext import WikitextBenchmark

REGISTRY: Dict[str, Benchmark] = {
    # "alpaca": AlpacaBenchmark,
    # "triviaqa": TriviaQABenchmark,
    # "narrativeqa": NarrativeQABenchmark,
    # "wikitext": WikitextBenchmark,
    # "humaneval": HumanEvalBenchmark,
    # "longbench_gov": LongBenchGovBenchmark,
    # "longbench_qmsum": LongBenchQMSumBenchmark,
    # "loogle": LooGLEBenchmark,
    # "leval": LEvalBenchmark,
    # "kvprobe": KVProbeBenchmark,
    # "sharegpt": ShareGPTBenchmark,
    "local_alpaca": LocalAlpacaBenchmark,
    "local_longbench_qmsum": LocalLongBenchQMSumBenchmark,
    "local_narrativeqa": LocalNarrativeQABenchmark,
    "local_sharegpt": LocalShareGPTBenchmark,
    "long_doc_qa": LongDocQABenchmark,
    "wmt16": WMT16Benchmark,
}


def list_all() -> Dict[str, Benchmark]:
    """Return list of all registered benchmark names."""
    return sorted(REGISTRY.keys())
