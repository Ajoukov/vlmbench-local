from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import requests

# prometheus metric names we care about
_PROMPT_METRIC = "vllm:prompt_tokens_total"
_GENERATION_METRIC = "vllm:generation_tokens_total"
_PREFIX_CACHED_METRIC = "vllm:prefix_cache_hits"

# line format:  metric_name{labels} value [timestamp]
_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{[^}]*\})?"
    r"\s+(?P<value>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
)


@dataclass
class MetricsSnapshot:
    """Cumulative counter values at a single point in time."""

    prompt_tokens_total: float
    generation_tokens_total: float
    prefix_cache_hits: float

    def delta(self, earlier: "MetricsSnapshot") -> dict:
        """
        Return per-request token counts relative to an *earlier* snapshot.

        Keys:
            prefill_tokens      : tokens that actually ran through prefill
            decode_tokens       : tokens generated during decode
            prefix_cache_hits   : tokens that were cached
        """

        return {
            "prefill_tokens": max(
                0, self.prompt_tokens_total - earlier.prompt_tokens_total
            ),
            "decode_tokens": max(
                0, self.generation_tokens_total - earlier.generation_tokens_total
            ),
            "prefix_cache_hits": max(
                0, self.prefix_cache_hits - earlier.prefix_cache_hits
            ),
        }


def _parse_counters(text: str) -> dict[str, float]:
    """Parse all samples from a Prometheus text payload.

    For multi-label counters (e.g. one row per finished_reason) the values are
    summed together under the bare metric name.
    """

    totals: dict[str, float] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)

        if m:
            name = m.group("name")
            value = float(m.group("value"))
            totals[name] = totals.get(name, 0.0) + value

    return totals


def fetch_snapshot(base_url: str, timeout: float = 5.0) -> Optional[MetricsSnapshot]:
    """GET {base_url}/metrics and return a MetricsSnapshot.
    Returns None if the endpoint is unreachable or the metrics are missing.

    """

    url = base_url.rstrip("/") + "/metrics"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception:
        return None

    counters = _parse_counters(resp.text)
    if (
        _PROMPT_METRIC not in counters
        or _GENERATION_METRIC not in counters
        or _PREFIX_CACHED_METRIC not in counters
    ):
        return None

    return MetricsSnapshot(
        prompt_tokens_total=counters[_PROMPT_METRIC],
        generation_tokens_total=counters[_GENERATION_METRIC],
        prefix_cache_hits=counters[_PREFIX_CACHED_METRIC],
    )
