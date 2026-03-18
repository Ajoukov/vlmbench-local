import re
from dataclasses import dataclass
from typing import Dict, Optional

import requests

# global variables
_PREFIX = "vllm:"  # prefix for all metrics to avoid name collisions
_METRICS = [
    "prefix_cache_queries_total",
    "prefix_cache_hits_total",
    "prompt_tokens_total",
    "generation_tokens_total",
    "time_to_first_token_seconds_count",
    "time_to_first_token_seconds_sum",
    "inter_token_latency_seconds_count",
    "inter_token_latency_seconds_sum",
    "request_time_per_output_token_seconds_count",
    "request_time_per_output_token_seconds_sum",
    "e2e_request_latency_seconds_count",
    "e2e_request_latency_seconds_sum",
    "request_inference_time_seconds_count",
    "request_inference_time_seconds_sum",
    "request_prefill_time_seconds_count",
    "request_prefill_time_seconds_sum",
    "request_decode_time_seconds_count",
    "request_decode_time_seconds_sum",
    "request_prefill_kv_computed_tokens_count",
    "request_prefill_kv_computed_tokens_sum",
]  # list of all metrics to fetch
_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{[^}]*\})?"
    r"\s+(?P<value>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
)  # regex to parse metric samples from Prometheus text format


@dataclass
class MetricsSnapshot:
    """Cumulative counter values at a single point in time."""

    metrics: Dict[str, float]

    @property
    def get_metric(self, name: str) -> float:
        """Get the value of a specific metric.

        Parameters
        ----------
        name : str
            The name of the metric to retrieve.

        Returns
        -------
        float
            The value of the specified metric, or 0.0 if it is not present.
        """

        return self.metrics.get(name, 0.0)

    def delta(self, earlier: "MetricsSnapshot") -> Dict[str, float]:
        """Return the deltas of all metrics compared to an earlier snapshot.

        Parameters
        ----------
        earlier : MetricsSnapshot
            The earlier snapshot to compare against.

        Returns
        -------
        Dict[str, float]
            A dictionary mapping metric names to their deltas (current - earlier).
        """

        deltas = {}
        for metric in _METRICS:
            deltas[metric] = max(
                0,
                self.metrics.get(metric, 0.0) - earlier.metrics.get(metric, 0.0),
            )

        return deltas


def _parse_counters(text: str) -> Dict[str, float]:
    """Parse all samples from a Prometheus text payload.

    For multi-label counters (e.g. one row per finished_reason) the values are
    summed together under the bare metric name.

    Parameters
    ----------
    text : str
        The raw text payload from the /metrics endpoint.

    Returns
    -------
    Dict[str, float]
        A dictionary mapping metric names to their cumulative counter values.
    """

    totals: Dict[str, float] = {}

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
    """Fetch the current cumulative counter values from the /metrics endpoint of a VLLM server.

    Parameters
    ----------
    base_url : str
        The base URL of the VLLM server (e.g. "http://localhost:8000").
    timeout : float, optional
        The timeout in seconds for the HTTP request (default is 5.0 seconds).

    Returns
    -------
    Optional[MetricsSnapshot]
        A MetricsSnapshot containing the current counter values, or None if the request failed.
    """

    url = base_url.rstrip("/") + "/metrics"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception:
        return None

    counters = _parse_counters(resp.text)

    values: Dict[str, float] = {}
    for metric in _METRICS:
        values[metric] = counters.get(f"{_PREFIX}{metric}", 0.0)

    return MetricsSnapshot(metrics=values)
