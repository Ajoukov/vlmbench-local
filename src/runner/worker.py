import json
import queue
import threading
import time
from typing import Any, Dict, Optional

import requests

from src.prometheus.metrics import fetch_snapshot


class WorkerStats:
    """Thread-safe statistics collector for Worker threads."""

    def __init__(self):
        # thread lock
        self._lock = threading.Lock()

        # counters
        self._total = 0
        self._success = 0
        self._http_error = 0
        self._timeout = 0
        self._exception = 0

        # bytes
        self._total_request_bytes = 0
        self._total_response_bytes = 0

        # latency stats
        self._latencies = []

        # token stats (vLLM usage fields)
        self._total_submitted_tokens = 0  # prompt_tokens sent by the client
        self._total_prefill_tokens = 0  # tokens that required prefill computation
        self._total_decode_tokens = 0  # tokens generated during decode
        self._total_cached_tokens = 0  # tokens served from KV cache

    def record_success(
        self,
        latency: float,
        request_size: int,
        response_size: int,
        llm_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a successful request.

        Parameters
        ----------
        latency : float
            The latency of the request in milliseconds.
        request_size : int
            The size of the request body in bytes.
        response_size : int
            The size of the response body in bytes.
        llm_meta : Optional[Dict[str, Any]]
            Optional metadata about the LLM request, such as token counts.
        """

        with self._lock:
            self._total += 1
            self._success += 1
            self._total_request_bytes += request_size
            self._total_response_bytes += response_size
            self._latencies.append(latency)
            self._accumulate_tokens(llm_meta)

    def record_http_error(
        self,
        latency: float,
        request_size: int,
        response_size: int,
        llm_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a failed request due to an HTTP error (status code >= 400).

        Parameters
        ----------
        latency : float
            The latency of the request in milliseconds.
        request_size : int
            The size of the request body in bytes.
        response_size : int
            The size of the response body in bytes.
        llm_meta : Optional[Dict[str, Any]]
            Optional metadata about the LLM request, such as token counts.
        """

        with self._lock:
            self._total += 1
            self._http_error += 1
            self._total_request_bytes += request_size
            self._total_response_bytes += response_size
            self._latencies.append(latency)
            self._accumulate_tokens(llm_meta)

    def record_timeout(self, request_size: int) -> None:
        """Record a request that resulted in a timeout.

        Parameters
        ----------
        request_size : int
            The size of the request body in bytes.
        """

        with self._lock:
            self._total += 1
            self._timeout += 1
            self._total_request_bytes += request_size

    def record_exception(self, request_size: int) -> None:
        """Record a request that resulted in an exception (e.g., network error).

        Parameters
        ----------
        request_size : int
            The size of the request body in bytes.
        """

        with self._lock:
            self._total += 1
            self._exception += 1
            self._total_request_bytes += request_size

    def stats(self) -> Dict[str, Any]:
        """Return a snapshot of the current statistics in a thread-safe manner.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the current statistics, including total requests,
            success count, error counts, average latency, token counts, etc.
        """

        with self._lock:
            return {
                "total_requests": self._total,
                "success": self._success,
                "http_error": self._http_error,
                "timeout": self._timeout,
                "exception": self._exception,
                "avg_latency_ms": self._avg_latency(),
                "p95_latency_ms": self._percentile(95),
                "total_request_bytes": self._total_request_bytes,
                "total_response_bytes": self._total_response_bytes,
                "total_submitted_tokens": self._total_submitted_tokens,
                "total_prefill_tokens": self._total_prefill_tokens,
                "total_decode_tokens": self._total_decode_tokens,
                "total_cached_tokens": self._total_cached_tokens,
            }

    def _accumulate_tokens(self, llm_meta: Optional[Dict[str, Any]]) -> None:
        """Calculate and accumulate token counts from the provided LLM metadata.

        Parameters
        ----------
        llm_meta : Optional[Dict[str, Any]]
            A dictionary containing LLM metadata, which may include token counts such as
            "submitted_tokens", "prefill_tokens", "decode_tokens", and "cached_tokens".
        """

        if not llm_meta:
            return

        submitted = llm_meta.get("submitted_tokens") or 0
        prefill = llm_meta.get("prefill_tokens") or 0
        decode = llm_meta.get("decode_tokens") or 0
        cached = llm_meta.get("cached_tokens") or 0

        self._total_submitted_tokens += submitted
        self._total_prefill_tokens += prefill
        self._total_decode_tokens += decode
        self._total_cached_tokens += cached

    def _avg_latency(self) -> float:
        """Calculate the average latency from the recorded latencies.

        Returns
        -------
        float
            The average latency in milliseconds. Returns 0.0 if no latencies are recorded.
        """

        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    def _percentile(self, p: int) -> float:
        """Calculate the p-th percentile latency from the recorded latencies.

        Parameters
        ----------
        p : int
            The desired percentile (e.g., 95 for the 95th percentile).

        Returns
        -------
        float
            The p-th percentile latency in milliseconds. Returns 0.0 if no latencies are recorded.
        """

        if not self._latencies:
            return 0.0

        sorted_lat = sorted(self._latencies)
        k = int(len(sorted_lat) * p / 100)

        return sorted_lat[min(k, len(sorted_lat) - 1)]


class Worker(threading.Thread):
    """Worker thread that processes jobs from a shared queue, send requests
    and records statistics."""

    def __init__(
        self,
        request_timeout: int,
        jobs: "queue.Queue[Optional[Dict[str, Any]]]",
        stats: WorkerStats,
        worker_id: int,
        metrics_base_url: Optional[str] = None,
    ):
        """Initialize the Worker thread.

        Parameters
        ----------
        request_timeout : int
            The timeout for each request in seconds.
        jobs : queue.Queue[Optional[Dict[str, Any]]]
            A thread-safe queue from which the worker will consume jobs. Each job is a dictionary
            containing the request details (e.g., name, url, headers, payload).
            A job of `None` is a signal to stop the worker.
        stats : WorkerStats
            An instance of WorkerStats for recording statistics about the requests processed by this worker.
        worker_id : int
            A unique identifier for this worker, used for logging purposes.
        metrics_base_url : Optional[str]
            An optional base URL for fetching metrics snapshots before and after requests.
            If provided, the worker will attempt to fetch snapshots from this URL to enrich
            the recorded statistics with LLM token breakdowns.
        """

        super().__init__(name=f"worker-{worker_id}", daemon=True)
        self.worker_id = worker_id
        self._rto = request_timeout
        self._jobs = jobs
        self._stats = stats
        self._metrics_base_url = metrics_base_url

    def run(self):
        """Main loop of the worker thread that continuously processes jobs from the queue until a
        `None` job is encountered, which signals the worker to stop."""

        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return

                self.process(
                    name=job["name"],
                    url=job["url"],
                    headers=job["headers"],
                    payload=job["payload"],
                )
            finally:
                self._jobs.task_done()

    def process(
        self,
        name: str,
        url: str,
        headers: Dict[str, str],
        payload: Any,
    ) -> Optional[Dict[str, Any]]:
        """Process a single job by sending a request to the specified URL with the given headers and payload,
        and recording the relevant statistics.

        Parameters
        ----------
        name : str
            A name for the request, used for logging purposes.
        url : str
            The URL to which the request will be sent.
        headers : Dict[str, str]
            A dictionary of HTTP headers to include in the request.
        payload : Any
            The body of the request, which will be JSON-encoded before sending.

        Returns
        -------
        Optional[Dict[str, Any]]
            A dictionary containing the response status, latency, request/response sizes, and LLM metadata
            if the request was successful or resulted in an HTTP error. Returns `None` if the request timed out
            or resulted in an exception.
        """

        request_body = json.dumps(payload)
        request_size = len(request_body.encode("utf-8"))

        start = time.perf_counter()

        try:
            # snapshot metrics BEFORE the request
            snap_before = (
                fetch_snapshot(self._metrics_base_url)
                if self._metrics_base_url
                else None
            )

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self._rto,
            )

            # dump the response in pretty format for debugging
            print(f"Response from {name} ({self.name}): {response.status_code}\n{json.dumps(response.json(), indent=2)}"
            )

            # snapshot metrics AFTER the request
            snap_after = (
                fetch_snapshot(self._metrics_base_url)
                if self._metrics_base_url
                else None
            )

            latency = (time.perf_counter() - start) * 1000

            status = response.status_code
            response_size = len(response.content)
            llm_meta = self._extract_llm_metadata(response)

            # Override prefill/decode/cached from /metrics deltas when available.
            # This is more reliable than the OpenAI usage field because vLLM's
            # prompt_tokens_total only counts tokens that actually ran prefill
            # (i.e. cache misses), while generation_tokens_total counts decode.
            if snap_before is not None and snap_after is not None:
                delta = snap_after.delta(snap_before)

                prefill_from_metrics = delta["prefill_tokens"]
                decode_from_metrics = delta["decode_tokens"]
                prefix_cached_from_metrics = delta["prefix_cache_hits"]

                submitted = llm_meta.get("submitted_tokens")  # from OpenAI usage
                cached_from_metrics = (
                    (submitted - prefill_from_metrics)
                    if submitted is not None
                    else None
                )

                llm_meta["prefill_tokens"] = prefill_from_metrics
                llm_meta["decode_tokens"] = decode_from_metrics
                llm_meta["cached_tokens"] = cached_from_metrics
                llm_meta["prefix_cache_hits"] = prefix_cached_from_metrics
                llm_meta["metrics_source"] = "vllm:/metrics"
            else:
                llm_meta["metrics_source"] = "openai:usage"

            if status < 400:
                self._stats.record_success(
                    latency, request_size, response_size, llm_meta
                )
            else:
                self._stats.record_http_error(
                    latency, request_size, response_size, llm_meta
                )

            submitted = llm_meta.get("submitted_tokens", "?")
            prefill = llm_meta.get("prefill_tokens", "?")
            decode = llm_meta.get("decode_tokens", "?")
            cached = llm_meta.get("cached_tokens", "?")
            prefix_cache_hits = llm_meta.get("prefix_cache_hits", "?")

            print(
                f"[{status}] {name} "
                f"{self.name} "
                f"latency={latency:.2f}ms "
                f"req={request_size}B "
                f"resp={response_size}B "
                f"tokens: submitted={submitted} prefill={prefill} decode={decode} cached={cached} prefix_cache_hits={prefix_cache_hits}"
            )

            return {
                "status": status,
                "latency_ms": latency,
                "request_bytes": request_size,
                "response_bytes": response_size,
                "llm_meta": llm_meta,
            }

        except requests.exceptions.Timeout:
            self._stats.record_timeout(request_size)
            print(
                f"[TIMEOUT] {name} {self.name} request timed out after {self._rto} seconds"
            )
            return None

        except Exception as e:
            self._stats.record_exception(request_size)
            print(f"[ERROR] {name} {self.name}: {e}")
            return None

    def _extract_llm_metadata(self, response) -> Dict[str, Any]:
        """Extract LLM-related metadata from the response, such as token counts from the OpenAI usage field.

        Parameters
        ----------
        response : requests.Response
            The HTTP response object from which to extract LLM metadata.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing LLM metadata such as "model", "prompt_tokens", "completion_tokens",
            "total_tokens", "finish_reason", "submitted_tokens", "prefill_tokens", "decode_tokens", "cached_tokens",
            and "prefix_cache_hits".
            If the response does not contain valid JSON or the expected fields, returns an empty dictionary.
        """

        try:
            data = response.json()
            usage = data.get("usage") or {}

            submitted = usage.get("prompt_tokens")
            decode = usage.get("completion_tokens")
            cached = (usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens"
            ) or usage.get("num_cached_tokens")
            prefill = (
                (submitted - cached)
                if (submitted is not None and cached is not None)
                else submitted
            )

            return {
                "model": data.get("model"),
                "prompt_tokens": submitted,
                "completion_tokens": decode,
                "total_tokens": usage.get("total_tokens"),
                "finish_reason": (
                    data.get("choices", [{}])[0].get("finish_reason")
                    if data.get("choices")
                    else None
                ),
                "submitted_tokens": submitted,
                "prefill_tokens": prefill,
                "decode_tokens": decode,
                "prefix_cache_hits": 0,
            }

        except Exception:
            return {}
