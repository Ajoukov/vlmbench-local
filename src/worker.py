import json
import queue
import threading
import time
from typing import Any, Dict, Optional

import requests

from src.metrics import fetch_snapshot


class WorkerStats:
    def __init__(self):
        self._lock = threading.Lock()

        # Counters
        self._total = 0
        self._success = 0
        self._http_error = 0
        self._timeout = 0
        self._exception = 0

        # Bytes
        self._total_request_bytes = 0
        self._total_response_bytes = 0

        # Latency stats
        self._latencies = []

        # Token stats (vLLM usage fields)
        self._total_submitted_tokens = 0   # prompt_tokens sent by the client
        self._total_prefill_tokens = 0     # tokens that required prefill computation
        self._total_decode_tokens = 0      # tokens generated during decode
        self._total_cached_tokens = 0      # tokens served from KV cache

    def record_success(self, latency: float, request_size: int, response_size: int,
                        llm_meta: Optional[Dict[str, Any]] = None):
        with self._lock:
            self._total += 1
            self._success += 1
            self._total_request_bytes += request_size
            self._total_response_bytes += response_size
            self._latencies.append(latency)
            self._accumulate_tokens(llm_meta)

    def record_http_error(self, latency: float, request_size: int, response_size: int,
                          llm_meta: Optional[Dict[str, Any]] = None):
        with self._lock:
            self._total += 1
            self._http_error += 1
            self._total_request_bytes += request_size
            self._total_response_bytes += response_size
            self._latencies.append(latency)
            self._accumulate_tokens(llm_meta)

    def record_timeout(self, request_size: int):
        with self._lock:
            self._total += 1
            self._timeout += 1
            self._total_request_bytes += request_size

    def record_exception(self, request_size: int):
        with self._lock:
            self._total += 1
            self._exception += 1
            self._total_request_bytes += request_size

    def stats(self) -> Dict[str, Any]:
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
                # Token counters
                "total_submitted_tokens": self._total_submitted_tokens,
                "total_prefill_tokens": self._total_prefill_tokens,
                "total_decode_tokens": self._total_decode_tokens,
                "total_cached_tokens": self._total_cached_tokens,
            }

    def _accumulate_tokens(self, llm_meta: Optional[Dict[str, Any]]):
        """Must be called while self._lock is held."""
        if not llm_meta:
            return
        submitted = llm_meta.get("submitted_tokens") or 0
        prefill  = llm_meta.get("prefill_tokens")   or 0
        decode   = llm_meta.get("decode_tokens")    or 0
        cached   = llm_meta.get("cached_tokens")    or 0
        self._total_submitted_tokens += submitted
        self._total_prefill_tokens   += prefill
        self._total_decode_tokens    += decode
        self._total_cached_tokens    += cached

    def _avg_latency(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    def _percentile(self, p: int) -> float:
        if not self._latencies:
            return 0.0
        sorted_lat = sorted(self._latencies)
        k = int(len(sorted_lat) * p / 100)
        return sorted_lat[min(k, len(sorted_lat) - 1)]


class Worker(threading.Thread):
    def __init__(
        self,
        request_timeout: int,
        jobs: "queue.Queue[Optional[Dict[str, Any]]]",
        stats: WorkerStats,
        worker_id: int,
        metrics_base_url: Optional[str] = None,
    ):
        super().__init__(name=f"worker-{worker_id}", daemon=True)
        self._rto = request_timeout
        self._jobs = jobs
        self._stats = stats
        self.worker_id = worker_id
        self._metrics_base_url = metrics_base_url

    def run(self):
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
        request_body = json.dumps(payload)
        request_size = len(request_body.encode("utf-8"))

        start = time.perf_counter()

        try:
            print(f"[REQUEST] {name} {self.name} sending request of size {request_size}B to {url}")

            # --- snapshot metrics BEFORE the request ---
            snap_before = (
                fetch_snapshot(self._metrics_base_url)
                if self._metrics_base_url else None
            )

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self._rto,
            )

            # --- snapshot metrics AFTER the request ---
            snap_after = (
                fetch_snapshot(self._metrics_base_url)
                if self._metrics_base_url else None
            )

            print(f"[RESPONSE] {name} {self.name} received response with status {response.status_code}")

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
                decode_from_metrics  = delta["decode_tokens"]
                submitted = llm_meta.get("submitted_tokens")   # from OpenAI usage
                cached_from_metrics = (
                    (submitted - prefill_from_metrics)
                    if submitted is not None
                    else None
                )
                llm_meta["prefill_tokens"] = prefill_from_metrics
                llm_meta["decode_tokens"]  = decode_from_metrics
                llm_meta["cached_tokens"]  = cached_from_metrics
                llm_meta["metrics_source"] = "vllm:/metrics"
            else:
                llm_meta["metrics_source"] = "openai:usage"

            if status < 400:
                self._stats.record_success(latency, request_size, response_size, llm_meta)
            else:
                self._stats.record_http_error(latency, request_size, response_size, llm_meta)

            submitted = llm_meta.get("submitted_tokens", "?")
            prefill   = llm_meta.get("prefill_tokens",   "?")
            decode    = llm_meta.get("decode_tokens",    "?")
            cached    = llm_meta.get("cached_tokens",    "?")

            print(
                f"[{status}] {name} "
                f"{self.name} "
                f"latency={latency:.2f}ms "
                f"req={request_size}B "
                f"resp={response_size}B "
                f"tokens: submitted={submitted} prefill={prefill} decode={decode} cached={cached}"
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
            print(f"[TIMEOUT] {name} {self.name} request timed out after {self._rto} seconds")
            return None

        except Exception as e:
            self._stats.record_exception(request_size)
            print(f"[ERROR] {name} {self.name}: {e}")
            return None

    def _extract_llm_metadata(self, response) -> Dict[str, Any]:
        """
        Extract LLM-specific metadata if available.
        Works for OpenAI-compatible APIs, with vLLM-specific token detail.

        vLLM token fields (per-request):
          submitted_tokens  – prompt tokens sent by the client
                              (usage.prompt_tokens)
          prefill_tokens    – submitted tokens that required actual prefill
                              computation (submitted - cached)
          decode_tokens     – tokens generated during the decode phase
                              (usage.completion_tokens)
          cached_tokens     – prompt tokens served from the KV prefix cache
                              (usage.prompt_tokens_details.cached_tokens)
        """
        try:
            data = response.json()
            usage = data.get("usage") or {}

            submitted = usage.get("prompt_tokens")
            decode    = usage.get("completion_tokens")
            cached    = (
                (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                # vLLM ≥ 0.4 also exposes num_cached_tokens at the top level
                or usage.get("num_cached_tokens")
            )
            prefill = (
                (submitted - cached)
                if (submitted is not None and cached is not None)
                else submitted
            )

            return {
                "model": data.get("model"),
                # legacy names kept for backward-compat
                "prompt_tokens":     submitted,
                "completion_tokens": decode,
                "total_tokens":      usage.get("total_tokens"),
                "finish_reason": (
                    data.get("choices", [{}])[0].get("finish_reason")
                    if data.get("choices")
                    else None
                ),
                # enriched token breakdown
                "submitted_tokens": submitted,
                "prefill_tokens":   prefill,
                "decode_tokens":    decode,
                "cached_tokens":    cached,
            }

        except Exception:
            return {}
