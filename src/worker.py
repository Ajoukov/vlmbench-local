import json
import queue
import threading
import time
from typing import Any, Dict, Optional

import requests


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

    def record_success(self, latency: float, request_size: int, response_size: int):
        with self._lock:
            self._total += 1
            self._success += 1
            self._total_request_bytes += request_size
            self._total_response_bytes += response_size
            self._latencies.append(latency)

    def record_http_error(self, latency: float, request_size: int, response_size: int):
        with self._lock:
            self._total += 1
            self._http_error += 1
            self._total_request_bytes += request_size
            self._total_response_bytes += response_size
            self._latencies.append(latency)

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
            }

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
    ):
        super().__init__(name=f"worker-{worker_id}", daemon=True)
        self._rto = request_timeout
        self._jobs = jobs
        self._stats = stats

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
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self._rto,
            )

            latency = (time.perf_counter() - start) * 1000

            status = response.status_code
            response_size = len(response.content)

            if status < 400:
                self._stats.record_success(latency, request_size, response_size)
            else:
                self._stats.record_http_error(latency, request_size, response_size)

            llm_meta = self._extract_llm_metadata(response)

            print(
                f"[{status}] {name} "
                f"{self.name} "
                f"latency={latency:.2f}ms "
                f"req={request_size}B "
                f"resp={response_size}B"
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
            print(f"[TIMEOUT] {name} {self.name}")
            return None

        except Exception as e:
            self._stats.record_exception(request_size)
            print(f"[ERROR] {name} {self.name}: {e}")
            return None

    def _extract_llm_metadata(self, response) -> Dict[str, Any]:
        """
        Extract LLM-specific metadata if available.
        Works for OpenAI-compatible APIs.
        """
        try:
            data = response.json()

            return {
                "model": data.get("model"),
                "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": data.get("usage", {}).get("completion_tokens"),
                "total_tokens": data.get("usage", {}).get("total_tokens"),
                "finish_reason": (
                    data.get("choices", [{}])[0].get("finish_reason")
                    if data.get("choices")
                    else None
                ),
            }

        except Exception:
            return {}
