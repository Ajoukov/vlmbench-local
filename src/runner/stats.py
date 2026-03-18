import threading
from typing import Any, Dict


class RunnerStats:
    """Thread-safe statistics collector for Runner threads."""

    def __init__(self):
        # thread lock
        self._lock = threading.Lock()

        # counters
        self._total = 0
        self._success = 0
        self._timeout = 0
        self._error = 0

        # bytes
        self._total_request_bytes = 0
        self._total_response_bytes = 0

        # latency stats
        self._latencies = []

    def record_success(
        self,
        latency: float,
        request_size: int,
        response_size: int,
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
        """

        with self._lock:
            self._total += 1
            self._success += 1
            self._total_request_bytes += request_size
            self._total_response_bytes += response_size
            self._latencies.append(latency)

    def record_error(
        self,
        latency: float,
        request_size: int,
        response_size: int,
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
        """

        with self._lock:
            self._total += 1
            self._error += 1
            self._total_request_bytes += request_size
            self._total_response_bytes += response_size
            self._latencies.append(latency)

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
                "error": self._error,
                "timeout": self._timeout,
                "avg_latency_ms": self._avg_latency(),
                "p95_latency_ms": self._percentile(95),
                "total_request_bytes": self._total_request_bytes,
                "total_response_bytes": self._total_response_bytes,
            }

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
