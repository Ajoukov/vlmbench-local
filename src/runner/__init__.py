import json
import queue
import threading
import time
from typing import Any, Dict, Optional

import requests

from src.prometheus import MetricsSnapshot, fetch_snapshot
from src.runner.stats import RunnerStats


class Runner(threading.Thread):
    """Runner thread that processes jobs from a shared queue, send requests and records statistics."""

    def __init__(
        self,
        runner_id: int,
        jobs: "queue.Queue[Optional[Dict[str, Any]]]",
        stats: RunnerStats,
        request_timeout: int,
        enable_metrics: bool = False,
    ):
        """Initialize the Runner thread.

        Parameters
        ----------
        runner_id : int
            A unique identifier for this runner thread.
        jobs : queue.Queue[Optional[Dict[str, Any]]]
            A thread-safe queue from which the runner will consume jobs.
        stats : RunnerStats
            A shared statistics collector that the runner will use to record request outcomes.
        request_timeout : int
            The timeout in seconds for each request sent by the runner.
        enable_metrics : bool, optional
            Whether to enable metrics collection from the /metrics endpoint (default is False).
        """

        super().__init__(name=f"runner-{runner_id}", daemon=True)

        self._runner_id = runner_id
        self._rto = request_timeout
        self._jobs = jobs
        self._stats = stats
        self._enable_metrics = enable_metrics

    def id(self) -> int:
        """Get the unique identifier of this runner.

        Returns
        -------
        int
            The unique identifier of this runner.
        """

        return self._runner_id

    def run(self):
        """Main loop of the runner thread.

        It continuously processes jobs from the queue until a `None` job is encountered, which signals the runner to stop.
        """

        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return

                self._process(
                    name=job["name"],
                    url=job["url"],
                    headers=job["headers"],
                    payload=job["payload"],
                )
            finally:
                self._jobs.task_done()

    def _process(
        self,
        name: str,
        url: str,
        headers: Dict[str, str],
        payload: Any,
    ):
        """Process a single job.

        Sending a request to the specified URL with the given headers and payload, and recording the relevant statistics.

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
        """

        # calculate request size in bytes
        request_body = json.dumps(payload)
        request_size = len(request_body.encode("utf-8"))

        # define metrics variables
        pre_metrics: MetricsSnapshot = None
        post_metrics: MetricsSnapshot = None

        # start the timer for latency measurement
        start = time.perf_counter()

        try:
            if self._enable_metrics:
                pre_metrics = fetch_snapshot(base_url=url, timeout=self._rto)

            # send the request
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self._rto,
            )

            if self._enable_metrics and pre_metrics:
                post_metrics = fetch_snapshot(base_url=url, timeout=self._rto)

            # calculate latency in milliseconds
            latency = (time.perf_counter() - start) * 1000

            status = response.status_code
            response_size = len(response.content)

            # record success or error based on status code
            if status == 200:
                self._stats.record_success(latency, request_size, response_size)
            else:
                self._stats.record_error(latency, request_size, response_size)

            print(
                f"[{status}] {name} "
                f"{self.name} "
                f"latency={latency:.2f}ms "
                f"req={request_size}B "
                f"resp={response_size}B "
            )

        except requests.exceptions.Timeout:
            self._stats.record_timeout(request_size)
            print(
                f"[TIMEOUT] {name} {self.name} request timed out after {self._rto} seconds"
            )

        except Exception as e:
            self._stats.record_error(request_size)
            print(f"[ERROR] {name} {self.name}: {e}")

        # calculate and print the differences in metrics values before and after the request
        if self._enable_metrics and pre_metrics and post_metrics:
            values = post_metrics.delta(pre_metrics)
            metrics_str = " ".join(
                f"{metric}={value:.2f}" for metric, value in values.items()
            )
            print(f"{metrics_str}")
