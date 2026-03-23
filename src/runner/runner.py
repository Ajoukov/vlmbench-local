import json
import queue
import threading
import time
from typing import Any, Dict, Optional

import requests

from src.prometheus import MetricsSnapshot, fetch_snapshot
from src.runner.stats import RunnerStats


class Runner(threading.Thread):
    """Runner thread processes jobs from a shared queue, send requests and records statistics."""

    def __init__(
        self,
        runner_id: int,
        endpoint: str,
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
        endpoint : str
            The base URL of the VLLM server to which the runner will send requests.
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
        self._endpoint = endpoint
        self._rto = request_timeout
        self._jobs = jobs
        self._stats = stats
        self._enable_metrics = enable_metrics
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Stop the thread by setting the stop event."""

        self._stop_event.set()

    def id(self) -> int:
        """Get the unique identifier of this runner.

        Returns
        -------
        int
            The unique identifier of this runner.
        """

        return self._runner_id

    def run(self) -> None:
        """Main loop of the runner thread.

        It continuously processes jobs from the queue until a `None` job is encountered, which signals the runner to stop.
        """

        while True:
            if self._stop_event.is_set():
                return

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
            except Exception as e:
                print(e)
            finally:
                self._jobs.task_done()

    def _process(
        self,
        name: str,
        url: str,
        headers: Dict[str, str],
        payload: Any,
    ) -> None:
        """Sending a request to the specified URL with the given headers and payload, and recording the relevant statistics.

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

        Raises
        ------
        RuntimeError
            If an unexpected error happens when sending the HTTP request.
        """

        # calculate request size in bytes
        request_body = json.dumps(payload)

        # define statistics variables
        http_status: int = 0
        http_req_bytes: int = len(request_body.encode("utf-8"))
        http_res_bytes: int = 0
        http_latency: float = 0
        pre_metrics: MetricsSnapshot = None
        post_metrics: MetricsSnapshot = None

        try:
            # if metrics collection is enabled, take a snapshot of metrics before sending the request
            if self._enable_metrics:
                pre_metrics = fetch_snapshot(base_url=self._endpoint, timeout=self._rto)

            # start the timer for latency measurement
            start = time.perf_counter()

            # send the request
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self._rto,
            )

            # calculate latency in milliseconds
            http_latency = (time.perf_counter() - start) * 1000

            # if metrics collection is enabled, take a snapshot of metrics after receiving the response
            if self._enable_metrics and pre_metrics:
                post_metrics = fetch_snapshot(
                    base_url=self._endpoint, timeout=self._rto
                )

            http_status = response.status_code
            http_res_bytes = len(response.content)

            # record success or error based on status code
            if http_status == 200:
                self._stats.record_success(http_latency, http_req_bytes, http_res_bytes)
            else:
                self._stats.record_error(http_latency, http_req_bytes, http_res_bytes)

        except requests.exceptions.Timeout:
            http_status = 408
            http_latency = (
                self._rto * 1000
            )  # set latency to the request timeout value in milliseconds

            # on timeout record a timeout, but do post metrics poll if enabled
            self._stats.record_timeout(http_req_bytes)

            # if metrics collection is enabled, take a snapshot of metrics after the timeout
            if self._enable_metrics and pre_metrics:
                post_metrics = fetch_snapshot(
                    base_url=self._endpoint, timeout=self._rto
                )

        except Exception as e:
            raise RuntimeError(f"Error while processing an entry: {e}")

        # calculate and print the differences in metrics values before and after the request
        metrics_str = ""
        if self._enable_metrics and pre_metrics and post_metrics:
            values = post_metrics.delta(pre_metrics)
            self._stats.record_vllm_metrics(values)

            metrics_str = "\n".join(
                f"vllm:{metric} = {value:.2f}" for metric, value in values.items()
            )

        # log the process
        print(
            f"[runner-{self.id()}] [{http_status}] {name} "
            f"latency={http_latency:.2f}ms "
            f"req={http_req_bytes}B "
            f"resp={http_res_bytes}B "
            f"\nstart={start} , end={start + (http_latency / 1000)}",
            f"\n{metrics_str}"
        )
