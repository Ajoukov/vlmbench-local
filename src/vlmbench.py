import os
import sys
import time
from typing import List, Optional

from benchmarks import REGISTRY as BENCHMARK_REGISTRY
from src import Benchmark
from src.runner import Runner, RunnerStats
from src.tokens import truncate_payload
from src.utils import assert_server_up, auto_detect_model, detect_max_model_len
from src.utils.args import build_parser
from src.utils.vars import init_vars


class VLMBench:
    """Orchestrates benchmark and plugin workloads."""

    def __init__(self, argv: Optional[List[str]] = None):
        """Initialize variables and parse command-line arguments."""

        self.vars = init_vars()
        self.argv = argv if argv is not None else sys.argv[1:]
        self.args = None

        # build the argument parser
        self.parser = build_parser(self.vars)

        # common variables
        self._endpoint = None
        self._data_dir = None

        # runners specific usage is with benchmarks
        self._clients = 1
        self._runners = []

    def _list_benchmarks(self) -> None:
        """List all available benchmarks."""

        from benchmarks import list_all

        print("Available benchmarks:")
        for name in list_all():
            print(f"  {name}")

    def _list_plugins(self) -> None:
        """List all available plugins."""

        from plugins import list_all

        print("Available plugins:")
        for name in list_all():
            print(f"  {name}")

    def _run_benchmark(
        self,
        name: str,
        benchmark: Benchmark,
        stop_after: int = 0,
        max_model_len: int = 0,
        truncate: bool = False,
        enable_metrics: bool = False,
    ):
        """Run a benchmark."""

        print(f"\n=== Benchmark: {name} (metrics={enable_metrics}) ===")
        print(f"--- start time: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

        # create a thread-safe stats keeper
        stats = RunnerStats()

        # create runner threads
        self._runners = [
            Runner(
                runner_id=index + 1,
                endpoint=self._endpoint,
                stats=stats,
                request_timeout=self.vars["REQUEST_TIMEOUT"],
                enable_metrics=enable_metrics,
            )
            for index in range(max(1, self._clients))
        ]

        # start each runner thread
        for runner in self._runners:
            runner.start()

        # keep track of request count and last runner
        request_count = 0
        last_runner = 0

        # call benchmark.run method to get benchmark cases sequentially
        for result in benchmark.run():
            # extract the uri and payload
            uri = result.get("uri", None)
            payload = result.get("payload", None)

            # validation checks
            if uri is None or payload is None:
                raise RuntimeError(
                    f"Benchmark `{name}` return must include a uri and a payload!"
                )

            if not payload.get("prompt") and not payload.get("messages"):
                continue

            # trucate payload if requested
            if truncate and max_model_len > 0:
                payload = truncate_payload(
                    endpoint=self._endpoint,
                    payload=payload,
                    max_model_len=max_model_len,
                    timeout_s=self.vars["REQUEST_TIMEOUT"],
                )

            # build a request template
            template = {
                "url": f"{self._endpoint.rstrip('/')}/v1{uri}",
                "headers": {"Content-Type": "application/json"},
                "payload": payload,
            }

            # select a runner using round-robin
            runner = self._runners[last_runner % len(self._runners)]
            last_runner += 1

            runner.queue_job(
                {
                    "name": name,
                    "url": template["url"],
                    "headers": template["headers"],
                    "payload": template["payload"],
                }
            )

            # check stop after condition
            request_count += 1
            if stop_after > 0 and request_count > stop_after:
                break

        # send a None job to stop runners after processing all requests
        for runner in self._runners:
            runner.queue_job(None)

        # wait for runners
        for runner in self._runners:
            runner.join()

        # print the summary of benchmark
        summary = stats.stats()

        n = summary["total_requests"]
        ok = summary["success"]
        fail = summary["error"] + summary["timeout"]
        total_request_bytes = summary["total_request_bytes"]
        total_response_bytes = summary["total_response_bytes"]
        average_latency = summary["avg_latency_ms"]
        p95_latency = summary["p95_latency_ms"]

        print(f"\n--- {name}: {n} requests, {ok} ok, {fail} failed ---")
        print(f"Total request bytes: {total_request_bytes}")
        print(f"Total response bytes: {total_response_bytes}")
        print(f"Average latency: {average_latency:.2f} ms")
        print(f"95th percentile latency: {p95_latency:.2f} ms")

        # print vLLM metrics if available
        vllm_metrics = stats.vllm_stats()
        for metric_name, value in vllm_metrics.items():
            print(f"vllm:{metric_name} = {value}")

        print(f"--- end time: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

        return n, ok, fail

    def _run_bench_command(self) -> None:
        """Run bench group commands."""

        # get input arguments
        args = self.args

        # check for list command
        if args.list:
            self._list_benchmarks()
            return

        # arguments validation checks
        if not args.benchmarks:
            raise RuntimeError("No benchmarks specified!")

        if args.clients < 1:
            raise RuntimeError("Invalid number of clients, --clients must be >= 1.")
        self._clients = args.clients

        for name in args.benchmarks:
            if name not in BENCHMARK_REGISTRY:
                raise RuntimeError(f"Unknown benchmark: {name}")

        # extract the endpoint and datasets directory
        self._endpoint = args.endpoint
        self._data_dir = os.environ.get("HF_HOME") or args.data_dir

        # check if server is healthy
        print(f"[CHECK] Checking server at {self._endpoint}")
        try:
            assert_server_up(self._endpoint)
        except Exception as e:
            raise RuntimeError(f"Cannot reach server at {self._endpoint}: {e}")
        print("[CHECK] Server is up.")

        # detect the model
        model = args.model or auto_detect_model(self._endpoint)
        print(f"[CHECK] Model: {model}")

        # get maximum model size (for truncation)
        max_model_len = 0
        if args.truncate:
            max_model_len = detect_max_model_len(
                self._endpoint, model, timeout_s=self.vars["REQUEST_TIMEOUT"]
            )
            print(f"[CHECK] Max model length: {max_model_len} (truncation enabled)")

        # check the data directory
        os.makedirs(self._data_dir, exist_ok=True)
        print(f"[CHECK] Data directory: {self._data_dir}")

        # keep track of benchmarks status
        total_n = 0
        total_ok = 0
        total_fail = 0

        print(
            f"\n=== Running {len(args.benchmarks)} benchmark(s) sequentially with {args.clients} client(s) each ==="
        )

        for name in args.benchmarks:
            # get the benchmark from registery
            bench_cls = BENCHMARK_REGISTRY[name]

            # create the benchmark
            benchmark = bench_cls.create(model=model, cache_dir=self._data_dir)

            # run and update the status
            n, ok, fail = self._run_benchmark(
                name=name,
                benchmark=benchmark,
                stop_after=args.stop_after,
                max_model_len=max_model_len,
                truncate=args.truncate,
                enable_metrics=args.enable_prometheus_metrics,
            )

            total_n += n
            total_ok += ok
            total_fail += fail

        print(
            f"\n=== All done: {total_n} total requests, {total_ok} ok, {total_fail} failed ==="
        )

    def _run_plugin_command(self) -> None:
        """Run plugin group commands."""

        # get input arguments
        args = self.args

        # check for list command
        if args.list:
            self._list_plugins()
            return

        # arguments validation checks
        if not getattr(args, "plugin_name", None):
            raise RuntimeError("No plugin specified!")

        if not hasattr(args, "plugin_runner"):
            raise RuntimeError("Invalid plugin specified!")

        # extract the datasets directory
        self._data_dir = os.environ.get("HF_HOME") or args.data_dir

        # check the data directory
        os.makedirs(self._data_dir, exist_ok=True)
        print(f"[CHECK] Data directory: {self._data_dir}")

        # run the plugin
        args.plugin_runner(args)

    def run(self) -> None:
        """Run VLMBench by the root subgroup."""

        # extract the arguments
        argv = self.argv
        if len(argv) >= 3 and argv[0] == "plugin" and argv[1] in ("-h", "--help"):
            argv = ["plugin", argv[2], "--help", *argv[3:]]

        # check the command
        self.args = self.parser.parse_args(argv)
        if self.args.command is None:
            raise RuntimeError("No command specified")

        # check if output export is given
        if hasattr(self.args, "export_output") and self.args.export_output:
            from src.tee import make_tee

            make_tee(self.args.export_output)
            print(f"[CHECK] Exporting output to {self.args.export_output}")

        # run `bench` group
        if self.args.command == "bench":
            self._run_bench_command()
            return

        # run `plugin` group
        if self.args.command == "plugin":
            self._run_plugin_command()
            return

    def shutdown(self) -> None:
        """Shutdown by calling stop on existing runners."""

        # the runners are generated by `bench` group
        for runner in self._runners:
            runner.stop()
            runner.join()
