import argparse
import os
import queue
import random
import sys
import time
from typing import Any, Dict, Optional

from benchmarks import REGISTRY as BENCHMARK_REGISTRY
from plugins import register_subcommands
from src import Benchmark
from src.runner import Runner, RunnerStats
from src.tokens import truncate_payload
from src.utils import assert_server_up, auto_detect_model, detect_max_model_len
from src.utils.vars import init_vars


class VLMBench:
    """Orchestrates benchmark and plugin workloads."""

    def __init__(self, argv: Optional[list[str]] = None):
        """Initialize variables and parse command-line arguments."""

        self.vars = init_vars()
        self.argv = argv if argv is not None else sys.argv[1:]
        self.args = None
        self.parser = self._build_parser()

        # runners specific usage is with benchmarks
        self._runners = []

    def _build_parser(self) -> argparse.ArgumentParser:
        """Builds the command-line argument parser with all subcommands.

        Returns
        -------
        ap : argparse.ArgumentParser
            The configured argument parser for CLI.
        """

        # the root argparser
        ap = argparse.ArgumentParser(
            description="VLMBench - benchmarking workloads for vLLM. By File Systems & Storage Lab @ Stony Brook University (2024-2026).",
        )

        # common flags
        common = argparse.ArgumentParser(add_help=False)
        common.add_argument(
            "--endpoint",
            default=self.vars["DEFAULT_ENDPOINT"],
            help=f"vLLM OpenAI API address (default: {self.vars['DEFAULT_ENDPOINT']})",
        )
        common.add_argument(
            "--model",
            default=None,
            help="Model name (auto-detected from API if omitted)",
        )
        common.add_argument(
            "--data-dir",
            default=self.vars["DEFAULT_DATA_DIR"],
            help=f"Local datasets and cache directory (default: {self.vars['DEFAULT_DATA_DIR']})",
        )
        common.add_argument(
            "--enable-prometheus-metrics",
            action="store_true",
            help="Enable Prometheus metrics collection (fetches cumulative values from /metrics API before and after benchmarks, and shows the differences)",
        )
        common.add_argument(
            "--export-output",
            type=str,
            default=None,
            help="Export all benchmark results to a file (in addition to console output; specify the file path as the argument)",
        )

        # subparser groups
        subparsers = ap.add_subparsers(dest="command")

        # `bench` group
        self._bench_parser = subparsers.add_parser(
            "bench",
            parents=[common],
            help="Run benchmarks",
            description="Run benchmark workloads against a vLLM instance using OpenAI API",
        )
        self._bench_parser.add_argument(
            "--list", action="store_true", help="List available benchmarks"
        )
        self._bench_parser.add_argument(
            "--stop-after",
            type=int,
            default=0,
            help="Stop after processing this many entries (for limiting tests; default: 0, meaning no limit, until the dataset is over)",
        )
        self._bench_parser.add_argument(
            "--truncate",
            action="store_true",
            help="Truncate inputs that exceed the model's context window (WARN: this might change a prompt to fit with model's context window)",
        )
        self._bench_parser.add_argument(
            "--clients",
            type=int,
            default=1,
            help="Number of concurrent clients (default: 1)",
        )
        self._bench_parser.add_argument(
            "--random-populate",
            action="store_true",
            help="Populate requests by random sampling from benchmark entries",
        )
        self._bench_parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Seed for random population (deterministic when used with --random-populate)",
        )
        self._bench_parser.add_argument(
            "--random-batch-size",
            type=int,
            default=100,
            help="Number of entries to buffer per batch in --random-populate mode (default: 100)",
        )
        self._bench_parser.add_argument(
            "benchmarks",
            nargs="*",
            help="Benchmark names to run",
        )

        # `plugin` group
        self._plugin_parser = subparsers.add_parser(
            "plugin",
            help="Run plugins",
            description="Run plugin workloads against a vLLM instance",
        )
        self._plugin_parser.add_argument(
            "--list", action="store_true", help="List available plugins"
        )

        # register plugin subcommands
        plugin_subparsers = self._plugin_parser.add_subparsers(dest="plugin_name")
        register_subcommands(plugin_subparsers, parents=[common])

        return ap

    def _list_benchmarks(self) -> None:
        from benchmarks import list_all

        print("Available benchmarks:")
        for name in list_all():
            print(f"  {name}")

    def _list_plugins(self) -> None:
        from plugins import list_all

        print("Available plugins:")
        for name in list_all():
            print(f"  {name}")

    def _run_benchmark(
        self,
        name: str,
        benchmark: Benchmark,
        endpoint: str,
        clients: int,
        truncate: bool = False,
        max_model_len: int = 0,
        enable_metrics: bool = False,
        random_populate: bool = False,
        seed: int | None = None,
        random_batch_size: int = 100,
    ):
        """Run a benchmark."""

        print(f"\n=== Benchmark: {name} (metrics={enable_metrics}) ===")
        print(f"--- start time: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

        # create a job queue
        jobs: "queue.Queue[Dict[str, Any] | None]" = queue.Queue()

        # create a thread-safe stats keeper
        stats = RunnerStats()

        # create runner threads
        self._runners = [
            Runner(
                runner_id=index + 1,
                endpoint=endpoint,
                jobs=jobs,
                stats=stats,
                request_timeout=self.vars["REQUEST_TIMEOUT"],
                enable_metrics=enable_metrics,
            )
            for index in range(max(1, clients))
        ]

        # start each runner thread
        for runner in self._runners:
            runner.start()

        # create a random range if random-populate is enabled
        rng = random.Random(seed) if random_populate else None
        if random_populate:
            seed_info = "None" if seed is None else str(seed)
            print(
                f"[CHECK] {name}: random populate enabled (seed={seed_info}, batch_size={random_batch_size})"
            )

        def _flush_random_batch(batch_templates: list[Dict[str, Any]]) -> None:
            # shuffle each client view independently while keeping batch memory bounded
            for _ in range(clients):
                shuffled = list(batch_templates)
                rng.shuffle(shuffled)

                for selected in shuffled:
                    jobs.put(
                        {
                            "name": name,
                            "url": selected["url"],
                            "headers": selected["headers"],
                            "payload": selected["payload"],
                        }
                    )

        # create a batch template to store prompts
        batch_templates: list[Dict[str, Any]] = []

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
                    endpoint=endpoint,
                    payload=payload,
                    max_model_len=max_model_len,
                    timeout_s=self.vars["REQUEST_TIMEOUT"],
                )

            # build a request template
            template = {
                "url": f"{endpoint.rstrip('/')}/v1{uri}",
                "headers": {"Content-Type": "application/json"},
                "payload": payload,
            }

            # if random populate is enabled, store it and flush it onces it reaches the batch size
            if random_populate:
                batch_templates.append(template)
                if len(batch_templates) >= random_batch_size:
                    _flush_random_batch(batch_templates)
                    batch_templates.clear()
            else:
                # in normal mode, send the job to each client
                for _ in range(clients):
                    jobs.put(
                        {
                            "name": name,
                            "url": template["url"],
                            "headers": template["headers"],
                            "payload": template["payload"],
                        }
                    )

        # flush the remaining populated benchmarks
        if random_populate and batch_templates:
            _flush_random_batch(batch_templates)

        # send a None job to stop runners after processing all requests
        for _ in self._runners:
            jobs.put(None)

        # wait for all jobs to receive
        jobs.join()

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

        if args.random_batch_size < 1:
            raise RuntimeError(
                "Invalid random batch size, --random-batch-size must be >= 1."
            )

        for name in args.benchmarks:
            if name not in BENCHMARK_REGISTRY:
                raise RuntimeError(f"Unknown benchmark: {name}")

        # extract the endpoint and datasets directory
        endpoint = args.endpoint
        data_dir = os.environ.get("HF_HOME") or args.data_dir

        # check if server is healthy
        print(f"[CHECK] Checking server at {endpoint}")
        try:
            assert_server_up(endpoint)
        except Exception as e:
            raise RuntimeError(f"Cannot reach server at {endpoint}: {e}")
        print("[CHECK] Server is up.")

        # detect the model
        model = args.model or auto_detect_model(endpoint)
        print(f"[CHECK] Model: {model}")

        # get maximum model size (for truncation)
        max_model_len = 0
        if args.truncate:
            max_model_len = detect_max_model_len(
                endpoint, model, timeout_s=self.vars["REQUEST_TIMEOUT"]
            )
            print(f"[CHECK] Max model length: {max_model_len} (truncation enabled)")

        # check the data directory
        os.makedirs(data_dir, exist_ok=True)
        print(f"[CHECK] Data directory: {data_dir}")

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
            benchmark = bench_cls.create(model=model, cache_dir=data_dir)

            # run and update the status
            n, ok, fail = self._run_benchmark(
                name=name,
                benchmark=benchmark,
                endpoint=endpoint,
                clients=args.clients,
                truncate=args.truncate,
                max_model_len=max_model_len,
                enable_metrics=args.enable_prometheus_metrics,
                random_populate=args.random_populate,
                seed=args.seed,
                random_batch_size=args.random_batch_size,
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
        data_dir = os.environ.get("HF_HOME") or args.data_dir

        # check the data directory
        os.makedirs(data_dir, exist_ok=True)
        print(f"[CHECK] Data directory: {data_dir}")

        # set the cache directory
        args.cache_dir = data_dir

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
