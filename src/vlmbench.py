import argparse
import os
import queue
import random
import sys
import time
from typing import Any, Dict, Optional

from benchmarks import REGISTRY as BENCHMARK_REGISTRY
from benchmarks import list_all as list_benchmarks
from plugins import list_all as list_plugins
from plugins import register_subcommands
from src.benchmark import Benchmark
from src.runner import Runner
from src.runner.stats import RunnerStats
from src.tokens import truncate_payload
from src.utils import assert_server_up, auto_detect_model, detect_max_model_len
from src.vars import init_vars


class VLMBench:
    """Orchestrates benchmark and plugin workloads for OpenAI-compatible endpoints."""

    def __init__(self, argv: Optional[list[str]] = None):
        # initialize variables and parse command-line arguments
        self.vars = init_vars()
        self.argv = argv if argv is not None else sys.argv[1:]
        self.args = None
        self.parser = self._build_parser()

    def _build_parser(self) -> argparse.ArgumentParser:
        """Builds the command-line argument parser with 'bench' and 'plugin' subcommands.

        Returns
        -------
        argparse.ArgumentParser
            The configured argument parser for VLMBench CLI.
        """

        ap = argparse.ArgumentParser(
            description="VLMBench - benchmarking and plugin workloads for OpenAI-compatible endpoints. By File Systems & Storage Lab @ Stony Brook University (2024-2026).",
        )

        common = argparse.ArgumentParser(add_help=False)
        common.add_argument(
            "--endpoint",
            default=self.vars["DEFAULT_ENDPOINT"],
            help=f"vLLM endpoint URL (default: {self.vars['DEFAULT_ENDPOINT']})",
        )
        common.add_argument(
            "--model",
            default=None,
            help="Model name (auto-detected from endpoint if omitted)",
        )
        common.add_argument(
            "--data-dir",
            default=self.vars["DEFAULT_DATA_DIR"],
            help=f"Dataset cache directory (default: {self.vars['DEFAULT_DATA_DIR']})",
        )
        common.add_argument(
            "--enable-metrics",
            action="store_true",
            help="Enable metrics collection (fetches cumulative counter values from /metrics endpoint before and after benchmarks/plugins, and prints the differences)",
        )

        subparsers = ap.add_subparsers(dest="command")

        bench_parser = subparsers.add_parser(
            "bench",
            parents=[common],
            help="Run benchmarks",
            description="Run benchmark workloads against a vLLM endpoint",
        )
        self._bench_parser = bench_parser
        bench_parser.add_argument(
            "--list", action="store_true", help="List available benchmarks and exit"
        )
        bench_parser.add_argument(
            "--stop-after",
            type=int,
            default=0,
            help="Stop after processing this many entries (for quick testing; default: 0, meaning no limit)",
        )
        bench_parser.add_argument(
            "--truncate",
            action="store_true",
            help="Truncate inputs that exceed the model's context window",
        )
        bench_parser.add_argument(
            "--clients",
            type=int,
            default=1,
            help="Number of concurrent client workers (default: 1)",
        )
        bench_parser.add_argument(
            "--random-populate",
            action="store_true",
            help="Populate requests by random sampling from benchmark entries",
        )
        bench_parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Seed for random population (deterministic when used with --random-populate)",
        )
        bench_parser.add_argument(
            "--random-batch-size",
            type=int,
            default=100,
            help="Number of entries to buffer per batch in --random-populate mode (default: 100)",
        )
        bench_parser.add_argument(
            "benchmarks",
            nargs="*",
            help="Benchmark names to run",
        )

        plugin_parser = subparsers.add_parser(
            "plugin",
            help="Run plugins",
            description="Run plugin workloads against a vLLM endpoint",
        )
        self._plugin_parser = plugin_parser
        plugin_parser.add_argument(
            "--list", action="store_true", help="List available plugins and exit"
        )

        plugin_subparsers = plugin_parser.add_subparsers(dest="plugin_name")
        register_subcommands(plugin_subparsers, parents=[common])

        return ap

    def _list_benchmarks(self) -> None:
        print("Available benchmarks:")
        for name in list_benchmarks():
            print(f"  {name}")

    def _list_plugins(self) -> None:
        print("Available plugins:")
        for name in list_plugins():
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
        print(f"\n=== Benchmark: {name} (metrics={enable_metrics}) ===")
        print(f"--- Start time: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

        jobs: "queue.Queue[Dict[str, Any] | None]" = queue.Queue()
        stats = RunnerStats()

        runners = [
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

        for runner in runners:
            runner.start()

        rng = random.Random(seed) if random_populate else None
        if random_populate:
            seed_info = "None" if seed is None else str(seed)
            print(
                f"{name}: random populate enabled (seed={seed_info}, batch_size={random_batch_size})"
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

        valid_entries = 0
        batch_templates: list[Dict[str, Any]] = []
        for result in benchmark.run():
            uri = result["uri"]
            payload = result["payload"]

            if not payload.get("prompt") and not payload.get("messages"):
                continue

            if truncate and max_model_len > 0:
                payload = truncate_payload(
                    endpoint=endpoint,
                    payload=payload,
                    max_model_len=max_model_len,
                    timeout_s=self.vars["REQUEST_TIMEOUT"],
                )

            template = {
                "url": f"{endpoint.rstrip('/')}/v1{uri}",
                "headers": {"Content-Type": "application/json"},
                "payload": payload,
            }
            valid_entries += 1

            if random_populate:
                batch_templates.append(template)
                if len(batch_templates) >= random_batch_size:
                    _flush_random_batch(batch_templates)
                    batch_templates.clear()
            else:
                for _ in range(clients):
                    jobs.put(
                        {
                            "name": name,
                            "url": template["url"],
                            "headers": template["headers"],
                            "payload": template["payload"],
                        }
                    )

        if random_populate and batch_templates:
            _flush_random_batch(batch_templates)

        if valid_entries == 0:
            print(f"{name}: no valid benchmark entries to enqueue")

        for _ in runners:
            jobs.put(None)

        jobs.join()

        for runner in runners:
            runner.join()

        summary = stats.stats()
        n = summary["total_requests"]
        ok = summary["success"]
        fail = summary["error"] + summary["timeout"]
        total_request_bytes = summary["total_request_bytes"]
        total_response_bytes = summary["total_response_bytes"]
        average_latency = summary["avg_latency_ms"]
        p95_latency = summary["p95_latency_ms"]

        print(f"--- {name}: {n} requests, {ok} ok, {fail} failed ---")
        print(f"Total request bytes: {total_request_bytes}")
        print(f"Total response bytes: {total_response_bytes}")
        print(f"Average latency: {average_latency:.2f} ms")
        print(f"95th percentile latency: {p95_latency:.2f} ms")

        vllm_metrics = stats.vllm_stats()
        for metric_name, value in vllm_metrics.items():
            print(f"vllm:'{metric_name}': {value}")

        print(f"--- End time: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

        return n, ok, fail

    def _run_bench_command(self) -> None:
        args = self.args

        if args.list:
            self._list_benchmarks()
            return

        if not args.benchmarks:
            self._bench_parser.print_usage()
            print(
                "Error: specify at least one benchmark (or use --list).",
                file=sys.stderr,
            )
            raise RuntimeError("No benchmarks specified")

        if args.clients < 1:
            print("Error: --clients must be >= 1.", file=sys.stderr)
            raise RuntimeError("Invalid number of clients")

        if args.random_batch_size < 1:
            print("Error: --random-batch-size must be >= 1.", file=sys.stderr)
            raise RuntimeError("Invalid random batch size")

        for name in args.benchmarks:
            if name not in BENCHMARK_REGISTRY:
                print(f"Error: Unknown benchmark '{name}'.", file=sys.stderr)
                print(f"Available: {list_benchmarks()}", file=sys.stderr)
                raise RuntimeError(f"Unknown benchmark: {name}")

        endpoint = args.endpoint
        data_dir = os.environ.get("HF_HOME") or args.data_dir

        try:
            print(f"Checking server at {endpoint} ...")
            assert_server_up(endpoint)
        except Exception as exc:
            print(f"Error: Cannot reach vLLM at {endpoint}: {exc}", file=sys.stderr)
            raise RuntimeError(f"Cannot reach server at {endpoint}: {exc}")
        print("Server is up.")

        model = args.model or auto_detect_model(endpoint)
        print(f"Model: {model}")

        max_model_len = 0
        if args.truncate:
            max_model_len = detect_max_model_len(
                endpoint, model, timeout_s=self.vars["REQUEST_TIMEOUT"]
            )
            print(f"Max model length: {max_model_len} (truncation enabled)")

        os.makedirs(data_dir, exist_ok=True)

        total_n = 0
        total_ok = 0
        total_fail = 0

        print(
            f"\n=== Running {len(args.benchmarks)} benchmark(s) sequentially with {args.clients} client(s) each ==="
        )

        for name in args.benchmarks:
            bench_cls = BENCHMARK_REGISTRY[name]

            benchmark = bench_cls.create(model=model, cache_dir=data_dir)
            benchmark.set_limit(args.stop_after)

            n, ok, fail = self._run_benchmark(
                name=name,
                benchmark=benchmark,
                endpoint=endpoint,
                clients=args.clients,
                truncate=args.truncate,
                max_model_len=max_model_len,
                enable_metrics=args.enable_metrics,
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
        args = self.args

        if args.list:
            self._list_plugins()
            return

        if not getattr(args, "plugin_name", None):
            self._plugin_parser.print_usage()
            print("Error: specify a plugin name (or use --list).", file=sys.stderr)
            raise RuntimeError("No plugin specified")

        if not hasattr(args, "plugin_runner"):
            print(
                f"Error: Plugin '{args.plugin_name}' has no runnable handler.",
                file=sys.stderr,
            )
            raise RuntimeError("Invalid plugin specified")

        endpoint = args.endpoint
        data_dir = os.environ.get("HF_HOME") or args.data_dir

        try:
            print(f"Checking server at {endpoint} ...")
            assert_server_up(endpoint)
        except Exception as exc:
            print(f"Error: Cannot reach vLLM at {endpoint}: {exc}", file=sys.stderr)
            raise RuntimeError(f"Cannot reach server at {endpoint}: {exc}")
        print("Server is up.")

        model = args.model or auto_detect_model(endpoint)
        print(f"Model: {model}")

        max_model_len = detect_max_model_len(
            endpoint, model, timeout_s=self.vars["REQUEST_TIMEOUT"]
        )
        print(f"Max model length: {max_model_len}")

        os.makedirs(data_dir, exist_ok=True)

        args.cache_dir = data_dir
        args.resolved_model = model
        args.max_model_len = max_model_len
        args.plugin_runner(args)

    def run(self) -> None:
        argv = self.argv
        if len(argv) >= 3 and argv[0] == "plugin" and argv[1] in ("-h", "--help"):
            argv = ["plugin", argv[2], "--help", *argv[3:]]

        self.args = self.parser.parse_args(argv)
        if self.args.command is None:
            self.parser.print_usage()
            print("Error: choose a command: 'bench' or 'plugin'.", file=sys.stderr)
            raise RuntimeError("No command specified")

        if self.args.command == "bench":
            self._run_bench_command()
            return

        if self.args.command == "plugin":
            self._run_plugin_command()
            return
