import argparse
import random
import os
import queue
import sys
import time
from typing import Any, Dict

from benchmarks import REGISTRY as BENCHMARK_REGISTRY
from benchmarks import list_all as list_benchmarks
from plugins import list_all as list_plugins
from plugins import register_subcommands
from src import Benchmark
from src.runner import Runner
from src.runner.stats import RunnerStats
from src.tokens import truncate_payload
from src.utils import assert_server_up, auto_detect_model, detect_max_model_len
from src.vars import init_vars


def _run_benchmark(
    vars: Dict[str, Any],
    name: str,
    benchmark: Benchmark,
    endpoint: str,
    clients: int,
    truncate: bool = False,
    max_model_len: int = 0,
    enable_metrics: bool = False,
    random_populate: bool = False,
):
    """Run a single benchmark: iterate its entries, send HTTP requests, and print the status code for each.

    Parameters
    ----------
    vars : Dict[str, Any]
        Global variables and configuration.
    name : str
        Benchmark name (for display).
    benchmark : Benchmark
        Benchmark instance to run.
    endpoint : str
        vLLM endpoint URL.
    clients : int
        Number of concurrent client workers to use.
    truncate : bool, optional
        Whether to truncate inputs that exceed the model's context window (default: False).
    max_model_len : int, optional
        Maximum model context length (required if truncate is True; default: 0).
    enable_metrics : bool, optional
        Whether to enable metrics collection (fetches cumulative counter values from /metrics endpoint before and after benchmarks, and prints the differences; default: False).
    """

    print(f"\n=== Benchmark: {name} (metrics={enable_metrics}) ===")
    print(f"--- Start time: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    # shared queue for jobs and stats collector
    jobs: "queue.Queue[Dict[str, Any] | None]" = queue.Queue()
    stats = RunnerStats()

    # create runner threads
    runners = [
        Runner(
            runner_id=index + 1,
            endpoint=endpoint,
            jobs=jobs,
            stats=stats,
            request_timeout=vars["REQUEST_TIMEOUT"],
            enable_metrics=enable_metrics,
        )
        for index in range(max(1, clients))
    ]

    # start runners
    for r in runners:
        r.start()

    # iterate benchmark entries and enqueue jobs
    for result in benchmark.run():
        uri = result["uri"]
        payload = result["payload"]

        # skip entries with empty prompts (filtered by build_input)
        if not payload.get("prompt") and not payload.get("messages"):
            continue

        # check if we need to truncate inputs that exceed the model's context window
        if truncate and max_model_len > 0:
            payload = truncate_payload(
                endpoint=endpoint,
                payload=payload,
                max_model_len=max_model_len,
                timeout_s=vars["REQUEST_TIMEOUT"],
            )

        url = f"{endpoint.rstrip('/')}/v1{uri}"
        headers = {"Content-Type": "application/json"}

        # enqueue one job per entry (runners pick jobs from the shared queue)
        if random_populate:
            random_range = random.randint(1, clients)
            for _ in range(random_range):
                jobs.put(
                    {
                        "name": name,
                        "url": url,
                        "headers": headers,
                        "payload": payload,
                    }
                )
        else:
            for _ in range(clients):
                jobs.put(
                    {
                        "name": name,
                        "url": url,
                        "headers": headers,
                        "payload": payload,
                    }
                )

    # signal runners to stop (one None per runner)
    for _ in runners:
        jobs.put(None)

    # wait for all jobs to be processed
    jobs.join()

    # wait for all runners to finish
    for r in runners:
        r.join()

    # print summary
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
    print(f"--- End time: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    return n, ok, fail


def _list_benchmarks() -> None:
    """Helper function to list benchmarks and exit (used for --list)."""

    print("Available benchmarks:")
    for name in list_benchmarks():
        print(f"  {name}")


def _list_plugins() -> None:
    """Helper function to list plugins and exit (used for --list)."""

    print("Available plugins:")
    for name in list_plugins():
        print(f"  {name}")


def main():
    # initialize global variables and configuration
    vars = init_vars()

    # parse command-line arguments
    ap = argparse.ArgumentParser(
        description="VLMBench — benchmarking and plugin workloads for OpenAI-compatible endpoints. By File Systems & Storage Lab @ Stony Brook University (2024-2026).",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--endpoint",
        default=vars["DEFAULT_ENDPOINT"],
        help=f"vLLM endpoint URL (default: {vars['DEFAULT_ENDPOINT']})",
    )
    common.add_argument(
        "--model",
        default=None,
        help="Model name (auto-detected from endpoint if omitted)",
    )
    common.add_argument(
        "--data-dir",
        default=vars["DEFAULT_DATA_DIR"],
        help=f"Dataset cache directory (default: {vars['DEFAULT_DATA_DIR']})",
    )
    common.add_argument(
        "--enable-metrics",
        action="store_true",
        help="Enable metrics collection (fetches cumulative counter values from /metrics endpoint before and after benchmarks/plugins, and prints the differences)",
    )
    common.add_argument(
        "--random-populate",
        action="store_true",
        help="Populate dataset items in random order",
    )

    # create subparsers for "bench" and "plugin" commands
    subparsers = ap.add_subparsers(dest="command")

    bench_parser = subparsers.add_parser(
        "bench",
        parents=[common],
        help="Run benchmarks",
        description="Run benchmark workloads against a vLLM endpoint",
    )
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
        "benchmarks",
        nargs="*",
        help="Benchmark names to run",
    )

    plugin_parser = subparsers.add_parser(
        "plugin",
        help="Run plugins",
        description="Run plugin workloads against a vLLM endpoint",
    )
    plugin_parser.add_argument(
        "--list", action="store_true", help="List available plugins and exit"
    )

    # create subparsers for each plugin
    plugin_subparsers = plugin_parser.add_subparsers(dest="plugin_name")
    register_subcommands(plugin_subparsers, parents=[common])

    # parse arguments
    argv = sys.argv[1:]
    if len(argv) >= 3 and argv[0] == "plugin" and argv[1] in ("-h", "--help"):
        argv = ["plugin", argv[2], "--help", *argv[3:]]

    # if no arguments are provided, show help
    args = ap.parse_args(argv)
    if args.command is None:
        ap.print_usage()
        print("Error: choose a command: 'bench' or 'plugin'.", file=sys.stderr)
        raise RuntimeError("No command specified")

    # handle "bench" and "plugin" commands
    if args.command == "bench":
        # if --list is specified, list benchmarks and exit
        if args.list:
            _list_benchmarks()
            return

        # if no benchmarks are specified, show usage and exit
        if not args.benchmarks:
            bench_parser.print_usage()
            print(
                "Error: specify at least one benchmark (or use --list).",
                file=sys.stderr,
            )
            raise RuntimeError("No benchmarks specified")

        # check that clients is a positive integer
        if args.clients < 1:
            print("Error: --clients must be >= 1.", file=sys.stderr)
            raise RuntimeError("Invalid number of clients")

        # check that specified benchmarks exist
        for name in args.benchmarks:
            if name not in BENCHMARK_REGISTRY:
                print(f"Error: Unknown benchmark '{name}'.", file=sys.stderr)
                print(f"Available: {list_benchmarks()}", file=sys.stderr)
                raise RuntimeError(f"Unknown benchmark: {name}")

        endpoint = args.endpoint
        data_dir = os.environ.get("HF_HOME") or args.data_dir

        # check server and detect model info before running benchmarks
        try:
            print(f"Checking server at {endpoint} ...")
            assert_server_up(endpoint)
        except Exception as e:
            print(f"Error: Cannot reach vLLM at {endpoint}: {e}", file=sys.stderr)
            raise RuntimeError(f"Cannot reach server at {endpoint}: {e}")
        print("Server is up.")

        # detect model name from endpoint if not provided (for better stats reporting and caching)
        model = args.model or auto_detect_model(endpoint)
        print(f"Model: {model}")

        # resolve max model length if truncation is needed (some benchmarks may require this)
        max_model_len = 0
        if args.truncate:
            max_model_len = detect_max_model_len(endpoint, model, timeout_s=vars["REQUEST_TIMEOUT"])
            print(f"Max model length: {max_model_len} (truncation enabled)")

        # ensure data directory exists for caching benchmark datasets
        os.makedirs(data_dir, exist_ok=True)

        total_n = 0
        total_ok = 0
        total_fail = 0

        print(
            f"\n=== Running {len(args.benchmarks)} benchmark(s) sequentially with {args.clients} client(s) each ==="
        )

        for name in args.benchmarks:
            bench_cls = BENCHMARK_REGISTRY[name]

            # create benchmark instance
            benchmark = bench_cls.create(model=model, cache_dir=data_dir)
            benchmark.set_limit(args.stop_after)

            # run benchmark and accumulate stats
            n, ok, fail = _run_benchmark(
                vars,
                name,
                benchmark,
                endpoint,
                clients=args.clients,
                truncate=args.truncate,
                max_model_len=max_model_len,
                enable_metrics=args.enable_metrics,
                random_populate=args.random_populate,
            )

            total_n += n
            total_ok += ok
            total_fail += fail

        print(
            f"\n=== All done: {total_n} total requests, {total_ok} ok, {total_fail} failed ==="
        )

        return

    if args.command == "plugin":
        # if --list is specified, list plugins and exit
        if args.list:
            _list_plugins()
            return

        # for plugins, we expect each plugin's subparser to set a "plugin_runner" attribute on args,
        # which is a function that takes args and runs the plugin workload
        if not getattr(args, "plugin_name", None):
            plugin_parser.print_usage()
            print("Error: specify a plugin name (or use --list).", file=sys.stderr)
            raise RuntimeError("No plugin specified")

        # check that the specified plugin exists and has a runner function
        if not hasattr(args, "plugin_runner"):
            print(
                f"Error: Plugin '{args.plugin_name}' has no runnable handler.",
                file=sys.stderr,
            )
            raise RuntimeError("Invalid plugin specified")

        endpoint = args.endpoint
        data_dir = os.environ.get("HF_HOME") or args.data_dir

        # check server and detect model info before running plugin workloads
        try:
            print(f"Checking server at {endpoint} ...")
            assert_server_up(endpoint)
        except Exception as e:
            print(f"Error: Cannot reach vLLM at {endpoint}: {e}", file=sys.stderr)
            raise RuntimeError(f"Cannot reach server at {endpoint}: {e}")
        print("Server is up.")

        # detect model name from endpoint if not provided (for better stats reporting and caching)
        model = args.model or auto_detect_model(endpoint)
        print(f"Model: {model}")

        # resolve max model length
        max_model_len = detect_max_model_len(endpoint, model, timeout_s=vars["REQUEST_TIMEOUT"])
        print(f"Max model length: {max_model_len}")

        # ensure data directory exists for caching benchmark datasets
        os.makedirs(data_dir, exist_ok=True)

        args.cache_dir = data_dir
        args.resolved_model = model
        args.max_model_len = max_model_len

        # run the plugin's workload function, which should handle everything internally (including printing its own status and results)
        args.plugin_runner(args)

        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
