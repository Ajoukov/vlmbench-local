#!/usr/bin/env python3

"""
VLMBench Coordinator

Usage:
    ./main.py bench --list
    ./main.py bench [--endpoint URL] [--model MODEL] [--data-dir DIR] benchmark1 [benchmark2 ...]
    ./main.py plugin --list
    ./main.py plugin simulator --total-kv-tokens N
    ./main.py plugin simulator --help

Examples:
    ./main.py bench --list
    ./main.py bench narrativeqa humaneval
    ./main.py plugin --list
    ./main.py plugin simulator --total-kv-tokens 8388608
    ./main.py plugin simulator --help

@authors:
    - Alexander "Sasha" Joukov (alexander.joukov@stonybrook.edu)
    - Amir Zadeh (anajafizadeh@cs.stonybrook.edu)

@year: 2026
@by: File Systems & Storage Lab @ Stony Brook University
"""

import argparse
import os
import queue
import sys
from typing import Any, Dict
import time

from benchmarks import REGISTRY as BENCHMARK_REGISTRY
from benchmarks import list_all as list_benchmarks
from plugins import list_all as list_plugins
from plugins import register_subcommands
from src import Benchmark
from src.utils import assert_server_up, detect_max_model_len, detect_model, token_count, truncate_payload
from src.worker import Worker, WorkerStats
from src.vars import init_vars


def run_benchmark(
    vars: Dict[str, Any],
    name: str,
    benchmark: Benchmark,
    endpoint: str,
    clients: int,
    stop_after: int = 0,
    truncate: bool = False,
    max_model_len: int = 0,
):
    """
    Run a single benchmark: iterate its entries, send HTTP requests,
    and print the status code for each.

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
    stop_after : int, optional
        Stop after processing this many entries (for quick testing; default: 0, meaning no limit).
    truncate : bool, optional
        Whether to truncate inputs that exceed the model's context window (default: False).
    max_model_len : int, optional
        Maximum model context length (required if truncate is True; default: 0).
    """

    print(f"\n=== Benchmark: {name} ===")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # shared queue for jobs and stats collector
    jobs: "queue.Queue[Dict[str, Any] | None]" = queue.Queue()
    stats = WorkerStats()

    # create worker threads
    workers = [
        Worker(
            request_timeout=vars["REQUEST_TIMEOUT"],
            jobs=jobs,
            stats=stats,
            worker_id=index + 1,
            metrics_base_url=endpoint,
        )
        for index in range(max(1, clients))
    ]

    # start workers
    for worker in workers:
        print(f"Starting worker {worker.worker_id} ...")
        worker.start()

    # iterate benchmark entries and enqueue jobs
    count = 0
    for result in benchmark.run():
        count += 1
        print(f"Processing entry {count} ...")
        if stop_after > 0 and count > stop_after:
            break

        uri = result["uri"]
        payload = result["payload"]

        # skip entries with empty prompts (filtered by build_input)
        if not payload.get("prompt") and not payload.get("messages"):
            count -= 1
            continue

        url = f"{endpoint.rstrip('/')}/v1{uri}"
        headers = {"Content-Type": "application/json"}

        # count tokens
        count, tokens = token_count(endpoint, payload.get("model"), payload.get("prompt", ""))
        if max_model_len > 0 and count > max_model_len:
            print(
                f"Entry {count} exceeds max model length ({count} > {max_model_len}).",
                file=sys.stderr,
            )

        # truncate payload if needed (and if we know the model's max context length)
        if truncate and max_model_len > 0:
            payload = truncate_payload(endpoint, payload, max_model_len, count, tokens)

        # each worker will process this job and update stats
        for _ in workers:
            print(f"Enqueuing job for entry {count} ...")
            jobs.put(
                {
                    "name": name,
                    "url": url,
                    "headers": headers,
                    "payload": payload,
                }
            )

    print(f"Finished enqueuing jobs for benchmark '{name}'. Total entries processed: {count}.")
    # signal workers to stop (one None per worker)
    for _ in workers:
        jobs.put(None)

    # wait for all jobs to be processed
    jobs.join()

    # wait for all workers to finish
    for worker in workers:
        worker.join()

    # print summary
    summary = stats.stats()
    n = summary["total_requests"]
    ok = summary["success"]
    fail = summary["http_error"] + summary["timeout"] + summary["exception"]

    print(f"--- {name}: {n} requests, {ok} ok, {fail} failed ---")
    print(
        f"    tokens: "
        f"submitted={summary['total_submitted_tokens']} "
        f"prefill={summary['total_prefill_tokens']} "
        f"decode={summary['total_decode_tokens']} "
        f"cached={summary['total_cached_tokens']}"
    )
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    return n, ok, fail


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
        if args.list:
            print("Available benchmarks:")
            for name in list_benchmarks():
                print(f"  {name}")
            return

        if not args.benchmarks:
            bench_parser.print_usage()
            print(
                "Error: specify at least one benchmark (or use --list).",
                file=sys.stderr,
            )
            raise RuntimeError("No benchmarks specified")

        if args.clients < 1:
            print("Error: --clients must be >= 1.", file=sys.stderr)
            raise RuntimeError("Invalid number of clients")
        
        for name in args.benchmarks:
            if name not in BENCHMARK_REGISTRY:
                print(f"Error: Unknown benchmark '{name}'.", file=sys.stderr)
                print(f"Available: {list_benchmarks()}", file=sys.stderr)
                raise RuntimeError(f"Unknown benchmark: {name}")

        endpoint = args.endpoint
        data_dir = os.environ.get("HF_HOME") or args.data_dir

        print(f"Checking server at {endpoint} ...")
        try:
            assert_server_up(endpoint)
        except Exception as e:
            print(f"Error: Cannot reach vLLM at {endpoint}: {e}", file=sys.stderr)
            raise RuntimeError(f"Cannot reach server at {endpoint}: {e}")
        print("Server is up.")

        model = args.model or detect_model(endpoint)
        print(f"Model: {model}")

        max_model_len = 0
        if args.truncate:
            max_model_len = detect_max_model_len(endpoint)
            print(f"Max model length: {max_model_len} (truncation enabled)")

        os.makedirs(data_dir, exist_ok=True)

        total_n = 0
        total_ok = 0
        total_fail = 0

        print(f"\n=== Running {len(args.benchmarks)} benchmark(s) sequentially with {args.clients} client(s) each ===")
        for name in args.benchmarks:
            bench_cls = BENCHMARK_REGISTRY[name]
            benchmark = bench_cls.create(model=model, cache_dir=data_dir)

            # run benchmark and accumulate stats
            n, ok, fail = run_benchmark(
                vars,
                name,
                benchmark,
                endpoint,
                clients=args.clients,
                stop_after=args.stop_after,
                truncate=args.truncate,
                max_model_len=max_model_len,
            )

            total_n += n
            total_ok += ok
            total_fail += fail

        print(
            f"\n=== All done: {total_n} total requests, {total_ok} ok, {total_fail} failed ==="
        )

        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        return

    if args.command == "plugin":
        if args.list:
            print("Available plugins:")
            for name in list_plugins():
                print(f"  {name}")
            return

        # for plugins, we expect each plugin's subparser to set a "plugin_runner" attribute on args,
        # which is a function that takes args and runs the plugin workload
        if not getattr(args, "plugin_name", None):
            plugin_parser.print_usage()
            print("Error: specify a plugin name (or use --list).", file=sys.stderr)
            raise RuntimeError("No plugin specified")

        if not hasattr(args, "plugin_runner"):
            print(f"Error: Plugin '{args.plugin_name}' has no runnable handler.", file=sys.stderr)
            raise RuntimeError("Invalid plugin specified")

        endpoint = args.endpoint
        data_dir = os.environ.get("HF_HOME") or args.data_dir

        # check server and detect model info before running plugin
        print(f"Checking server at {endpoint} ...")
        try:
            assert_server_up(endpoint)
        except Exception as e:
            print(f"Error: Cannot reach vLLM at {endpoint}: {e}", file=sys.stderr)
            raise RuntimeError(f"Cannot reach server at {endpoint}: {e}")
        print("Server is up.")

        # reverse compatibility: if --model is not provided, try to auto-detect it from the endpoint
        resolved_model = args.model or detect_model(endpoint)
        print(f"Model: {resolved_model}")

        # resolve max model length if truncation is needed (some plugins may require this)
        max_model_len = detect_max_model_len(endpoint)
        print(f"Max model length: {max_model_len}")

        os.makedirs(data_dir, exist_ok=True)

        args.cache_dir = data_dir
        args.resolved_model = resolved_model
        args.max_model_len = max_model_len

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
