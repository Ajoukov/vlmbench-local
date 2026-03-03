#!/usr/bin/env python3

"""
VLMBench Coordinator

Usage:
  ./main.py --list
  ./main.py [--endpoint URL] [--model MODEL] [--data-dir DIR] benchmark1 [benchmark2 ...]

Examples:
  ./main.py --list
  ./main.py narrativeqa humaneval
  ./main.py --endpoint http://127.0.0.1:8080 --model facebook/opt-125m alpaca triviaqa

@authors:
    - Alexander "Sasha" Joukov (alexander.joukov@stonybrook.edu)
    - Amir Zadeh (anajafizadeh@cs.stonybrook.edu)

@year: 2026
@by: File Systems & Storage Lab @ Stony Brook University
"""

import argparse
import os
import sys
import time
from typing import Any, Dict

from benchmarks import REGISTRY, list_all
from src import Benchmark
from src.utils import assert_server_up, detect_model
from src.worker import Worker
from vars import init_vars


def run_benchmark(
    vars: Dict[str, Any],
    name: str,
    benchmark: Benchmark,
    endpoint: str,
    stop_after: int = 0,
):
    """
    Run a single benchmark: iterate its entries, send HTTP requests,
    and print the status code for each.
    """
    print(f"\n=== Benchmark: {name} ===")
    w = Worker(request_timeout=vars["REQUEST_TIMEOUT"])

    count = 0
    for result in benchmark.run():
        count += 1
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

        t0 = time.monotonic()
        w.process(name=name, url=url, headers=headers, payload=payload)
        elapsed = time.monotonic() - t0

        remaining = vars["REQUEST_TIMEOUT"] - elapsed
        if remaining > 0:
            # print(f"Sleeping {remaining:.1f}s (ran {elapsed:.1f}s / {vars['REQUEST_TIMEOUT']}s timeout)")
            time.sleep(remaining)

    stats = w.stats()
    n = stats["total_requests"]
    ok = stats["success"]
    fail = stats["http_error"] + stats["timeout"] + stats["exception"]

    print(f"--- {name}: {n} requests, {ok} ok, {fail} failed ---")

    return n, ok, fail


def main():
    vars = init_vars()

    ap = argparse.ArgumentParser(
        description="VLMBench — connects to a running vLLM instance"
    )
    ap.add_argument(
        "--list", action="store_true", help="List available benchmarks and exit"
    )
    ap.add_argument(
        "--endpoint",
        default=vars["DEFAULT_ENDPOINT"],
        help=f"vLLM endpoint URL (default: {vars['DEFAULT_ENDPOINT']})",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Model name (auto-detected from endpoint if omitted)",
    )
    ap.add_argument(
        "--data-dir",
        default=vars["DEFAULT_DATA_DIR"],
        help=f"Dataset cache directory (default: {vars['DEFAULT_DATA_DIR']})",
    )
    ap.add_argument(
        "--stop-after",
        type=int,
        default=0,
        help="Stop after processing this many entries (for quick testing; default: 0, meaning no limit)",
    )
    ap.add_argument(
        "benchmarks",
        nargs="*",
        help="Benchmark names to run",
    )

    args = ap.parse_args()

    if args.list:
        print("Available benchmarks:")
        for name in list_all():
            print(f"  {name}")
        return

    if not args.benchmarks:
        ap.print_usage()
        print("Error: specify at least one benchmark (or use --list).", file=sys.stderr)
        sys.exit(2)

    # Validate benchmark names
    for name in args.benchmarks:
        if name not in REGISTRY:
            print(f"Error: Unknown benchmark '{name}'.", file=sys.stderr)
            print(f"Available: {list_all()}", file=sys.stderr)
            sys.exit(2)

    endpoint = args.endpoint
    data_dir = os.environ.get("HF_HOME") or args.data_dir

    # Check server
    print(f"Checking server at {endpoint} ...")
    try:
        assert_server_up(endpoint)
    except Exception as e:
        print(f"Error: Cannot reach vLLM at {endpoint}: {e}", file=sys.stderr)
        sys.exit(1)
    print("Server is up.")

    # Detect or use provided model
    model = args.model or detect_model(endpoint)
    print(f"Model: {model}")

    os.makedirs(data_dir, exist_ok=True)

    # Run benchmarks sequentially
    total_n = 0
    total_ok = 0
    total_fail = 0

    for name in args.benchmarks:
        bench_cls = REGISTRY[name]
        benchmark = bench_cls.create(model=model, cache_dir=data_dir)
        n, ok, fail = run_benchmark(
            vars, name, benchmark, endpoint, stop_after=args.stop_after
        )
        total_n += n
        total_ok += ok
        total_fail += fail

    print(
        f"\n=== All done: {total_n} total requests, {total_ok} ok, {total_fail} failed ==="
    )

    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
