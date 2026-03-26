import argparse
from typing import Dict

from plugins import register_subcommands


def build_parser(vars: Dict[str, str]) -> argparse.ArgumentParser:
    """Builds the command-line argument parser with all subcommands."""

    # the root argparser
    ap = argparse.ArgumentParser(
        description="VLMBench - benchmarking workloads for vLLM. By File Systems & Storage Lab @ Stony Brook University (2024-2026).",
    )

    # common flags
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--endpoint",
        default=vars["DEFAULT_ENDPOINT"],
        help=f"vLLM OpenAI API address (default: {vars['DEFAULT_ENDPOINT']})",
    )
    common.add_argument(
        "--model",
        default=None,
        help="Model name (auto-detected from API if omitted)",
    )
    common.add_argument(
        "--data-dir",
        default=vars["DEFAULT_DATA_DIR"],
        help=f"Local datasets and cache directory (default: {vars['DEFAULT_DATA_DIR']})",
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
    bench_parser = subparsers.add_parser(
        "bench",
        parents=[common],
        help="Run benchmarks",
        description="Run benchmark workloads against a vLLM instance using OpenAI API",
    )
    bench_parser.add_argument(
        "--list", action="store_true", help="List available benchmarks"
    )
    bench_parser.add_argument(
        "--stop-after",
        type=int,
        default=0,
        help="Stop after processing this many entries (for limiting tests; default: 0, meaning no limit, until the dataset is over)",
    )
    bench_parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate inputs that exceed the model's context window (WARN: this might change a prompt to fit with model's context window)",
    )
    bench_parser.add_argument(
        "--clients",
        type=int,
        default=1,
        help="Number of concurrent clients (default: 1)",
    )
    bench_parser.add_argument(
        "benchmarks",
        nargs="*",
        help="Benchmark names to run",
    )

    # `plugin` group
    plugin_parser = subparsers.add_parser(
        "plugin",
        help="Run plugins",
        description="Run plugin workloads against a vLLM instance",
    )
    plugin_parser.add_argument(
        "--list", action="store_true", help="List available plugins"
    )

    # register plugin subcommands
    plugin_subparsers = plugin_parser.add_subparsers(dest="plugin_name")
    register_subcommands(plugin_subparsers, parents=[common])

    return ap
