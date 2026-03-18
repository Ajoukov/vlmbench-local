import argparse
import time

from src.utils import assert_server_up


def register_parser(
    subparsers: argparse._SubParsersAction,
    parents: list[argparse.ArgumentParser],
) -> None:
    """Register readiness plugin subcommand and its arguments."""

    parser = subparsers.add_parser(
        "readiness",
        parents=parents,
        help="Check readiness of API",
        description="Check readiness of OpenAI-compatible API",
    )
    parser.add_argument(
        "--retrys",
        type=int,
        default=5,
        help="Number of times to retry the readiness check before giving up (default: 5)",
    )
    parser.set_defaults(plugin_runner=run_from_args)


def run_from_args(args: argparse.Namespace) -> None:
    """Run the readiness plugin using the provided command-line arguments."""

    for i in range(args.retrys):
        try:
            assert_server_up(args.endpoint)
            print("API is ready!")
            return
        except Exception as e:
            print(f"Readiness check failed (attempt {i + 1}/{args.retrys}): {e}")
            time.sleep(2)  # wait a bit before retrying

    print(
        f"API is not ready after {args.retrys} attempts. Please check the server and try again."
    )
