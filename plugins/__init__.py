"""Plugin registry."""

import argparse

from plugins.readiness.readiness import register_parser as register_readiness_parser
from plugins.simulator.simulator import register_parser as register_simulator_parser

REGISTRY = {
    "readiness": register_readiness_parser,
    "simulator": register_simulator_parser,
}


def list_all():
    """Return list of all registered plugin names."""
    return sorted(REGISTRY.keys())


def register_subcommands(
    subparsers: argparse._SubParsersAction,
    parents: list[argparse.ArgumentParser],
) -> None:
    """Register all plugin-specific subcommands."""
    for name in list_all():
        REGISTRY[name](subparsers, parents)
