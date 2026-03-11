import math
import random
import time
import argparse
from typing import Optional

import requests

from src.utils import truncate_payload
from plugins.simulator.text_sources import (
    TaskType,
    TextSource,
    build_prompt_pair,
    make_source,
)

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

PROMPT_RATIO = 1.0 / 3.0
REQUEST_INTERVAL_S = 1.0
RUN_INTERVAL_S = 2.0
DEFAULT_REQUEST_TIMEOUT_S = 10.0
MIN_PROMPT_TOKENS = 16
MIN_GEN_TOKENS = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_tokens(total_tokens: int) -> tuple[int, int]:
    """Split *total_tokens* into (prompt_tokens, gen_tokens)."""
    usable = max(1, total_tokens - 2)
    prompt_tokens = max(MIN_PROMPT_TOKENS, int(usable * PROMPT_RATIO))
    gen_tokens = max(MIN_GEN_TOKENS, usable - prompt_tokens)

    # Safety clamp
    if prompt_tokens + gen_tokens > usable:
        gen_tokens = max(MIN_GEN_TOKENS, usable - prompt_tokens)
    if prompt_tokens + gen_tokens > usable:
        prompt_tokens = max(MIN_PROMPT_TOKENS, usable - gen_tokens)

    return prompt_tokens, gen_tokens


def _build_prompt(prefix_text: str, suffix: str) -> str:
    """Concatenate the shared *prefix_text* with a task-instruction *suffix*."""
    return f"{prefix_text}\n\n{suffix}"


def _send_request(
    endpoint: str,
    completions_url: str,
    model: str,
    prompt: str,
    prompt_tokens: int,
    gen_tokens: int,
    request_timeout_s: float,
) -> None:
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "max_tokens": gen_tokens,
        "temperature": 0,
        "stream": True,
    }
    payload = truncate_payload(
        endpoint, payload, max_model_len=prompt_tokens + gen_tokens + 2
    )
    try:
        with requests.post(
            completions_url,
            json=payload,
            stream=True,
            timeout=request_timeout_s,
        ) as r:
            r.raise_for_status()
            for _ in r.iter_lines():
                pass
    except Exception as e:
        print(f"  [ERROR] Simulator request failed: {e}")


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

def run_simulator(
    endpoint: str,
    model: str,
    max_model_len: int,
    total_kv_tokens: int,
    prefix_length_perc: float = 50.0,
    n_runs: int = 1,
    source_type: str = "wikitext",
    task: Optional[TaskType] = None,
    cache_dir: Optional[str] = None,
    utilization_perc: float = 100.0,
    request_interval_s: float = REQUEST_INTERVAL_S,
    run_interval_s: float = RUN_INTERVAL_S,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
) -> None:
    """
    Fire synthetic requests to simulate KV-cache usage with prefix sharing.

    The shared prefix is a real English passage drawn from *source_type*.
    Each request appends a different task instruction (summarise, QA, explain,
    chat, continue) as its unique suffix, ensuring no full cache hits while the
    prefix region is maximally reused.

    Parameters
    ----------
    endpoint:
        Base URL of the OpenAI-compatible inference server.
    model:
        Model identifier served at *endpoint*.
    max_model_len:
        Maximum context length (in tokens) supported by the model.
    total_kv_tokens:
        Target number of KV tokens to fill per run.
    prefix_length_perc:
        Percentage (0–100) of each prompt that is the *shared prefix*.
        The remaining percentage is the task-instruction suffix.
    n_runs:
        How many full simulation cycles to execute.
    source_type:
        Text backend for generating the English passage.
        ``"wikitext"`` (default) | ``"squad"`` | ``"wikipedia"``.
    task:
        Force a specific :class:`~text_sources.TaskType` for every request.
        ``None`` (default) rotates through all task types randomly.
    cache_dir:
        HuggingFace dataset cache directory (ignored for ``"wikipedia"``).
    utilization_perc:
        Fraction of *total_kv_tokens* to actually target (0–100).
    request_interval_s:
        Seconds to wait between requests within a run.
    run_interval_s:
        Seconds to wait between runs.
    request_timeout_s:
        Per-request HTTP timeout in seconds.
    """
    completions_url = f"{endpoint.rstrip('/')}/v1/completions"

    # ---- KV-token budget -------------------------------------------------
    effective_kv = max(1, int(math.ceil(total_kv_tokens * (utilization_perc / 100.0))))
    max_single = max(1, max_model_len - 2)
    target_tokens = min(effective_kv, max_single)

    # ---- Prompt / generation split ----------------------------------------
    prompt_tokens, gen_tokens = _split_tokens(target_tokens + 2)

    # ---- Prefix / suffix split (in tokens) --------------------------------
    prefix_frac = max(0.0, min(1.0, prefix_length_perc / 100.0))
    prefix_tokens = max(1, int(prompt_tokens * prefix_frac))
    suffix_tokens = max(1, prompt_tokens - prefix_tokens)

    # ---- How many requests per run? ---------------------------------------
    actual_req_tokens = prompt_tokens + gen_tokens
    requests_per_run = max(1, math.ceil(effective_kv / max(1, actual_req_tokens)))

    # ---- Approximate character budget for the passage --------------------
    # ~4 chars per sub-word token is a reasonable heuristic.
    _CHARS_PER_TOKEN = 4
    prefix_chars = prefix_tokens * _CHARS_PER_TOKEN

    # ---- Report ----------------------------------------------------------
    print("=" * 56)
    print("  KV-Cache Prefix Simulator")
    print("=" * 56)
    print(f"  Text source               : {source_type}")
    print(f"  Task type                 : {task.value if task else 'random'}")
    print(f"  Total KV tokens target    : {total_kv_tokens}")
    print(f"  Utilization (%)           : {utilization_perc:.1f}")
    print(f"  Effective KV tokens       : {effective_kv}")
    print(f"  Max model length          : {max_model_len}")
    print(f"  Target tokens / request   : {target_tokens}")
    print(f"  Prompt tokens             : {prompt_tokens}")
    print(f"    ├─ Prefix  (~{prefix_length_perc:.0f}%)           : ~{prefix_tokens} tokens")
    print(f"    └─ Suffix  (~{100 - prefix_length_perc:.0f}%)           : ~{suffix_tokens} tokens")
    print(f"  Generation tokens         : {gen_tokens}")
    print(f"  Actual KV / request       : {actual_req_tokens}")
    print(f"  Requests per run          : {requests_per_run}")
    print(f"  Number of runs (N)        : {n_runs}")
    if effective_kv > max_single:
        print(
            "  [WARN] Target exceeds single-request capacity; "
            "capped at max_model_len."
        )
    print("=" * 56)

    # ---- Build the shared prefix via the text source ---------------------
    print(f"\nLoading text source '{source_type}'…")
    source: TextSource = make_source(source_type, cache_dir=cache_dir, seed=42)
    # Deterministic seed=42 → same passage every time this function is called,
    # which is exactly what we want so the KV cache prefix is stable across runs.
    seed_rng = random.Random(42)
    pair = build_prompt_pair(
        source,
        task=task,
        min_prefix_chars=max(100, prefix_chars // 2),
        max_prefix_chars=prefix_chars * 2,
        rng=seed_rng,
    )
    prefix_text = pair.prefix

    print(f"Shared prefix preview (~{prefix_tokens} tokens, source={source_type}):")
    print(f"  «{prefix_text[:160].rstrip()}…»")
    print(f"  Task: {pair.task.value}\n")

    suffix_rng = random.Random()  # unseeded → fresh suffix templates each run

    # ---- Run loop --------------------------------------------------------
    for run_idx in range(n_runs):
        print(f"─── Run {run_idx + 1}/{n_runs} " + "─" * 40)
        completed = 0

        for req_idx in range(requests_per_run):
            # Build a fresh suffix for each request (different task template
            # each time so the suffix region always causes a cache miss).
            req_pair = build_prompt_pair(
                source,
                task=task,
                min_prefix_chars=max(100, prefix_chars // 2),
                max_prefix_chars=prefix_chars * 2,
                rng=suffix_rng,
            )
            # Override prefix with the fixed deterministic one so the server
            # sees the exact same prefix tokens on every request.
            prompt = _build_prompt(prefix_text, req_pair.suffix)

            label = f"  [{req_idx + 1:>3}/{requests_per_run}] task={req_pair.task.value:<10}"
            print(label, end=" ", flush=True)

            _send_request(
                endpoint=endpoint,
                completions_url=completions_url,
                model=model,
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                gen_tokens=gen_tokens,
                request_timeout_s=request_timeout_s,
            )

            completed += actual_req_tokens
            kv_filled = min(completed, effective_kv)
            print(f"KV filled: {kv_filled:>8} / {effective_kv}")

            if req_idx < requests_per_run - 1:
                time.sleep(request_interval_s)

        print(f"  Run {run_idx + 1} complete. Total KV filled: {min(completed, effective_kv)}/{effective_kv}")

        if run_idx < n_runs - 1:
            time.sleep(run_interval_s)

    print("\n" + "=" * 56)
    print("  Simulation complete.")
    print("=" * 56)


def simulate(
    endpoint: str,
    model: str,
    max_model_len: int,
    total_kv_tokens: int,
    prefix_length_perc: float = 50.0,
    n_runs: int = 1,
    source_type: str = "wikitext",
    task: Optional[TaskType] = None,
    cache_dir: Optional[str] = None,
    utilization_perc: float = 100.0,
) -> None:
    """Public entry point for the KV-cache prefix simulator."""
    
    run_simulator(
        endpoint=endpoint,
        model=model,
        max_model_len=max_model_len,
        total_kv_tokens=total_kv_tokens,
        prefix_length_perc=prefix_length_perc,
        n_runs=n_runs,
        source_type=source_type,
        task=task,
        cache_dir=cache_dir,
        utilization_perc=utilization_perc,
    )


def register_parser(
    subparsers: argparse._SubParsersAction,
    parents: list[argparse.ArgumentParser],
) -> None:
    """Register simulator plugin subcommand and its arguments."""

    parser = subparsers.add_parser(
        "simulator",
        parents=parents,
        help="Run KV-cache prefix simulator",
        description="Run synthetic KV-cache prefix-sharing simulation",
    )
    parser.add_argument(
        "--total-kv-tokens",
        type=int,
        required=True,
        help="Total KV cache tokens to target",
    )
    parser.add_argument(
        "--prefix-length-perc",
        type=float,
        default=70.0,
        help="Shared prefix percentage for simulator requests (default: 70)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of simulation runs (default: 1)",
    )
    parser.add_argument(
        "--source-type",
        default="wikitext",
        help="Text source: wikitext | squad | wikipedia (default: wikitext)",
    )
    parser.add_argument(
        "--task",
        default=None,
        choices=[t.value for t in TaskType],
        help="Task type for simulator requests (default: random)",
    )
    parser.add_argument(
        "--utilization-perc",
        type=float,
        default=100.0,
        help="Percent of total KV tokens to target (default: 100)",
    )
    parser.add_argument(
        "--request-interval-s",
        type=float,
        default=REQUEST_INTERVAL_S,
        help=f"Seconds to wait between requests (default: {REQUEST_INTERVAL_S})",
    )
    parser.add_argument(
        "--run-interval-s",
        type=float,
        default=RUN_INTERVAL_S,
        help=f"Seconds to wait between runs (default: {RUN_INTERVAL_S})",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_S,
        help=f"HTTP timeout per request in seconds (default: {DEFAULT_REQUEST_TIMEOUT_S})",
    )
    parser.set_defaults(plugin_runner=run_from_args)


def run_from_args(args: argparse.Namespace) -> None:
    """Run simulator from parsed CLI args."""
    simulate_task = TaskType(args.task) if args.task else None
    run_simulator(
        endpoint=args.endpoint,
        model=args.resolved_model,
        max_model_len=args.max_model_len,
        total_kv_tokens=args.total_kv_tokens,
        prefix_length_perc=args.prefix_length_perc,
        n_runs=args.n_runs,
        source_type=args.source_type,
        task=simulate_task,
        cache_dir=args.cache_dir,
        utilization_perc=args.utilization_perc,
        request_interval_s=args.request_interval_s,
        run_interval_s=args.run_interval_s,
        request_timeout_s=args.request_timeout_s,
    )
