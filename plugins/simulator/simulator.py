import argparse
import math
import queue
import random
import time
from typing import Optional, Tuple, Dict

from .text_sources import (
    TaskType,
    TextSource,
    build_prompt_pair,
    make_source,
)
from src.runner import Runner
from src.runner.stats import RunnerStats
from src.tokens import truncate_payload
from src.utils import assert_server_up, auto_detect_model, detect_max_model_len

PROMPT_RATIO = 1.0 / 3.0
REQUEST_INTERVAL_S = 1.0
RUN_INTERVAL_S = 2.0
DEFAULT_REQUEST_TIMEOUT_S = 10.0
MIN_PROMPT_TOKENS = 16
MIN_GEN_TOKENS = 16


def _split_tokens(total_tokens: int) -> Tuple[int, int]:
    """Split total tokens into (prompt_tokens, gen_tokens).
    
    Parameters
    ----------
    total_tokens : int
        Total number of tokens.

    Returns
    -------
    tuple[int, int]
        Prompt tokens and output tokens number.
    """

    usable = max(1, total_tokens - 2)
    prompt_tokens = max(MIN_PROMPT_TOKENS, int(usable * PROMPT_RATIO))
    gen_tokens = max(MIN_GEN_TOKENS, usable - prompt_tokens)

    if prompt_tokens + gen_tokens > usable:
        gen_tokens = max(MIN_GEN_TOKENS, usable - prompt_tokens)
    if prompt_tokens + gen_tokens > usable:
        prompt_tokens = max(MIN_PROMPT_TOKENS, usable - gen_tokens)

    return prompt_tokens, gen_tokens


def _build_prompt(prefix_text: str, suffix: str) -> str:
    """Concatenate the shared *prefix_text* with a task-instruction *suffix*.
    
    Parameters
    ----------
    prefix_test : str
    suffix : str

    Returns
    -------
    prompt : str
    """

    return f"{prefix_text}\n\n{suffix}"


def _build_payload(
    endpoint: str,
    model: str,
    prompt: str,
    prompt_tokens: int,
    gen_tokens: int,
) -> Dict:
    """Build the payload of request.

    Parameters
    ----------
    endpoint : str
        vLLM address.
    model : str
        Model name.
    prompt : str
        Payload prompt.
    prompt_tokens : int
        Prompt token size.
    gen_tokens : int
        Response token size.

    Returns
    -------
    payload : dict
    """
    
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "max_tokens": gen_tokens,
        "temperature": 0,
    }

    payload = truncate_payload(
        endpoint, payload, max_model_len=prompt_tokens + gen_tokens + 2
    )

    return payload


def run_simulator(
    endpoint: str,
    model: str,
    max_model_len: int,
    total_kv_tokens: int,
    prefix_length_perc: float = 50.0,
    n_runs: int = 1,
    source_type: str = "wikitext",
    enable_metrics: bool = False,
    task: Optional[TaskType] = None,
    cache_dir: Optional[str] = None,
    seed: int = 42,
    utilization_perc: float = 100.0,
    request_interval_s: float = REQUEST_INTERVAL_S,
    run_interval_s: float = RUN_INTERVAL_S,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
) -> None:
    """Fire synthetic requests to simulate KV-cache usage with prefix sharing.

    The shared prefix is a real English passage drawn from *source_type*.
    Each request appends a different task instruction (summarise, QA, explain,
    chat, continue) as its unique suffix, ensuring no full cache hits while the
    prefix region is maximally reused.

    Parameters
    ----------
    endpoint : str
        Base URL of the OpenAI-compatible inference server.
    model : str
        Model identifier served at *endpoint*.
    max_model_len : int
        Maximum context length (in tokens) supported by the model.
    total_kv_tokens : int
        Target number of KV tokens to fill per run.
    prefix_length_perc : float
        Percentage (0 - 100) of each prompt that is the *shared prefix*.
        The remaining percentage is the task-instruction suffix.
    n_runs : int
        How many full simulation cycles to execute.
    source_type : str
        Text backend for generating the English passage.
        ``"wikitext"`` (default) | ``"squad"`` | ``"wikipedia"``.
    task : TaskType
        Force a specific :class:`~text_sources.TaskType` for every request.
        ``None`` (default) rotates through all task types randomly.
    cache_dir : str
        HuggingFace dataset cache directory (ignored for ``"wikipedia"``).
    seed : int
        Seed for random population (default 42).
    utilization_perc : float
        Fraction of *total_kv_tokens* to actually target (0 - 100).
    request_interval_s : float
        Seconds to wait between requests within a run.
    run_interval_s : float
        Seconds to wait between runs.
    request_timeout_s : float
        Per-request HTTP timeout in seconds.
    """

    completions_url = f"{endpoint.rstrip('/')}/v1/completions"

    # KV-token budget
    effective_kv = max(1, int(math.ceil(total_kv_tokens * (utilization_perc / 100.0))))
    max_single = max(1, max_model_len - 2)
    target_tokens = min(effective_kv, max_single)

    # prompt / generation split
    prompt_tokens, gen_tokens = _split_tokens(target_tokens + 2)

    # prefix / suffix split (in tokens)
    prefix_frac = max(0.0, min(1.0, prefix_length_perc / 100.0))
    prefix_tokens = max(1, int(prompt_tokens * prefix_frac))
    suffix_tokens = max(1, prompt_tokens - prefix_tokens)

    # how many requests per run
    actual_req_tokens = prompt_tokens + gen_tokens
    requests_per_run = max(1, math.ceil(effective_kv / max(1, actual_req_tokens)))

    # approximate character budget for the passage
    # ~4 chars per sub-word token is a reasonable heuristic.
    _CHARS_PER_TOKEN = 4
    prefix_chars = prefix_tokens * _CHARS_PER_TOKEN

    # Report
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
    print(
        f"    ├─ Prefix  (~{prefix_length_perc:.0f}%)           : ~{prefix_tokens} tokens"
    )
    print(
        f"    └─ Suffix  (~{100 - prefix_length_perc:.0f}%)           : ~{suffix_tokens} tokens"
    )
    print(f"  Generation tokens         : {gen_tokens}")
    print(f"  Actual KV / request       : {actual_req_tokens}")
    print(f"  Requests per run          : {requests_per_run}")
    print(f"  Number of runs (N)        : {n_runs}")
    if effective_kv > max_single:
        print(
            "  [WARN] Target exceeds single-request capacity; capped at max_model_len."
        )
    print("=" * 56)

    # build the shared prefix via the text source
    print(f"\nLoading text source '{source_type}'…")
    source: TextSource = make_source(source_type, cache_dir=cache_dir, seed=42)
    
    # deterministic seed, same passage every time this function is called,
    # which is exactly what we want so the KV cache prefix is stable across runs.
    seed_rng = random.Random(seed)
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

    suffix_rng = random.Random()  # unseeded, fresh suffix templates each run

    # reuse the common request execution path used by benchmarks.
    jobs: "queue.Queue[dict | None]" = queue.Queue()
    stats = RunnerStats()
    runner = Runner(
        runner_id=1,
        endpoint=endpoint,
        jobs=jobs,
        stats=stats,
        request_timeout=request_timeout_s,
        enable_metrics=enable_metrics,
    )
    runner.start()

    # run loop
    for run_idx in range(n_runs):
        print(f"─── Run {run_idx + 1}/{n_runs} " + "─" * 40)
        completed = 0

        for req_idx in range(requests_per_run):
            # build a fresh suffix for each request (different task template
            # each time so the suffix region always causes a cache miss).
            req_pair = build_prompt_pair(
                source,
                task=task,
                min_prefix_chars=max(100, prefix_chars // 2),
                max_prefix_chars=prefix_chars * 2,
                rng=suffix_rng,
            )

            # override prefix with the fixed deterministic one so the server
            # sees the exact same prefix tokens on every request.
            prompt = _build_prompt(prefix_text, req_pair.suffix)

            label = f"  [{req_idx + 1:>3}/{requests_per_run}] task={req_pair.task.value:<10}"
            print(label, end=" ", flush=True)

            # build a payload
            payload = _build_payload(
                endpoint=endpoint,
                model=model,
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                gen_tokens=gen_tokens,
            )

            # send it to runner
            jobs.put(
                {
                    "name": "simulator",
                    "url": completions_url,
                    "headers": {"Content-Type": "application/json"},
                    "payload": payload,
                }
            )

            # wait for job to be finished
            jobs.join()

            completed += actual_req_tokens
            kv_filled = min(completed, effective_kv)
            print(f"KV filled: {kv_filled:>8} / {effective_kv}")

            if req_idx < requests_per_run - 1:
                time.sleep(request_interval_s)

        print(
            f"  Run {run_idx + 1} complete. Total KV filled: {min(completed, effective_kv)}/{effective_kv}"
        )

        if run_idx < n_runs - 1:
            time.sleep(run_interval_s)

    # terminate the runner thread
    jobs.put(None)
    jobs.join()
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

    print(f"--- {n} requests, {ok} ok, {fail} failed ---")
    print(f"Total request bytes: {total_request_bytes}")
    print(f"Total response bytes: {total_response_bytes}")
    print(f"Average latency: {average_latency:.2f} ms")
    print(f"95th percentile latency: {p95_latency:.2f} ms")

    # print vLLM metrics if available
    vllm_metrics = stats.vllm_stats()
    for metric_name, value in vllm_metrics.items():
        print(f"vllm:'{metric_name}': {value}")


def register_parser(
    subparsers: argparse._SubParsersAction,
    parents: list[argparse.ArgumentParser],
) -> None:
    """Register simulator plugin subcommand and its arguments."""

    parser = subparsers.add_parser(
        "simulator",
        parents=parents,
        help="Run KV-cache prefix simulator",
        description="Run synthetic KV-cache prefix-sharing simulation.",
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
        "--seed",
        type=int,
        default=42,
        help="Seed for random population",
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

    # get simulate task
    simulate_task = TaskType(args.task) if args.task else None

    # extract the endpoint and datasets directory
    endpoint = args.endpoint

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

    # get maximum model size
    max_model_len = detect_max_model_len(
        endpoint, model, timeout_s=args.request_timeout_s
    )
    print(f"[CHECK] Max model length: {max_model_len}")

    # call run simulator
    run_simulator(
        endpoint=args.endpoint,
        model=model,
        max_model_len=max_model_len,
        total_kv_tokens=args.total_kv_tokens,
        prefix_length_perc=args.prefix_length_perc,
        n_runs=args.n_runs,
        source_type=args.source_type,
        task=simulate_task,
        cache_dir=args.cache_dir,
        seed=args.seed,
        utilization_perc=args.utilization_perc,
        request_interval_s=args.request_interval_s,
        run_interval_s=args.run_interval_s,
        request_timeout_s=args.request_timeout_s,
        enable_metrics=args.enable_metrics,
    )
