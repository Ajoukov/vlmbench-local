import requests
import time
import math
import uuid
from src.utils import truncate_payload

PROMPT_RATIO = 1.0 / 3.0
REQUEST_INTERVAL_S = 1.0
DEFAULT_REQUEST_TIMEOUT_S = 10.0
MIN_PROMPT_TOKENS = 16
MIN_GEN_TOKENS = 16


def _split_tokens_from_max_len(max_model_len: int) -> tuple[int, int]:
    usable = max(1, max_model_len - 2)
    prompt_tokens = max(MIN_PROMPT_TOKENS, int(usable * PROMPT_RATIO))
    gen_tokens = max(MIN_GEN_TOKENS, usable - prompt_tokens)

    if prompt_tokens + gen_tokens > usable:
        gen_tokens = max(MIN_GEN_TOKENS, usable - prompt_tokens)
        if prompt_tokens + gen_tokens > usable:
            prompt_tokens = max(MIN_PROMPT_TOKENS, usable - gen_tokens)

    return prompt_tokens, gen_tokens


def _keep_request_alive(
    endpoint: str,
    completions_url: str,
    model: str,
    prompt_tokens: int,
    gen_tokens: int,
    request_timeout_s: float,
) -> None:
    unique_tag = f"warmup-{uuid.uuid4().hex}-{time.time_ns()}"
    prompt = f"{unique_tag} " + ("warmup " * max(prompt_tokens * 3, prompt_tokens))

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": gen_tokens,
        "temperature": 0,
        "stream": True,
    }

    payload = truncate_payload(endpoint, payload, max_model_len=prompt_tokens + gen_tokens + 2)

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
        print(f"Warmup request failed: {e}")


def run_warmup_plugin(
    endpoint: str,
    model: str,
    max_model_len: int,
    total_kv_tokens: int,
    utilization_perc: float = 100.0,
    request_interval_s: float = REQUEST_INTERVAL_S,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
) -> None:
    completions_url = f"{endpoint.rstrip('/')}/v1/completions"
    effective_kv_tokens = max(1, int(math.ceil(total_kv_tokens * (utilization_perc / 100.0))))
    max_single_request_tokens = max(1, max_model_len - 2)
    target_request_tokens = min(effective_kv_tokens, max_single_request_tokens)
    prompt_tokens, gen_tokens = _split_tokens_from_max_len(target_request_tokens + 2)

    print("=== Warmup plugin ===")
    print("Total KV tokens:", total_kv_tokens)
    print("Requested utilization (%):", utilization_perc)
    print("Effective KV tokens:", effective_kv_tokens)
    print("Max model length:", max_model_len)
    print("Max single-request tokens:", max_single_request_tokens)
    print("Target single-request tokens:", target_request_tokens)
    print("Prompt tokens:", prompt_tokens)
    print("Generation tokens:", gen_tokens)
    print("Actual request KV tokens:", prompt_tokens + gen_tokens)

    if effective_kv_tokens > max_single_request_tokens:
        print(
            "Requested KV tokens exceed a single request capacity; "
            "using the largest request allowed by max_model_len."
        )

    actual_request_tokens = prompt_tokens + gen_tokens
    required_requests = max(1, math.ceil(effective_kv_tokens / max(1, actual_request_tokens)))
    print("Required sequential warmup requests:", required_requests)

    completed_tokens = 0
    for request_index in range(required_requests):
        print(f"Submitting warmup request {request_index + 1}/{required_requests}")
        _keep_request_alive(
            endpoint=endpoint,
            completions_url=completions_url,
            model=model,
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            request_timeout_s=request_timeout_s,
        )
        completed_tokens += actual_request_tokens
        print("Accumulated KV tokens:", min(completed_tokens, effective_kv_tokens), "/", effective_kv_tokens)
        if request_index < required_requests - 1:
            time.sleep(request_interval_s)

    print("Warmup requests completed")


def warmup(
    endpoint: str,
    model: str,
    max_model_len: int,
    total_kv_tokens: int,
    utilization_perc: float = 100.0,
) -> None:
    """Main entry point for the warmup plugin."""

    run_warmup_plugin(
        endpoint=endpoint,
        model=model,
        max_model_len=max_model_len,
        total_kv_tokens=total_kv_tokens,
        utilization_perc=utilization_perc,
    )
