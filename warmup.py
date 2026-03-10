import requests
import threading
import time
import math

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


def _estimate_concurrency(total_kv_tokens: int, prompt_tokens: int, gen_tokens: int) -> int:
    per_request_tokens = max(1, prompt_tokens + gen_tokens)
    return max(1, math.ceil(total_kv_tokens / per_request_tokens))


def _keep_request_alive(
    completions_url: str,
    model: str,
    prompt_tokens: int,
    gen_tokens: int,
    request_timeout_s: float,
) -> None:
    prompt = "warmup " * prompt_tokens

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": gen_tokens,
        "temperature": 0,
        "stream": True,
    }

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
    request_interval_s: float = REQUEST_INTERVAL_S,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
) -> None:
    completions_url = f"{endpoint.rstrip('/')}/v1/completions"
    prompt_tokens, gen_tokens = _split_tokens_from_max_len(max_model_len)
    estimated_concurrency = _estimate_concurrency(
        total_kv_tokens=total_kv_tokens,
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
    )

    print("=== Warmup plugin ===")
    print("Total KV tokens:", total_kv_tokens)
    print("Max model length:", max_model_len)
    print("Prompt tokens:", prompt_tokens)
    print("Generation tokens:", gen_tokens)
    print("Estimated per-request KV tokens:", prompt_tokens + gen_tokens)
    print("Estimated warmup concurrency:", estimated_concurrency)

    threads = []
    for _ in range(estimated_concurrency):
        t = threading.Thread(
            target=_keep_request_alive,
            args=(
                completions_url,
                model,
                prompt_tokens,
                gen_tokens,
                request_timeout_s,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)
        time.sleep(request_interval_s)

    print("Warmup requests launched")


def warmup(
    endpoint: str,
    model: str,
    max_model_len: int,
    total_kv_tokens: int,
) -> None:
    run_warmup_plugin(
        endpoint=endpoint,
        model=model,
        max_model_len=max_model_len,
        total_kv_tokens=total_kv_tokens,
    )
