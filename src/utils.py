import sys

import requests


def detect_model(endpoint: str) -> str:
    """Auto-detect the served model from the /v1/models endpoint."""
    r = requests.get(f"{endpoint.rstrip('/')}/v1/models", timeout=10)
    r.raise_for_status()
    models = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
    if not models:
        print("Error: No models found at endpoint.", file=sys.stderr)
        sys.exit(1)
    if len(models) > 1:
        print(
            f"Multiple models found: {models}. Using first: {models[0]}",
            file=sys.stderr,
        )
    return models[0]


def assert_server_up(endpoint: str, timeout_s: float = 5.0):
    r = requests.get(f"{endpoint.rstrip('/')}/health", timeout=timeout_s)
    r.raise_for_status()


def detect_max_model_len(endpoint: str) -> int:
    """Get max_model_len from the /v1/models endpoint."""
    r = requests.get(f"{endpoint.rstrip('/')}/v1/models", timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        print("Error: No models found at endpoint.", file=sys.stderr)
        sys.exit(1)
    return data[0].get("max_model_len", 0)


def truncate_payload(endpoint: str, payload: dict, max_model_len: int) -> dict:
    """
    Tokenize the input; if it exceeds max_model_len - max_tokens, truncate and detokenize.
    Works for both chat completions (messages) and completions (prompt).
    """
    base = endpoint.rstrip("/")
    model = payload.get("model", "")
    generation_tokens = payload.get("max_tokens")
    if generation_tokens is None:
        raise ValueError("Payload must include 'max_tokens' when using --truncate")
    # Subtract a small buffer to account for special tokens (e.g. BOS)
    # added during the detokenize → re-tokenize round-trip
    limit = max_model_len - generation_tokens - 2

    # Extract the raw text to tokenize (always use prompt format to avoid
    # chat template issues with models that don't support one)
    messages = payload.get("messages")
    prompt = payload.get("prompt")

    if messages:
        # Extract text from the last user message
        text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                text = msg.get("content", "")
                break
        if not text:
            return payload
    elif prompt:
        text = prompt
    else:
        return payload

    tok_r = requests.post(
        f"{base}/tokenize",
        json={"model": model, "prompt": text},
        timeout=10,
    )

    tok_r.raise_for_status()
    tokens = tok_r.json().get("tokens", [])
    count = tok_r.json().get("count", len(tokens))

    if count <= limit:
        return payload

    # Truncate: keep first limit tokens, detokenize back
    truncated_tokens = tokens[:limit]
    detok_r = requests.post(
        f"{base}/detokenize",
        json={"model": model, "tokens": truncated_tokens},
        timeout=10,
    )
    detok_r.raise_for_status()
    truncated_text = detok_r.json().get("prompt", "")

    print(f"[TRUNCATE] {count} -> {limit} tokens")

    payload = dict(payload)
    if messages:
        new_messages = list(messages)
        for i in range(len(new_messages) - 1, -1, -1):
            if new_messages[i].get("role") == "user":
                new_messages[i] = {**new_messages[i], "content": truncated_text}
                break
        payload["messages"] = new_messages
    elif prompt:
        payload["prompt"] = truncated_text

    return payload
