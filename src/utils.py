from typing import Dict, List, Tuple

import requests


def _message_to_prompt(messages: List[Dict]) -> str:
    """Convert a list of messages to a single prompt string.

    This is used to ensure that the token counting and truncation logic
    works correctly for both chat and non-chat models, by always using
    the same prompt format.

    Parameters
    ----------
    messages : list[dict]
        A list of messages in the format [{"role": "user", "content": "..."}, ...].

    Returns
    -------
    str
        A single string containing the concatenated user messages.
    """

    prompt = ""
    for msg in messages:
        prompt += msg.get("role", "") + ": " + msg.get("content", "") + "\n"
    return prompt.strip()


def _prompt_to_messages(prompt: str) -> List[Dict]:
    """Convert a prompt string back to a list of messages.

    This is used to convert the truncated prompt back to messages format
    after truncation.

    Parameters
    ----------
    prompt : str
        A single string containing the concatenated user messages.

    Returns
    -------
    list[dict]
        A list of messages in the format [{"role": "user", "content": "..."}, ...].
    """

    messages = []
    for line in prompt.split("\n"):
        if ": " in line:
            role, content = line.split(": ", 1)
            messages.append({"role": role.strip(), "content": content.strip()})
        else:
            messages.append({"role": "user", "content": line.strip()})
    return messages


def assert_server_up(endpoint: str, timeout_s: float = 5.0) -> None:
    """Assert that the server is up by calling the /health endpoint.

    Parameters
    ----------
    endpoint : str
        The base URL of the model server (e.g. http://localhost:8000).
    timeout_s : float, optional
        The timeout in seconds for the health check request (default is 5.0).

    Raises
    ------
    HTTPError
        If the /health endpoint returns a non-200 status code or is unreachable.
    """

    r = requests.get(f"{endpoint.rstrip('/')}/health", timeout=timeout_s)
    r.raise_for_status()


def auto_detect_model(endpoint: str) -> str:
    """Auto-detect the served model from the /v1/models endpoint.

    Parameters
    ----------
    endpoint : str
        The base URL of the model server (e.g. http://localhost:8000).

    Returns
    -------
    str
        The name of the first model found at the endpoint.

    Raises
    ------
    HTTPError
        If no models are found at the endpoint.
    RuntimeError
        If the /v1/models endpoint returns an unexpected response.
    """

    # call /v1/models to get the list of available models
    r = requests.get(f"{endpoint.rstrip('/')}/v1/models", timeout=10)
    r.raise_for_status()

    # extract the model names from the response
    models = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
    if not models or len(models) == 0:
        raise RuntimeError("No models found at endpoint.")

    return models[0]


def detect_max_model_len(endpoint: str, model: str, timeout_s: float = 10.0) -> int:
    """Get max_model_len from the /v1/models endpoint.

    Parameters
    ----------
    endpoint : str
        The base URL of the model server (e.g. http://localhost:8000).
    model : str
        The name of the model for which to get max_model_len.
    timeout_s : float, optional
        The timeout in seconds for the request (default is 10.0).

    Returns
    -------
    int
        The max_model_len of the specified model.

    Raises
    ------
    HTTPError
        If the /v1/models endpoint returns a non-200 status code or is unreachable.
    RuntimeError
        If no models are found at the endpoint or if the response is malformed.
    """

    # call /v1/models to get the list of available models and their max_model_len
    r = requests.get(f"{endpoint.rstrip('/')}/v1/models", timeout=timeout_s)
    r.raise_for_status()

    # extract the max_model_len from the first model in the response
    data = r.json().get("data", [])
    if not data:
        raise RuntimeError("No models found at endpoint.")

    # find the specified model and return its max_model_len
    for model_data in data:
        if model_data.get("id") == model:
            return model_data.get("max_model_len", 0)

    raise RuntimeError(f"Model '{model}' not found at endpoint.")


def token_count(
    endpoint: str, model: str, payload: Dict, timeout_s: float = 10.0
) -> Tuple[int, List[int]]:
    """Get token count for a given prompt and model.

    Parameters
    ----------
    endpoint : str
        The base URL of the model server (e.g. http://localhost:8000).
    model : str
        The name of the model to use for tokenization.
    payload : dict
        The payload containing the input text to tokenize.
    timeout_s : float, optional
        The timeout in seconds for the request (default is 10.0).

    Returns
    -------
    tuple[int, list[int]]
        A tuple containing the token count and the list of token IDs.

    Raises
    ------
    HTTPError
        If the /tokenize endpoint returns a non-200 status code or is unreachable.
    """

    # extract the raw text to tokenize
    prompt = payload.get("prompt", payload.get("messages", ""))
    if isinstance(prompt, list):
        prompt = _message_to_prompt(prompt)

    # call /tokenize to get the token count and tokens for the prompt
    r = requests.post(
        f"{endpoint.rstrip('/')}/tokenize",
        json={"model": model, "prompt": prompt},
        timeout=timeout_s,
    )
    r.raise_for_status()

    # extract the token count and tokens from the response
    tokens = r.json().get("tokens", [])
    count = r.json().get("count", len(tokens))

    return count, tokens


def truncate_payload(
    endpoint: str,
    model: str,
    payload: dict,
    max_model_len: int,
    count: int,
    tokens: List[int],
    timeout_s: float = 10.0,
) -> Dict:
    """Truncate the input prompt in the payload if the token count exceeds the model's max_model_len.

    Parameters
    ----------
    endpoint : str
        The base URL of the model server (e.g. http://localhost:8000).
    model : str
        The name of the model to use for truncation.
    payload : dict
        The original payload to be sent to the model server.
    max_model_len : int
        The maximum token length supported by the model.
    count : int
        The token count of the input prompt.
    tokens : list[int]
        The list of token IDs for the input prompt.
    timeout_s : float, optional
        The timeout in seconds for the detokenize request (default is 10.0).

    Returns
    -------
    dict
        The modified payload with the truncated prompt if truncation was necessary, otherwise the original payload.

    Raises
    ------
    HTTPError
        If the /detokenize endpoint returns a non-200 status code or is unreachable during truncation.
    """

    generation_tokens = payload.get("max_tokens")
    if generation_tokens is None:
        raise ValueError("Payload must include 'max_tokens' when using --truncate")

    # calculate the limit
    limit = max_model_len - generation_tokens - 2
    if count <= limit:
        return payload

    # truncate, keep first limit tokens, detokenize back
    truncated_tokens = tokens[:limit]
    r = requests.post(
        f"{endpoint.rstrip('/')}/detokenize",
        json={"model": model, "tokens": truncated_tokens},
        timeout=timeout_s,
    )
    r.raise_for_status()

    # extract the truncated text from the response and update the payload
    truncated_text = r.json().get("prompt", "")

    # update the payload with the truncated prompt
    if "messages" in payload:
        payload["messages"] = _prompt_to_messages(truncated_text)
    else:
        payload["prompt"] = truncated_text

    return payload
