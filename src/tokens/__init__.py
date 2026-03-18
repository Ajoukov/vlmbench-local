from typing import Dict, List, Tuple

import requests


def _render_messages(messages: List[Dict]) -> str:
    """Render a list of messages into a single prompt string.

    Parameters
    ----------
    messages : list[dict]
        A list of messages, where each message is a dict with 'role' and 'content' keys.

    Returns
    -------
    str
        The rendered prompt string.
    """

    return "\n".join(
        [
            f"{message.get('role', '')}: {message.get('content', '')}"
            for message in messages
        ]
    )


def _token_count(
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
        prompt = _render_messages(prompt)

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


def _truncate_prompt(
    endpoint: str,
    model: str,
    tokens: List[int],
    limit: int,
    timeout_s: float = 10.0,
) -> str:
    """Truncate the prompt tokens to fit within the specified limit and detokenize back to text.

    Parameters
    ----------
    endpoint : str
        The base URL of the model server (e.g. http://localhost:8000).
    model : str
        The name of the model to use for detokenization.
    tokens : list[int]
        The list of token IDs for the input prompt.
    limit : int
        The maximum number of tokens allowed for the prompt after truncation.
    timeout_s : float, optional
        The timeout in seconds for the detokenize request (default is 10.0).

    Returns
    -------
    str
        The truncated prompt text.

    Raises
    ------
    HTTPError
        If the /detokenize endpoint returns a non-200 status code or is unreachable.
    """

    truncated_tokens = tokens[:limit]

    r = requests.post(
        f"{endpoint.rstrip('/')}/detokenize",
        json={"model": model, "tokens": truncated_tokens},
        timeout=timeout_s,
    )
    r.raise_for_status()

    return r.json().get("prompt", "")


def _truncate_messages(
    endpoint: str,
    model: str,
    msgs: List[Dict],
    limit: int,
    timeout_s: float = 10.0,
) -> List[Dict]:
    """Truncate the list of messages to fit within the specified token limit.

    Parameters
    ----------
    endpoint : str
        The base URL of the model server (e.g. http://localhost:8000).
    model : str
        The name of the model to use for tokenization and detokenization.
    msgs : list[dict]
        The list of messages to be truncated, where each message is a dict with 'role' and 'content' keys.
    limit : int
        The maximum number of tokens allowed for the messages after truncation.
    timeout_s : float, optional
        The timeout in seconds for the tokenization and detokenization requests (default is 10.0).

    Returns
    -------
    list[dict]
        The truncated list of messages that fits within the token limit.

    Raises
    ------
    HTTPError
        If the /tokenize or /detokenize endpoints return a non-200 status code or are unreachable during truncation.
    """

    while msgs:
        count, _ = _token_count(
            endpoint,
            model,
            {"messages": msgs},
            timeout_s,
        )

        if count <= limit:
            return msgs

        # preserve first system message
        if len(msgs) > 1 and msgs[0]["role"] == "system":
            msgs.pop(1)
        else:
            msgs.pop(0)

    return []


def truncate_payload(
    endpoint: str,
    payload: dict,
    max_model_len: int,
    timeout_s: float = 10.0,
) -> Dict:
    """Truncate the input prompt in the payload if the token count exceeds the model's max_model_len.

    Parameters
    ----------
    endpoint : str
        The base URL of the model server (e.g. http://localhost:8000).
    payload : dict
        The original payload to be sent to the model server.
    max_model_len : int
        The maximum token length supported by the model.
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

    # extract the model
    model = payload.get("model")
    if model is None:
        raise ValueError("Payload must include 'model' key to determine tokenization for truncation")

    generation_tokens = payload.get("max_tokens")
    if generation_tokens is None:
        raise ValueError("Payload must include 'max_tokens' when using --truncate")

    # calculate the limit
    limit = max_model_len - generation_tokens - 32

    # get the token count for the input prompt
    count, tokens = _token_count(
        endpoint,
        model,
        payload,
        timeout_s,
    )
    if count <= limit:
        return payload

    if "prompt" in payload:
        truncated_prompt = _truncate_prompt(endpoint, model, tokens, limit, timeout_s)
        payload["prompt"] = truncated_prompt
    elif "messages" in payload:
        truncated_msgs = _truncate_messages(
            endpoint, model, payload["messages"], limit, timeout_s
        )
        payload["messages"] = truncated_msgs

    return payload
