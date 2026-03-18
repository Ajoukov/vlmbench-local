import requests


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
