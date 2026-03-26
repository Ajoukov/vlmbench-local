import os
from typing import Any, Dict

# default values for environment variables
REQUEST_TIMEOUT = 12  # seconds
DEFAULT_ENDPOINT = "http://127.0.0.1:8080"  # openai endpoint
DEFAULT_DATA_DIR = "llm-datasets"  # default data directory for datasets


def init_vars() -> Dict[str, Any]:
    """Read environment variables and return a dictionary of variables.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the environment variables and their values.
    """

    vars = {
        "REQUEST_TIMEOUT": int(os.getenv("REQUEST_TIMEOUT", REQUEST_TIMEOUT)),
        "DEFAULT_ENDPOINT": os.getenv("DEFAULT_ENDPOINT", DEFAULT_ENDPOINT),
        "DEFAULT_DATA_DIR": os.getenv("DEFAULT_DATA_DIR", DEFAULT_DATA_DIR),
    }

    return vars
