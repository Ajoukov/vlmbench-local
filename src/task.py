from abc import ABC, abstractmethod


class Task(ABC):
    """A task is a specific type of problem that we want to solve with a language model."""

    def __init__(self, model: str):
        """Initialize the task with a model name (e.g. "gpt-4")."""

        self._model = model

    def model(self) -> str:
        """Return the model name."""

        return self._model

    @abstractmethod
    def payload(self, prompt: str, opts: dict) -> dict:
        """Build an HTTP-ready request dict.

        Returns
        -------
        dict
            A dictionary containing the keys "model", "prompt", and any additional options.
        """

        pass
