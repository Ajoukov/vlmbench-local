import sys


class Tee:
    """A class that duplicates the output to both the console and a file."""

    def __init__(self, file_path: str):
        """Initialize the Tee object with the specified file path."""

        self._file = open(file_path, "a")
        self._stdout = sys.stdout

    def write(self, data: str) -> None:
        """Write data to both the file and the console."""

        self._file.write(data)
        self._stdout.write(data)

    def flush(self) -> None:
        self._file.flush()
        self._stdout.flush()


def make_tee(file_path: str) -> None:
    """Redirect the standard output to a Tee object that writes to both the console and a file.
    
    Parameters
    ----------
    file_path : str
        The path to the file where the output should be written.
    """

    sys.stdout = Tee(file_path)
