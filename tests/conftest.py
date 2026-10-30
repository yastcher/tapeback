import importlib.util
import os
import tempfile
from pathlib import Path

import pytest

# A test run's temporary files, pytest's tmp_path and its subprocesses' mktemp alike, go
# under the repository's tmp/, never /tmp: on some machines /tmp is a small tmpfs with a
# per-user quota, and a full one makes every command fail. TMPDIR is what both read, and
# pytest has made no temporary directory yet when this file loads; gettempdir() caches
# its first answer, so the cache goes too.
_TMP = Path(__file__).resolve().parents[1] / "tmp"
_TMP.mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(_TMP)
tempfile.tempdir = None

pytest_plugins = [
    "tests.fixtures",
]


def _pyannote_available() -> bool:
    """Whether pyannote is installed, asked without importing it: the import loads
    torch, about 760 MB in every pytest process, wanted or not."""
    try:
        return importlib.util.find_spec("pyannote.audio") is not None
    except ModuleNotFoundError:  # no `pyannote` package at all
        return False


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip the tests marked `pyannote` where pyannote is not installed."""
    if _pyannote_available():
        return
    skip = pytest.mark.skip(reason="pyannote-audio not installed (install tapeback[diarize])")
    for item in items:
        if item.get_closest_marker("pyannote"):
            item.add_marker(skip)
