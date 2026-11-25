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

# The run's home is under tmp/ as well. isolate_user_dirs keeps tapeback's files out of
# the developer's home through the XDG variables, which holds only while the code reads
# them, and a mutation run breaks exactly that code: a mutant's fallback to the home
# wrote a test's voice profile over the developer's own. Set before tapeback is imported,
# so the paths it fixes at import (Settings' default vault) land here too. Only the caches
# stay the developer's, pinned before HOME moves: the e2e suite loads its Whisper and
# pyannote models from there, and tapeback keeps nothing in a cache.
os.environ.setdefault("XDG_CACHE_HOME", str(Path.home() / ".cache"))
_HOME = _TMP / "home"
_HOME.mkdir(exist_ok=True)
os.environ["HOME"] = str(_HOME)

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
