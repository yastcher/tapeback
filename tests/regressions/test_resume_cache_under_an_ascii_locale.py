"""Regression: the resume cache wrote with the locale's encoding.

`store()` wrote its JSON with `Path.write_text` and no encoding, Cyrillic kept as is.
Under a locale whose encoding is not UTF-8 the write raised UnicodeEncodeError, which
its `except OSError` let through, and the run that had just transcribed failed. The C
locale with UTF-8 mode and locale coercion turned off is such a locale on every machine.
"""

import os
import subprocess
import sys

# The command line itself must be ASCII under that locale, hence the escapes.
_STORE_AND_LOAD = """
import sys
from pathlib import Path

from tapeback import _resume
from tapeback.models import Segment

key = _resume.ResumeKey("0123456789abcdef0123456789abcdef")
directory = Path(sys.argv[1])
said = "\\u0440\\u0435\\u043f\\u043b\\u0438\\u043a\\u0430"
stored = _resume.store(key, directory, [Segment(start=0.0, end=5.0, text=said)], {})
loaded, _ = _resume.load(key, directory)
print(stored == directory / key.filename, ascii(loaded[0].text))
"""


def test_the_resume_cache_keeps_cyrillic_under_an_ascii_locale(tmp_path):
    env = {**os.environ, "LC_ALL": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"}

    result = subprocess.run(
        [sys.executable, "-c", _STORE_AND_LOAD, str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    said = "'\\u0440\\u0435\\u043f\\u043b\\u0438\\u043a\\u0430'"
    assert result.stdout == f"True {said}\n", result.stderr
