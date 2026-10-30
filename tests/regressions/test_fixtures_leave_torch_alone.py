"""Regression: importing the test fixtures loaded torch.

`tests/fixtures.py` asked whether pyannote is installed by importing `pyannote.audio`,
which loads torch: every pytest process, even one that ran a single unit test, grew from
about 65 MB to about 830 MB. mutmut runs pytest in a worker per core, and its first run
on a laptop with 8 GB exhausted the memory: the kernel's OOM killer took the IDE with it.
"""

import json
import subprocess
import sys

from tests.fixtures import REPO_ROOT

_LOADED = (
    "import json, sys, tests.fixtures\n"
    "print(json.dumps([m for m in ('torch', 'pyannote.audio') if m in sys.modules]))"
)


def test_importing_the_fixtures_loads_neither_torch_nor_pyannote():
    result = subprocess.run(
        [sys.executable, "-c", _LOADED], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )

    assert json.loads(result.stdout) == []
