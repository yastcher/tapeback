"""Regression: freeing the GPU loaded torch.

`free_gpu_memory()` imported torch to empty its CUDA cache. A process that has not
imported torch holds no such cache, yet paid several hundred MB of native code to find
that out: a run with remote transcription and no diarization, and every test that
reached the call. Only a torch already imported can hold one.
"""

import subprocess
import sys
from types import SimpleNamespace

from tapeback import _gpu

_FREE_IN_A_FRESH_INTERPRETER = (
    "import sys\n"
    "from tapeback._gpu import free_gpu_memory\n"
    "free_gpu_memory()\n"
    "print('torch' in sys.modules)"
)


def test_freeing_the_gpu_does_not_load_torch():
    result = subprocess.run(
        [sys.executable, "-c", _FREE_IN_A_FRESH_INTERPRETER],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "False"


def test_a_loaded_torch_has_its_cache_emptied(monkeypatch):
    emptied = []
    cuda = SimpleNamespace(is_available=lambda: True, empty_cache=lambda: emptied.append(True))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))

    _gpu.free_gpu_memory()

    assert emptied == [True]


def test_a_loaded_torch_without_cuda_is_left_alone(monkeypatch):
    emptied = []
    cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: emptied.append(True))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))

    _gpu.free_gpu_memory()

    assert emptied == []
