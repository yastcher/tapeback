"""Regression tests for VRAM release when CUDA fails and we fall back to CPU."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tapeback.transcriber import Transcriber

OOM = "CUDA failed with error out of memory"


def _info():
    info = MagicMock()
    info.language, info.language_probability, info.duration = "ru", 0.9, 10.0
    return info


@pytest.fixture
def clear_gpu(monkeypatch):
    monkeypatch.setattr("tapeback._whisper_device.wait_for_clamp_release", lambda *_a, **_k: True)
    monkeypatch.setattr("tapeback._whisper_device.get_free_vram_mib", lambda: 4096)


def test_low_free_vram_skips_cuda_entirely(settings, monkeypatch, capsys):
    """The leak is unrecoverable, so a card that cannot fit a model must not be touched.

    A CUDA out-of-memory during load leaves its allocation behind for the life of the
    process — measured 3674 MiB -> 95 MiB, unrecovered by dropping the traceback or by
    CT2_CUDA_ALLOCATOR=cuda_malloc_async. Attempting a load that cannot fit therefore
    costs the whole card, so it must be refused up front.
    """
    monkeypatch.setattr("tapeback._whisper_device.get_free_vram_mib", lambda: 95)
    clamp_calls: list[object] = []
    monkeypatch.setattr(
        "tapeback._whisper_device.wait_for_clamp_release",
        lambda *a, **k: clamp_calls.append(a) or True,
    )
    s = settings.model_copy(update={"device": "cuda", "min_free_vram_mib": 1200})

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        transcriber = Transcriber(s)

    assert transcriber.describe() == "Whisper: large-v3-turbo on cpu/int8"
    assert mock_model_cls.call_args_list[0].kwargs["device"] == "cpu"
    assert "95 MiB VRAM free" in capsys.readouterr().err
    # No point asking about the clamp on a card we are not going to use.
    assert clamp_calls == []


def test_enough_free_vram_uses_cuda(settings, monkeypatch):
    monkeypatch.setattr("tapeback._whisper_device.get_free_vram_mib", lambda: 1200)
    monkeypatch.setattr("tapeback._whisper_device.wait_for_clamp_release", lambda *_a, **_k: True)
    s = settings.model_copy(update={"device": "cuda", "min_free_vram_mib": 1200})

    with patch("tapeback.transcriber.WhisperModel"):
        assert Transcriber(s).describe() == "Whisper: large-v3-turbo on cuda/int8_float16"


def test_unknown_free_vram_does_not_block_cuda(settings, monkeypatch):
    """No nvidia-smi means no answer — never refuse the GPU over a check we cannot run."""
    monkeypatch.setattr("tapeback._whisper_device.get_free_vram_mib", lambda: None)
    monkeypatch.setattr("tapeback._whisper_device.wait_for_clamp_release", lambda *_a, **_k: True)
    s = settings.model_copy(update={"device": "cuda"})

    with patch("tapeback.transcriber.WhisperModel"):
        assert Transcriber(s).describe() == "Whisper: large-v3-turbo on cuda/int8_float16"


def test_vram_is_released_before_the_cpu_model_is_built_at_load(settings, clear_gpu, monkeypatch):
    """After an OOM at load, the dead GPU allocation must go before the CPU model is built.

    Bug: free VRAM went 3674 MiB -> 95 MiB after a failed CUDA load and stayed there for
    the rest of the process. ctranslate2 has its own allocator, and the exception's
    traceback kept the failed model's frame — and therefore its memory — reachable. The
    diarizer's VRAM check then found nothing free and dropped to CPU too, so a single
    transcription failure degraded the whole run.
    """
    calls: list[str] = []
    monkeypatch.setattr("tapeback.transcriber.free_gpu_memory", lambda: calls.append("free"))

    s = settings.model_copy(update={"device": "cuda", "thermal_clamp_wait": 60.0})

    def _construct(*_args, **kwargs):
        calls.append(f"build:{kwargs['device']}")
        if kwargs["device"] == "cuda":
            raise RuntimeError(OOM)
        return MagicMock()

    with patch("tapeback.transcriber.WhisperModel", side_effect=_construct):
        transcriber = Transcriber(s)

    assert calls == ["build:cuda", "free", "build:cpu"], calls
    assert transcriber.describe() == "Whisper: large-v3-turbo on cpu/int8"


def test_vram_is_released_before_the_cpu_model_is_built_at_inference(
    settings, clear_gpu, monkeypatch
):
    """Same ordering when the OOM happens while transcribing, not while loading."""
    calls: list[str] = []
    monkeypatch.setattr("tapeback.transcriber.free_gpu_memory", lambda: calls.append("free"))

    s = settings.model_copy(update={"device": "cuda", "thermal_clamp_wait": 60.0})

    gpu_model = MagicMock()
    gpu_model.transcribe.side_effect = RuntimeError(OOM)
    cpu_model = MagicMock()
    cpu_model.transcribe.return_value = (iter([]), _info())

    def _construct(*_args, **kwargs):
        calls.append(f"build:{kwargs['device']}")
        return gpu_model if kwargs["device"] == "cuda" else cpu_model

    with patch("tapeback.transcriber.WhisperModel", side_effect=_construct):
        transcriber = Transcriber(s)
        transcriber.transcribe(Path("/fake/audio.wav"))

    assert calls == ["build:cuda", "free", "build:cpu"], calls
    assert transcriber.describe() == "Whisper: large-v3-turbo on cpu/int8"


def test_failed_gpu_model_is_not_referenced_after_fallback(settings, clear_gpu):
    """The transcriber must not still be holding the GPU model it gave up on."""
    s = settings.model_copy(update={"device": "cuda", "thermal_clamp_wait": 60.0})

    gpu_model = MagicMock()
    gpu_model.transcribe.side_effect = RuntimeError(OOM)
    cpu_model = MagicMock()
    cpu_model.transcribe.return_value = (iter([]), _info())

    with patch(
        "tapeback.transcriber.WhisperModel",
        side_effect=lambda *_a, **k: gpu_model if k["device"] == "cuda" else cpu_model,
    ):
        transcriber = Transcriber(s)
        transcriber.transcribe(Path("/fake/audio.wav"))

    assert transcriber._model is cpu_model
    assert transcriber._model is not gpu_model


def test_non_cuda_runtime_error_still_propagates(settings, clear_gpu):
    """The cleanup must not turn an unrelated failure into a silent CPU fallback."""
    s = settings.model_copy(update={"device": "cuda", "thermal_clamp_wait": 60.0})

    with (
        patch("tapeback.transcriber.WhisperModel", side_effect=RuntimeError("corrupt audio")),
        pytest.raises(RuntimeError, match="corrupt audio"),
    ):
        Transcriber(s)
