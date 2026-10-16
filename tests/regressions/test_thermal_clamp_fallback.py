"""Regression tests for behaviour under a GPU thermal clamp."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tapeback.transcriber import Transcriber


def _info():
    info = MagicMock()
    info.language, info.language_probability, info.duration = "ru", 0.9, 10.0
    return info


@pytest.fixture
def clamped(monkeypatch):
    """Pretend the GPU is thermally clamped and never releases."""
    monkeypatch.setattr("tapeback._device.wait_for_clamp_release", lambda *_a, **_k: False)


@pytest.fixture
def clear_gpu(monkeypatch):
    monkeypatch.setattr("tapeback._device.wait_for_clamp_release", lambda *_a, **_k: True)


def test_clamped_gpu_falls_back_to_cpu(settings, clamped, capsys):
    """A clamped card is ~8x slower than the CPU, so it must not be used.

    Bug: transcription persisted on a GPU whose power budget the controller had cut
    from 50 W to 5 W, pinning it to 300 MHz. Measured on the same clip, CPU ran at
    2.39x real time against 0.31x clamped — this is what turned a fifteen-minute job
    into a multi-hour one that never appeared to finish.
    """
    s = settings.model_copy(
        update={"device": "cuda", "thermal_clamp_check": True, "thermal_clamp_wait": 60.0}
    )

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        transcriber = Transcriber(s)

    assert transcriber.describe() == "Whisper: large-v3-turbo on cpu/int8"
    assert mock_model_cls.call_args_list[0].kwargs["device"] == "cpu"
    assert "thermally clamped" in capsys.readouterr().err


def test_clear_gpu_is_used_normally(settings, clear_gpu):
    s = settings.model_copy(
        update={"device": "cuda", "thermal_clamp_check": True, "thermal_clamp_wait": 60.0}
    )

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        transcriber = Transcriber(s)

    assert transcriber.describe() == "Whisper: large-v3-turbo on cuda/int8_float16"
    assert mock_model_cls.call_args_list[0].kwargs["device"] == "cuda"


def test_fallback_can_be_declined(settings, clamped, capsys):
    """Opting out keeps the GPU, but must say plainly that it will be slow."""
    s = settings.model_copy(
        update={
            "device": "cuda",
            "thermal_clamp_check": True,
            "thermal_clamp_wait": 60.0,
            "thermal_clamp_cpu_fallback": False,
        }
    )

    with patch("tapeback.transcriber.WhisperModel"):
        transcriber = Transcriber(s)

    assert transcriber.describe() == "Whisper: large-v3-turbo on cuda/int8_float16"
    assert "very slow" in capsys.readouterr().err


def test_the_gpu_is_reclaimed_once_the_clamp_clears(settings, monkeypatch):
    """A run stranded on the CPU must return to the GPU when the card frees up.

    The decision is retaken for every stage — each channel runs in its own worker,
    which resolves the device for itself. Without that, one clamp at the wrong moment
    would hold the whole recording on the CPU even after the card recovered.
    """
    clamped = iter([True, False])  # first stage clamped, second clear
    monkeypatch.setattr(
        "tapeback._device.wait_for_clamp_release", lambda *_a, **_k: not next(clamped)
    )
    monkeypatch.setattr("tapeback._device.get_free_vram_mib", lambda: 4096)
    s = settings.model_copy(update={"device": "cuda", "thermal_clamp_check": True})

    with patch("tapeback.transcriber.WhisperModel"):
        first = Transcriber(s).describe()
        second = Transcriber(s).describe()

    assert first == "Whisper: large-v3-turbo on cpu/int8"
    assert second == "Whisper: large-v3-turbo on cuda/int8_float16"


def test_zero_wait_still_checks_the_clamp(settings, monkeypatch):
    """Not waiting is not the same as not looking.

    The default is a zero wait: the clamp clears on system idle and the shortest
    release measured was 451 s, so waiting rarely pays. The check itself is one query
    and is what makes returning to the GPU at the next stage possible.
    """
    waits: list[float] = []
    monkeypatch.setattr(
        "tapeback._device.wait_for_clamp_release",
        lambda timeout, **_k: waits.append(timeout) or False,
    )
    monkeypatch.setattr("tapeback._device.get_free_vram_mib", lambda: 4096)
    s = settings.model_copy(
        update={"device": "cuda", "thermal_clamp_check": True, "thermal_clamp_wait": 0.0}
    )

    with patch("tapeback.transcriber.WhisperModel"):
        assert Transcriber(s).describe() == "Whisper: large-v3-turbo on cpu/int8"

    assert waits == [0.0]


def test_clamp_check_skipped_when_disabled(settings, monkeypatch):
    """thermal_clamp_check=false must not touch the GPU at all."""
    calls: list[object] = []
    monkeypatch.setattr(
        "tapeback._device.wait_for_clamp_release",
        lambda *a, **k: calls.append(a) or True,
    )
    s = settings.model_copy(update={"device": "cuda", "thermal_clamp_check": False})

    with patch("tapeback.transcriber.WhisperModel"):
        Transcriber(s)

    assert calls == []


def test_clamp_check_skipped_on_cpu(settings, monkeypatch):
    calls: list[object] = []
    monkeypatch.setattr(
        "tapeback._device.wait_for_clamp_release",
        lambda *a, **k: calls.append(a) or True,
    )
    s = settings.model_copy(update={"device": "cpu", "thermal_clamp_check": True})

    with patch("tapeback.transcriber.WhisperModel"):
        Transcriber(s)

    assert calls == []


def test_stage_pause_runs_between_channels(settings, clear_gpu):
    """Pacing sheds heat between channels instead of driving into the clamp."""
    s = settings.model_copy(
        update={"device": "cuda", "thermal_clamp_check": True, "stage_pause_seconds": 7.0}
    )
    seg = MagicMock()
    seg.start, seg.end, seg.text, seg.words = 0.0, 5.0, "text", []

    slept: list[float] = []
    messages: list[str] = []
    with (
        patch("tapeback.transcriber.WhisperModel") as mock_model_cls,
        patch("tapeback.transcriber.time.sleep", slept.append),
    ):
        instance = mock_model_cls.return_value
        instance.transcribe.side_effect = [
            (iter([seg]), _info()),
            (iter([seg]), _info()),
        ]
        Transcriber(s).transcribe_stereo(
            Path("/fake/mic.wav"), Path("/fake/monitor.wav"), on_status=messages.append
        )

    assert slept == [7.0]
    assert any("Pausing 7s" in m for m in messages)


def test_no_pause_when_disabled(settings, clear_gpu):
    s = settings.model_copy(
        update={"device": "cuda", "thermal_clamp_check": True, "stage_pause_seconds": 0.0}
    )
    seg = MagicMock()
    seg.start, seg.end, seg.text, seg.words = 0.0, 5.0, "text", []

    slept: list[float] = []
    with (
        patch("tapeback.transcriber.WhisperModel") as mock_model_cls,
        patch("tapeback.transcriber.time.sleep", slept.append),
    ):
        instance = mock_model_cls.return_value
        instance.transcribe.side_effect = [
            (iter([seg]), _info()),
            (iter([seg]), _info()),
        ]
        Transcriber(s).transcribe_stereo(Path("/fake/mic.wav"), Path("/fake/monitor.wav"))

    assert slept == []
