import json
import os
import wave
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from click.testing import CliRunner

from tapeback.recorder import Recorder
from tapeback.settings import Settings


def _pyannote_available() -> bool:
    try:
        import pyannote.audio  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


requires_pyannote = pytest.mark.skipif(
    not _pyannote_available(),
    reason="pyannote-audio not installed (install tapeback[diarize])",
)


def _pystray_available() -> bool:
    try:
        import pystray  # noqa: F401, PLC0415

        return True
    except Exception:
        # ImportError when not installed; Xlib.error.DisplayNameError on headless CI
        return False


requires_pystray = pytest.mark.skipif(
    not _pystray_available(),
    reason="pystray not installed (install tapeback[tray])",
)

# --- pytest fixtures ---


@pytest.fixture
def tmp_vault(tmp_path):
    """Temporary Obsidian vault for tests."""
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


@pytest.fixture
def vault_env(tmp_vault, monkeypatch):
    """tmp_vault + TAPEBACK_VAULT_PATH env var — for CLI tests that call get_settings()."""
    monkeypatch.setenv("TAPEBACK_VAULT_PATH", str(tmp_vault))
    return tmp_vault


@pytest.fixture
def settings(tmp_vault):
    """Settings with temporary vault."""
    return Settings(vault_path=tmp_vault)


@pytest.fixture
def session_wavs(tmp_path):
    """Factory: create session_dir with mic.wav + monitor.wav sine WAVs.

    Returns callable(session_name, duration=2.0) -> (session_dir, monitor_wav, mic_wav).
    """

    def _create(session_name: str, duration: float = 2.0) -> tuple[Path, Path, Path]:
        session_dir = tmp_path / session_name
        session_dir.mkdir()
        monitor_wav = session_dir / "monitor.wav"
        mic_wav = session_dir / "mic.wav"
        create_mono_wav(monitor_wav, duration=duration, sample_rate=48000, amplitude=0.5)
        create_mono_wav(mic_wav, duration=duration, sample_rate=48000, amplitude=0.5)
        return session_dir, monitor_wav, mic_wav

    return _create


@pytest.fixture
def recorder(tmp_path):
    """Recorder with temporary state directory."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    return Recorder(state_dir=state_dir)


@pytest.fixture
def session_file(recorder):
    """Path to session file."""
    return recorder.session_file


@pytest.fixture
def runner():
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def stereo_wav(tmp_path):
    """Factory fixture: returns callable(segments_spec) -> Path to stereo WAV."""

    def _create(segments_spec, sample_rate=16000):
        path = tmp_path / "stereo.wav"
        create_stereo_wav_segments(path, sample_rate, segments_spec)
        return path

    return _create


@pytest.fixture
def e2e_settings(tmp_path):
    """Settings for e2e tests with HF token from environment."""
    vault = tmp_path / "vault"
    vault.mkdir()
    return Settings(
        vault_path=vault,
        hf_token=os.environ.get("TAPEBACK_HF_TOKEN", os.environ.get("HF_TOKEN", "")),
    )


@pytest.fixture
def summarize_settings(tmp_vault):
    """Settings pre-configured for summarizer tests."""
    return Settings(
        vault_path=tmp_vault,
        llm_provider="anthropic",
        llm_api_key="sk-ant-test-key",
    )


@pytest.fixture
def tray_app(settings):
    """TrayApp with mocked pystray icon and recorder for testing state transitions."""
    from tapeback.tray import TrayApp  # noqa: PLC0415 — optional dep

    app = TrayApp(settings)
    app._icon = MagicMock()
    app._recorder = MagicMock()
    app._recorder.is_recording.return_value = False
    app._recorder.get_session_info.return_value = None
    return app


@pytest.fixture
def e2e_output_dir(tmp_path):
    """Temporary directory for intermediate pipeline files."""
    d = tmp_path / "e2e_output"
    d.mkdir()
    return d


# --- WAV file helpers ---


def create_silent_wav(path: Path, duration: float = 1.0, sample_rate: int = 48000) -> None:
    """Create a silent (zero-amplitude) WAV file for testing."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        n_frames = int(duration * sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)


def create_mono_wav(
    path: Path, duration: float = 1.0, sample_rate: int = 48000, amplitude: float = 0.5
) -> None:
    """Create a mono WAV with a 440Hz sine wave at given amplitude."""
    n_frames = int(duration * sample_rate)
    t = np.linspace(0, duration, n_frames, dtype=np.float32)
    samples = (amplitude * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def create_stereo_wav(path, duration, sample_rate, left_amplitude, right_amplitude):
    """Create a stereo WAV with sine wave on each channel at given amplitudes."""
    n_frames = int(duration * sample_rate)
    t = np.linspace(0, duration, n_frames, dtype=np.float32)

    left = (left_amplitude * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    right = (right_amplitude * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    stereo = np.empty(n_frames * 2, dtype=np.int16)
    stereo[0::2] = left
    stereo[1::2] = right

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())


def create_stereo_wav_segments(path, sample_rate, segments_spec):
    """Create a stereo WAV with different amplitudes per time segment.

    segments_spec: list of (duration, left_amp, right_amp)
    """
    all_left = []
    all_right = []

    for duration, left_amp, right_amp in segments_spec:
        n_frames = int(duration * sample_rate)
        t = np.linspace(0, duration, n_frames, dtype=np.float32)
        all_left.append((left_amp * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16))
        all_right.append((right_amp * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16))

    left = np.concatenate(all_left)
    right = np.concatenate(all_right)

    stereo = np.empty(len(left) * 2, dtype=np.int16)
    stereo[0::2] = left
    stereo[1::2] = right

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())


# --- Test data constants ---


SAMPLE_MD = """\
---
date: 2026-03-20
time: "14:30"
duration: "00:10:00"
language: en
---

# Meeting 2026-03-20 14:30

[00:00:01] **You:** Hello there.
[00:00:05] **Speaker 1:** Hi, let's discuss the plan.
"""

SAMPLE_MD_WITH_SUMMARY = """\
---
date: 2026-03-20
time: "14:30"
duration: "00:10:00"
language: en
---

## Summary

Old summary here.

---

# Meeting 2026-03-20 14:30

[00:00:01] **You:** Hello there.
"""

SAMPLE_TRANSCRIPT_MD = """\
---
date: 2026-03-20
time: "14:30"
duration: "00:10:00"
language: en
tags:
  - meeting
  - transcript
audio: "[[attachments/audio/2026-03-20_14-30-00.wav]]"
---

# Meeting 2026-03-20 14:30

[00:00:01] **You:** Hello there.
[00:00:05] **Speaker 1:** Hi, let's discuss the plan.
"""

VALID_LLM_RESPONSE = json.dumps(
    {
        "brief": "Discussed the project plan and assigned tasks.",
        "action_items": [
            {"assignee": "You", "action": "Send the report", "deadline": "Friday"},
            {"assignee": "Speaker 1", "action": "Review the code", "deadline": None},
        ],
        "key_decisions": ["Use PostgreSQL instead of MongoDB"],
        "is_trivial": False,
    }
)

VALID_LLM_RESPONSE_MINIMAL = json.dumps(
    {
        "brief": "Test summary.",
        "action_items": [],
        "key_decisions": [],
        "is_trivial": False,
    }
)


class HttpError(Exception):
    """Exception with status_code attribute, mimicking SDK exceptions (anthropic, openai)."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


# --- Test helpers ---


def clear_all_provider_env_vars(monkeypatch) -> None:
    """Remove all provider-specific API key env vars."""
    from tapeback.summarizer import _PROVIDER_ENV_VARS  # noqa: PLC0415

    for env_var in _PROVIDER_ENV_VARS.values():
        monkeypatch.delenv(env_var, raising=False)


def mock_anthropic_response(text: str) -> MagicMock:
    """Create a mock anthropic Messages.create response."""
    response = MagicMock()
    response.content = [MagicMock(text=text)]
    return response


def voice_signal(duration: float, sr: int, fundamental: float) -> np.ndarray:
    """Generate a synthetic voice signal with harmonics."""
    t = np.linspace(0, duration, int(duration * sr), dtype=np.float32)
    return (
        np.sin(2 * np.pi * fundamental * t)
        + 0.5 * np.sin(2 * np.pi * fundamental * 2 * t)
        + 0.25 * np.sin(2 * np.pi * fundamental * 3 * t)
    ) * 10000


# --- Session file helpers ---


def create_session_file(session_file, **overrides):
    """Write a session.json file with sensible defaults, overridden by kwargs."""
    data = {
        "pid_monitor": 99998,
        "pid_mic": 99999,
        "session_name": "test_session",
        "monitor_path": "/tmp/tapeback/test_session/monitor.wav",
        "mic_path": "/tmp/tapeback/test_session/mic.wav",
        "started_at": "2026-03-17T14:30:00",
        **overrides,
    }
    session_file.write_text(json.dumps(data))
    return data


# --- Mock helpers ---


def mock_whisper_transcribe(segments_data):
    """Create a mock WhisperModel that returns given segments.

    segments_data: list of (start, end, text) tuples
    """
    mock_model = MagicMock()

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_info.duration = 10.0

    mock_segments = []
    for start, end, text in segments_data:
        seg = MagicMock()
        seg.start = start
        seg.end = end
        seg.text = f" {text} "
        seg.words = []
        mock_segments.append(seg)

    # Each call to transcribe() returns a fresh iterator
    mock_model.transcribe.side_effect = lambda *a, **kw: (iter(list(mock_segments)), mock_info)

    return mock_model


def mock_pyannote_annotation(tracks):
    """Create a mock pyannote annotation from (start, end, speaker) tuples."""
    mock_annotation = MagicMock()
    itertracks_result = []
    for start, end, speaker in tracks:
        turn = MagicMock()
        turn.start = start
        turn.end = end
        itertracks_result.append((turn, None, speaker))
    mock_annotation.itertracks.return_value = itertracks_result
    return mock_annotation
