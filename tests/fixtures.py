import json
import wave
from pathlib import Path

import numpy as np
import pytest

from meetrec.recorder import Recorder
from meetrec.settings import Settings

# --- pytest fixtures ---


@pytest.fixture
def tmp_vault(tmp_path):
    """Temporary Obsidian vault for tests."""
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


@pytest.fixture
def settings(tmp_vault):
    """Settings with temporary vault."""
    return Settings(vault_path=tmp_vault)


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


# --- WAV file helpers ---


def create_silent_wav(path: Path, duration: float = 1.0, sample_rate: int = 48000) -> None:
    """Create a silent (zero-amplitude) WAV file for testing."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        n_frames = int(duration * sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)


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


def create_session_file(session_file, **overrides):
    """Write a session.json file with sensible defaults, overridden by kwargs."""
    data = {
        "pid_monitor": 99998,
        "pid_mic": 99999,
        "session_name": "test_session",
        "monitor_path": "/tmp/meetrec/test_session/monitor.wav",
        "mic_path": "/tmp/meetrec/test_session/mic.wav",
        "started_at": "2026-03-17T14:30:00",
        **overrides,
    }
    session_file.write_text(json.dumps(data))
    return data


@pytest.fixture
def stereo_wav(tmp_path):
    """Factory fixture: returns callable(segments_spec) -> Path to stereo WAV."""

    def _create(segments_spec, sample_rate=16000):
        path = tmp_path / "stereo.wav"
        create_stereo_wav_segments(path, sample_rate, segments_spec)
        return path

    return _create
