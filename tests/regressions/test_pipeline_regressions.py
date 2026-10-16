"""Regression tests for pipeline bugs."""

import shutil
from unittest.mock import patch

import pytest

from tapeback.pipeline import process_stereo_file
from tapeback.settings import Settings
from tests.fixtures import create_stereo_wav_segments, mock_whisper_transcribe


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_stereo_times_every_heavy_stage(tmp_path):
    """Every heavy stage must report its own timing.

    Bug: load_stereo_channels (reads the whole 48 kHz stereo file into two float32
    arrays) and gate_wav_inactive (a per-100ms Python loop over the mic channel)
    ran outside any stage_timer, so their cost showed up as unexplained dead time
    between "Stage 'merge'" and "Stage 'load model'" and could not be attributed.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(vault_path=vault, gate_mic_silence=True)

    stereo = tmp_path / "stereo.wav"
    create_stereo_wav_segments(stereo, 48000, [(1.0, 0.8, 0.003), (1.0, 0.003, 0.8)])

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_model = mock_whisper_transcribe([(0.0, 1.0, "Speech.")])

    messages: list[str] = []
    with patch("tapeback.transcriber.WhisperModel", return_value=mock_model):
        process_stereo_file(stereo, output_dir, settings, diarize=False, on_status=messages.append)

    timed = {m.split("'")[1] for m in messages if m.startswith("Stage '")}
    assert {"load channels", "split", "gate mic", "load model"} <= timed


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_stereo_reports_resolved_device(tmp_path):
    """The pipeline must state where Whisper landed.

    Bug: a run that silently fell back to CPU (roughly 10x slower) produced the
    same output as a healthy GPU run — the only signal was a stderr warning that
    scrolled past between other status lines.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(vault_path=vault, device="cpu", compute_type="int8")

    stereo = tmp_path / "stereo.wav"
    create_stereo_wav_segments(stereo, 48000, [(1.0, 0.8, 0.003)])

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_model = mock_whisper_transcribe([(0.0, 1.0, "Speech.")])

    messages: list[str] = []
    with patch("tapeback.transcriber.WhisperModel", return_value=mock_model):
        process_stereo_file(stereo, output_dir, settings, diarize=False, on_status=messages.append)

    assert any(m == "Whisper: large-v3-turbo on cpu/int8" for m in messages)
