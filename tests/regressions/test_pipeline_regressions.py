"""Regression tests for pipeline bugs."""

import shutil
from unittest.mock import patch

import pytest

from tapeback.pipeline import process_mono_file, process_stereo_file
from tapeback.settings import Settings
from tests.fixtures import (
    create_mono_wav,
    create_stereo_wav_segments,
    mock_whisper_transcribe,
)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_stereo_no_diarize_returns_no_raw_segments(tmp_path):
    """Without diarization, raw_segments must be None to avoid duplicate markdown sections.

    Bug: process_stereo_file always returned raw_segments, causing format_markdown
    to render two identical "## Transcript" + "## Diarized Transcript" sections
    when --no-diarize was used.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(vault_path=vault)

    stereo = tmp_path / "stereo.wav"
    create_stereo_wav_segments(stereo, 48000, [(1.0, 0.8, 0.003), (1.0, 0.003, 0.8)])

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_model = mock_whisper_transcribe([(0.0, 1.0, "Speech.")])

    with patch("tapeback.transcriber.WhisperModel", return_value=mock_model):
        _segments, _info, raw_segments = process_stereo_file(
            stereo, output_dir, settings, diarize=False
        )

    assert raw_segments is None


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_stereo_diarize_without_hf_token_returns_no_raw_segments(tmp_path):
    """When diarize=True but no HF token is configured, diarization is skipped —
    so raw_segments must be None (no duplicate section)."""
    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(vault_path=vault, hf_token="")

    stereo = tmp_path / "stereo.wav"
    create_stereo_wav_segments(stereo, 48000, [(1.0, 0.8, 0.003), (1.0, 0.003, 0.8)])

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_model = mock_whisper_transcribe([(0.0, 1.0, "Speech.")])

    with patch("tapeback.transcriber.WhisperModel", return_value=mock_model):
        _segments, _info, raw_segments = process_stereo_file(
            stereo, output_dir, settings, diarize=True
        )

    assert raw_segments is None


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_mono_no_diarize_returns_no_raw_segments(tmp_path):
    """Mono pipeline must also return None raw_segments when diarization is off."""
    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(vault_path=vault)

    mono = tmp_path / "mono.wav"
    create_mono_wav(mono, duration=1.0, sample_rate=48000, amplitude=0.5)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_model = mock_whisper_transcribe([(0.0, 1.0, "Speech.")])

    with patch("tapeback.transcriber.WhisperModel", return_value=mock_model):
        _segments, _info, raw_segments = process_mono_file(
            mono, output_dir, settings, diarize=False
        )

    assert raw_segments is None
