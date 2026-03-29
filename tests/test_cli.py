"""CLI integration tests — invoke click commands via CliRunner.

Mock only ML models (WhisperModel, pyannote Pipeline). Everything else
(ffmpeg, file I/O, formatting) runs for real.
"""

import json
import shutil
from unittest.mock import MagicMock, patch

import pytest

from tapeback.cli import cli
from tapeback.pipeline import _maybe_diarize_segments, process_stereo_file, stop_and_process
from tapeback.models import Segment
from tapeback.settings import Settings
from tapeback.summarizer import _PROVIDER_ENV_VARS
from tests.fixtures import (
    create_mono_wav,
    create_silent_wav,
    create_stereo_wav_segments,
    mock_pyannote_annotation,
    mock_whisper_transcribe,
)

# --- process command ---


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_mono_pipeline(runner, tmp_path, monkeypatch):
    """process command: mono WAV → transcribe → save markdown + audio to vault.
    Also tests --name for custom output filename."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("TAPEBACK_VAULT_PATH", str(vault))
    monkeypatch.setenv("TAPEBACK_DIARIZE", "false")

    audio = tmp_path / "2026-03-20_10-00-00.wav"
    create_silent_wav(audio, duration=2.0, sample_rate=48000)

    mock_model = mock_whisper_transcribe(
        [
            (0.0, 5.0, "Hello from the meeting."),
            (5.0, 10.0, "Second sentence here."),
        ]
    )

    with patch("tapeback.transcriber.WhisperModel", return_value=mock_model):
        result = runner.invoke(cli, ["process", str(audio), "--no-diarize"])

    assert result.exit_code == 0, result.output + str(result.exception or "")

    # Markdown saved to vault with correct content
    md_path = vault / "meetings" / "2026-03-20_10-00-00.md"
    assert md_path.exists()
    md_content = md_path.read_text()
    assert "Hello from the meeting." in md_content
    assert "Second sentence here." in md_content
    assert "date: 2026-03-20" in md_content

    # Audio copied to vault
    assert (vault / "attachments" / "audio" / "2026-03-20_10-00-00.wav").exists()

    # Test --name option
    audio2 = tmp_path / "recording.wav"
    create_silent_wav(audio2, duration=2.0)
    mock_model2 = mock_whisper_transcribe([(0.0, 5.0, "Speech.")])

    with patch("tapeback.transcriber.WhisperModel", return_value=mock_model2):
        result2 = runner.invoke(
            cli, ["process", str(audio2), "--name", "my-meeting", "--no-diarize"]
        )

    assert result2.exit_code == 0
    assert (vault / "meetings" / "my-meeting.md").exists()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_with_diarization(runner, tmp_path, monkeypatch):
    """process command with diarization: transcribe → diarize → speakers in markdown."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("TAPEBACK_VAULT_PATH", str(vault))
    monkeypatch.setenv("TAPEBACK_HF_TOKEN", "hf_fake")

    audio = tmp_path / "2026-03-20_10-00-00.wav"
    create_silent_wav(audio, duration=2.0)

    mock_model = mock_whisper_transcribe(
        [
            (0.0, 5.0, "First speaker."),
            (5.0, 10.0, "Second speaker."),
        ]
    )
    annotation = mock_pyannote_annotation(
        [
            (0.0, 5.0, "SPEAKER_00"),
            (5.0, 10.0, "SPEAKER_01"),
        ]
    )

    with (
        patch("tapeback.transcriber.WhisperModel", return_value=mock_model),
        patch("pyannote.audio.Pipeline") as mock_pipeline_cls,
    ):
        mock_pipeline = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_pipeline.return_value = annotation

        result = runner.invoke(cli, ["process", str(audio)])

    assert result.exit_code == 0, result.output
    md_content = (vault / "meetings" / "2026-03-20_10-00-00.md").read_text()
    assert "Speaker 1" in md_content
    assert "Speaker 2" in md_content


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_stereo_dual_channel(runner, tmp_path, monkeypatch):
    """process command with stereo WAV: auto-detects stereo, uses dual-channel pipeline.

    Dual-channel pipeline: split channels → transcribe each → mic gets "You" label.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("TAPEBACK_VAULT_PATH", str(vault))
    monkeypatch.setenv("TAPEBACK_DIARIZE", "false")

    audio = tmp_path / "2026-03-20_10-00-00.wav"
    create_stereo_wav_segments(audio, 48000, [(1.0, 0.8, 0.003), (1.0, 0.003, 0.8)])

    mock_model = mock_whisper_transcribe(
        [
            (0.0, 1.0, "Channel speech."),
        ]
    )

    with patch("tapeback.transcriber.WhisperModel", return_value=mock_model):
        result = runner.invoke(cli, ["process", str(audio), "--no-diarize"])

    assert result.exit_code == 0, result.output + str(result.exception or "")

    # Dual-channel pipeline transcribes each channel separately (2 calls)
    assert mock_model.transcribe.call_count == 2

    md_content = (vault / "meetings" / "2026-03-20_10-00-00.md").read_text()
    # Mic channel gets "You" label automatically in stereo pipeline
    assert "**You:**" in md_content
    assert "Stereo file detected" in result.output


# --- status command ---


def test_status_command(runner, tmp_path, monkeypatch):
    """status command: shows 'Not recording' or session info."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("TAPEBACK_VAULT_PATH", str(vault))

    with patch("tapeback.cli.get_settings") as mock_settings:
        mock_settings.return_value = Settings(vault_path=vault)

        # Not recording
        with patch("tapeback.recorder.Recorder.get_session_info", return_value=None):
            result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "Not recording." in result.output
        assert str(vault) in result.output

        # While recording
        session_info = {
            "session_name": "2026-03-20_10-00-00",
            "started_at": "2026-03-20T10:00:00",
            "pid_monitor": 12345,
            "pid_mic": 12346,
            "monitor_path": "/tmp/tapeback/test/monitor.wav",
            "mic_path": "/tmp/tapeback/test/mic.wav",
        }
        with patch("tapeback.recorder.Recorder.get_session_info", return_value=session_info):
            result = runner.invoke(cli, ["status"])
        assert "Recording in progress: 2026-03-20_10-00-00" in result.output


# --- _stop_and_process (dual-channel pipeline) ---


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_stop_and_process_pipeline(tmp_path):
    """_stop_and_process: full dual-channel pipeline with mocked ML models."""
    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(vault_path=vault, hf_token="hf_fake")

    session_dir = tmp_path / "2026-03-20_10-00-00"
    session_dir.mkdir()
    monitor_wav = session_dir / "monitor.wav"
    mic_wav = session_dir / "mic.wav"
    create_mono_wav(monitor_wav, duration=2.0, sample_rate=48000, amplitude=0.5)
    create_mono_wav(mic_wav, duration=2.0, sample_rate=48000, amplitude=0.5)

    mock_recorder = MagicMock()
    mock_recorder.stop.return_value = (monitor_wav, mic_wav)

    mock_model = mock_whisper_transcribe([(0.0, 1.0, "Hello."), (1.0, 2.0, "World.")])
    annotation = mock_pyannote_annotation([(0.0, 2.0, "SPEAKER_00")])

    with (
        patch("tapeback.transcriber.WhisperModel", return_value=mock_model),
        patch("pyannote.audio.Pipeline") as mock_pipeline_cls,
    ):
        mock_pipeline = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_pipeline.return_value = annotation
        stop_and_process(mock_recorder, settings, diarize=True)

    md_path = vault / "meetings" / "2026-03-20_10-00-00.md"
    assert md_path.exists()
    md_content = md_path.read_text()
    assert "Hello." in md_content
    assert "World." in md_content
    assert (vault / "attachments" / "audio" / "2026-03-20_10-00-00.wav").exists()
    assert not session_dir.exists()  # temp files cleaned up


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_stop_and_process_no_diarize(tmp_path):
    """_stop_and_process with diarize=False skips pyannote entirely."""
    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(vault_path=vault)

    session_dir = tmp_path / "2026-03-20_11-00-00"
    session_dir.mkdir()
    monitor_wav = session_dir / "monitor.wav"
    mic_wav = session_dir / "mic.wav"
    create_mono_wav(monitor_wav, duration=2.0, sample_rate=48000, amplitude=0.5)
    create_mono_wav(mic_wav, duration=2.0, sample_rate=48000, amplitude=0.5)

    mock_recorder = MagicMock()
    mock_recorder.stop.return_value = (monitor_wav, mic_wav)
    mock_model = mock_whisper_transcribe([(0.0, 5.0, "No diarize.")])

    with (
        patch("tapeback.transcriber.WhisperModel", return_value=mock_model),
        patch("pyannote.audio.Pipeline") as mock_pipeline_cls,
    ):
        stop_and_process(mock_recorder, settings, diarize=False)
        mock_pipeline_cls.from_pretrained.assert_not_called()

    assert (vault / "meetings" / "2026-03-20_11-00-00.md").exists()


# --- _maybe_diarize ---


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_stereo_file_function(tmp_path):
    """_process_stereo_file: splits channels, transcribes each, merges."""
    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(vault_path=vault)

    stereo = tmp_path / "stereo.wav"
    create_stereo_wav_segments(stereo, 48000, [(1.0, 0.8, 0.003), (1.0, 0.003, 0.8)])

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_model = mock_whisper_transcribe([(0.0, 1.0, "Speech.")])

    with patch("tapeback.transcriber.WhisperModel", return_value=mock_model):
        segments, _info = process_stereo_file(stereo, output_dir, settings, diarize=False)

    assert len(segments) > 0
    # Mic segments get "You" from transcribe_stereo
    you_segments = [s for s in segments if s.speaker == "You"]
    assert len(you_segments) > 0
    assert mock_model.transcribe.call_count == 2


def test_maybe_diarize_skips_and_warns(runner, tmp_path):
    """_maybe_diarize returns segments unchanged when disabled or token missing."""
    vault = tmp_path / "vault"
    vault.mkdir()

    segments = [Segment(start=0.0, end=5.0, text="Hello")]

    # Disabled
    settings_off = Settings(vault_path=vault)
    result = _maybe_diarize_segments(segments, settings_off, tmp_path / "a.wav", None, diarize=False)
    assert result is segments

    # No token
    settings_no_token = Settings(vault_path=vault, hf_token="", diarize=True)
    result = _maybe_diarize_segments(segments, settings_no_token, tmp_path / "a.wav", None, diarize=True)
    assert result is segments


# --- summarize command ---


VALID_LLM_RESPONSE = json.dumps(
    {
        "brief": "Test summary.",
        "action_items": [],
        "key_decisions": [],
        "is_trivial": False,
    }
)

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


def test_summarize_command_rewrites_file(runner, tmp_path, monkeypatch):
    """summarize command: mock LLM → file gets summary section."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("TAPEBACK_VAULT_PATH", str(vault))
    monkeypatch.setenv("TAPEBACK_LLM_API_KEY", "sk-test")

    md_file = tmp_path / "transcript.md"
    md_file.write_text(SAMPLE_TRANSCRIPT_MD)

    with patch("tapeback.summarizer._call_llm", return_value=VALID_LLM_RESPONSE):
        result = runner.invoke(cli, ["summarize", str(md_file)])

    assert result.exit_code == 0, result.output
    content = md_file.read_text()
    assert "## Summary" in content
    assert "Test summary." in content
    assert "# Meeting 2026-03-20 14:30" in content


def test_summarize_command_no_api_key(runner, tmp_path, monkeypatch):
    """No API key → error, file unchanged."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("TAPEBACK_VAULT_PATH", str(vault))
    monkeypatch.delenv("TAPEBACK_LLM_API_KEY", raising=False)
    for env_var in _PROVIDER_ENV_VARS.values():
        monkeypatch.delenv(env_var, raising=False)

    md_file = tmp_path / "transcript.md"
    md_file.write_text(SAMPLE_TRANSCRIPT_MD)

    with patch("tapeback.summarizer._build_provider_chain", return_value=[]):
        result = runner.invoke(cli, ["summarize", str(md_file)])

    assert result.exit_code != 0
    assert md_file.read_text() == SAMPLE_TRANSCRIPT_MD


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_with_summarization(runner, tmp_path, monkeypatch):
    """Full pipeline: process → transcribe → summarize → file has summary."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("TAPEBACK_VAULT_PATH", str(vault))
    monkeypatch.setenv("TAPEBACK_DIARIZE", "false")
    monkeypatch.setenv("TAPEBACK_LLM_API_KEY", "sk-test")

    audio = tmp_path / "2026-03-20_10-00-00.wav"
    create_silent_wav(audio, duration=2.0, sample_rate=48000)

    mock_model = mock_whisper_transcribe([(0.0, 5.0, "Hello from the meeting.")])

    with (
        patch("tapeback.transcriber.WhisperModel", return_value=mock_model),
        patch("tapeback.summarizer._call_llm", return_value=VALID_LLM_RESPONSE),
    ):
        result = runner.invoke(cli, ["process", str(audio), "--no-diarize"])

    assert result.exit_code == 0, result.output
    md_content = (vault / "meetings" / "2026-03-20_10-00-00.md").read_text()
    assert "## Summary" in md_content
    assert "Test summary." in md_content
    assert "Hello from the meeting." in md_content


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_no_summarize_flag(runner, tmp_path, monkeypatch):
    """--no-summarize → no LLM call."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("TAPEBACK_VAULT_PATH", str(vault))
    monkeypatch.setenv("TAPEBACK_DIARIZE", "false")

    audio = tmp_path / "2026-03-20_10-00-00.wav"
    create_silent_wav(audio, duration=2.0, sample_rate=48000)
    mock_model = mock_whisper_transcribe([(0.0, 5.0, "Speech.")])

    with (
        patch("tapeback.transcriber.WhisperModel", return_value=mock_model),
        patch("tapeback.summarizer._call_llm") as mock_llm,
    ):
        result = runner.invoke(cli, ["process", str(audio), "--no-diarize", "--no-summarize"])

    assert result.exit_code == 0
    mock_llm.assert_not_called()
    md_content = (vault / "meetings" / "2026-03-20_10-00-00.md").read_text()
    assert "## Summary" not in md_content


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_stop_and_process_summarization_failure(tmp_path):
    """LLM fails → warning printed, transcript still saved."""
    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(vault_path=vault, llm_api_key="sk-test")

    session_dir = tmp_path / "2026-03-20_12-00-00"
    session_dir.mkdir()
    monitor_wav = session_dir / "monitor.wav"
    mic_wav = session_dir / "mic.wav"
    create_mono_wav(monitor_wav, duration=2.0, sample_rate=48000, amplitude=0.5)
    create_mono_wav(mic_wav, duration=2.0, sample_rate=48000, amplitude=0.5)

    mock_recorder = MagicMock()
    mock_recorder.stop.return_value = (monitor_wav, mic_wav)
    mock_model = mock_whisper_transcribe([(0.0, 5.0, "Important content.")])

    with (
        patch("tapeback.transcriber.WhisperModel", return_value=mock_model),
        patch("pyannote.audio.Pipeline"),
        patch("tapeback.summarizer._call_llm", side_effect=RuntimeError("API error")),
    ):
        stop_and_process(mock_recorder, settings, diarize=False, do_summarize=True)

    # Transcript still saved despite summarization failure
    md_path = vault / "meetings" / "2026-03-20_12-00-00.md"
    assert md_path.exists()
    assert "Important content." in md_path.read_text()
    assert "## Summary" not in md_path.read_text()
