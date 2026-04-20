"""CLI integration tests — invoke click commands via CliRunner.

Mock only ML models (WhisperModel, pyannote Pipeline). Everything else
(ffmpeg, file I/O, formatting) runs for real.
"""

import shutil
from unittest.mock import MagicMock, patch

import pytest

from tapeback.cli import cli
from tapeback.models import Segment
from tapeback.pipeline import _maybe_diarize_segments, process_stereo_file, stop_and_process
from tapeback.settings import Settings
from tapeback.summarizer import _PROVIDER_ENV_VARS
from tests.fixtures import (
    SAMPLE_TRANSCRIPT_MD,
    VALID_LLM_RESPONSE_MINIMAL,
    create_silent_wav,
    create_stereo_wav_segments,
    mock_pyannote_annotation,
    mock_whisper_transcribe,
)

# --- process command ---


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_mono_pipeline(runner, tmp_path, monkeypatch, vault_env):
    """process command: mono WAV → transcribe → save markdown + audio to vault.
    Also tests --name for custom output filename."""
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
    md_path = vault_env / "meetings" / "2026-03-20_10-00-00.md"
    assert md_path.exists()
    md_content = md_path.read_text()
    assert "Hello from the meeting." in md_content
    assert "Second sentence here." in md_content
    assert "date: 2026-03-20" in md_content

    # Audio copied to vault
    assert (vault_env / "attachments" / "audio" / "2026-03-20_10-00-00.wav").exists()

    # Test --name option
    audio2 = tmp_path / "recording.wav"
    create_silent_wav(audio2, duration=2.0)
    mock_model2 = mock_whisper_transcribe([(0.0, 5.0, "Speech.")])

    with patch("tapeback.transcriber.WhisperModel", return_value=mock_model2):
        result2 = runner.invoke(
            cli, ["process", str(audio2), "--name", "my-meeting", "--no-diarize"]
        )

    assert result2.exit_code == 0
    assert (vault_env / "meetings" / "my-meeting.md").exists()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_with_diarization(runner, tmp_path, monkeypatch, vault_env):
    """process command with diarization: transcribe → diarize → speakers in markdown."""
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
    md_content = (vault_env / "meetings" / "2026-03-20_10-00-00.md").read_text()
    assert "Speaker 1" in md_content
    assert "Speaker 2" in md_content


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_stereo_dual_channel(runner, tmp_path, monkeypatch, vault_env):
    """process command with stereo WAV: auto-detects stereo, uses dual-channel pipeline.

    Dual-channel pipeline: split channels → transcribe each → mic gets "You" label.
    """
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

    md_content = (vault_env / "meetings" / "2026-03-20_10-00-00.md").read_text()
    # Mic channel gets "You" label automatically in stereo pipeline
    assert "**You:**" in md_content
    assert "Stereo file detected" in result.output


# --- status command ---


def test_status_command(runner, vault_env):
    """status command: shows 'Not recording' or session info."""
    with patch("tapeback.cli.get_settings") as mock_settings:
        mock_settings.return_value = Settings(vault_path=vault_env)

        # Not recording
        with patch("tapeback.recorder.Recorder.get_session_info", return_value=None):
            result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "Not recording." in result.output
        assert str(vault_env) in result.output

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
def test_stop_and_process_pipeline(tmp_vault, session_wavs):
    """_stop_and_process: full dual-channel pipeline with mocked ML models."""
    settings = Settings(vault_path=tmp_vault, hf_token="hf_fake")
    session_dir, monitor_wav, mic_wav = session_wavs("2026-03-20_10-00-00")

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

    md_path = tmp_vault / "meetings" / "2026-03-20_10-00-00.md"
    assert md_path.exists()
    md_content = md_path.read_text()
    assert "Hello." in md_content
    assert "World." in md_content
    assert (tmp_vault / "attachments" / "audio" / "2026-03-20_10-00-00.wav").exists()
    assert not session_dir.exists()  # temp files cleaned up


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_stop_and_process_no_diarize(tmp_vault, session_wavs):
    """_stop_and_process with diarize=False skips pyannote entirely."""
    settings = Settings(vault_path=tmp_vault)
    _session_dir, monitor_wav, mic_wav = session_wavs("2026-03-20_11-00-00")

    mock_recorder = MagicMock()
    mock_recorder.stop.return_value = (monitor_wav, mic_wav)
    mock_model = mock_whisper_transcribe([(0.0, 5.0, "No diarize.")])

    with (
        patch("tapeback.transcriber.WhisperModel", return_value=mock_model),
        patch("pyannote.audio.Pipeline") as mock_pipeline_cls,
    ):
        stop_and_process(mock_recorder, settings, diarize=False)
        mock_pipeline_cls.from_pretrained.assert_not_called()

    assert (tmp_vault / "meetings" / "2026-03-20_11-00-00.md").exists()


# --- _maybe_diarize ---


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_stereo_file_function(tmp_path, tmp_vault):
    """_process_stereo_file: splits channels, transcribes each, merges."""
    settings = Settings(vault_path=tmp_vault)

    stereo = tmp_path / "stereo.wav"
    create_stereo_wav_segments(stereo, 48000, [(1.0, 0.8, 0.003), (1.0, 0.003, 0.8)])

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_model = mock_whisper_transcribe([(0.0, 1.0, "Speech.")])

    with patch("tapeback.transcriber.WhisperModel", return_value=mock_model):
        segments, _info, _raw = process_stereo_file(stereo, output_dir, settings, diarize=False)

    assert len(segments) > 0
    # Mic segments get "You" from transcribe_stereo
    you_segments = [s for s in segments if s.speaker == "You"]
    assert len(you_segments) > 0
    assert mock_model.transcribe.call_count == 2


def test_maybe_diarize_skips_and_warns(tmp_path, tmp_vault):
    """_maybe_diarize returns segments unchanged when disabled or token missing."""
    segments = [Segment(start=0.0, end=5.0, text="Hello")]

    # Disabled
    settings_off = Settings(vault_path=tmp_vault)
    result = _maybe_diarize_segments(
        segments, settings_off, tmp_path / "a.wav", None, diarize=False
    )
    assert result is segments

    # No token
    settings_no_token = Settings(vault_path=tmp_vault, hf_token="", diarize=True)
    result = _maybe_diarize_segments(
        segments, settings_no_token, tmp_path / "a.wav", None, diarize=True
    )
    assert result is segments


# --- summarize command ---


def test_summarize_command_rewrites_file(runner, tmp_path, monkeypatch, vault_env):
    """summarize command: mock LLM → file gets summary section."""
    monkeypatch.setenv("TAPEBACK_LLM_API_KEY", "sk-test")

    md_file = tmp_path / "transcript.md"
    md_file.write_text(SAMPLE_TRANSCRIPT_MD)

    with patch("tapeback.summarizer._call_llm", return_value=VALID_LLM_RESPONSE_MINIMAL):
        result = runner.invoke(cli, ["summarize", str(md_file)])

    assert result.exit_code == 0, result.output
    content = md_file.read_text()
    assert "## Summary" in content
    assert "Test summary." in content
    assert "# Meeting 2026-03-20 14:30" in content


def test_summarize_command_no_api_key(runner, tmp_path, monkeypatch, vault_env):
    """No API key → error, file unchanged."""
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
def test_process_with_summarization(runner, tmp_path, monkeypatch, vault_env):
    """Full pipeline: process → transcribe → summarize → file has summary."""
    monkeypatch.setenv("TAPEBACK_DIARIZE", "false")
    monkeypatch.setenv("TAPEBACK_LLM_API_KEY", "sk-test")

    audio = tmp_path / "2026-03-20_10-00-00.wav"
    create_silent_wav(audio, duration=2.0, sample_rate=48000)

    mock_model = mock_whisper_transcribe([(0.0, 5.0, "Hello from the meeting.")])

    with (
        patch("tapeback.transcriber.WhisperModel", return_value=mock_model),
        patch("tapeback.summarizer._call_llm", return_value=VALID_LLM_RESPONSE_MINIMAL),
    ):
        result = runner.invoke(cli, ["process", str(audio), "--no-diarize"])

    assert result.exit_code == 0, result.output
    md_content = (vault_env / "meetings" / "2026-03-20_10-00-00.md").read_text()
    assert "## Summary" in md_content
    assert "Test summary." in md_content
    assert "Hello from the meeting." in md_content


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_no_summarize_flag(runner, tmp_path, monkeypatch, vault_env):
    """--no-summarize → no LLM call."""
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
    md_content = (vault_env / "meetings" / "2026-03-20_10-00-00.md").read_text()
    assert "## Summary" not in md_content


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_stop_and_process_summarization_failure(tmp_vault, session_wavs):
    """LLM fails → warning printed, transcript still saved."""
    settings = Settings(vault_path=tmp_vault, llm_api_key="sk-test")
    _session_dir, monitor_wav, mic_wav = session_wavs("2026-03-20_12-00-00")

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
    md_path = tmp_vault / "meetings" / "2026-03-20_12-00-00.md"
    assert md_path.exists()
    assert "Important content." in md_path.read_text()
    assert "## Summary" not in md_path.read_text()


def test_summarize_command_with_provider_and_model_overrides(
    runner, tmp_path, monkeypatch, vault_env
):
    """--provider and --model override settings for the summarize call."""
    monkeypatch.setenv("TAPEBACK_LLM_API_KEY", "sk-test")

    md_file = tmp_path / "transcript.md"
    md_file.write_text(SAMPLE_TRANSCRIPT_MD)

    with patch("tapeback.summarizer._call_llm", return_value=VALID_LLM_RESPONSE_MINIMAL):
        result = runner.invoke(
            cli,
            ["summarize", str(md_file), "--provider", "groq", "--model", "custom-model"],
        )

    assert result.exit_code == 0, result.output
    content = md_file.read_text()
    assert "## Summary" in content


def test_summarize_command_empty_transcript(runner, tmp_path, monkeypatch, vault_env):
    """summarize with no transcript content → error exit."""
    monkeypatch.setenv("TAPEBACK_LLM_API_KEY", "sk-test")

    md_file = tmp_path / "no_transcript.md"
    md_file.write_text("---\ndate: 2026-03-20\n---\n\nJust notes, no meeting header.\n")

    result = runner.invoke(cli, ["summarize", str(md_file)])

    assert result.exit_code != 0
    assert "No transcript content" in result.output


def test_status_command_with_pactl(runner, vault_env):
    """status command: shows audio sources when pactl is available."""
    with (
        patch("tapeback.cli.get_settings") as mock_settings,
        patch("tapeback.recorder.Recorder.get_session_info", return_value=None),
        patch("shutil.which", return_value="/usr/bin/pactl"),
        patch("subprocess.run") as mock_run,
    ):
        mock_settings.return_value = Settings(vault_path=vault_env)
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="alsa_output.pci-0000_00_1f.3.analog-stereo.monitor\n",
        )
        result = runner.invoke(cli, ["status"])

    assert result.exit_code == 0
    assert "Audio sources:" in result.output
    assert "alsa_output" in result.output


# --- tray command ---


def test_tray_missing_dependency(runner):
    """tapeback tray → helpful error when pystray not installed."""
    with patch.dict("sys.modules", {"tapeback.tray": None}):
        result = runner.invoke(cli, ["tray"])

    assert result.exit_code != 0
    assert "pystray" in result.output
