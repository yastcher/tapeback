"""Regression: "Summarizing..." was printed twice.

The pipeline reports the stage through its status callback — which is also what lands
in the run record — and summarizer.maybe_summarize printed the same line again on its
own, so every summarized meeting showed it twice and the run record only once.
"""

import shutil
from unittest.mock import patch

import pytest

from tapeback.cli import cli
from tests.fixtures import VALID_LLM_RESPONSE_MINIMAL, create_silent_wav, mock_whisper_transcribe


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_summarizing_is_reported_once(runner, tmp_path, monkeypatch, vault_env):
    monkeypatch.setenv("TAPEBACK_DIARIZE", "false")
    monkeypatch.setenv("TAPEBACK_LLM_API_KEY", "sk-test")
    audio = tmp_path / "2026-03-20_10-00-00.wav"
    create_silent_wav(audio, duration=2.0, sample_rate=48000)

    with (
        patch(
            "tapeback.transcriber.WhisperModel",
            return_value=mock_whisper_transcribe([(0.0, 2.0, "Hello from the meeting.")]),
        ),
        patch("tapeback.summarizer._call_llm", return_value=VALID_LLM_RESPONSE_MINIMAL),
    ):
        result = runner.invoke(cli, ["process", str(audio), "--no-diarize"])

    assert result.exit_code == 0, result.output
    assert result.stderr.count("Summarizing...") == 1
    assert result.stderr.count("Summary added.") == 1
