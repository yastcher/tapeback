"""End-to-end quality tests for the transcription + diarization pipeline.

These tests use real audio files and real ML models (faster-whisper, pyannote).
They are slow (minutes) and need HF_TOKEN for pyannote; a GPU makes them faster.
The recordings live in tests/data/, outside git.

Run with: TAPEBACK_RUN_E2E=1 HF_TOKEN=... uv run pytest tests/test_e2e_quality.py -v
"""

import os
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from tapeback.cli import cli
from tapeback.pipeline import process_stereo_file

_RUN_E2E = os.environ.get("TAPEBACK_RUN_E2E", "").lower() in ("1", "true", "yes")
_TEST_DATA = Path(__file__).parent / "data"
# You on the mic, two other people on the monitor channel.
_STEREO_WAV = _TEST_DATA / "2026-04-02_22-01-53.wav"

pytestmark = [
    pytest.mark.skipif(not _RUN_E2E, reason="Set TAPEBACK_RUN_E2E=1 to run e2e tests"),
    pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required"),
    pytest.mark.skipif(not _STEREO_WAV.exists(), reason="Test WAV not found"),
]


def test_stereo_pipeline_produces_segments(e2e_settings, e2e_output_dir):
    """Full stereo pipeline: transcribe + merge. No diarization."""
    segments, info, _raw = process_stereo_file(
        _STEREO_WAV, e2e_output_dir, e2e_settings, diarize=False
    )

    assert len(segments) > 0, "Pipeline must produce at least one segment"
    assert float(info.get("duration", 0)) > 0, "Duration must be positive"

    # Mic channel segments should have "You" speaker
    you_segments = [s for s in segments if s.speaker == "You"]
    assert len(you_segments) > 0, "Stereo pipeline must identify mic channel as 'You'"


def test_stereo_pipeline_with_diarization(e2e_settings, e2e_output_dir):
    """Full stereo pipeline with diarization: "You" on the mic, and the two people on
    the monitor channel as two speakers — neither merged into one nor split into more.
    """
    if not e2e_settings.hf_token.get_secret_value():
        pytest.skip("HF_TOKEN required for diarization test")

    segments, _info, _raw = process_stereo_file(
        _STEREO_WAV, e2e_output_dir, e2e_settings, diarize=True
    )

    assert len(segments) > 0

    speakers = {s.speaker for s in segments if s.speaker is not None}
    assert "You" in speakers, "Must detect user on mic channel"

    remote = speakers - {"You"}
    assert len(remote) == 2, f"Expected the 2 people on the monitor channel, got {sorted(remote)}"


def test_process_command_with_real_audio(e2e_settings):
    """tapeback process command with real stereo WAV end-to-end."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "process",
            str(_STEREO_WAV),
            "--name",
            "e2e-test",
            "--no-diarize",
            "--no-summarize",
        ],
        env={
            "TAPEBACK_VAULT_PATH": str(e2e_settings.vault_path),
            "TAPEBACK_DIARIZE": "false",
        },
    )

    assert result.exit_code == 0, result.output + str(result.exception or "")
    assert "Stereo file detected" in result.output

    md_path = e2e_settings.vault_path / "meetings" / "e2e-test.md"
    assert md_path.exists(), "Markdown file must be created"

    md_content = md_path.read_text()
    assert "**You:**" in md_content, "Mic segments must be labeled as 'You'"
    assert "date:" in md_content, "Markdown must have YAML frontmatter"
