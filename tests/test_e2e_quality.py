"""End-to-end quality tests for the transcription + diarization pipeline.

These tests use real audio files and real ML models (faster-whisper, pyannote).
They are slow (minutes) and require GPU + HF token.

Run with: MEETREC_RUN_E2E=1 uv run pytest tests/test_e2e_quality.py -v
"""

import os
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from meetrec.cli import _process_stereo_file, cli

_RUN_E2E = os.environ.get("MEETREC_RUN_E2E", "").lower() in ("1", "true", "yes")
_TEST_DATA = Path(__file__).parent / "data"
_STEREO_WAV = _TEST_DATA / "2026-03-24_18-31-58.wav"

pytestmark = [
    pytest.mark.skipif(not _RUN_E2E, reason="Set MEETREC_RUN_E2E=1 to run e2e tests"),
    pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required"),
    pytest.mark.skipif(not _STEREO_WAV.exists(), reason="Test WAV not found"),
]


def test_stereo_pipeline_produces_segments(e2e_settings, e2e_output_dir):
    """Full stereo pipeline: transcribe + merge. No diarization."""
    segments, info = _process_stereo_file(_STEREO_WAV, e2e_output_dir, e2e_settings, diarize=False)

    assert len(segments) > 0, "Pipeline must produce at least one segment"
    assert float(info.get("duration", 0)) > 0, "Duration must be positive"

    # Mic channel segments should have "You" speaker
    you_segments = [s for s in segments if s.speaker == "You"]
    assert len(you_segments) > 0, "Stereo pipeline must identify mic channel as 'You'"


def test_stereo_pipeline_with_diarization(e2e_settings, e2e_output_dir):
    """Full stereo pipeline with diarization.

    Test audio: user speaks into mic, YouTube video plays on monitor.
    Expected: "You" for mic segments, "Speaker N" for monitor segments.
    The monitor channel has ONE speaker (video narrator) — diarization
    should not split it into multiple speakers.
    """
    if not e2e_settings.hf_token:
        pytest.skip("MEETREC_HF_TOKEN required for diarization test")

    segments, _info = _process_stereo_file(_STEREO_WAV, e2e_output_dir, e2e_settings, diarize=True)

    assert len(segments) > 0

    speakers = {s.speaker for s in segments if s.speaker is not None}
    assert "You" in speakers, "Must detect user on mic channel"

    non_you_speakers = speakers - {"You"}
    assert len(non_you_speakers) >= 1, "Must detect at least one remote speaker"

    # Quality check: monitor channel should have exactly ONE speaker
    # The test audio has a single video narrator — splitting into multiple
    # speakers indicates a diarization quality issue.
    assert len(non_you_speakers) == 1, (
        f"Expected 1 remote speaker, got {len(non_you_speakers)}: {non_you_speakers}. "
        "Diarization incorrectly split a single speaker into multiple."
    )


def test_process_command_with_real_audio(e2e_settings):
    """meetrec process command with real stereo WAV end-to-end."""
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
            "MEETREC_VAULT_PATH": str(e2e_settings.vault_path),
            "MEETREC_DIARIZE": "false",
        },
    )

    assert result.exit_code == 0, result.output + str(result.exception or "")
    assert "Stereo file detected" in result.output

    md_path = e2e_settings.vault_path / "meetings" / "e2e-test.md"
    assert md_path.exists(), "Markdown file must be created"

    md_content = md_path.read_text()
    assert "**You:**" in md_content, "Mic segments must be labeled as 'You'"
    assert "date:" in md_content, "Markdown must have YAML frontmatter"
