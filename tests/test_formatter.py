"""Formatter tests — markdown generation and vault I/O pipelines."""

from meetrec.formatter import format_markdown
from meetrec.models import Segment
from meetrec.vault import save_to_vault


def test_format_markdown_pipeline():
    """format_markdown should produce complete markdown with frontmatter, timecodes,
    speaker labels, short-segment filtering, and consecutive-speaker merging."""
    segments = [
        # Short segment — should be filtered out (< 1s)
        Segment(start=0.0, end=0.5, text="Too short.", speaker="You"),
        # Two consecutive "You" segments — should merge
        Segment(start=1.0, end=5.0, text="Hello there.", speaker="You"),
        Segment(start=5.0, end=10.0, text="How are you?", speaker="You"),
        # Speaker change
        Segment(start=83.0, end=90.0, text="I'm fine.", speaker="Speaker 1"),
        Segment(start=90.0, end=95.0, text="Thanks.", speaker="Speaker 1"),
        # Back to You — separate block
        Segment(start=165.0, end=170.0, text="Great.", speaker="You"),
    ]

    result = format_markdown(
        segments=segments,
        session_name="2026-03-17_14-30-00",
        audio_rel_path="attachments/audio/2026-03-17_14-30-00.wav",
        duration_seconds=5025.0,
        language="en",
    )

    # Frontmatter
    assert result.startswith("---\n")
    assert "date: 2026-03-17" in result
    assert 'time: "14:30"' in result
    assert 'duration: "01:23:45"' in result
    assert "language: en" in result
    assert "[[attachments/audio/2026-03-17_14-30-00.wav]]" in result
    assert "  - meeting" in result
    assert "  - transcript" in result

    # Short segment filtered
    assert "Too short" not in result

    # Consecutive speakers merged
    assert "**You:** Hello there. How are you?" in result
    assert "**Speaker 1:** I'm fine. Thanks." in result
    assert "**You:** Great." in result

    # Timecodes
    assert "[00:00:01]" in result
    assert "[00:01:23]" in result
    assert "[00:02:45]" in result

    # 3 timecoded lines (not 5 or 6)
    assert result.count("[00:") == 3


def test_format_markdown_preserves_pauses():
    """Consecutive segments from the same speaker separated by a pause should NOT merge."""
    segments = [
        Segment(start=0.0, end=5.0, text="First block.", speaker="You"),
        # 3-second gap (5.0 → 8.0) — exceeds default pause_threshold=1.0
        Segment(start=8.0, end=13.0, text="After pause.", speaker="You"),
    ]

    result = format_markdown(
        segments=segments,
        session_name="2026-03-17_14-30-00",
        audio_rel_path="audio.wav",
        duration_seconds=13.0,
        language="en",
    )

    assert "**You:** First block." in result
    assert "**You:** After pause." in result
    assert "First block. After pause." not in result
    assert result.count("[00:") == 2


def test_save_to_vault_pipeline(settings, tmp_vault, tmp_path):
    """save_to_vault should create directories, save .md and .wav, and avoid overwrites."""
    stereo_wav = tmp_path / "stereo.wav"
    stereo_wav.write_bytes(b"fake wav data")

    meetings_dir = tmp_vault / settings.meetings_dir
    attachments_dir = tmp_vault / settings.attachments_dir
    assert not meetings_dir.exists()
    assert not attachments_dir.exists()

    # First save — creates directories and files
    md_path = save_to_vault(
        markdown="# Test",
        stereo_wav=stereo_wav,
        settings=settings,
        session_name="2026-03-17_14-30-00",
    )

    assert md_path.exists()
    assert md_path.read_text() == "# Test"
    assert md_path.name == "2026-03-17_14-30-00.md"
    assert meetings_dir.exists()
    assert attachments_dir.exists()

    audio_path = tmp_vault / "attachments" / "audio" / "2026-03-17_14-30-00.wav"
    assert audio_path.exists()

    # Second save — gets _1 suffix (no overwrite)
    md_path_2 = save_to_vault("# Second", stereo_wav, settings, "2026-03-17_14-30-00")
    assert md_path_2.name == "2026-03-17_14-30-00_1.md"
    assert md_path_2.read_text() == "# Second"
