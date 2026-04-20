"""Formatter tests — markdown generation and vault I/O pipelines."""

from tapeback.formatter import _mark_low_confidence_words, format_live_markdown, format_markdown
from tapeback.models import Segment, Word
from tapeback.vault import remove_live_markdown, save_live_markdown, save_to_vault


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


def test_format_markdown_with_raw_segments():
    """When raw_segments is provided, output should have two sections:
    '## Transcript' (raw) and '## Diarized Transcript' (diarized)."""
    raw_segments = [
        Segment(start=1.0, end=5.0, text="Hello there.", speaker="You"),
        Segment(start=5.0, end=10.0, text="I'm fine.", speaker="Other"),
    ]

    diarized_segments = [
        Segment(start=1.0, end=5.0, text="Hello there.", speaker="You"),
        Segment(start=5.0, end=10.0, text="I'm fine.", speaker="Speaker 1"),
    ]

    result = format_markdown(
        segments=diarized_segments,
        session_name="2026-04-04_14-30-00",
        audio_rel_path="audio.wav",
        duration_seconds=10.0,
        language="en",
        raw_segments=raw_segments,
    )

    # Both section headers present
    assert "## Transcript" in result
    assert "## Diarized Transcript" in result

    # Raw section has Other, diarized has Speaker 1
    transcript_idx = result.index("## Transcript")
    diarized_idx = result.index("## Diarized Transcript")
    assert transcript_idx < diarized_idx

    raw_section = result[transcript_idx:diarized_idx]
    diarized_section = result[diarized_idx:]

    assert "**Other:**" in raw_section
    assert "**Speaker 1:**" in diarized_section
    assert "**You:**" in raw_section
    assert "**You:**" in diarized_section


def test_format_markdown_without_raw_segments_unchanged():
    """When raw_segments is None, output should be single-section (backwards compat)."""
    segments = [
        Segment(start=1.0, end=5.0, text="Hello.", speaker="You"),
    ]

    result = format_markdown(
        segments=segments,
        session_name="2026-04-04_14-30-00",
        audio_rel_path="audio.wav",
        duration_seconds=5.0,
        language="en",
    )

    assert "## Transcript" not in result
    assert "## Diarized Transcript" not in result
    assert "**You:** Hello." in result


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


def test_mark_low_confidence_words_mixed():
    """Low-confidence words should be wrapped in italics; consecutive ones grouped."""
    segment = Segment(
        start=0.0,
        end=5.0,
        text="Sorry could you repeat passive note",
        speaker="You",
        words=[
            Word(start=0.0, end=0.3, word="Sorry", probability=0.13),
            Word(start=0.3, end=0.5, word="could", probability=0.25),
            Word(start=0.5, end=0.7, word="you", probability=0.20),
            Word(start=0.7, end=1.0, word="repeat", probability=0.85),
            Word(start=1.0, end=1.3, word="passive", probability=0.15),
            Word(start=1.3, end=1.6, word="note", probability=0.92),
        ],
    )

    result = _mark_low_confidence_words(segment)

    # Consecutive low-confidence words grouped into single italic span
    assert result.text == "*Sorry could you* repeat *passive* note"
    assert result.speaker == "You"
    assert result.start == 0.0
    assert result.end == 5.0
    # Original words preserved
    assert result.words == segment.words


def test_mark_low_confidence_words_all_confident():
    """All words above threshold — text unchanged."""
    segment = Segment(
        start=0.0,
        end=2.0,
        text="Hello world",
        speaker="You",
        words=[
            Word(start=0.0, end=0.5, word="Hello", probability=0.95),
            Word(start=0.5, end=1.0, word="world", probability=0.88),
        ],
    )

    result = _mark_low_confidence_words(segment)
    assert result.text == "Hello world"


def test_mark_low_confidence_words_all_low():
    """All words below threshold — entire text in single italic span."""
    segment = Segment(
        start=0.0,
        end=2.0,
        text="uh um ah",
        speaker="You",
        words=[
            Word(start=0.0, end=0.3, word="uh", probability=0.10),
            Word(start=0.3, end=0.6, word="um", probability=0.15),
            Word(start=0.6, end=0.9, word="ah", probability=0.20),
        ],
    )

    result = _mark_low_confidence_words(segment)
    assert result.text == "*uh um ah*"


def test_mark_low_confidence_words_no_words():
    """Segment without word-level data — returned as-is."""
    segment = Segment(start=0.0, end=2.0, text="No words here.", speaker="You")

    result = _mark_low_confidence_words(segment)
    assert result.text == "No words here."


# --- format_live_markdown ---


def test_format_live_markdown_with_segments():
    """Live markdown should include segments with timecodes and speaker labels."""
    segments = [
        Segment(start=1.0, end=5.0, text="Hello there.", speaker="You"),
        Segment(start=5.0, end=10.0, text="Hi, how are you?", speaker="Other"),
    ]

    result = format_live_markdown(
        segments=segments,
        session_name="2026-04-18_14-30-00",
        language="en",
    )

    assert "# Live Transcript 2026-04-18 14:30" in result
    assert "recording in progress" in result
    assert "**You:** Hello there." in result
    assert "**Other:** Hi, how are you?" in result
    assert "Final transcript with diarization" in result
    # No YAML front matter
    assert "---\ndate:" not in result
    assert "duration:" not in result


def test_format_live_markdown_empty_segments():
    """Live markdown with no segments shows waiting message."""
    result = format_live_markdown(
        segments=[],
        session_name="2026-04-18_14-30-00",
        language="en",
    )

    assert "Waiting for first transcription cycle" in result
    assert "# Live Transcript" in result


# --- save_live_markdown / remove_live_markdown ---


def test_save_live_markdown_creates_and_overwrites(settings, tmp_vault):
    """save_live_markdown should create the file and overwrite on subsequent calls."""
    path1 = save_live_markdown("# First", settings, "test-session")
    assert path1.exists()
    assert path1.read_text() == "# First"
    assert path1.name == "test-session_live.md"

    # Overwrites, not unique suffix
    path2 = save_live_markdown("# Second", settings, "test-session")
    assert path2 == path1
    assert path2.read_text() == "# Second"


def test_remove_live_markdown(settings, tmp_vault):
    """remove_live_markdown should delete the file without error."""
    save_live_markdown("# Content", settings, "test-session")
    md_path = tmp_vault / "meetings" / "test-session_live.md"
    assert md_path.exists()

    remove_live_markdown(settings, "test-session")
    assert not md_path.exists()

    # Should not raise on missing file
    remove_live_markdown(settings, "test-session")
