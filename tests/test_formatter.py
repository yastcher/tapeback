"""Formatter tests — markdown generation and vault I/O pipelines."""

import pytest

from tapeback.formatter import (
    TranscriptMeta,
    _format_duration_hms,
    _format_duration_human,
    _format_timecode,
    _mark_low_confidence_words,
    _merge_consecutive_speakers,
    format_live_markdown,
    format_markdown,
)
from tapeback.models import Segment, Word
from tapeback.vault import remove_live_markdown, save_live_markdown, save_to_vault


@pytest.mark.parametrize(
    ("seconds", "hms", "human", "timecode"),
    [
        (59.0, "00:00:59", "59s", "[00:00:59]"),
        (60.0, "00:01:00", "1m 0s", "[00:01:00]"),
        (61.0, "00:01:01", "1m 1s", "[00:01:01]"),
        (3599.0, "00:59:59", "59m 59s", "[00:59:59]"),
        (3600.0, "01:00:00", "1h 0s", "[01:00:00]"),
        (3601.0, "01:00:01", "1h 1s", "[01:00:01]"),
    ],
)
def test_seconds_render_across_minute_and_hour_boundaries(seconds, hms, human, timecode):
    assert _format_duration_hms(seconds) == hms
    assert _format_duration_human(seconds) == human
    assert _format_timecode(seconds) == timecode


def test_format_markdown_pipeline():
    """format_markdown should produce complete markdown with frontmatter, timecodes,
    speaker labels, short-segment filtering, and consecutive-speaker merging."""
    segments = [
        # Short segment — should be filtered out (< 1s). Not at 0, where its end
        # minus its start equals its end plus its start.
        Segment(start=0.3, end=0.8, text="Too short.", speaker="You"),
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
        meta=TranscriptMeta(
            session_name="2026-03-17_14-30-00",
            audio_rel_path="attachments/audio/2026-03-17_14-30-00.wav",
            duration_seconds=5025.0,
            language="en",
        ),
    )

    # The short segment is gone, and consecutive segments of a speaker merged.
    assert result == (
        "---\n"
        "date: 2026-03-17\n"
        'time: "14:30"\n'
        'duration: "01:23:45"\n'
        "language: en\n"
        'audio: "[[attachments/audio/2026-03-17_14-30-00.wav]]"\n'
        "tags:\n"
        "  - meeting\n"
        "  - transcript\n"
        "---\n"
        "\n"
        "# Meeting 2026-03-17 14:30\n"
        "\n"
        "**Duration:** 1h 23m 45s | **Language:** en\n"
        "\n"
        "---\n"
        "\n"
        "[00:00:01] **You:** Hello there. How are you?\n"
        "\n"
        "[00:01:23] **Speaker 1:** I'm fine. Thanks.\n"
        "\n"
        "[00:02:45] **You:** Great.\n"
    )


def test_a_session_without_a_time_shows_midnight():
    """`start --name standup` names a session with no date and no time in it."""
    lines = format_markdown(
        segments=[],
        meta=TranscriptMeta(
            session_name="standup",
            audio_rel_path="attachments/audio/standup.wav",
            duration_seconds=5.0,
            language="en",
        ),
    ).splitlines()

    assert lines[1] == "date: standup"
    assert lines[2] == 'time: "00:00"'
    assert lines[11] == "# Meeting standup 00:00"
    assert format_live_markdown([], "standup", "en").splitlines()[0] == (
        "# Live Transcript standup 00:00"
    )


@pytest.mark.parametrize(
    ("second_start", "blocks"),
    [
        (5.5, [(0.0, "You", "First. Second.")]),
        (6.0, [(0.0, "You", "First."), (6.0, "You", "Second.")]),
        (6.5, [(0.0, "You", "First."), (6.5, "You", "Second.")]),
    ],
)
def test_a_pause_of_a_second_splits_a_speakers_blocks(second_start, blocks):
    segments = [
        Segment(start=0.0, end=5.0, text="First.", speaker="You"),
        Segment(start=second_start, end=second_start + 5.0, text="Second.", speaker="You"),
    ]

    assert _merge_consecutive_speakers(segments) == blocks


@pytest.mark.parametrize(
    ("second_end", "blocks"),
    [
        (59.0, [(0.0, "You", "First. Second.")]),
        (60.0, [(0.0, "You", "First."), (30.0, "You", "Second.")]),
        (61.0, [(0.0, "You", "First."), (30.0, "You", "Second.")]),
    ],
)
def test_a_block_closes_at_sixty_seconds(second_end, blocks):
    segments = [
        Segment(start=0.0, end=30.0, text="First.", speaker="You"),
        Segment(start=30.0, end=second_end, text="Second.", speaker="You"),
    ]

    assert _merge_consecutive_speakers(segments) == blocks


def test_format_markdown_preserves_pauses():
    """Consecutive segments from the same speaker separated by a pause should NOT merge."""
    segments = [
        Segment(start=0.0, end=5.0, text="First block.", speaker="You"),
        # 3-second gap (5.0 → 8.0) — exceeds default pause_threshold=1.0
        Segment(start=8.0, end=13.0, text="After pause.", speaker="You"),
    ]

    result = format_markdown(
        segments=segments,
        meta=TranscriptMeta(
            session_name="2026-03-17_14-30-00",
            audio_rel_path="audio.wav",
            duration_seconds=13.0,
            language="en",
        ),
    )

    assert "**You:** First block." in result
    assert "**You:** After pause." in result
    assert "First block. After pause." not in result
    assert result.count("[00:") == 2


def test_format_markdown_writes_one_transcript_without_a_heading():
    """The transcript follows the header rule directly, with no section headings."""
    segments = [
        Segment(start=1.0, end=5.0, text="Hello.", speaker="You"),
    ]

    result = format_markdown(
        segments=segments,
        meta=TranscriptMeta(
            session_name="2026-04-04_14-30-00",
            audio_rel_path="audio.wav",
            duration_seconds=5.0,
            language="en",
        ),
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


def test_a_word_at_the_threshold_is_not_marked():
    segment = Segment(
        start=0.0,
        end=3.0,
        text="below at above",
        speaker="You",
        words=[
            Word(start=0.0, end=1.0, word=" below", probability=0.34),
            Word(start=1.0, end=2.0, word=" at", probability=0.35),
            Word(start=2.0, end=3.0, word=" above", probability=0.36),
        ],
    )

    assert _mark_low_confidence_words(segment).text == "*below* at above"


def test_a_blank_word_is_skipped_not_taken_for_the_end():
    segment = Segment(
        start=0.0,
        end=3.0,
        text="Hello world",
        speaker="You",
        words=[
            Word(start=0.0, end=1.0, word=" Hello", probability=0.9),
            Word(start=1.0, end=1.5, word=" ", probability=0.9),
            Word(start=1.5, end=3.0, word=" world", probability=0.2),
        ],
    )

    assert _mark_low_confidence_words(segment).text == "Hello *world*"


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

    # No front matter and no duration: the recording is still going.
    assert result == (
        "# Live Transcript 2026-04-18 14:30\n"
        "\n"
        "**Language:** en | **Status:** recording in progress\n"
        "\n"
        "---\n"
        "\n"
        "[00:00:01] **You:** Hello there.\n"
        "\n"
        "[00:00:05] **Other:** Hi, how are you?\n"
        "\n"
        "---\n"
        "\n"
        "*Live preview. Final transcript with diarization will replace this file.*\n"
    )


def test_format_live_markdown_empty_segments():
    """Live markdown with no segments shows waiting message."""
    result = format_live_markdown(
        segments=[],
        session_name="2026-04-18_14-30-00",
        language="en",
    )

    assert result == (
        "# Live Transcript 2026-04-18 14:30\n"
        "\n"
        "**Language:** en | **Status:** recording in progress\n"
        "\n"
        "---\n"
        "\n"
        "*Waiting for first transcription cycle...*\n"
        "\n"
        "---\n"
        "\n"
        "*Live preview. Final transcript with diarization will replace this file.*\n"
    )


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
