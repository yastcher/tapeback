"""Regression tests for transcript formatting bugs."""

from tapeback.formatter import TranscriptMeta, format_markdown
from tapeback.models import Segment, Word


def _speech(start: float, end: float, speaker: str | None = None) -> Segment:
    return Segment(start=start, end=end, text="реплика.", words=None, speaker=speaker)


def test_long_single_speaker_stretch_is_not_one_block():
    """A long uninterrupted stretch must stay navigable by timecode.

    Bug: _merge_consecutive_speakers merged while the speaker was unchanged and the
    gap stayed under pause_threshold, with no upper bound on the result. A 31-minute
    single-speaker recording rendered as two blocks whose last timecode was
    [00:00:45] — the text was all there, but there was no way to find anything in it.
    """
    # 30 minutes of back-to-back speech from one speaker, gaps far below the
    # pause threshold so nothing else would split it.
    segments = [_speech(i * 10.0, i * 10.0 + 9.9, "You") for i in range(180)]

    markdown = format_markdown(
        segments=segments,
        meta=TranscriptMeta(
            session_name="2026-08-06_12-00-00",
            audio_rel_path="attachments/audio/x.wav",
            duration_seconds=1800.0,
            language="ru",
        ),
    )

    timecodes = [line.split("]")[0] + "]" for line in markdown.splitlines() if line.startswith("[")]
    assert len(timecodes) > 1, "30 minutes collapsed into a single block"
    # Navigation is only useful if the timecodes track the audio: the last one must
    # be near the end, not near the beginning.
    assert timecodes[-1] > "[00:25:00]"


def test_short_stretch_is_still_merged_into_one_block():
    """The cap must not fragment ordinary short turns."""
    segments = [_speech(i * 2.0, i * 2.0 + 1.9, "You") for i in range(5)]

    markdown = format_markdown(
        segments=segments,
        meta=TranscriptMeta(
            session_name="2026-08-06_12-00-00",
            audio_rel_path="attachments/audio/x.wav",
            duration_seconds=10.0,
            language="ru",
        ),
    )

    timecodes = [line for line in markdown.splitlines() if line.startswith("[")]
    assert len(timecodes) == 1


def _said(text: str, start: float = 0.0, end: float = 5.0) -> Segment:
    return Segment(start=start, end=end, text=text, words=None, speaker="You")


def test_subtitle_corpus_hallucinations_are_stripped():
    """Whisper's subtitle-corpus phrases must not reach the transcript.

    Bug: Whisper emits credits from its training data over long pauses. Real
    occurrences across the project's own transcripts included "Субтитры DimaTorzok"
    (twice), "Редактор субтитров .Семкин", "Корректор .Кулакова" and six of
    "Продолжение следует...". None were filtered, so they read as things people said.
    """
    segments = [
        _said("Ведь решение. Субтитры DimaTorzok штука, куда ты вставляешь summary.", 0.0, 5.0),
        _said("Как будто нам сейчас не нужно. Редактор субтитров .Семкин", 6.0, 11.0),
        _said("поскольку Корректор .Кулакова общаются", 12.0, 17.0),
        _said("Продолжение следует...", 18.0, 23.0),
    ]

    markdown = format_markdown(
        segments=segments,
        meta=TranscriptMeta(
            session_name="2026-08-06_12-00-00",
            audio_rel_path="attachments/audio/x.wav",
            duration_seconds=30.0,
            language="ru",
        ),
    )

    lowered = markdown.lower()
    for marker in (
        "dimatorzok",
        "субтитры",
        "редактор субтитров",
        "корректор",
        "продолжение следует",
    ):
        assert marker not in lowered, marker
    # The real speech around the hallucination must survive.
    assert "куда ты вставляешь summary" in markdown
    assert "как будто нам сейчас не нужно" in markdown.lower()


def test_a_stripped_phrase_stays_out_of_a_segment_with_words():
    """Local Whisper gives every segment its words, and they must not bring the phrase back.

    Bug: the italic pass rebuilds a segment's text from its words, so a credit cut out of
    the middle of a sentence returned with them. The test above passes words=None and
    could not see it; on the default local path the credit stayed in the note.
    """
    words = [
        Word(start=0.0, end=0.5, word=" Ведь", probability=0.9),
        Word(start=0.5, end=1.0, word=" решение.", probability=0.9),
        Word(start=1.0, end=2.0, word=" Субтитры", probability=0.9),
        Word(start=2.0, end=3.0, word=" DimaTorzok", probability=0.9),
        Word(start=3.0, end=5.0, word=" штука.", probability=0.9),
    ]
    segment = Segment(
        start=0.0,
        end=5.0,
        text="Ведь решение. Субтитры DimaTorzok штука.",
        words=words,
        speaker="You",
    )

    markdown = format_markdown(
        segments=[segment],
        meta=TranscriptMeta(
            session_name="2026-08-06_12-00-00",
            audio_rel_path="attachments/audio/x.wav",
            duration_seconds=5.0,
            language="ru",
        ),
    )

    lines = [line for line in markdown.splitlines() if line.startswith("[")]
    assert lines == ["[00:00:00] **You:** Ведь решение. штука"]


def test_segment_that_is_only_a_hallucination_is_dropped_entirely():
    segments = [
        _said("Продолжение следует...", 0.0, 5.0),
        _said("Реальная реплика про pipeline.", 6.0, 11.0),
    ]

    markdown = format_markdown(
        segments=segments,
        meta=TranscriptMeta(
            session_name="2026-08-06_12-00-00",
            audio_rel_path="attachments/audio/x.wav",
            duration_seconds=15.0,
            language="ru",
        ),
    )

    lines = [line for line in markdown.splitlines() if line.startswith("[")]
    assert len(lines) == 1
    assert "pipeline" in lines[0]


def test_speaker_change_still_splits_regardless_of_length():
    segments = [_speech(0.0, 5.0, "You"), _speech(5.0, 10.0, "Other")]

    markdown = format_markdown(
        segments=segments,
        meta=TranscriptMeta(
            session_name="2026-08-06_12-00-00",
            audio_rel_path="attachments/audio/x.wav",
            duration_seconds=10.0,
            language="ru",
        ),
    )

    lines = [line for line in markdown.splitlines() if line.startswith("[")]
    assert len(lines) == 2
    assert "**You:**" in lines[0]
    assert "**Other:**" in lines[1]
