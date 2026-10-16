"""Regression: diarization lost short stretches of speech, then made speakers of them.

Diarization cuts a Whisper segment wherever pyannote hears another voice. A piece under
a second was dropped as a VAD artifact, words and all ("Agreed."). Kept under its own
pyannote speaker instead, such a piece lost nothing, but on a recorded meeting it added
a speaker who was not there: lines of a word or three cut out of the others' sentences.

A piece under a second now joins the speech around it, in the same Whisper segment.
"""

import shutil
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from tapeback.diarizer import assign_speakers, merge_channel_segments
from tapeback.formatter import TranscriptMeta, format_markdown
from tapeback.models import DiarizationSegment, Segment, Word
from tapeback.pipeline import process_file
from tapeback.settings import Settings
from tests.fixtures import (
    create_mono_wav,
    create_stereo_wav_segments,
    mock_pyannote_annotation,
    mock_whisper_by_channel,
    requires_pyannote,
)

# One Whisper segment in which pyannote hears another voice for 0.4 s.
EXCHANGE = (
    0.0,
    6.0,
    "We ship Friday? Agreed. Then tests first.",
    [
        (0.0, 0.6, "We"),
        (0.6, 1.4, "ship"),
        (1.4, 2.4, "Friday?"),
        (2.6, 3.0, "Agreed."),
        (3.2, 4.0, "Then"),
        (4.0, 5.0, "tests"),
        (5.0, 6.0, "first."),
    ],
)
TURNS = [(0.0, 2.5, "SPEAKER_00"), (2.5, 3.1, "SPEAKER_01"), (3.1, 6.0, "SPEAKER_00")]

META = TranscriptMeta(
    session_name="2026-10-15_10-00-00",
    audio_rel_path="attachments/2026-10-15_10-00-00.wav",
    duration_seconds=6.0,
    language="en",
)


class _FakePipeline:
    def __call__(self, _audio, **_kwargs):
        return mock_pyannote_annotation(TURNS)


def _word(start: float, end: float, word: str) -> Word:
    return Word(start=start, end=end, word=f" {word}", probability=0.9)


def _turn(speaker: str, start: float, end: float) -> DiarizationSegment:
    return DiarizationSegment(speaker=speaker, start=start, end=end)


def _transcript_lines(markdown: str) -> list[str]:
    return [line for line in markdown.splitlines() if line.startswith("[")]


@requires_pyannote
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
@pytest.mark.parametrize("source", ["stereo", "mono"])
def test_a_short_piece_keeps_its_words_and_adds_no_speaker(tmp_path, source):
    audio = tmp_path / f"{source}.wav"
    if source == "stereo":
        # The mic (left) stays silent; the call (right) speaks throughout.
        create_stereo_wav_segments(audio, 48000, [(7.0, 0.003, 0.8)])
    else:
        create_mono_wav(audio, duration=7.0)
    settings = Settings(
        vault_path=tmp_path,
        hf_token=SecretStr("hf_fake"),
        device="cpu",
        # Both voices are the same synthetic tone, which the spectral merge would fuse.
        spectral_merge_threshold=0.0,
    )

    with (
        patch(
            "tapeback.transcriber.WhisperModel",
            return_value=mock_whisper_by_channel([], [EXCHANGE]),
        ),
        patch("pyannote.audio.Pipeline.from_pretrained", return_value=_FakePipeline()),
    ):
        note = process_file(audio, settings, diarize=True, do_summarize=False).read_text()

    assert _transcript_lines(note) == [
        "[00:00:00] **Speaker 1:** We ship Friday? Agreed. Then tests first."
    ]


def test_a_short_piece_opening_a_segment_joins_the_speech_after_it():
    segment = Segment(
        start=0.0,
        end=3.0,
        text="Okay, let us begin.",
        words=[_word(0.0, 0.4, "Okay,"), _word(0.5, 1.5, "let"), _word(1.5, 3.0, "us begin.")],
    )
    turns = [_turn("SPEAKER_00", 0.0, 0.45), _turn("SPEAKER_01", 0.45, 3.0)]

    markdown = format_markdown(assign_speakers([segment], turns), META)

    assert _transcript_lines(markdown) == ["[00:00:00] **Speaker 1:** Okay, let us begin."]


@pytest.mark.parametrize(
    ("reply_seconds", "expected"),
    [
        (0.9, ["[00:00:00] **Speaker 1:** We ship Friday? Agreed, fully. Then tests first."]),
        (
            1.0,
            [
                "[00:00:00] **Speaker 1:** We ship Friday?",
                "[00:00:02] **Speaker 2:** Agreed, fully.",
                "[00:00:04] **Speaker 1:** Then tests first.",
            ],
        ),
        (
            1.1,
            [
                "[00:00:00] **Speaker 1:** We ship Friday?",
                "[00:00:02] **Speaker 2:** Agreed, fully.",
                "[00:00:04] **Speaker 1:** Then tests first.",
            ],
        ),
    ],
)
def test_a_second_of_another_voice_is_a_speaker_of_its_own(reply_seconds, expected):
    reply_end = 2.5 + reply_seconds
    segment = Segment(
        start=0.0,
        end=6.0,
        text="We ship Friday? Agreed, fully. Then tests first.",
        words=[
            _word(0.0, 2.4, "We ship Friday?"),
            _word(2.5, reply_end - 0.5, "Agreed,"),
            _word(reply_end - 0.5, reply_end, "fully."),
            _word(4.0, 6.0, "Then tests first."),
        ],
    )
    turns = [
        _turn("SPEAKER_00", 0.0, 2.45),
        _turn("SPEAKER_01", 2.45, reply_end + 0.05),
        _turn("SPEAKER_00", reply_end + 0.05, 6.0),
    ]

    markdown = format_markdown(assign_speakers([segment], turns), META)

    assert _transcript_lines(markdown) == expected


def test_sub_second_segment_cut_by_diarization_is_still_dropped():
    """An artifact stays one when its words straddle a change of speaker."""
    blip = Segment(
        start=0.0,
        end=0.8,
        text="Uh, okay.",
        words=[_word(0.0, 0.4, "Uh,"), _word(0.5, 0.8, "okay.")],
    )
    speech = Segment(
        start=1.0,
        end=4.0,
        text="Let us begin.",
        words=[_word(1.0, 2.0, "Let"), _word(2.0, 3.0, "us"), _word(3.0, 4.0, "begin.")],
    )
    turns = [
        _turn("SPEAKER_00", 0.0, 0.45),
        _turn("SPEAKER_01", 0.45, 0.8),
        _turn("SPEAKER_00", 1.0, 4.0),
    ]

    markdown = format_markdown(assign_speakers([blip, speech], turns), META)

    assert _transcript_lines(markdown) == ["[00:00:01] **Speaker 1:** Let us begin."]


def test_segments_whose_words_span_under_a_second_survive_consolidation():
    """Whisper returned each segment longer than a second; their words, rebuilt into
    pieces and joined by consolidation, still span less."""
    first = Segment(start=0.0, end=1.2, text="Yes,", words=[_word(0.8, 1.1, "Yes,")])
    second = Segment(start=1.2, end=2.4, text="sure.", words=[_word(1.2, 1.6, "sure.")])
    turns = [_turn("SPEAKER_00", 0.0, 2.4)]

    segments = merge_channel_segments([], assign_speakers([first, second], turns))

    assert _transcript_lines(format_markdown(segments, META)) == [
        "[00:00:00] **Speaker 1:** Yes, sure."
    ]
