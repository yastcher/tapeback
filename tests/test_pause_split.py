"""Splitting the mic's segments at pauses: where a pause is found, and what each piece keeps."""

import numpy as np
import pytest

from tapeback.channel import _pause_points, split_on_silence
from tapeback.models import Segment, Word

SR = 16000


def _level(seconds: float, value: int) -> np.ndarray:
    return np.full(round(seconds * SR), value, dtype=np.int16)


def _word(start: float, end: float, text: str) -> Word:
    return Word(start=start, end=end, word=text, probability=0.9)


# A second of speech, a second of silence, a second of speech.
SPEECH_PAUSE_SPEECH = np.concatenate([_level(1, 7000), _level(1, 0), _level(1, 7000)])


def _one_two() -> Segment:
    return Segment(
        start=0.0,
        end=3.0,
        text="One. Two.",
        words=[_word(0.2, 0.8, " One."), _word(2.2, 2.8, " Two.")],
        speaker="You",
    )


def _one_and_two() -> list[Segment]:
    return [
        Segment(start=0.2, end=0.8, text="One.", words=[_word(0.2, 0.8, " One.")], speaker="You"),
        Segment(start=2.2, end=2.8, text="Two.", words=[_word(2.2, 2.8, " Two.")], speaker="You"),
    ]


def test_a_missing_monitor_reads_as_silence():
    segment = _one_two()

    split = split_on_silence(
        [segment], _level(3, 0), SR, 1.0, monitor_samples=np.zeros(0, dtype=np.int16)
    )

    assert split == [segment]


@pytest.mark.parametrize(
    ("mic", "segment", "points"),
    [
        (SPEECH_PAUSE_SPEECH, Segment(start=0.0, end=3.0, text="x"), [1.5]),
        (
            np.concatenate([_level(2.1, 7000), _level(0.9, 0)]),
            Segment(start=0.0, end=3.0, text="x"),
            [],
        ),
        (
            np.concatenate([_level(2.0, 7000), _level(1.0, 0)]),
            Segment(start=0.0, end=3.0, text="x"),
            [2.5],
        ),
        (
            np.concatenate([_level(1.9, 7000), _level(1.1, 0)]),
            Segment(start=0.0, end=3.0, text="x"),
            [2.45],
        ),
        (
            np.concatenate([_level(2, 0), _level(3, 7000)]),
            Segment(start=2.0, end=5.0, text="x"),
            [],
        ),
    ],
    ids=["middle", "short of the threshold", "at the end", "past the threshold", "late start"],
)
def test_a_pause_is_cut_in_its_middle(mic, segment, points):
    assert _pause_points(segment, mic, SR, 1.0, None) == points


@pytest.mark.parametrize(
    ("end", "points"),
    [(1.0999375, []), (1.1, [1.05]), (1.1000625, [1.05003125])],
)
def test_a_pause_needs_one_full_window(end, points):
    segment = Segment(start=1.0, end=end, text="x")

    assert _pause_points(segment, _level(2, 0), SR, 0.05, _level(2, 1000)) == points


@pytest.mark.parametrize(
    ("level", "pieces"),
    [(2999, 2), (3000, 1), (3001, 1)],
)
def test_a_pause_where_the_monitor_drowns_the_mic(level, pieces):
    """The monitor is 10000 there: under 0.3 of it the mic only listens."""
    mic = np.concatenate([_level(1, 7000), _level(1, level), _level(1, 7000)])
    monitor = np.concatenate([_level(1, 0), _level(1, 10000), _level(1, 0)])

    split = split_on_silence([_one_two()], mic, SR, 1.0, monitor_samples=monitor)

    assert split == (_one_and_two() if pieces == 2 else [_one_two()])


def test_a_pause_as_long_as_the_threshold_cuts():
    words = [
        _word(0.1, 0.5, " one"),
        _word(0.5, 0.95, " two"),
        _word(2.05, 2.5, " three"),
        _word(2.5, 3.0, " four"),
    ]
    segment = Segment(start=0.0, end=3.0, text="One two three four.", words=words, speaker="You")

    assert split_on_silence([segment], SPEECH_PAUSE_SPEECH, SR, pause_threshold=1.0) == [
        Segment(start=0.1, end=0.95, text="one two", words=words[:2], speaker="You"),
        Segment(start=2.05, end=3.0, text="three four", words=words[2:], speaker="You"),
    ]


@pytest.mark.parametrize(
    ("start", "end"),
    [(1.45, 2.0), (1.46, 2.0), (1.0, 1.55), (1.0, 1.54)],
)
def test_a_word_within_a_twentieth_of_a_second_of_the_cut_stays(start, end):
    word = _word(start, end, " edge")
    segment = Segment(start=0.0, end=3.0, text="Edge.", words=[word], speaker="You")

    assert split_on_silence([segment], SPEECH_PAUSE_SPEECH, SR, 1.0) == [
        Segment(start=start, end=end, text="edge", words=[word], speaker="You")
    ]
