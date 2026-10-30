"""Splitting the mic's segments at pauses: where a pause is found, and what each piece keeps."""

import numpy as np
import pytest

from tapeback.channel import _pauses, split_on_silence
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
    ("mic", "segment", "pauses"),
    [
        (SPEECH_PAUSE_SPEECH, Segment(start=0.0, end=3.0, text="x"), [(1.0, 2.0)]),
        (
            np.concatenate([_level(2.1, 7000), _level(0.9, 0)]),
            Segment(start=0.0, end=3.0, text="x"),
            [],
        ),
        (
            np.concatenate([_level(2.0, 7000), _level(1.0, 0)]),
            Segment(start=0.0, end=3.0, text="x"),
            [(2.0, 3.0)],
        ),
        (
            np.concatenate([_level(1.9, 7000), _level(1.1, 0)]),
            Segment(start=0.0, end=3.0, text="x"),
            [(1.9, 3.0)],
        ),
        (
            np.concatenate([_level(2, 0), _level(3, 7000)]),
            Segment(start=2.0, end=5.0, text="x"),
            [],
        ),
        (
            np.concatenate([_level(1.0, 7000), _level(0.5, 0), _level(1.5, 7000)]),
            Segment(start=0.0, end=3.0, text="x"),
            [],
        ),
    ],
    ids=[
        "middle",
        "short of the threshold",
        "at the end",
        "past the threshold",
        "late start",
        "a short silence between speech",
    ],
)
def test_a_pause_is_a_quiet_stretch_as_long_as_the_threshold(mic, segment, pauses):
    assert _pauses(segment, mic, SR, 1.0, None) == pauses


@pytest.mark.parametrize(
    ("end", "pauses"),
    [(1.0999375, []), (1.1, [(1.0, 1.1)]), (1.1000625, [(1.0, 1.1000625)])],
)
def test_a_pause_needs_one_full_window(end, pauses):
    segment = Segment(start=1.0, end=end, text="x")

    assert _pauses(segment, _level(2, 0), SR, 0.05, _level(2, 1000)) == pauses


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


def _texts(segment: Segment, mic: np.ndarray) -> list[str]:
    return [piece.text for piece in split_on_silence([segment], mic, SR, 1.0)]


def _said(*words: Word) -> Segment:
    return Segment(start=0.0, end=words[-1].end + 0.1, text="x", words=list(words), speaker="You")


def test_a_single_word_is_never_cut():
    segment = Segment(
        start=0.0, end=3.0, text="Hi.", words=[_word(0.1, 2.9, " Hi.")], speaker="You"
    )

    assert split_on_silence([segment], SPEECH_PAUSE_SPEECH, SR, 1.0) == [segment]


def test_the_widest_gap_inside_the_pause_is_the_cut():
    """Two words fell inside a 2 s pause; the longest silence between words wins."""
    mic = np.concatenate([_level(1.5, 7000), _level(2.0, 0), _level(1.5, 7000)])
    segment = _said(
        _word(0.1, 1.4, " Hello"),
        _word(1.6, 2.4, " um"),
        _word(2.6, 2.7, " er"),
        _word(3.6, 4.9, " Next."),
    )

    assert _texts(segment, mic) == ["Hello um er", "Next."]


@pytest.mark.parametrize(
    ("start", "end", "texts"),
    [
        (0.7, 2.1, ["One long", "Two."]),
        (0.9, 2.1, ["One", "long Two."]),
        (0.95, 2.4, ["One", "long Two."]),
    ],
    ids=["middle before the pause's", "at the pause's middle", "after it"],
)
def test_a_word_across_the_whole_pause_goes_by_its_middle(start, end, texts):
    """The pause runs from 1.0 to 2.0, so its middle is at 1.5."""
    segment = _said(_word(0.1, 0.6, " One"), _word(start, end, " long"), _word(2.5, 2.9, " Two."))

    assert _texts(segment, SPEECH_PAUSE_SPEECH) == texts


def test_a_pause_before_the_first_word_cuts_nothing():
    mic = np.concatenate([_level(1.2, 0), _level(2.8, 7000)])
    segment = Segment(
        start=0.0,
        end=4.0,
        text="First words.",
        words=[_word(1.3, 2.0, " First"), _word(2.1, 3.9, " words.")],
        speaker="You",
    )

    assert split_on_silence([segment], mic, SR, 1.0) == [segment]


def test_a_pause_after_the_last_word_cuts_nothing():
    mic = np.concatenate([_level(2.0, 7000), _level(1.2, 0)])
    segment = Segment(
        start=0.0,
        end=3.2,
        text="Both words.",
        words=[_word(0.1, 0.9, " Both"), _word(1.0, 1.9, " words.")],
        speaker="You",
    )

    assert split_on_silence([segment], mic, SR, 1.0) == [segment]


def test_two_pauses_in_one_gap_cut_once():
    """A click between two silences splits the pause in two; both pick the same gap."""
    mic = np.concatenate(
        [_level(2.0, 7000), _level(1.2, 0), _level(0.1, 7000), _level(1.2, 0), _level(2.0, 7000)]
    )
    segment = _said(_word(0.1, 1.9, " One."), _word(4.6, 6.4, " Two."))

    assert _texts(segment, mic) == ["One.", "Two."]
