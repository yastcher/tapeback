"""Regression: a pause cut ran through a word.

Pause splitting cut at the middle of a pause and gave each piece the words within
0.05 s of its bounds, so a word across the cut went into both pieces when short and
into neither when longer: the note repeated it or lost it.
"""

import numpy as np
import pytest

from tapeback.channel import split_on_silence
from tapeback.models import Segment, Word

SR = 16000


def _word(start: float, end: float, text: str) -> Word:
    return Word(start=start, end=end, word=text, probability=0.9)


@pytest.mark.parametrize(
    ("start", "end"), [(1.47, 1.53), (1.4, 1.6)], ids=["near the cut", "across it"]
)
def test_a_word_in_the_middle_of_a_pause_lands_in_one_piece(start, end):
    # Speech, a second of silence, speech: the pause runs from 1.0 to 2.0.
    speech = np.full(SR, 7000, dtype=np.int16)
    mic = np.concatenate([speech, np.zeros(SR, dtype=np.int16), speech])
    words = [_word(0.1, 0.9, " One."), _word(start, end, " um"), _word(2.1, 2.9, " Two.")]
    segment = Segment(start=0.0, end=3.0, text="One. um Two.", words=words, speaker="You")

    pieces = split_on_silence([segment], mic, SR, pause_threshold=1.0)

    assert [piece.text for piece in pieces] == ["One.", "um Two."]
