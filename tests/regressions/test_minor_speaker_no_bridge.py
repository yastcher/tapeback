"""Regression: a small cluster merged two speakers into one.

The spectral merge's minor-speaker absorption merged pair by pair, so a short cluster
that resembled BOTH real speakers joined them: A with the minor, the minor with B, so A
with B. On recordings with two remote speakers, notes showed one speaker for this reason.
A fragment now joins at most one speaker, and only one it clearly belongs to: a fragment
as close to two voices as to each other is two people at once and joins neither.
"""

from tapeback.models import Diarization, DiarizationSegment
from tapeback.speaker_merge import merge_speakers_by_voice


def test_a_fragment_close_to_two_speakers_joins_neither_and_bridges_nothing():
    voices = {
        "SPEAKER_00": (1.0, 0.0, 0.0),
        "SPEAKER_01": (0.0, 1.0, 0.0),
        "SPEAKER_02": (0.7, 0.7, 0.141),
    }
    segments = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=50.0),
        DiarizationSegment(speaker="SPEAKER_01", start=50.0, end=100.0),
        DiarizationSegment(speaker="SPEAKER_02", start=100.0, end=101.0),
    ]

    merged = merge_speakers_by_voice(Diarization(segments=segments, voices=voices))

    assert [(segment.speaker, segment.start) for segment in merged] == [
        ("SPEAKER_00", 0.0),
        ("SPEAKER_01", 50.0),
    ]
