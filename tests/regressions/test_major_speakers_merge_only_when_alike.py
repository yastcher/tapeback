"""Regression: speakers merged in chains.

The spectral merge joined the clusters of any two speakers whose profiles reached the
threshold, so A like B and B like C put A with C, whom A did not resemble. The voice merge
keeps the guarantee: a speaker joins a person only through that person's largest part.
"""

from tapeback.models import Diarization, DiarizationSegment
from tapeback.speaker_merge import merge_speakers_by_voice


def test_a_speaker_alike_to_one_part_of_a_person_and_not_another_stays_apart():
    # 0.866 between neighbours, 0.5 between the ends: the first two are one person, the
    # last two would be too, but the first and the last are not.
    voices = {
        "SPEAKER_00": (1.0, 0.0),
        "SPEAKER_01": (0.866, 0.5),
        "SPEAKER_02": (0.5, 0.866),
    }
    segments = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=30.0),
        DiarizationSegment(speaker="SPEAKER_01", start=30.0, end=50.0),
        DiarizationSegment(speaker="SPEAKER_02", start=50.0, end=70.0),
    ]

    merged = merge_speakers_by_voice(Diarization(segments=segments, voices=voices))

    assert [segment.speaker for segment in merged] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_02"]
