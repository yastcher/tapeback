"""Regression: real speakers merged in chains.

`merge_similar_speakers` joined the clusters of any two real speakers whose profiles
reached the threshold, so A like B and B like C put A with C, whom A did not resemble.
"""

import numpy as np

from tapeback.models import DiarizationSegment
from tapeback.speaker_merge import merge_similar_speakers

SR = 16000


def _tones(seconds: float, weight: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (np.sin(2 * np.pi * 400 * t) + weight * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)


def test_a_speaker_alike_to_one_member_and_not_another_stays_apart():
    # At 0.96, 00-01 (0.982), 00-02 (0.979) and 01-03 (0.976) reach it pair by pair;
    # 01-02 (0.923), 00-03 (0.917) and 02-03 (0.817) do not.
    audio = np.concatenate([_tones(1, weight) for weight in (0.6, 0.9, 0.35, 1.4)])
    segments = [
        DiarizationSegment(speaker=f"SPEAKER_0{index}", start=float(index), end=index + 1.0)
        for index in range(4)
    ]

    merged = merge_similar_speakers(segments, audio, SR, similarity_threshold=0.96)

    assert [segment.speaker for segment in merged] == [
        "SPEAKER_00",
        "SPEAKER_00",
        "SPEAKER_02",
        "SPEAKER_03",
    ]
