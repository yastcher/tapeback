"""Regression: a minor cluster merged two real speakers into one.

Minor-speaker absorption compares a short cluster (echo, cross-talk) with each other
speaker at a relaxed threshold, and merges pair by pair into one group. A 1-second
cluster that resembles BOTH real speakers therefore joined them: A with the minor,
the minor with B, so A with B — although A and B were never close enough to merge
on their own. On four real recordings with two remote speakers each, the note showed
one speaker in two of them for exactly this reason.

The minor cluster must join the one speaker it resembles most, never bridge two.
"""

import numpy as np

from tapeback.models import DiarizationSegment
from tapeback.speaker_merge import _speaker_spectral_profile, merge_similar_speakers

SR = 16000


def _two_tones(seconds: float, overtone: float) -> np.ndarray:
    """400 Hz plus a 1200 Hz overtone of the given weight: how much of it a "voice"
    carries sets how alike two of them look to the spectral comparison."""
    t = np.arange(int(seconds * SR)) / SR
    return (np.sin(2 * np.pi * 400 * t) + overtone * np.sin(2 * np.pi * 1200 * t)).astype(
        np.float32
    )


def _cosine(audio, segments, a: str, b: str) -> float:
    pa = _speaker_spectral_profile(audio, SR, segments, a)
    pb = _speaker_spectral_profile(audio, SR, segments, b)
    return float(np.dot(pa, pb) / (np.linalg.norm(pa) * np.linalg.norm(pb)))


def test_a_minor_cluster_joins_one_speaker_and_never_bridges_two():
    # A speaks 20 s, B 10 s, the minor cluster 1 s (under 15 s and 20% of A).
    audio = np.concatenate([_two_tones(20, 0.2), _two_tones(10, 0.6), _two_tones(1, 0.5)])
    segments = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=20.0),
        DiarizationSegment(speaker="SPEAKER_01", start=20.0, end=30.0),
        DiarizationSegment(speaker="SPEAKER_02", start=30.0, end=31.0),
    ]
    # The scenario: A and B apart at the default 0.96, the minor within the relaxed
    # 0.92 of both, and closer to B.
    assert 0.92 < _cosine(audio, segments, "SPEAKER_00", "SPEAKER_01") < 0.96
    assert _cosine(audio, segments, "SPEAKER_02", "SPEAKER_00") > 0.92
    assert _cosine(audio, segments, "SPEAKER_02", "SPEAKER_00") < _cosine(
        audio, segments, "SPEAKER_02", "SPEAKER_01"
    )

    merged = merge_similar_speakers(segments, audio, SR, similarity_threshold=0.96)

    assert [s.speaker for s in merged] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01"]
