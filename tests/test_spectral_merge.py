"""Spectral profiles and the merge of over-segmented speakers, on synthetic signals."""

import numpy as np
import pytest

from tapeback.models import DiarizationSegment
from tapeback.speaker_merge import (
    _similarity,
    _speaker_spectral_profile,
    merge_similar_speakers,
)

SR = 16000


def _tones(seconds: float, weight: float) -> np.ndarray:
    """400 Hz plus some 1200 Hz: the weight sets how alike two of these look."""
    t = np.arange(int(seconds * SR)) / SR
    return (np.sin(2 * np.pi * 400 * t) + weight * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)


def _impulse() -> np.ndarray:
    """One sample at 1024 in 3073: two frames, the first sees it at its centre."""
    samples = np.zeros(3073, dtype=np.float32)
    samples[1024] = 1.0
    return samples


def _segment(speaker: str, start: float, end: float) -> DiarizationSegment:
    return DiarizationSegment(speaker=speaker, start=start, end=end)


def _speakers(segments: list[DiarizationSegment]) -> list[str]:
    return [segment.speaker for segment in segments]


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([9.999999999999999e-11], [1.0], None),
        ([1e-10], [1.0], 1.0),
        ([1.0000000000000002e-10], [1.0], 1.0),
        ([1.0], [9.999999999999999e-11], None),
        ([1.0], [1e-10], 1.0),
        ([1.0], [1.0000000000000002e-10], 1.0),
    ],
)
def test_a_profile_is_silent_below_the_floor_on_either_side(a, b, expected):
    assert _similarity(np.array(a), np.array(b)) == expected


def test_frames_overlap_by_half_from_the_segment_start():
    profile = _speaker_spectral_profile(
        _impulse(), SR, [_segment("SPEAKER_00", 0.0, 1.0)], "SPEAKER_00"
    )

    assert profile == pytest.approx(np.full(501, 0.5), abs=1e-6)


def test_a_frame_that_would_run_past_the_segment_is_not_taken():
    """3071 samples hold the frame at 0; one at 1024 would be a sample short."""
    profile = _speaker_spectral_profile(
        _impulse()[:3071], SR, [_segment("SPEAKER_00", 0.0, 1.0)], "SPEAKER_00"
    )

    assert profile == pytest.approx(np.full(501, 1.0), abs=1e-5)


@pytest.mark.parametrize(
    ("sample_rate", "bins"),
    [(16000, 501), (8000, 1000), (4000, 974), (204800, 40), (409600, 20)],
)
def test_a_profile_spans_the_voice_band(sample_rate, bins):
    """8 kHz puts the band's top at Nyquist and 4 kHz clips it there; above 204.8 kHz
    the lowest bin stays 1, so the DC bin never counts."""
    voiced = _speaker_spectral_profile(
        _impulse(), sample_rate, [_segment("SPEAKER_00", 0.0, 1.0)], "SPEAKER_00"
    )
    silent = _speaker_spectral_profile(_impulse(), sample_rate, [], "SPEAKER_00")

    assert len(voiced) == bins
    assert silent.tolist() == [0.0] * bins


@pytest.mark.parametrize(
    ("sample_rate", "profile"),
    [(199, [0.0]), (10_000_000, [0.0]), (8_192_000, [0.5])],
    ids=["Nyquist under 100 Hz", "one bin wider than the band", "one bin fits"],
)
def test_a_band_of_at_most_one_bin(sample_rate, profile):
    computed = _speaker_spectral_profile(
        _impulse(), sample_rate, [_segment("SPEAKER_00", 0.0, 1.0)], "SPEAKER_00"
    )

    assert computed.tolist() == pytest.approx(profile, abs=1e-6)


@pytest.mark.parametrize(
    ("threshold", "speakers"),
    [
        (-0.01, ["SPEAKER_00", "SPEAKER_01"]),
        (0.0, ["SPEAKER_00", "SPEAKER_01"]),
        (0.01, ["SPEAKER_00", "SPEAKER_00"]),
    ],
)
def test_a_threshold_of_zero_or_below_turns_merging_off(threshold, speakers):
    audio = np.concatenate([_tones(1, 0.2), _tones(1, 0.2)])
    segments = [_segment("SPEAKER_00", 0.0, 1.0), _segment("SPEAKER_01", 1.0, 2.0)]

    merged = merge_similar_speakers(segments, audio, SR, similarity_threshold=threshold)

    assert _speakers(merged) == speakers


@pytest.mark.parametrize(
    ("first_total", "second_end", "second"),
    [
        (1.0, 1.1875, "SPEAKER_00"),
        (76.0, 16.0, "SPEAKER_01"),
        (76.0, 15.5, "SPEAKER_00"),
        (100.0, 16.5, "SPEAKER_01"),
        (10.0, 3.0, "SPEAKER_01"),
        (10.0, 2.9, "SPEAKER_00"),
        (10.0, 3.5, "SPEAKER_01"),
    ],
)
def test_a_minor_speaker_has_under_15_seconds_and_under_a_fifth(first_total, second_end, second):
    """The pair's profiles sit at 0.94: under the standard threshold, over the minor one.
    Speech past the end of the audio adds time but no frames."""
    audio = np.concatenate([_tones(1, 0.2), _tones(1, 0.6)])
    segments = [_segment("SPEAKER_00", 0.0, 1.0)]
    if first_total > 1.0:
        segments.append(_segment("SPEAKER_00", 10.0, 9.0 + first_total))
    segments.append(_segment("SPEAKER_01", 1.0, second_end))

    merged = merge_similar_speakers(segments, audio, SR, similarity_threshold=0.96)

    assert merged[-1].speaker == second


def test_a_merged_pair_keeps_the_first_label():
    audio = np.concatenate([_tones(1, 0.2), _tones(1, 0.35), _tones(1, 1.4)])
    segments = [
        _segment("SPEAKER_00", 0.0, 1.0),
        _segment("SPEAKER_01", 1.0, 2.0),
        _segment("SPEAKER_02", 2.0, 3.0),
    ]

    merged = merge_similar_speakers(segments, audio, SR, similarity_threshold=0.96)

    assert _speakers(merged) == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_02"]


def test_the_threshold_itself_merges():
    audio = np.concatenate([_tones(1, 0.2), _tones(1, 0.6)])
    segments = [_segment("SPEAKER_00", 0.0, 1.0), _segment("SPEAKER_01", 1.0, 2.0)]
    # Measured rather than written down: its last bits depend on the BLAS build.
    at = _similarity(
        _speaker_spectral_profile(audio, SR, segments, "SPEAKER_00"),
        _speaker_spectral_profile(audio, SR, segments, "SPEAKER_01"),
    )
    assert at is not None

    def merged_at(threshold: float) -> list[str]:
        return _speakers(merge_similar_speakers(segments, audio, SR, threshold))

    assert merged_at(at) == ["SPEAKER_00", "SPEAKER_00"]
    assert merged_at(float(np.nextafter(at, 2.0))) == ["SPEAKER_00", "SPEAKER_01"]
    assert merged_at(float(np.nextafter(at, 0.0))) == ["SPEAKER_00", "SPEAKER_00"]
