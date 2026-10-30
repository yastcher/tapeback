"""Regression tests for channel loading: integer overflow and memory footprint."""

import wave

import numpy as np
import pytest

from tapeback.channel import (
    _rms_for_range,
    classify_segment_by_channel,
    identify_user_speaker,
    load_stereo_channels,
)
from tapeback.models import DiarizationSegment


def _write_stereo(path, left: np.ndarray, right: np.ndarray, sample_rate: int = 48000):
    interleaved = np.empty(left.size + right.size, dtype=np.int16)
    interleaved[0::2] = left
    interleaved[1::2] = right
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved.tobytes())


def test_rms_does_not_overflow_on_loud_audio(tmp_path):
    """RMS must widen before squaring, or int16 samples wrap around.

    A sample near full scale squares to ~1.07e9, far outside int16. Squaring in the
    sample dtype silently wraps and reports a *quiet* channel for the loudest audio,
    which would invert every mic-vs-monitor decision built on it.
    """
    path = tmp_path / "loud.wav"
    loud = np.full(48000, 32000, dtype=np.int16)
    quiet = np.full(48000, 10, dtype=np.int16)
    _write_stereo(path, loud, quiet)

    mic, monitor, sample_rate = load_stereo_channels(path)

    assert _rms_for_range(0.0, 1.0, mic, sample_rate) == pytest.approx(32000.0)
    assert _rms_for_range(0.0, 1.0, monitor, sample_rate) == pytest.approx(10.0)


def test_channel_classification_survives_full_scale_audio(tmp_path):
    """The same overflow would flip which channel is judged louder."""
    path = tmp_path / "loud.wav"
    loud = np.full(48000, 32000, dtype=np.int16)
    quiet = np.full(48000, 100, dtype=np.int16)
    _write_stereo(path, loud, quiet)

    mic, monitor, sample_rate = load_stereo_channels(path)

    assert classify_segment_by_channel(0.0, 1.0, mic, monitor, sample_rate) == "mic"
    assert classify_segment_by_channel(0.0, 1.0, monitor, mic, sample_rate) == "monitor"


def test_user_identification_survives_full_scale_audio(tmp_path):
    """The accumulating variant overflows harder than the averaging ones.

    `identify_user_speaker` sums squares across every segment of a speaker rather
    than averaging one slice, so int16 does not merely wrap — the total goes negative
    and its square root comes back complex, which fails as a TypeError rather than a
    wrong answer. Widening is required in all four squaring sites, not just the
    obvious ones.
    """
    path = tmp_path / "loud.wav"
    loud = np.full(48000, 32000, dtype=np.int16)
    quiet = np.full(48000, 100, dtype=np.int16)
    _write_stereo(path, loud, quiet)

    segments = [
        DiarizationSegment(start=0.0, end=0.5, speaker="SPEAKER_00"),
        DiarizationSegment(start=0.5, end=1.0, speaker="SPEAKER_01"),
    ]

    # Must return an answer or None — never raise.
    result = identify_user_speaker(segments, path)
    assert result is None or result.startswith("SPEAKER_")


def test_channels_are_stored_as_int16(tmp_path):
    """Keeping the samples as int16 halves what the pipeline holds for a long recording.

    A 37-minute stereo recording at 48 kHz is 434 MB of int16. Converting the whole
    thing to float32 up front doubled that to 869 MB and, because the raw buffer was
    still alive during the conversion, peaked at 1.3 GB — on a machine with 1.4 GB
    free that is the difference between running and swapping.
    """
    path = tmp_path / "s.wav"
    _write_stereo(path, np.zeros(1000, dtype=np.int16), np.zeros(1000, dtype=np.int16))

    mic, monitor, _ = load_stereo_channels(path)

    assert mic.dtype == np.int16
    assert monitor.dtype == np.int16
    # Contiguous, not a strided view into the interleaved buffer: the RMS loop walks
    # these tens of thousands of times, and a stride-2 view roughly halves that.
    assert mic.flags["C_CONTIGUOUS"]
    assert monitor.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("half_frame", [0, 2], ids=["at a frame", "inside a frame"])
def test_a_file_cut_inside_a_frame_keeps_its_whole_frames(tmp_path, half_frame):
    """A recording a killed process left behind can end anywhere, half a frame included.

    Bug: the half frame's single sample made the chunk odd, and reshaping it into
    pairs raised ValueError, so the whole recording could not be loaded.
    """
    path = tmp_path / "cut.wav"
    _write_stereo(path, np.arange(1000, dtype=np.int16), np.arange(1000, 2000, dtype=np.int16))
    # The 44-byte header, 600 whole frames of 4 bytes, and maybe half of the next one.
    path.write_bytes(path.read_bytes()[: 44 + 600 * 4 + half_frame])

    mic, monitor, sample_rate = load_stereo_channels(path)

    assert mic.tolist() == list(range(600))
    assert monitor.tolist() == list(range(1000, 1600))
    assert sample_rate == 48000


def test_channels_are_split_correctly(tmp_path):
    """Left is the mic, right is the monitor — the split must survive the rework."""
    path = tmp_path / "s.wav"
    left = np.arange(100, dtype=np.int16)
    right = np.arange(100, 200, dtype=np.int16)
    _write_stereo(path, left, right)

    mic, monitor, sample_rate = load_stereo_channels(path)

    assert sample_rate == 48000
    assert np.array_equal(mic, left)
    assert np.array_equal(monitor, right)
