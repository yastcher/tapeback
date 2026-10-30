"""Which diarized speaker is the user: the one whose mic stands out from the monitor most."""

import wave
from pathlib import Path

import numpy as np
import pytest

from tapeback.channel import identify_user_speaker
from tapeback.models import DiarizationSegment

SR = 16000


def _stereo_arrays(path: Path, mic: np.ndarray, monitor: np.ndarray) -> Path:
    interleaved = np.empty(mic.size + monitor.size, dtype=np.int16)
    interleaved[0::2] = mic
    interleaved[1::2] = monitor
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(interleaved.tobytes())
    return path


def _stereo(path: Path, regions: list[tuple[float, int, int]]) -> Path:
    """Constant levels per (seconds, mic, monitor) region."""

    def channel(index: int) -> np.ndarray:
        return np.concatenate(
            [np.full(round(region[0] * SR), region[index], dtype=np.int16) for region in regions]
        )

    return _stereo_arrays(path, channel(1), channel(2))


def _said(speaker: str, start: float, end: float) -> DiarizationSegment:
    return DiarizationSegment(speaker=speaker, start=start, end=end)


@pytest.mark.parametrize(
    ("mic", "user"), [(1999, None), (2000, "SPEAKER_00"), (2001, "SPEAKER_00")]
)
def test_the_user_needs_twice_the_others_ratio(tmp_path, mic, user):
    wav = _stereo(tmp_path / "s.wav", [(1.0, mic, 1000), (1.0, 1000, 1000)])
    segments = [_said("SPEAKER_00", 0.0, 1.0), _said("SPEAKER_01", 1.0, 2.0)]

    assert identify_user_speaker(segments, wav) == user


def test_the_ratio_is_of_energies_whatever_the_lengths(tmp_path):
    wav = _stereo(tmp_path / "s.wav", [(2.0, 3600, 1000), (0.5, 3000, 1500)])
    segments = [_said("SPEAKER_00", 0.0, 2.0), _said("SPEAKER_01", 2.0, 2.5)]

    assert identify_user_speaker(segments, wav) is None


def test_with_a_silent_monitor_the_user_is_the_louder_mic(tmp_path):
    wav = _stereo(tmp_path / "s.wav", [(3.0, 2000, 0), (1.0, 6000, 0)])
    segments = [
        _said("SPEAKER_01", 0.0, 1.0),
        _said("SPEAKER_01", 1.0, 2.0),
        _said("SPEAKER_01", 2.0, 3.0),
        _said("SPEAKER_00", 3.0, 4.0),
    ]

    assert identify_user_speaker(segments, wav) == "SPEAKER_00"


def test_energy_adds_up_over_every_segment_of_a_speaker(tmp_path):
    wav = _stereo(
        tmp_path / "s.wav",
        [(1.0, 6000, 1000), (1.0, 1500, 3000), (1.0, 1000, 1000), (1.0, 1500, 300)],
    )
    segments = [
        _said("SPEAKER_00", 0.0, 1.0),
        _said("SPEAKER_01", 1.0, 2.0),
        _said("SPEAKER_00", 2.0, 3.0),
        _said("SPEAKER_01", 3.0, 4.0),
    ]

    assert identify_user_speaker(segments, wav) == "SPEAKER_00"


def test_an_empty_segment_does_not_end_the_speakers_count(tmp_path):
    wav = _stereo(tmp_path / "s.wav", [(1.0, 6000, 300), (1.0, 300, 6000)])
    segments = [
        _said("SPEAKER_00", 0.1, 0.10001),
        _said("SPEAKER_00", 0.1, 1.0),
        _said("SPEAKER_01", 1.0, 2.0),
    ]

    assert identify_user_speaker(segments, wav) == "SPEAKER_00"


def test_a_single_frame_at_the_start_counts(tmp_path):
    mic = np.array([1000] + [100] * 15999, dtype=np.int16)
    monitor = np.array([0] + [1000] * 15999, dtype=np.int16)
    wav = _stereo_arrays(tmp_path / "s.wav", mic, monitor)
    segments = [_said("SPEAKER_00", 0.0, 0.0000625), _said("SPEAKER_01", 0.0000625, 1.0)]

    assert identify_user_speaker(segments, wav) == "SPEAKER_00"
