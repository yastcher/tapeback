"""Diarizer tests — speaker identification, assignment, channel analysis, silence splitting."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tapeback.diarizer import (
    Diarizer,
    assign_speakers,
    classify_segment_by_channel,
    identify_user_speaker,
    load_stereo_channels,
    merge_channel_segments,
    merge_similar_speakers,
    split_on_silence,
)
from tapeback.models import DiarizationSegment, Segment, Word
from tapeback.settings import Settings
from tests.fixtures import create_stereo_wav, create_stereo_wav_segments, requires_pyannote

# --- Diarizer init / diarize ---


@requires_pyannote
def test_diarizer_init_without_token(tmp_vault):
    """Diarizer should raise RuntimeError when hf_token is empty."""

    settings = Settings(vault_path=tmp_vault, hf_token="")

    with pytest.raises(RuntimeError, match="HuggingFace token required"):
        Diarizer(settings)


@requires_pyannote
def test_diarizer_clustering_threshold(tmp_vault):
    """Diarizer should apply custom clustering_threshold, skip when None."""

    # With threshold — should call instantiate
    settings_with = Settings(
        vault_path=tmp_vault, hf_token="hf_fake", device="cpu", clustering_threshold=0.85
    )
    with patch("pyannote.audio.Pipeline") as mock_cls:
        mock_pipeline = MagicMock()
        mock_cls.from_pretrained.return_value = mock_pipeline
        mock_pipeline.parameters.return_value = {
            "clustering": {"method": "centroid", "min_cluster_size": 12, "threshold": 0.7},
            "segmentation": {"min_duration_off": 0.0},
        }
        Diarizer(settings_with)

    mock_pipeline.instantiate.assert_called_once()
    assert mock_pipeline.instantiate.call_args[0][0]["clustering"]["threshold"] == 0.85

    # Without threshold — should not touch params
    settings_without = Settings(
        vault_path=tmp_vault, hf_token="hf_fake", device="cpu", clustering_threshold=None
    )
    with patch("pyannote.audio.Pipeline") as mock_cls:
        mock_pipeline = MagicMock()
        mock_cls.from_pretrained.return_value = mock_pipeline
        Diarizer(settings_without)

    mock_pipeline.parameters.assert_not_called()
    mock_pipeline.instantiate.assert_not_called()


@requires_pyannote
def test_diarize_returns_segments(tmp_vault):
    """Diarizer.diarize should return list of DiarizationSegment."""

    settings = Settings(vault_path=tmp_vault, hf_token="hf_fake", device="cpu")

    mock_turn_1 = MagicMock()
    mock_turn_1.start = 0.0
    mock_turn_1.end = 3.0
    mock_turn_2 = MagicMock()
    mock_turn_2.start = 3.5
    mock_turn_2.end = 7.0

    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = [
        (mock_turn_1, None, "SPEAKER_00"),
        (mock_turn_2, None, "SPEAKER_01"),
    ]

    with patch("pyannote.audio.Pipeline") as mock_cls:
        mock_pipeline = MagicMock()
        mock_cls.from_pretrained.return_value = mock_pipeline
        mock_pipeline.return_value = mock_annotation

        diarizer = Diarizer(settings)
        result = diarizer.diarize(Path("/fake/audio.wav"))

    assert len(result) == 2
    assert result[0].speaker == "SPEAKER_00"
    assert result[0].start == 0.0
    assert result[0].end == 3.0
    assert result[1].speaker == "SPEAKER_01"


# --- identify_user_speaker ---


@pytest.mark.parametrize(
    "wav_setup,dsegs,expected",
    [
        pytest.param(
            {"type": "uniform", "left": 0.8, "right": 0.05},
            [DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=1.0)],
            None,
            id="single_speaker",
        ),
        pytest.param(
            {"type": "uniform", "left": 0.5, "right": 0.5},
            [
                DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=0.5),
                DiarizationSegment(speaker="SPEAKER_01", start=0.5, end=1.0),
            ],
            None,
            id="ambiguous",
        ),
        pytest.param(
            {"type": "segments", "spec": [(1.0, 0.8, 0.05), (1.0, 0.05, 0.8)]},
            [
                DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=1.0),
                DiarizationSegment(speaker="SPEAKER_01", start=1.0, end=2.0),
            ],
            "SPEAKER_00",
            id="distinct_speakers",
        ),
    ],
)
def test_identify_user_speaker(tmp_path, wav_setup, dsegs, expected):
    """identify_user_speaker: single→None, ambiguous→None, distinct→mic speaker."""
    wav_path = tmp_path / "stereo.wav"
    if wav_setup["type"] == "uniform":
        create_stereo_wav(wav_path, 1.0, 16000, wav_setup["left"], wav_setup["right"])
    else:
        create_stereo_wav_segments(wav_path, 16000, wav_setup["spec"])

    assert identify_user_speaker(dsegs, wav_path) == expected


# --- classify_segment_by_channel ---


@pytest.mark.parametrize(
    "left_amp,right_amp,expected",
    [
        pytest.param(0.8, 0.1, "mic", id="mic_dominant"),
        pytest.param(0.1, 0.8, "monitor", id="monitor_dominant"),
        pytest.param(0.5, 0.5, None, id="ambiguous"),
    ],
)
def test_classify_segment_by_channel(tmp_path, left_amp, right_amp, expected):
    """Channel classification based on RMS energy ratio."""
    wav_path = tmp_path / "stereo.wav"
    create_stereo_wav(wav_path, 2.0, 16000, left_amp, right_amp)
    mic, monitor, sr = load_stereo_channels(wav_path)
    assert classify_segment_by_channel(0.0, 2.0, mic, monitor, sr) == expected


# --- assign_speakers ---


@pytest.mark.parametrize(
    "stereo_spec,user_speaker,expected_labels",
    [
        pytest.param(
            [(1.0, 0.1, 0.8), (1.0, 0.8, 0.1)],
            "SPEAKER_00",
            ["Speaker 1", "You"],
            id="channel_overrides_pyannote",
        ),
        pytest.param(
            None,
            "SPEAKER_00",
            ["You", "Speaker 1"],
            id="pyannote_with_user_speaker",
        ),
        pytest.param(
            None,
            None,
            ["Speaker 1", "Speaker 2"],
            id="no_user_speaker",
        ),
    ],
)
def test_assign_speakers_labeling(tmp_path, stereo_spec, user_speaker, expected_labels):
    """assign_speakers: channel override, pyannote assignment, no-user numbering."""
    wav_path = None
    if stereo_spec is not None:
        wav_path = tmp_path / "stereo.wav"
        create_stereo_wav_segments(wav_path, 16000, stereo_spec)

    segments = [
        Segment(start=0.0, end=1.0, text="First", words=None),
        Segment(start=1.0, end=2.0, text="Second", words=None),
    ]
    dsegs = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=1.0),
        DiarizationSegment(speaker="SPEAKER_01", start=1.0, end=2.0),
    ]

    result = assign_speakers(segments, dsegs, user_speaker=user_speaker, stereo_wav=wav_path)
    assert [s.speaker for s in result] == expected_labels
    # Input not mutated
    assert all(s.speaker is None for s in segments)


def test_assign_speakers_ambiguous_falls_back_to_pyannote(tmp_path):
    """Ambiguous channel energy should fall back to pyannote + user_speaker."""
    wav_path = tmp_path / "stereo.wav"
    create_stereo_wav(wav_path, 2.0, 16000, 0.5, 0.5)

    segments = [
        Segment(start=0.0, end=1.0, text="A", words=None),
        Segment(start=1.0, end=2.0, text="B", words=None),
    ]
    dsegs = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=1.0),
        DiarizationSegment(speaker="SPEAKER_01", start=1.0, end=2.0),
    ]

    result = assign_speakers(segments, dsegs, user_speaker="SPEAKER_00", stereo_wav=wav_path)
    assert result[0].speaker == "You"
    assert result[1].speaker == "Speaker 1"


def test_assign_speakers_with_words_and_gaps():
    """Word-level voting and gap handling in speaker assignment."""
    segments = [
        Segment(
            start=0.0,
            end=2.0,
            text="Hello there",
            words=[Word(start=0.0, end=1.0, word="Hello", probability=0.9)],
        ),
        # Word in gap between diarization segments → nearest speaker
        Segment(
            start=2.0,
            end=4.0,
            text="Gap word",
            words=[Word(start=2.5, end=3.0, word="Gap", probability=0.9)],
        ),
        Segment(
            start=5.0,
            end=7.0,
            text="Hi back",
            words=[Word(start=5.0, end=6.0, word="Hi", probability=0.9)],
        ),
    ]

    dsegs = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.0),
        # Gap at 2.0-4.5
        DiarizationSegment(speaker="SPEAKER_01", start=4.5, end=7.0),
    ]

    result = assign_speakers(segments, dsegs, user_speaker="SPEAKER_00")
    assert result[0].speaker == "You"
    assert result[1].speaker == "You"  # gap word closer to SPEAKER_00
    assert result[2].speaker == "Speaker 1"

    # Speakers numbered by appearance, not by pyannote label
    segments_reversed = [
        Segment(start=0.0, end=2.0, text="A", words=None),
        Segment(start=3.0, end=5.0, text="B", words=None),
    ]
    dsegs_reversed = [
        DiarizationSegment(speaker="SPEAKER_02", start=0.0, end=2.5),
        DiarizationSegment(speaker="SPEAKER_00", start=2.5, end=5.0),
    ]
    result2 = assign_speakers(segments_reversed, dsegs_reversed, user_speaker=None)
    assert result2[0].speaker == "Speaker 1"  # SPEAKER_02 appeared first
    assert result2[1].speaker == "Speaker 2"


# --- merge_channel_segments ---


@pytest.mark.parametrize(
    "mic_segs,mon_segs,expected_starts",
    [
        pytest.param(
            [
                Segment(start=0.0, end=3.0, text="A", speaker="You"),
                Segment(start=5.0, end=7.0, text="B", speaker="You"),
            ],
            [
                Segment(start=2.0, end=4.0, text="C", speaker="Speaker 1"),
                Segment(start=6.0, end=8.0, text="D", speaker="Speaker 1"),
            ],
            [0.0, 2.0, 5.0, 6.0],
            id="interleaved",
        ),
        pytest.param(
            [Segment(start=0.0, end=5.0, text="A", speaker="You")],
            [Segment(start=2.0, end=6.0, text="B", speaker="Speaker 1")],
            [0.0, 2.0],
            id="overlapping",
        ),
    ],
)
def test_merge_channel_segments(mic_segs, mon_segs, expected_starts):
    """Segments from both channels merged and sorted by start time."""
    result = merge_channel_segments(mic_segs, mon_segs)
    assert [s.start for s in result] == expected_starts


# --- split_on_silence ---


def test_split_on_silence_at_real_pause(stereo_wav):
    """Segment spanning speech-silence-speech should be split at the silence gap."""
    wav_path = stereo_wav([(1.0, 0.8, 0.0), (1.5, 0.0, 0.0), (1.5, 0.8, 0.0)])
    mic_raw, _monitor_raw, sr = load_stereo_channels(wav_path)

    seg = Segment(
        start=0.0,
        end=4.0,
        text="hello world",
        words=[
            Word(start=0.0, end=0.8, word="hello", probability=0.9),
            Word(start=2.7, end=3.5, word="world", probability=0.9),
        ],
        speaker="You",
    )

    result = split_on_silence([seg], mic_raw, sr, pause_threshold=1.0)
    assert len(result) == 2
    assert result[0].text == "hello"
    assert result[1].text == "world"
    assert result[0].speaker == "You"


def test_split_on_silence_no_pause(stereo_wav):
    """Continuous speech should not be split."""
    wav_path = stereo_wav([(3.0, 0.8, 0.0)])
    mic_raw, _monitor_raw, sr = load_stereo_channels(wav_path)

    seg = Segment(
        start=0.0,
        end=3.0,
        text="continuous speech here",
        words=[
            Word(start=0.0, end=1.0, word="continuous", probability=0.9),
            Word(start=1.0, end=2.0, word="speech", probability=0.9),
            Word(start=2.0, end=3.0, word="here", probability=0.9),
        ],
    )
    assert len(split_on_silence([seg], mic_raw, sr, pause_threshold=1.0)) == 1


def test_split_on_silence_short_pause_ignored(stereo_wav):
    """Silence shorter than pause_threshold should not cause a split."""
    wav_path = stereo_wav([(1.0, 0.8, 0.0), (0.5, 0.0, 0.0), (1.5, 0.8, 0.0)])
    mic_raw, _monitor_raw, sr = load_stereo_channels(wav_path)

    seg = Segment(start=0.0, end=3.0, text="no split", speaker="You")
    assert len(split_on_silence([seg], mic_raw, sr, pause_threshold=1.0)) == 1


def test_split_on_silence_monitor_relative(stereo_wav):
    """When mic is quiet but monitor is loud, detect pause via monitor ratio."""
    wav_path = stereo_wav([(2.0, 0.8, 0.0), (2.0, 0.05, 0.8), (2.0, 0.8, 0.0)])
    mic_raw, monitor_raw, sr = load_stereo_channels(wav_path)

    seg = Segment(
        start=0.0,
        end=6.0,
        text="hello world",
        words=[
            Word(start=0.5, end=1.5, word="hello", probability=0.9),
            Word(start=4.5, end=5.5, word="world", probability=0.9),
        ],
        speaker="You",
    )

    result = split_on_silence([seg], mic_raw, sr, pause_threshold=1.0, monitor_samples=monitor_raw)
    assert len(result) == 2
    assert result[0].text == "hello"
    assert result[1].text == "world"


# --- merge_similar_speakers ---


def _voice_signal(duration: float, sr: int, fundamental: float) -> np.ndarray:
    """Generate a synthetic voice signal with harmonics."""
    t = np.linspace(0, duration, int(duration * sr), dtype=np.float32)
    return (
        np.sin(2 * np.pi * fundamental * t)
        + 0.5 * np.sin(2 * np.pi * fundamental * 2 * t)
        + 0.25 * np.sin(2 * np.pi * fundamental * 3 * t)
    ) * 10000


def test_merge_similar_speakers_same_voice():
    """Two segments from the same voice should be merged into one speaker."""
    sr = 16000
    audio = _voice_signal(5.0, sr, fundamental=200.0)

    segments = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.0),
        DiarizationSegment(speaker="SPEAKER_01", start=2.5, end=5.0),
    ]

    merged = merge_similar_speakers(segments, audio, sr)
    speakers = {seg.speaker for seg in merged}
    assert len(speakers) == 1


def test_merge_similar_speakers_different_voices():
    """Two spectrally distinct speakers should NOT be merged."""
    sr = 16000
    n_samples = int(5.0 * sr)
    audio = np.zeros(n_samples, dtype=np.float32)

    # Speaker A: 200 Hz + harmonics (0-2 s)
    audio[: int(2.0 * sr)] = _voice_signal(2.0, sr, fundamental=200.0)
    # Speaker B: 800 Hz + harmonics (2.5-5 s)
    start2 = int(2.5 * sr)
    sig_b = _voice_signal(2.5, sr, fundamental=800.0)
    audio[start2 : start2 + len(sig_b)] = sig_b

    segments = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.0),
        DiarizationSegment(speaker="SPEAKER_01", start=2.5, end=5.0),
    ]

    merged = merge_similar_speakers(segments, audio, sr)
    speakers = {seg.speaker for seg in merged}
    assert len(speakers) == 2


def test_merge_similar_speakers_single_speaker():
    """Single speaker should pass through unchanged."""
    sr = 16000
    audio = np.zeros(int(3.0 * sr), dtype=np.float32)

    segments = [DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=3.0)]

    merged = merge_similar_speakers(segments, audio, sr)
    assert len(merged) == 1
    assert merged[0].speaker == "SPEAKER_00"


def test_merge_similar_speakers_empty():
    """Empty input returns empty output."""
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    assert merge_similar_speakers([], audio, sr) == []
