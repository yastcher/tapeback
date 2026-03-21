from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meetrec.diarizer import (
    DiarizationSegment,
    assign_speakers,
    classify_segment_by_channel,
    identify_user_speaker,
    load_stereo_channels,
    merge_channel_segments,
)
from meetrec.transcriber import Segment, Word
from tests.fixtures import create_stereo_wav, create_stereo_wav_segments

# --- Diarizer init / diarize ---


def test_diarizer_init_without_token(tmp_vault):
    """Diarizer should raise RuntimeError when hf_token is empty."""
    from meetrec.diarizer import Diarizer
    from meetrec.settings import Settings

    settings = Settings(vault_path=tmp_vault, hf_token="")

    with pytest.raises(RuntimeError, match="HuggingFace token required"):
        Diarizer(settings)


def test_diarizer_applies_clustering_threshold(tmp_vault):
    """Diarizer should apply custom clustering_threshold to pipeline."""
    from meetrec.diarizer import Diarizer
    from meetrec.settings import Settings

    settings = Settings(
        vault_path=tmp_vault,
        hf_token="hf_fake_token",
        device="cpu",
        clustering_threshold=0.85,
    )

    with patch("pyannote.audio.Pipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_pipeline.parameters.return_value = {
            "clustering": {"method": "centroid", "min_cluster_size": 12, "threshold": 0.7},
            "segmentation": {"min_duration_off": 0.0},
        }

        Diarizer(settings)

    mock_pipeline.parameters.assert_called_once_with(instantiated=True)
    mock_pipeline.instantiate.assert_called_once()
    call_params = mock_pipeline.instantiate.call_args[0][0]
    assert call_params["clustering"]["threshold"] == 0.85


def test_diarizer_no_clustering_override_by_default(tmp_vault):
    """Diarizer should not touch clustering params when threshold is None."""
    from meetrec.diarizer import Diarizer
    from meetrec.settings import Settings

    settings = Settings(
        vault_path=tmp_vault,
        hf_token="hf_fake_token",
        device="cpu",
        clustering_threshold=None,
    )

    with patch("pyannote.audio.Pipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline

        Diarizer(settings)

    mock_pipeline.parameters.assert_not_called()
    mock_pipeline.instantiate.assert_not_called()


def test_diarize_returns_segments(tmp_vault):
    """Diarizer.diarize should return list of DiarizationSegment."""
    from meetrec.diarizer import Diarizer
    from meetrec.settings import Settings

    settings = Settings(vault_path=tmp_vault, hf_token="hf_fake_token", device="cpu")

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

    with patch("pyannote.audio.Pipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
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
            {
                "type": "segments",
                "spec": [(1.0, 0.8, 0.05), (1.0, 0.05, 0.8)],
            },
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
        create_stereo_wav(
            wav_path,
            duration=1.0,
            sample_rate=16000,
            left_amplitude=wav_setup["left"],
            right_amplitude=wav_setup["right"],
        )
    else:
        create_stereo_wav_segments(wav_path, sample_rate=16000, segments_spec=wav_setup["spec"])

    result = identify_user_speaker(dsegs, wav_path)
    assert result == expected


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
    create_stereo_wav(
        wav_path,
        duration=2.0,
        sample_rate=16000,
        left_amplitude=left_amp,
        right_amplitude=right_amp,
    )
    mic, monitor, sr = load_stereo_channels(wav_path)
    result = classify_segment_by_channel(0.0, 2.0, mic, monitor, sr)
    assert result == expected


# --- assign_speakers ---


def test_assign_speakers_with_stereo_override(tmp_path):
    """Channel energy should override pyannote speaker assignment."""
    wav_path = tmp_path / "stereo.wav"
    # 0-1s: monitor dominant (not You), 1-2s: mic dominant (You)
    create_stereo_wav_segments(
        wav_path,
        sample_rate=16000,
        segments_spec=[
            (1.0, 0.1, 0.8),  # monitor dominant
            (1.0, 0.8, 0.1),  # mic dominant
        ],
    )

    segments = [
        Segment(start=0.0, end=1.0, text="Remote speech", words=None),
        Segment(start=1.0, end=2.0, text="My speech", words=None),
    ]

    dsegs = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=1.0),
        DiarizationSegment(speaker="SPEAKER_01", start=1.0, end=2.0),
    ]

    # pyannote thinks SPEAKER_00 is user — but channel says otherwise
    result = assign_speakers(segments, dsegs, user_speaker="SPEAKER_00", stereo_wav=wav_path)

    assert result[0].speaker == "Speaker 1"  # monitor → not You
    assert result[1].speaker == "You"  # mic → You


def test_assign_speakers_stereo_fallback_on_ambiguous(tmp_path):
    """Ambiguous channel energy should fall back to pyannote + user_speaker."""
    wav_path = tmp_path / "stereo.wav"
    create_stereo_wav(
        wav_path, duration=2.0, sample_rate=16000, left_amplitude=0.5, right_amplitude=0.5
    )

    segments = [
        Segment(start=0.0, end=1.0, text="Ambiguous", words=None),
        Segment(start=1.0, end=2.0, text="Also ambiguous", words=None),
    ]

    dsegs = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=1.0),
        DiarizationSegment(speaker="SPEAKER_01", start=1.0, end=2.0),
    ]

    result = assign_speakers(segments, dsegs, user_speaker="SPEAKER_00", stereo_wav=wav_path)

    # Falls back to pyannote: SPEAKER_00 = user_speaker → "You"
    assert result[0].speaker == "You"
    assert result[1].speaker == "Speaker 1"


def test_assign_speakers_without_stereo_backward_compat():
    """Without stereo_wav, behavior should be identical to before."""
    segments = [
        Segment(start=0.0, end=2.0, text="Hello", words=None),
        Segment(start=3.0, end=5.0, text="World", words=None),
    ]

    dsegs = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.5),
        DiarizationSegment(speaker="SPEAKER_01", start=2.5, end=5.0),
    ]

    result = assign_speakers(segments, dsegs, user_speaker="SPEAKER_00")

    assert result[0].speaker == "You"
    assert result[1].speaker == "Speaker 1"


def test_assign_speakers_with_words():
    """Words within SPEAKER_00 -> 'You', within SPEAKER_01 -> 'Speaker 1'."""
    segments = [
        Segment(
            start=0.0,
            end=2.0,
            text="Hello there",
            words=[Word(start=0.0, end=1.0, word="Hello", probability=0.9)],
        ),
        Segment(
            start=3.0,
            end=5.0,
            text="Hi back",
            words=[Word(start=3.0, end=4.0, word="Hi", probability=0.9)],
        ),
    ]

    dsegs = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.5),
        DiarizationSegment(speaker="SPEAKER_01", start=2.5, end=5.0),
    ]

    result = assign_speakers(segments, dsegs, user_speaker="SPEAKER_00")

    assert result[0].speaker == "You"
    assert result[1].speaker == "Speaker 1"
    # Input not mutated
    assert segments[0].speaker is None
    assert segments[1].speaker is None


def test_assign_speakers_no_user():
    """Without user_speaker, all speakers get 'Speaker N' labels."""
    segments = [
        Segment(start=0.0, end=2.0, text="A", words=None),
        Segment(start=3.0, end=5.0, text="B", words=None),
    ]

    dsegs = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.5),
        DiarizationSegment(speaker="SPEAKER_01", start=2.5, end=5.0),
    ]

    result = assign_speakers(segments, dsegs, user_speaker=None)

    assert result[0].speaker == "Speaker 1"
    assert result[1].speaker == "Speaker 2"


def test_assign_speakers_preserves_order():
    """Speakers numbered by appearance in transcription, not by pyannote label."""
    segments = [
        Segment(start=0.0, end=2.0, text="First", words=None),
        Segment(start=3.0, end=5.0, text="Second", words=None),
    ]

    dsegs = [
        DiarizationSegment(speaker="SPEAKER_02", start=0.0, end=2.5),
        DiarizationSegment(speaker="SPEAKER_00", start=2.5, end=5.0),
    ]

    result = assign_speakers(segments, dsegs, user_speaker=None)

    assert result[0].speaker == "Speaker 1"  # SPEAKER_02 appeared first
    assert result[1].speaker == "Speaker 2"  # SPEAKER_00 appeared second


def test_assign_speakers_gap_handling():
    """Word in gap between diarization segments -> assigned to nearest."""
    segments = [
        Segment(
            start=2.0,
            end=4.0,
            text="Gap word",
            words=[Word(start=2.5, end=3.0, word="Gap", probability=0.9)],
        ),
    ]

    dsegs = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.0),
        DiarizationSegment(speaker="SPEAKER_01", start=4.0, end=6.0),
    ]

    result = assign_speakers(segments, dsegs, user_speaker=None)

    # Word at 2.5-3.0 is closer to SPEAKER_00 (ends at 2.0) than SPEAKER_01 (starts at 4.0)
    assert result[0].speaker == "Speaker 1"


# --- merge_channel_segments ---


@pytest.mark.parametrize(
    "mic_segs,mon_segs,expected_count,expected_starts",
    [
        pytest.param(
            [
                Segment(start=0.0, end=3.0, text="My first", speaker="You"),
                Segment(start=5.0, end=7.0, text="My second", speaker="You"),
            ],
            [
                Segment(start=2.0, end=4.0, text="Their first", speaker="Speaker 1"),
                Segment(start=6.0, end=8.0, text="Their second", speaker="Speaker 1"),
            ],
            4,
            [0.0, 2.0, 5.0, 6.0],
            id="interleaved",
        ),
        pytest.param(
            [Segment(start=0.0, end=5.0, text="Talking over", speaker="You")],
            [Segment(start=2.0, end=6.0, text="Also talking", speaker="Speaker 1")],
            2,
            [0.0, 2.0],
            id="overlapping",
        ),
    ],
)
def test_merge_channel_segments(mic_segs, mon_segs, expected_count, expected_starts):
    """Segments from both channels merged and sorted by start time."""
    result = merge_channel_segments(mic_segs, mon_segs)

    assert len(result) == expected_count
    assert [s.start for s in result] == expected_starts
