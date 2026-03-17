import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meetrec.diarizer import (
    DiarizationSegment,
    assign_speakers,
    identify_user_speaker,
)
from meetrec.transcriber import Segment, Word


def _create_stereo_wav(path, duration, sample_rate, left_amplitude, right_amplitude):
    """Create a stereo WAV with sine wave on each channel at given amplitudes."""
    n_frames = int(duration * sample_rate)
    t = np.linspace(0, duration, n_frames, dtype=np.float32)

    left = (left_amplitude * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    right = (right_amplitude * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    stereo = np.empty(n_frames * 2, dtype=np.int16)
    stereo[0::2] = left
    stereo[1::2] = right

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())


def _create_stereo_wav_segments(path, sample_rate, segments_spec):
    """Create a stereo WAV with different amplitudes per time segment.

    segments_spec: list of (duration, left_amp, right_amp)
    """
    all_left = []
    all_right = []

    for duration, left_amp, right_amp in segments_spec:
        n_frames = int(duration * sample_rate)
        t = np.linspace(0, duration, n_frames, dtype=np.float32)
        all_left.append((left_amp * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16))
        all_right.append((right_amp * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16))

    left = np.concatenate(all_left)
    right = np.concatenate(all_right)

    stereo = np.empty(len(left) * 2, dtype=np.int16)
    stereo[0::2] = left
    stereo[1::2] = right

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())


# --- Diarizer init / diarize ---


def test_diarizer_init_without_token(tmp_vault):
    """Diarizer should raise RuntimeError when hf_token is empty."""
    from meetrec.diarizer import Diarizer
    from meetrec.settings import Settings

    settings = Settings(vault_path=tmp_vault, hf_token="")

    with pytest.raises(RuntimeError, match="HuggingFace token required"):
        Diarizer(settings)


def test_diarizer_passes_token_correctly(tmp_vault):
    """Pipeline.from_pretrained must receive 'token' kwarg, not 'use_auth_token'."""
    from meetrec.diarizer import Diarizer
    from meetrec.settings import Settings

    settings = Settings(vault_path=tmp_vault, hf_token="hf_test_123", device="cpu")

    with patch("pyannote.audio.Pipeline") as mock_pipeline_cls:
        mock_pipeline_cls.from_pretrained.return_value = MagicMock()
        Diarizer(settings)

    _args, kwargs = mock_pipeline_cls.from_pretrained.call_args
    assert "use_auth_token" not in kwargs
    assert kwargs["token"] == "hf_test_123"


def test_diarize_returns_segments(tmp_vault):
    """Diarizer.diarize should return list of DiarizationSegment."""
    from meetrec.diarizer import Diarizer
    from meetrec.settings import Settings

    settings = Settings(vault_path=tmp_vault, hf_token="hf_fake_token", device="cpu")

    # Mock pyannote turn objects
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
        from pathlib import Path

        result = diarizer.diarize(Path("/fake/audio.wav"))

    assert len(result) == 2
    assert result[0].speaker == "SPEAKER_00"
    assert result[0].start == 0.0
    assert result[0].end == 3.0
    assert result[1].speaker == "SPEAKER_01"


def test_diarize_cuda_fallback(tmp_vault, capsys):
    """Diarizer should fall back to CPU when CUDA is not available."""
    from meetrec.diarizer import Diarizer
    from meetrec.settings import Settings

    settings = Settings(vault_path=tmp_vault, hf_token="hf_fake_token", device="cuda")

    with patch("pyannote.audio.Pipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.side_effect = RuntimeError("CUDA not available")

        Diarizer(settings)

    captured = capsys.readouterr()
    assert "CUDA not available for diarization" in captured.err


def test_diarize_cuda_oom_fallback(tmp_vault, capsys):
    """Diarizer.diarize should fall back to CPU on CUDA OOM during inference."""
    from meetrec.diarizer import Diarizer
    from meetrec.settings import Settings

    settings = Settings(vault_path=tmp_vault, hf_token="hf_fake_token", device="cuda")

    mock_turn = MagicMock()
    mock_turn.start = 0.0
    mock_turn.end = 3.0
    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = [(mock_turn, None, "SPEAKER_00")]

    call_count = 0

    def pipeline_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("CUDA out of memory.")
        return mock_annotation

    with patch("pyannote.audio.Pipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_pipeline.side_effect = pipeline_call

        diarizer = Diarizer(settings)
        from pathlib import Path

        result = diarizer.diarize(Path("/fake/audio.wav"))

    assert len(result) == 1
    assert result[0].speaker == "SPEAKER_00"
    # Verify pipeline was moved to CPU
    mock_pipeline.to.assert_called()
    captured = capsys.readouterr()
    assert "CUDA out of memory" in captured.err


# --- identify_user_speaker ---


def test_identify_user_speaker(tmp_path):
    """Speaker with most energy on mic channel should be identified as user."""
    wav_path = tmp_path / "stereo.wav"
    # left=mic: loud (0.8), right=monitor: quiet (0.05)
    _create_stereo_wav(
        wav_path, duration=1.0, sample_rate=16000, left_amplitude=0.8, right_amplitude=0.05
    )

    dseg = [DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=1.0)]

    # Only one speaker → None (can't determine)
    result = identify_user_speaker(dseg, wav_path)
    assert result is None


def test_identify_user_speaker_ambiguous(tmp_path):
    """Equal energy on both channels → None (ambiguous)."""
    wav_path = tmp_path / "stereo.wav"
    _create_stereo_wav(
        wav_path, duration=1.0, sample_rate=16000, left_amplitude=0.5, right_amplitude=0.5
    )

    dsegs = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=0.5),
        DiarizationSegment(speaker="SPEAKER_01", start=0.5, end=1.0),
    ]

    result = identify_user_speaker(dsegs, wav_path)
    assert result is None


def test_identify_user_speaker_multiple(tmp_path):
    """With distinct channel energy, the mic-dominant speaker is identified."""
    wav_path = tmp_path / "stereo.wav"
    # 0-1s: mic=loud, monitor=quiet (user speaking)
    # 1-2s: mic=quiet, monitor=loud (other person speaking)
    _create_stereo_wav_segments(
        wav_path,
        sample_rate=16000,
        segments_spec=[
            (1.0, 0.8, 0.05),  # SPEAKER_00: mic dominant
            (1.0, 0.05, 0.8),  # SPEAKER_01: monitor dominant
        ],
    )

    dsegs = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=1.0),
        DiarizationSegment(speaker="SPEAKER_01", start=1.0, end=2.0),
    ]

    result = identify_user_speaker(dsegs, wav_path)
    assert result == "SPEAKER_00"


# --- assign_speakers ---


def test_assign_speakers_with_words():
    """Words within SPEAKER_00 → 'You', within SPEAKER_01 → 'Speaker 1'."""
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


def test_assign_speakers_without_words():
    """Segments without word timestamps use segment boundaries for alignment."""
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
    """Word in gap between diarization segments → assigned to nearest."""
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
