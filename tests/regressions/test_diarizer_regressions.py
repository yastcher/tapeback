"""Regression tests for diarizer bugs."""

import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tapeback.audio import split_channels_16k
from tapeback.diarizer import (
    Diarizer,
    assign_speakers,
    filter_silent_segments,
    load_stereo_channels,
    merge_similar_speakers,
    split_on_silence,
)
from tapeback.models import DiarizationSegment, Segment, Word
from tapeback.settings import Settings
from tests.fixtures import create_stereo_wav_segments, requires_pyannote


@requires_pyannote
def test_diarizer_passes_token_correctly(tmp_vault):
    """Pipeline.from_pretrained must receive 'token' kwarg, not 'use_auth_token'.

    Bug: pyannote 4.x changed API from use_auth_token to token.
    Old code passed use_auth_token which caused TypeError.
    """

    settings = Settings(vault_path=tmp_vault, hf_token="hf_test_123", device="cpu")

    with patch("pyannote.audio.Pipeline") as mock_pipeline_cls:
        mock_pipeline_cls.from_pretrained.return_value = MagicMock()
        Diarizer(settings)

    _args, kwargs = mock_pipeline_cls.from_pretrained.call_args
    assert "use_auth_token" not in kwargs
    assert kwargs["token"] == "hf_test_123"


@requires_pyannote
def test_diarize_handles_diarize_output(tmp_vault):
    """Diarizer.diarize should handle DiarizeOutput (pyannote 4.x) wrapping Annotation.

    Bug: pyannote 4.x returns DiarizeOutput instead of Annotation.
    Code called .itertracks() directly which raised AttributeError.
    """

    settings = Settings(vault_path=tmp_vault, hf_token="hf_fake_token", device="cpu")

    mock_turn = MagicMock()
    mock_turn.start = 0.0
    mock_turn.end = 3.0

    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = [(mock_turn, None, "SPEAKER_00")]

    # Simulate DiarizeOutput: no itertracks, but has speaker_diarization
    mock_diarize_output = MagicMock(spec=[])
    mock_diarize_output.speaker_diarization = mock_annotation

    with patch("pyannote.audio.Pipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_pipeline.return_value = mock_diarize_output

        diarizer = Diarizer(settings)
        result = diarizer.diarize(Path("/fake/audio.wav"))

    assert len(result) == 1
    assert result[0].speaker == "SPEAKER_00"


@requires_pyannote
def test_diarize_cuda_init_fallback(tmp_vault, capsys):
    """Diarizer should fall back to CPU when CUDA is not available at init.

    Bug: Diarizer crashed if torch.device('cuda') failed.
    """

    settings = Settings(vault_path=tmp_vault, hf_token="hf_fake_token", device="cuda")

    with patch("pyannote.audio.Pipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.side_effect = RuntimeError("CUDA not available")

        Diarizer(settings)

    captured = capsys.readouterr()
    assert "CUDA not available for diarization" in captured.err


@requires_pyannote
def test_diarize_cuda_oom_fallback(tmp_vault, capsys):
    """Diarizer.diarize should fall back to CPU on CUDA OOM during inference.

    Bug: GPU with 3.6 GiB couldn't hold both Whisper and pyannote models.
    Pipeline loaded on CUDA but crashed with OutOfMemoryError during inference.
    """

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
        result = diarizer.diarize(Path("/fake/audio.wav"))

    assert len(result) == 1
    assert result[0].speaker == "SPEAKER_00"
    mock_pipeline.to.assert_called()
    captured = capsys.readouterr()
    assert "CUDA out of memory" in captured.err


def test_monitor_speech_not_attributed_to_you(tmp_path):
    """Monitor-dominant segment must not be labeled 'You' even if pyannote thinks so.

    Bug: pyannote clustered monitor speech with mic speech into one speaker.
    The channel-based override should catch this and label it correctly.
    """
    wav_path = tmp_path / "stereo.wav"
    # 0-1s: monitor dominant (remote speaker), 1-2s: mic dominant (user)
    create_stereo_wav_segments(
        wav_path,
        sample_rate=16000,
        segments_spec=[
            (1.0, 0.1, 0.8),  # remote speaker on monitor
            (1.0, 0.8, 0.1),  # user on mic
        ],
    )

    segments = [
        Segment(start=0.0, end=1.0, text="Remote person talking", words=None),
        Segment(start=1.0, end=2.0, text="I am talking", words=None),
    ]

    # Pyannote wrongly puts both segments under SPEAKER_00
    dsegs = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.0),
    ]

    # identify_user_speaker would return SPEAKER_00 (mic energy is high overall)
    result = assign_speakers(segments, dsegs, user_speaker="SPEAKER_00", stereo_wav=wav_path)

    # Channel override: first segment is monitor-dominant → NOT "You"
    assert result[0].speaker != "You"
    # Second segment is mic-dominant → "You"
    assert result[1].speaker == "You"


# --- Dual-channel pipeline ---

MULTI_SPEAKER_WAV = Path(__file__).parent.parent / "data" / "2026-04-01_18-41-14.wav"


@pytest.mark.skipif(not MULTI_SPEAKER_WAV.exists(), reason="test WAV not available")
def test_spectral_merge_default_preserves_different_speakers(tmp_path):
    """Default threshold (0.96) must preserve distinct speakers from same channel.

    Bug: Power-spectrum cosine similarity is dominated by channel frequency
    response, not voice identity.  Two male YouTube voices had cosine ~0.92.
    Default threshold 0.96 is above this, so they stay separate.
    Low thresholds (0.92) incorrectly merge them.
    """
    _, monitor_16k = split_channels_16k(MULTI_SPEAKER_WAV, tmp_path)

    with wave.open(str(monitor_16k), "rb") as wf:
        sr = wf.getframerate()
        raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)

    segments = [
        DiarizationSegment(speaker="SPEAKER_00", start=16.03, end=22.73),
        DiarizationSegment(speaker="SPEAKER_02", start=30.47, end=37.07),
        DiarizationSegment(speaker="SPEAKER_00", start=45.49, end=56.48),
    ]

    # Default threshold (0.96) preserves both speakers (cosine ~0.92)
    result = merge_similar_speakers(segments, raw, sr, similarity_threshold=0.96)
    assert {s.speaker for s in result} == {"SPEAKER_00", "SPEAKER_02"}

    # Low threshold merges them — documents the risk of aggressive thresholds
    merged = merge_similar_speakers(segments, raw, sr, similarity_threshold=0.92)
    assert len({s.speaker for s in merged}) == 1


MULTI_SPEAKER_WAV_2 = Path(__file__).parent.parent / "data" / "2026-04-01_23-15-26.wav"


@pytest.mark.skipif(not MULTI_SPEAKER_WAV_2.exists(), reason="test WAV not available")
def test_spectral_merge_096_preserves_three_speakers(tmp_path):
    """Default threshold (0.96) must preserve 3 distinct speakers (cosines 0.94-0.95).

    Bug: At threshold=0.95, SPEAKER_01 vs SPEAKER_02 (cosine=0.9504) get merged.
    Default 0.96 is safely above all observed different-speaker cosines.
    """
    _, monitor_16k = split_channels_16k(MULTI_SPEAKER_WAV_2, tmp_path)

    with wave.open(str(monitor_16k), "rb") as wf:
        sr = wf.getframerate()
        raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)

    # Real pyannote segments from 2026-04-01_23-15-26.wav
    segments = [
        DiarizationSegment(speaker="SPEAKER_01", start=13.77, end=13.97),
        DiarizationSegment(speaker="SPEAKER_02", start=13.97, end=25.02),
        DiarizationSegment(speaker="SPEAKER_01", start=25.02, end=25.78),
        DiarizationSegment(speaker="SPEAKER_02", start=25.78, end=25.82),
        DiarizationSegment(speaker="SPEAKER_01", start=40.82, end=41.04),
        DiarizationSegment(speaker="SPEAKER_00", start=41.04, end=41.75),
        DiarizationSegment(speaker="SPEAKER_01", start=41.75, end=42.02),
        DiarizationSegment(speaker="SPEAKER_00", start=42.02, end=44.48),
        DiarizationSegment(speaker="SPEAKER_00", start=45.26, end=49.47),
        DiarizationSegment(speaker="SPEAKER_00", start=51.04, end=52.29),
        DiarizationSegment(speaker="SPEAKER_01", start=52.29, end=52.33),
        DiarizationSegment(speaker="SPEAKER_01", start=59.58, end=62.97),
        DiarizationSegment(speaker="SPEAKER_02", start=62.97, end=65.98),
    ]

    # Default threshold (0.96) preserves all 3 speakers
    result = merge_similar_speakers(segments, raw, sr, similarity_threshold=0.96)
    assert {s.speaker for s in result} == {"SPEAKER_00", "SPEAKER_01", "SPEAKER_02"}

    # threshold=0.95 loses SPEAKER_02 (cosine 0.9504 with SPEAKER_01)
    merged_095 = merge_similar_speakers(segments, raw, sr, similarity_threshold=0.95)
    assert len({s.speaker for s in merged_095}) < 3


TEST_WAV = Path(__file__).parent.parent / "data" / "2026-03-19_02-45-24.wav"


@pytest.mark.skipif(not TEST_WAV.exists(), reason="test WAV not available")
def test_dual_channel_speaker_attribution():
    """filter_silent_segments should keep speech and drop silence per channel.

    Uses raw (pre-loudnorm) stereo channels for RMS filtering.

    Test file channel analysis (RMS per second):
      0-9s:  L(mic)=350-445   R(mon)=0        -> mic=speech, monitor=silence
      10-16s: L(mic)=95-136   R(mon)=2050-2950 -> mic=silence, monitor=speech
      17-18s: L(mic)=460-665  R(mon)=0         -> mic=speech, monitor=silence
      19-29s: L(mic)=260-770  R(mon)=1120-2580 -> both channels=speech
    """

    mic_raw, monitor_raw, raw_sr = load_stereo_channels(TEST_WAV)

    # Fake segments covering each time region
    fake_segments = [
        Segment(start=0.0, end=9.0, text="Region 0-9s"),
        Segment(start=10.0, end=16.0, text="Region 10-16s"),
        Segment(start=17.0, end=18.0, text="Region 17-18s"),
        Segment(start=19.0, end=29.0, text="Region 19-29s"),
    ]

    # Filter mic channel
    mic_kept = filter_silent_segments(fake_segments, mic_raw, raw_sr)
    mic_starts = {s.start for s in mic_kept}

    # Mic should keep: 0-9s (speech), 17-18s (speech), 19-29s (simultaneous)
    assert 0.0 in mic_starts, "Mic speech at 0-9s should pass filter"
    assert 17.0 in mic_starts, "Mic speech at 17-18s should pass filter"
    assert 19.0 in mic_starts, "Simultaneous speech at 19-29s should pass mic filter"
    # Mic should drop: 10-16s (silence/background noise)
    assert 10.0 not in mic_starts, "Mic silence at 10-16s should be filtered out"

    # Filter monitor channel
    monitor_kept = filter_silent_segments(fake_segments, monitor_raw, raw_sr)
    monitor_starts = {s.start for s in monitor_kept}

    # Monitor should keep: 10-16s (speech), 19-29s (simultaneous)
    assert 10.0 in monitor_starts, "Monitor speech at 10-16s should pass filter"
    assert 19.0 in monitor_starts, "Simultaneous speech at 19-29s should pass monitor filter"
    # Monitor should drop: 0-9s (silence), 17-18s (silence)
    assert 0.0 not in monitor_starts, "Monitor silence at 0-9s should be filtered out"
    assert 17.0 not in monitor_starts, "Monitor silence at 17-18s should be filtered out"


@pytest.mark.skipif(not TEST_WAV.exists(), reason="test WAV not available")
def test_split_on_silence_detects_real_pause():
    """split_on_silence should split mic segment at 9-10s pause.

    Test WAV: user speaks 0-9s, pauses 10-16s (monitor active), resumes 17s+.
    A segment spanning 0-20s should be split around the pause.
    """

    mic_raw, monitor_raw, sr = load_stereo_channels(TEST_WAV)

    # Simulate Whisper returning one big mic segment covering speech + pause + speech
    seg = Segment(
        start=0.0,
        end=20.0,
        text="speech before pause speech after pause",
        words=[
            Word(start=1.0, end=2.0, word="speech", probability=0.9),
            Word(start=3.0, end=4.0, word="before", probability=0.9),
            Word(start=5.0, end=6.0, word="pause", probability=0.9),
            Word(start=17.0, end=18.0, word="speech", probability=0.9),
            Word(start=18.0, end=19.0, word="after", probability=0.9),
            Word(start=19.0, end=20.0, word="pause", probability=0.9),
        ],
        speaker="You",
    )

    result = split_on_silence(
        [seg],
        mic_raw,
        sr,
        pause_threshold=1.0,
        monitor_samples=monitor_raw,
    )

    assert len(result) >= 2, (
        f"Expected split at ~10s pause, got {len(result)} segment(s): "
        f"{[(s.start, s.end, s.text) for s in result]}"
    )
    # First sub-segment should end before the pause
    assert result[0].end <= 10.0, f"First segment should end before pause, got end={result[0].end}"
    # Last sub-segment should start after the pause
    assert result[-1].start >= 15.0, (
        f"Last segment should start after pause, got start={result[-1].start}"
    )
