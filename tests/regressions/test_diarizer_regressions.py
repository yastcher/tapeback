"""Regression tests for diarizer bugs."""

import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import SecretStr

from tapeback.audio import split_channels_16k
from tapeback.channel import load_stereo_channels
from tapeback.diarizer import (
    Diarizer,
    assign_speakers,
    merge_similar_speakers,
)
from tapeback.models import DiarizationSegment, Segment
from tapeback.settings import Settings
from tests.fixtures import create_stereo_wav_segments, requires_pyannote


@requires_pyannote
def test_diarizer_passes_token_correctly(tmp_vault):
    """Pipeline.from_pretrained must receive 'token' kwarg, not 'use_auth_token'.

    Bug: pyannote 4.x changed API from use_auth_token to token.
    Old code passed use_auth_token which caused TypeError.
    """

    settings = Settings(vault_path=tmp_vault, hf_token=SecretStr("hf_test_123"), device="cpu")

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

    settings = Settings(vault_path=tmp_vault, hf_token=SecretStr("hf_fake_token"), device="cpu")

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

    settings = Settings(vault_path=tmp_vault, hf_token=SecretStr("hf_fake_token"), device="cuda")

    with (
        patch("pyannote.audio.Pipeline") as mock_pipeline_cls,
        # Isolation by construction: without this the test reads the machine's real
        # free VRAM, so anything else using the GPU sends Diarizer down the
        # not-enough-VRAM branch and this assertion fails for an unrelated reason.
        patch("tapeback._device.get_free_vram_mib", return_value=4096),
    ):
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

    settings = Settings(vault_path=tmp_vault, hf_token=SecretStr("hf_fake_token"), device="cuda")

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

    with (
        patch("pyannote.audio.Pipeline") as mock_pipeline_cls,
        # The assertion below requires the CUDA branch to be taken, which depends on
        # free VRAM — do not let another process on the GPU decide this test's result.
        patch("tapeback._device.get_free_vram_mib", return_value=4096),
    ):
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


ECHO_WAV = Path(__file__).parent.parent / "data" / "2026-04-03_15-22-38.wav"


@pytest.mark.skipif(not ECHO_WAV.exists(), reason="test WAV not available")
def test_minor_speaker_absorbed_into_dominant():
    """Minor speaker (echo/crosstalk) must be absorbed into the dominant speaker.

    Bug: pyannote split one remote speaker into SPEAKER_00 (80.5s) and SPEAKER_01
    (8.6s). Cosine similarity = 0.9466 — below 0.96 threshold, so merge didn't
    trigger. SPEAKER_01 was headphone echo bleeding into the mic channel.

    Fix: speakers with very little speech (< 15s and < 20% of dominant) use
    a lower merge threshold (0.92) because their spectral profiles are unreliable.
    """
    # Use raw stereo channel (same as pipeline does), not 16kHz processed version
    _, raw, sr = load_stereo_channels(ECHO_WAV)

    # Real pyannote segments from this recording
    segments = [
        DiarizationSegment(speaker="SPEAKER_00", start=1.40, end=9.26),
        DiarizationSegment(speaker="SPEAKER_00", start=15.93, end=21.56),
        DiarizationSegment(speaker="SPEAKER_00", start=25.01, end=25.68),
        DiarizationSegment(speaker="SPEAKER_01", start=59.04, end=60.70),
        DiarizationSegment(speaker="SPEAKER_00", start=73.03, end=73.61),
        DiarizationSegment(speaker="SPEAKER_01", start=73.61, end=73.86),
        DiarizationSegment(speaker="SPEAKER_00", start=74.50, end=82.47),
        DiarizationSegment(speaker="SPEAKER_01", start=102.19, end=104.03),
        DiarizationSegment(speaker="SPEAKER_00", start=105.25, end=112.18),
        DiarizationSegment(speaker="SPEAKER_01", start=128.28, end=129.02),
        DiarizationSegment(speaker="SPEAKER_00", start=129.02, end=140.60),
        DiarizationSegment(speaker="SPEAKER_01", start=197.92, end=199.02),
        DiarizationSegment(speaker="SPEAKER_00", start=236.58, end=255.70),
        DiarizationSegment(speaker="SPEAKER_00", start=423.49, end=441.99),
        DiarizationSegment(speaker="SPEAKER_01", start=456.48, end=456.80),
        DiarizationSegment(speaker="SPEAKER_01", start=463.05, end=464.26),
        DiarizationSegment(speaker="SPEAKER_00", start=466.76, end=473.66),
    ]

    # SPEAKER_00: ~80s, SPEAKER_01: ~8.6s, cosine = 0.9466
    # Without minor logic, 0.96 threshold keeps both speakers
    # With minor logic, SPEAKER_01 is absorbed (< 15s, < 20% of dominant, cosine > 0.92)
    result = merge_similar_speakers(segments, raw, sr, similarity_threshold=0.96)
    speakers = {s.speaker for s in result}

    assert len(speakers) == 1, (
        f"Expected 1 speaker after minor absorption, got {len(speakers)}: {speakers}"
    )
