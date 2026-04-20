"""Transcriber tests — Whisper integration and segment processing."""

import locale
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tapeback.transcriber import VRAM_INT8_THRESHOLD_MIB, Transcriber, _resolve_compute_type


@pytest.mark.parametrize(
    "requested,device,vram_mib,expected",
    [
        # Explicit values pass through unchanged
        pytest.param("float16", "cuda", None, "float16", id="explicit_float16"),
        pytest.param("int8", "cuda", None, "int8", id="explicit_int8"),
        pytest.param("float32", "cpu", None, "float32", id="explicit_float32"),
        # Auto + CPU → int8
        pytest.param("auto", "cpu", None, "int8", id="auto_cpu"),
        # Auto + CUDA + VRAM signal
        pytest.param("auto", "cuda", 3600, "int8", id="auto_cuda_low_vram"),
        pytest.param("auto", "cuda", 8000, "float16", id="auto_cuda_high_vram"),
        pytest.param("auto", "cuda", None, "float16", id="auto_cuda_no_nvidia_smi"),
        # Threshold boundary
        pytest.param("auto", "cuda", VRAM_INT8_THRESHOLD_MIB, "float16", id="boundary_at"),
        pytest.param("auto", "cuda", VRAM_INT8_THRESHOLD_MIB - 1, "int8", id="boundary_below"),
    ],
)
def test_resolve_compute_type(requested, device, vram_mib, expected):
    """Pure compute-type resolution: explicit passthrough, auto + device/VRAM branching."""
    with patch("tapeback.transcriber._get_free_vram_mib", return_value=vram_mib):
        assert _resolve_compute_type(requested, device) == expected


def test_lc_messages_set_for_pyav_locale_workaround():
    """Transcriber module must set LC_MESSAGES=C to prevent PyAV crash on non-ASCII locales.

    PyAV's Cython code uses c_string_encoding=ascii. On non-English locales,
    FFmpeg's av_strerror() may return non-ASCII text via strerror_r(), causing
    UnicodeDecodeError in PyAV's err_check().
    Both env var and C locale must be set — env var alone is not enough.
    """
    assert os.environ.get("LC_MESSAGES") == "C"
    lc = locale.getlocale(locale.LC_MESSAGES)
    # locale.getlocale returns (None, None) for "C" locale
    assert lc == (None, None) or lc[0] == "C"


def test_transcribe_stereo_pipeline(settings):
    """transcribe_stereo should transcribe both channels, assign 'You' to mic,
    and correctly map words from faster-whisper output to Segment dataclasses."""
    mock_word = MagicMock()
    mock_word.start = 0.0
    mock_word.end = 0.5
    mock_word.word = "Hello"
    mock_word.probability = 0.95

    mock_seg_mic = MagicMock()
    mock_seg_mic.start = 0.0
    mock_seg_mic.end = 3.0
    mock_seg_mic.text = " My speech "
    mock_seg_mic.words = [mock_word]

    mock_seg_monitor = MagicMock()
    mock_seg_monitor.start = 1.0
    mock_seg_monitor.end = 4.0
    mock_seg_monitor.text = " Their speech "
    mock_seg_monitor.words = []

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_info.duration = 5.0

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.side_effect = [
            (iter([mock_seg_mic]), mock_info),
            (iter([mock_seg_monitor]), mock_info),
        ]

        transcriber = Transcriber(settings)
        mic_segs, monitor_segs, info = transcriber.transcribe_stereo(
            Path("/fake/mic.wav"), Path("/fake/monitor.wav")
        )

    # Whisper called twice (mic + monitor)
    assert instance.transcribe.call_count == 2

    # Mic segments: speaker="You", words mapped to Word dataclass
    assert len(mic_segs) == 1
    assert mic_segs[0].speaker == "You"
    assert mic_segs[0].text == "My speech"
    assert mic_segs[0].words is not None
    assert mic_segs[0].words[0].word == "Hello"
    assert mic_segs[0].words[0].probability == 0.95

    # Monitor segments: speaker=None
    assert len(monitor_segs) == 1
    assert monitor_segs[0].speaker is None
    assert monitor_segs[0].text == "Their speech"

    # Info dict
    assert info["language"] == "en"
    assert info["duration"] == 5.0
