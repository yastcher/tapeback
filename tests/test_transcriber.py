"""Transcriber tests — Whisper integration and segment processing."""

import locale
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from tapeback.transcriber import VRAM_INT8_THRESHOLD_MIB, Transcriber, _resolve_compute_type


def test_resolve_compute_type_explicit_passthrough():
    """Explicit compute_type values pass through unchanged."""
    assert _resolve_compute_type("float16", "cuda") == "float16"
    assert _resolve_compute_type("int8", "cuda") == "int8"
    assert _resolve_compute_type("float32", "cpu") == "float32"


def test_resolve_compute_type_auto_cpu():
    """Auto on CPU always resolves to int8."""
    assert _resolve_compute_type("auto", "cpu") == "int8"


def test_resolve_compute_type_auto_cuda_low_vram():
    """Auto on CUDA with low VRAM resolves to int8."""
    with patch("tapeback.transcriber._get_free_vram_mib", return_value=3600):
        assert _resolve_compute_type("auto", "cuda") == "int8"


def test_resolve_compute_type_auto_cuda_high_vram():
    """Auto on CUDA with enough VRAM resolves to float16."""
    with patch("tapeback.transcriber._get_free_vram_mib", return_value=8000):
        assert _resolve_compute_type("auto", "cuda") == "float16"


def test_resolve_compute_type_auto_cuda_no_nvidia_smi():
    """Auto on CUDA without nvidia-smi falls back to float16."""
    with patch("tapeback.transcriber._get_free_vram_mib", return_value=None):
        assert _resolve_compute_type("auto", "cuda") == "float16"


def test_resolve_compute_type_auto_cuda_boundary():
    """VRAM exactly at threshold gets float16, below gets int8."""
    with patch("tapeback.transcriber._get_free_vram_mib", return_value=VRAM_INT8_THRESHOLD_MIB):
        assert _resolve_compute_type("auto", "cuda") == "float16"
    with patch("tapeback.transcriber._get_free_vram_mib", return_value=VRAM_INT8_THRESHOLD_MIB - 1):
        assert _resolve_compute_type("auto", "cuda") == "int8"


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
