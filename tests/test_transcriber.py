"""Transcriber tests — Whisper integration and segment processing."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from tapeback.transcriber import Transcriber


def test_lc_messages_set_for_pyav_locale_workaround():
    """Transcriber module must set LC_MESSAGES=C to prevent PyAV crash on non-ASCII locales.

    PyAV's Cython code uses c_string_encoding=ascii. On non-English locales,
    FFmpeg's av_strerror() may return non-ASCII text via strerror_r(), causing
    UnicodeDecodeError in PyAV's err_check().
    """
    assert os.environ.get("LC_MESSAGES") == "C"


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
