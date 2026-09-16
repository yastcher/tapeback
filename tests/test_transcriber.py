"""Transcriber tests — Whisper integration and segment processing."""

import locale
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from tapeback.transcriber import Transcriber, _resolve_compute_type


@pytest.mark.parametrize(
    "requested,device,expected",
    [
        # Explicit values pass through unchanged
        pytest.param("float16", "cuda", "float16", id="explicit_float16"),
        pytest.param("int8", "cuda", "int8", id="explicit_int8"),
        pytest.param("float32", "cpu", "float32", id="explicit_float32"),
        # Auto: device-driven only — no VRAM probing
        pytest.param("auto", "cpu", "int8", id="auto_cpu"),
        # int8_float16, not float16: measured 14.16x vs 3.90x real time on the same
        # clip and half the VRAM, with no quality cost.
        pytest.param("auto", "cuda", "int8_float16", id="auto_cuda"),
        # A GPU-only type must be translated when we end up on the CPU: ctranslate2
        # raises ValueError rather than degrading, so carrying TAPEBACK_COMPUTE_TYPE
        # across a thermal or VRAM fallback used to crash the run.
        pytest.param("int8_float16", "cpu", "int8", id="gpu_only_type_on_cpu"),
        pytest.param("float16", "cpu", "int8", id="float16_on_cpu"),
    ],
)
def test_resolve_compute_type(requested, device, expected):
    """Pure compute-type resolution: explicit passthrough, auto + device branching."""
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
        # Monitor is transcribed FIRST so its detected language can be reused for the
        # gated mic channel — see tests/regressions/test_language_detection.py.
        instance.transcribe.side_effect = [
            (iter([mock_seg_monitor]), mock_info),
            (iter([mock_seg_mic]), mock_info),
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


def test_load_model_prefers_local_cache(settings):
    """Model load tries the local cache first — no HuggingFace round-trip per start."""
    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        Transcriber(settings)

    assert mock_model_cls.call_count == 1
    assert mock_model_cls.call_args_list[0].kwargs["local_files_only"] is True


def test_load_model_downloads_when_not_cached(settings):
    """If the model isn't cached, fall back to a network download (local_files_only=False)."""
    instance = MagicMock()
    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        mock_model_cls.side_effect = [LocalEntryNotFoundError("not cached"), instance]
        transcriber = Transcriber(settings)

    assert mock_model_cls.call_count == 2
    assert mock_model_cls.call_args_list[0].kwargs["local_files_only"] is True
    assert mock_model_cls.call_args_list[1].kwargs["local_files_only"] is False
    assert transcriber._model is instance


def test_transcribe_stereo_reports_per_channel_timings(settings):
    """transcribe_stereo emits separate timing lines for the mic and monitor channels."""
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_info.duration = 5.0

    mic_seg = MagicMock()
    mic_seg.start, mic_seg.end, mic_seg.text, mic_seg.words = 0.0, 3.0, "mine", []
    mon_seg = MagicMock()
    mon_seg.start, mon_seg.end, mon_seg.text, mon_seg.words = 0.0, 2.0, "theirs", []

    messages: list[str] = []
    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.side_effect = [
            (iter([mic_seg]), mock_info),
            (iter([mon_seg]), mock_info),
        ]
        transcriber = Transcriber(settings)
        transcriber.transcribe_stereo(
            Path("/fake/mic.wav"), Path("/fake/monitor.wav"), on_status=messages.append
        )

    assert any(m.startswith("Stage 'transcribe mic'") for m in messages)
    assert any(m.startswith("Stage 'transcribe monitor'") for m in messages)


def test_transcribe_passes_language_and_hallucination_settings(settings):
    """multilingual / language_detection_segments / hallucination threshold reach Whisper."""
    s = settings.model_copy(
        update={
            "device": "cpu",
            "multilingual": True,
            "language_detection_segments": 4,
            "hallucination_silence_threshold": 2.0,
        }
    )
    info = MagicMock()
    info.language, info.language_probability, info.duration = "en", 0.9, 1.0

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.return_value = (iter([]), info)

        Transcriber(s).transcribe(Path("/fake/audio.wav"))
        kwargs = instance.transcribe.call_args.kwargs

    assert kwargs["multilingual"] is True
    assert kwargs["language_detection_segments"] == 4
    assert kwargs["hallucination_silence_threshold"] == 2.0


def test_transcribe_passes_beam_size_and_temperature(settings):
    """beam_size and the temperature fallback ladder reach Whisper."""
    s = settings.model_copy(update={"device": "cpu", "beam_size": 3, "temperature": (0.0, 0.2)})
    info = MagicMock()
    info.language, info.language_probability, info.duration = "en", 0.9, 1.0

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.return_value = (iter([]), info)

        Transcriber(s).transcribe(Path("/fake/audio.wav"))
        kwargs = instance.transcribe.call_args.kwargs

    assert kwargs["beam_size"] == 3
    assert kwargs["temperature"] == (0.0, 0.2)


def test_batched_inference_used_when_batch_size_positive(settings):
    """batch_size > 0 routes transcription through BatchedInferencePipeline."""
    s = settings.model_copy(update={"device": "cpu", "batch_size": 8})
    info = MagicMock()
    info.language, info.language_probability, info.duration = "en", 0.9, 1.0

    with (
        patch("tapeback.transcriber.WhisperModel") as mock_model_cls,
        patch("tapeback.transcriber.BatchedInferencePipeline") as mock_batched_cls,
    ):
        batched = mock_batched_cls.return_value
        batched.transcribe.return_value = (iter([]), info)

        Transcriber(s).transcribe(Path("/fake/audio.wav"))

    assert batched.transcribe.call_args.kwargs["batch_size"] == 8
    mock_model_cls.return_value.transcribe.assert_not_called()


def test_hotwords_reach_whisper_when_configured(settings):
    """The glossary bias must actually be passed through, not just stored."""
    s = settings.model_copy(update={"device": "cpu", "hotwords": "RAG, ONNX, OpenVINO"})
    info = MagicMock()
    info.language, info.language_probability, info.duration = "ru", 0.9, 1.0

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.return_value = (iter([]), info)

        Transcriber(s).transcribe(Path("/fake/audio.wav"))

    assert instance.transcribe.call_args.kwargs["hotwords"] == "RAG, ONNX, OpenVINO"


def test_default_hotwords_cover_the_terms_the_model_mangles(settings):
    """The shipped glossary must contain the vocabulary that measurably failed."""
    glossary = settings.hotwords.lower()
    for term in ("tapeback", "whisper", "obsidian", "llm", "rag", "onnx", "jira"):
        assert term in glossary, term


def test_hotwords_omitted_when_empty(settings):
    """An empty glossary must not be sent — faster-whisper would tokenise it per window."""
    s = settings.model_copy(update={"device": "cpu", "hotwords": ""})
    info = MagicMock()
    info.language, info.language_probability, info.duration = "ru", 0.9, 1.0

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.return_value = (iter([]), info)

        Transcriber(s).transcribe(Path("/fake/audio.wav"))

    assert "hotwords" not in instance.transcribe.call_args.kwargs


def test_batching_warns_which_settings_it_drops(settings, capsys):
    """Batching silently reverts anti-hallucination settings — the user must be told."""
    s = settings.model_copy(
        update={
            "device": "cpu",
            "batch_size": 8,
            "no_speech_threshold": 0.4,
            "temperature": (0.0, 0.2, 0.4),
        }
    )
    with (
        patch("tapeback.transcriber.WhisperModel"),
        patch("tapeback.transcriber.BatchedInferencePipeline"),
    ):
        Transcriber(s)

    warning = capsys.readouterr().err
    assert "TAPEBACK_BATCH_SIZE=8" in warning
    assert "no_speech_threshold" in warning
    assert "condition_on_previous_text" in warning
    assert "temperature (only the first value is used)" in warning
    # hallucination_silence_threshold defaults to None — nothing is lost, so it is
    # not listed; naming settings the user never set would be noise.
    assert "hallucination_silence_threshold" not in warning


def test_no_batching_warning_when_batching_is_off(settings, capsys):
    s = settings.model_copy(update={"device": "cpu", "batch_size": 0})
    with patch("tapeback.transcriber.WhisperModel"):
        Transcriber(s)

    assert "TAPEBACK_BATCH_SIZE" not in capsys.readouterr().err


def test_describe_reports_resolved_device_and_compute_type(settings):
    """describe() states where the model actually landed, not what was requested."""
    s = settings.model_copy(
        update={
            "device": "cpu",
            "compute_type": "auto",
            "stt_model": "large-v3",
            "batch_size": 0,
        }
    )
    with patch("tapeback.transcriber.WhisperModel"):
        description = Transcriber(s).describe()

    assert description == "Whisper: large-v3 on cpu/int8"


def test_describe_mentions_batch_size_when_batching_enabled(settings):
    """Batched mode changes which parameters faster-whisper honours — make it visible."""
    s = settings.model_copy(
        update={"device": "cpu", "compute_type": "int8", "stt_model": "tiny", "batch_size": 8}
    )
    with (
        patch("tapeback.transcriber.WhisperModel"),
        patch("tapeback.transcriber.BatchedInferencePipeline"),
    ):
        description = Transcriber(s).describe()

    assert description == "Whisper: tiny on cpu/int8, batch_size=8"


def test_describe_reflects_cpu_fallback_after_cuda_failure(settings):
    """After a CUDA fallback describe() must say cpu/int8, not the requested cuda/float16."""
    s = settings.model_copy(update={"device": "cuda", "compute_type": "float16"})
    info = MagicMock()
    info.language, info.language_probability, info.duration = "en", 0.9, 1.0

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.side_effect = [
            RuntimeError("CUDA failed with error out of memory"),
            (iter([]), info),
        ]
        transcriber = Transcriber(s)
        assert transcriber.describe() == "Whisper: large-v3-turbo on cuda/float16"

        transcriber.transcribe(Path("/fake/audio.wav"))

    assert transcriber.describe() == "Whisper: large-v3-turbo on cpu/int8"


def test_transcribe_reports_progress_through_on_status(settings):
    """Long runs must show movement; faster-whisper's own tqdm bypasses on_status."""
    s = settings.model_copy(update={"device": "cpu"})
    info = MagicMock()
    info.language, info.language_probability, info.duration = "en", 0.9, 600.0

    segs = []
    for end in (60.0, 300.0, 540.0):
        seg = MagicMock()
        seg.start, seg.end, seg.text, seg.words = end - 10.0, end, "text", []
        segs.append(seg)

    messages: list[str] = []
    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.return_value = (iter(segs), info)
        # min_interval defaults to 10s of wall clock; a fake clock makes every
        # segment clear the gap so the mapping position -> percent is testable.
        with patch("tapeback._timing.time.monotonic", side_effect=[0.0, 100.0, 200.0, 300.0]):
            Transcriber(s).transcribe(
                Path("/fake/audio.wav"), stage="transcribe mic", on_status=messages.append
            )

    assert messages == [
        "  transcribe mic: 10% (1:00 / 10:00)",
        "  transcribe mic: 50% (5:00 / 10:00)",
        "  transcribe mic: 90% (9:00 / 10:00)",
    ]


def test_transcribe_progress_silent_by_default(settings):
    """A caller that passes no reporter (e.g. live mode) gets no progress output."""
    s = settings.model_copy(update={"device": "cpu"})
    info = MagicMock()
    info.language, info.language_probability, info.duration = "en", 0.9, 600.0
    seg = MagicMock()
    seg.start, seg.end, seg.text, seg.words = 0.0, 60.0, "text", []

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.return_value = (iter([seg]), info)
        segments, _info = Transcriber(s).transcribe(Path("/fake/audio.wav"))

    assert len(segments) == 1


def test_plain_inference_when_batch_size_zero(settings):
    """batch_size == 0 keeps the plain (non-batched) path."""
    s = settings.model_copy(update={"device": "cpu", "batch_size": 0})
    info = MagicMock()
    info.language, info.language_probability, info.duration = "en", 0.9, 1.0

    with (
        patch("tapeback.transcriber.WhisperModel") as mock_model_cls,
        patch("tapeback.transcriber.BatchedInferencePipeline") as mock_batched_cls,
    ):
        instance = mock_model_cls.return_value
        instance.transcribe.return_value = (iter([]), info)

        Transcriber(s).transcribe(Path("/fake/audio.wav"))

    instance.transcribe.assert_called_once()
    mock_batched_cls.assert_not_called()
