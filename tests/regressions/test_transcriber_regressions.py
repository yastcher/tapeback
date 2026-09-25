"""Regression tests for transcriber bugs."""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

from tapeback.settings import Settings
from tapeback.transcriber import Transcriber


def test_cuda_fallback_to_cpu(tmp_vault):
    """Should fall back to CPU when CUDA is not available at model load time.

    Bug: Transcriber crashed on machines without CUDA support.
    """
    s = Settings(vault_path=tmp_vault, compute_type="float16")
    call_args = []

    def mock_init(model_name, device="cuda", compute_type="float16", local_files_only=True):
        call_args.append(device)
        if device == "cuda":
            raise RuntimeError("CUDA not available")
        return MagicMock()

    with patch("tapeback.transcriber.WhisperModel", side_effect=mock_init):
        Transcriber(s)

    assert call_args == ["cuda", "cpu"]


def test_cuda_inference_fallback_to_cpu(tmp_vault, capsys):
    """Model loads on CUDA but fails during inference (e.g. missing libcublas).

    Bug: Model loaded fine on CUDA, but crashed during transcription
    when libcublas.so.12 was missing. No recovery, lost recording.
    """
    s = Settings(vault_path=tmp_vault, compute_type="float16")

    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 5.0
    mock_segment.text = "Hello"
    mock_segment.words = []

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_info.duration = 5.0

    def failing_iter():
        """Generator that raises RuntimeError on first iteration (simulates CUDA failure)."""
        raise RuntimeError("Library libcublas.so.12 is not found")
        yield  # make it a generator

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        cuda_model = MagicMock()
        cpu_model = MagicMock()
        mock_model_cls.side_effect = [cuda_model, cpu_model]

        cuda_model.transcribe.return_value = (failing_iter(), mock_info)
        cpu_model.transcribe.return_value = (iter([mock_segment]), mock_info)

        transcriber = Transcriber(s)
        segments, info = transcriber.transcribe(Path("/fake/audio.wav"))

    assert mock_model_cls.call_count == 2
    assert mock_model_cls.call_args_list[0] == call(
        s.stt_model, device="cuda", compute_type="float16", local_files_only=True
    )
    assert mock_model_cls.call_args_list[1] == call(
        s.stt_model, device="cpu", compute_type="int8", local_files_only=True
    )

    assert len(segments) == 1
    assert segments[0].text == "Hello"
    assert info["duration"] == 5.0

    captured = capsys.readouterr()
    assert "CUDA runtime error" in captured.err


def test_cuda_fallback_when_transcribe_call_raises(tmp_vault, capsys):
    """faster-whisper raises RuntimeError synchronously from transcribe() (before iteration).

    Bug: missing libcublas.so.12 makes faster-whisper raise during eager language
    detection inside transcribe() itself — not while iterating the segment generator.
    The previous fallback only wrapped _collect_segments(), so this error escaped
    and aborted the recording.
    """
    s = Settings(vault_path=tmp_vault, compute_type="float16")

    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 5.0
    mock_segment.text = "Hello"
    mock_segment.words = []

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_info.duration = 5.0

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        cuda_model = MagicMock()
        cpu_model = MagicMock()
        mock_model_cls.side_effect = [cuda_model, cpu_model]

        cuda_model.transcribe.side_effect = RuntimeError(
            "Library libcublas.so.12 is not found or cannot be loaded"
        )
        cpu_model.transcribe.return_value = (iter([mock_segment]), mock_info)

        transcriber = Transcriber(s)
        segments, info = transcriber.transcribe(Path("/fake/audio.wav"))

    assert mock_model_cls.call_count == 2
    assert mock_model_cls.call_args_list[1] == call(
        s.stt_model, device="cpu", compute_type="int8", local_files_only=True
    )
    assert len(segments) == 1
    assert segments[0].text == "Hello"
    assert info["duration"] == 5.0

    captured = capsys.readouterr()
    assert "CUDA runtime error" in captured.err


def test_cuda_inference_fallback_preserves_auto_language(tmp_vault):
    """CPU fallback must pass language=None (not "auto" string) when auto-detection is on.

    Bug: transcriber.py line 145 used self._settings.language ("auto") instead
    of the resolved variable (None). Whisper doesn't accept "auto" as a language code.
    """
    s = Settings(vault_path=tmp_vault, compute_type="float16", language="auto")

    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 5.0
    mock_segment.text = "Привет"
    mock_segment.words = []

    mock_info = MagicMock()
    mock_info.language = "ru"
    mock_info.language_probability = 0.99
    mock_info.duration = 5.0

    def failing_iter():
        raise RuntimeError("CUDA OOM")
        yield  # make it a generator

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        cuda_model = MagicMock()
        cpu_model = MagicMock()
        mock_model_cls.side_effect = [cuda_model, cpu_model]

        cuda_model.transcribe.return_value = (failing_iter(), mock_info)
        cpu_model.transcribe.return_value = (iter([mock_segment]), mock_info)

        transcriber = Transcriber(s)
        segments, info = transcriber.transcribe(Path("/fake/audio.wav"))

    # CPU retry must pass language=None (auto-detect), not "auto" string
    retry_kwargs = cpu_model.transcribe.call_args.kwargs
    assert retry_kwargs["language"] is None

    assert len(segments) == 1
    assert info["language"] == "ru"


def test_empty_transcription(settings, capsys):
    """Empty transcription result should return empty list with warning.

    Bug: No speech in audio caused silent empty output without any feedback.
    """
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_info.duration = 10.0

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.return_value = (iter([]), mock_info)

        transcriber = Transcriber(settings)
        segments, _info = transcriber.transcribe(Path("/fake/audio.wav"))

    assert segments == []
    captured = capsys.readouterr()
    assert "No speech detected" in captured.err
