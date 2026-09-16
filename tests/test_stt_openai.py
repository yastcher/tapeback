"""OpenAI remote STT backend — mocked at the SDK / ffmpeg boundary."""

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from tapeback._lazy import load_transcriber
from tapeback._stt_media import _UploadJob
from tapeback._stt_openai import OpenAITranscriber, _resolve_api_key
from tapeback._stt_openai_fmt import _info_from_response, _segments_from_verbose
from tapeback.models import Segment
from tapeback.settings import Settings
from tests.fixtures import create_silent_wav


def test_resolve_api_key_from_stt_api_key(tmp_vault):
    s = Settings(vault_path=tmp_vault, stt_api_key=SecretStr("sk-from-stt"))
    assert _resolve_api_key(s) == "sk-from-stt"


def test_resolve_api_key_from_openai_env(monkeypatch, settings):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    assert _resolve_api_key(settings) == "sk-from-env"


def test_resolve_api_key_from_llm_key_when_provider_openai(monkeypatch, tmp_vault):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    s = Settings(
        vault_path=tmp_vault,
        llm_provider="openai",
        llm_api_key=SecretStr("sk-from-llm"),
    )
    assert _resolve_api_key(s) == "sk-from-llm"


def test_resolve_api_key_ignores_llm_key_for_non_openai_provider(monkeypatch, tmp_vault):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    s = Settings(
        vault_path=tmp_vault,
        llm_provider="anthropic",
        llm_api_key=SecretStr("sk-ant-not-for-stt"),
    )
    with pytest.raises(RuntimeError, match="TAPEBACK_STT_API_KEY"):
        _resolve_api_key(s)


def test_resolve_api_key_missing_raises(monkeypatch, settings):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TAPEBACK_STT_API_KEY"):
        _resolve_api_key(settings)


def test_openai_transcriber_describe(monkeypatch, settings):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    t = OpenAITranscriber(settings.model_copy(update={"stt_model": "whisper-1"}))
    assert t.describe() == "OpenAI STT: whisper-1 (timestamps; local diarize ok)"

    t2 = OpenAITranscriber(settings.model_copy(update={"stt_model": "gpt-transcribe"}))
    assert t2.describe() == "OpenAI STT: gpt-transcribe (text; local diarize skipped)"

    t3 = OpenAITranscriber(settings.model_copy(update={"stt_model": "gpt-4o-transcribe-diarize"}))
    assert t3.describe() == "OpenAI STT: gpt-4o-transcribe-diarize (remote diarize)"


def test_openai_transcriber_missing_api_key_raises(monkeypatch, settings):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TAPEBACK_STT_API_KEY"):
        OpenAITranscriber(settings)


def test_openai_transcriber_calls_api_and_maps_segments(monkeypatch, settings, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    audio = tmp_path / "clip.wav"
    create_silent_wav(audio, duration=0.5, sample_rate=16000)

    api_response = SimpleNamespace(
        language="en",
        duration=0.5,
        segments=[
            SimpleNamespace(
                start=0.0,
                end=0.5,
                text="hi",
                words=[SimpleNamespace(word="hi", start=0.0, end=0.5)],
            )
        ],
    )
    mock_create = MagicMock(return_value=api_response)
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create = mock_create

    s = settings.model_copy(
        update={
            "stt_backend": "openai",
            "stt_model": "whisper-1",
            "resume_cache": False,
            "language": "en",
            "hotwords": "tapeback",
        }
    )

    with (
        patch("tapeback._stt_openai._encode_mp3") as encode,
        patch("openai.OpenAI", return_value=mock_client) as openai_cls,
    ):

        def _fake_encode(_wav: Path, mp3: Path) -> None:
            mp3.write_bytes(b"fake-mp3-bytes")

        encode.side_effect = _fake_encode
        segments, info = OpenAITranscriber(s).transcribe(audio, stage="transcribe")

    assert openai_cls.call_args.kwargs["api_key"] == "sk-test"
    assert mock_create.called
    kwargs = mock_create.call_args.kwargs
    assert kwargs["model"] == "whisper-1"
    assert kwargs["language"] == "en"
    assert kwargs["response_format"] == "verbose_json"
    assert kwargs["timestamp_granularities"] == ["word", "segment"]
    assert kwargs["prompt"] == "tapeback"
    assert len(segments) == 1
    assert segments[0].text == "hi"
    assert info["language"] == "en"
    assert info["partial"] is False


def test_openai_transcriber_text_only_model_skips_verbose(monkeypatch, settings, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    audio = tmp_path / "clip.wav"
    create_silent_wav(audio, duration=0.5, sample_rate=16000)

    api_response = SimpleNamespace(text="plain text only", languages=[SimpleNamespace(code="en")])
    mock_create = MagicMock(return_value=api_response)
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create = mock_create

    s = settings.model_copy(
        update={
            "stt_model": "gpt-transcribe",
            "resume_cache": False,
            "language": "en",
            "hotwords": "tapeback, RabbitMQ",
        }
    )

    with (
        patch("tapeback._stt_openai._encode_mp3") as encode,
        patch("openai.OpenAI", return_value=mock_client),
    ):

        def _fake_encode(_wav: Path, mp3: Path) -> None:
            mp3.write_bytes(b"x")

        encode.side_effect = _fake_encode
        segments, info = OpenAITranscriber(s).transcribe(audio)

    kwargs = mock_create.call_args.kwargs
    assert "response_format" not in kwargs
    assert "timestamp_granularities" not in kwargs
    assert "language" not in kwargs
    assert kwargs["extra_body"]["languages"] == ["en"]
    assert kwargs["extra_body"]["keywords"] == ["tapeback", "RabbitMQ"]
    assert kwargs["prompt"] == "tapeback, RabbitMQ"
    assert len(segments) == 1
    assert segments[0].text == "plain text only"
    assert segments[0].words is None
    assert info["language"] == "en"


def test_openai_transcriber_diarize_model_maps_speakers(monkeypatch, settings, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    audio = tmp_path / "clip.wav"
    create_silent_wav(audio, duration=0.5, sample_rate=16000)

    api_response = SimpleNamespace(
        language="en",
        segments=[
            SimpleNamespace(speaker="A", start=0.0, end=0.2, text="hello"),
            SimpleNamespace(speaker="B", start=0.2, end=0.5, text="world"),
            SimpleNamespace(speaker="A", start=0.5, end=0.8, text="again"),
        ],
    )
    mock_create = MagicMock(return_value=api_response)
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create = mock_create

    s = settings.model_copy(
        update={
            "stt_model": "gpt-4o-transcribe-diarize",
            "resume_cache": False,
            "language": "en",
            "hotwords": "should-not-prompt",
        }
    )

    with (
        patch("tapeback._stt_openai._encode_mp3") as encode,
        patch("tapeback._stt_openai._wav_duration", return_value=45.0),
        patch("openai.OpenAI", return_value=mock_client),
    ):

        def _fake_encode(_wav: Path, mp3: Path) -> None:
            mp3.write_bytes(b"x")

        encode.side_effect = _fake_encode
        segments, _info = OpenAITranscriber(s).transcribe(audio)

    kwargs = mock_create.call_args.kwargs
    assert kwargs["response_format"] == "diarized_json"
    assert kwargs["chunking_strategy"] == "auto"
    assert "prompt" not in kwargs
    assert [seg.text for seg in segments] == ["hello", "world", "again"]
    assert [seg.speaker for seg in segments] == ["Speaker 1", "Speaker 2", "Speaker 1"]


def test_openai_transcriber_chunks_long_audio(monkeypatch, settings, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    audio = tmp_path / "long.wav"
    create_silent_wav(audio, duration=1.0, sample_rate=16000)

    s = settings.model_copy(
        update={
            "stt_model": "whisper-1",
            "stt_concurrency": 4,
            "resume_cache": False,
            "language": "en",
        }
    )
    t = OpenAITranscriber(s)

    responses = {
        0.0: SimpleNamespace(
            language="en",
            duration=100.0,
            segments=[SimpleNamespace(start=0.0, end=1.0, text="first", words=None)],
        ),
        100.0: SimpleNamespace(
            language="en",
            duration=50.0,
            segments=[SimpleNamespace(start=0.0, end=1.0, text="second", words=None)],
        ),
    }
    upload_calls: list[float] = []

    def _fake_one_upload(job: _UploadJob, on_status):
        upload_calls.append(job.time_offset)
        resp = responses[job.time_offset]
        return (
            _segments_from_verbose(resp, job.time_offset),
            _info_from_response(resp, job.duration, language_fallback=job.language),
        )

    with (
        patch("tapeback._stt_openai._wav_duration", return_value=150.0),
        patch("tapeback._stt_openai._max_chunk_seconds", return_value=100.0),
        patch("tapeback._stt_openai._slice_wav") as slice_wav,
        patch.object(OpenAITranscriber, "_transcribe_one_upload", side_effect=_fake_one_upload),
    ):
        segments, info = t.transcribe(audio)

    assert slice_wav.call_count == 2
    assert sorted(upload_calls) == [0.0, 100.0]
    assert [seg.text for seg in segments] == ["first", "second"]
    assert segments[0].start == 0.0
    assert segments[1].start == 100.0
    assert info["duration"] == 150.0
    assert info["partial"] is False


def test_openai_transcriber_chunks_run_in_parallel(monkeypatch, settings, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    audio = tmp_path / "long.wav"
    create_silent_wav(audio, duration=1.0, sample_rate=16000)

    s = settings.model_copy(
        update={
            "stt_model": "whisper-1",
            "stt_concurrency": 3,
            "resume_cache": False,
            "language": "en",
        }
    )
    t = OpenAITranscriber(s)

    active = 0
    max_active = 0
    lock = threading.Lock()
    barrier = threading.Barrier(3)

    def _fake_one_upload(job: _UploadJob, on_status):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        barrier.wait(timeout=2.0)
        with lock:
            active -= 1
        return (
            [
                Segment(
                    start=job.time_offset,
                    end=job.time_offset + 1.0,
                    text=f"at-{job.time_offset:.0f}",
                    words=None,
                )
            ],
            {
                "language": "en",
                "language_probability": 1.0,
                "duration": job.duration,
                "partial": False,
            },
        )

    with (
        patch("tapeback._stt_openai._wav_duration", return_value=300.0),
        patch("tapeback._stt_openai._max_chunk_seconds", return_value=100.0),
        patch("tapeback._stt_openai._slice_wav"),
        patch.object(OpenAITranscriber, "_transcribe_one_upload", side_effect=_fake_one_upload),
    ):
        segments, info = t.transcribe(audio)

    assert max_active == 3
    assert [seg.text for seg in segments] == ["at-0", "at-100", "at-200"]
    assert info["duration"] == 300.0


def test_diarize_model_chunks_under_api_duration_cap(monkeypatch, settings, tmp_path):
    """Size-only slicing leaves ~50min MP3s; diarize API rejects >1400s — cap by duration."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    audio = tmp_path / "long.wav"
    create_silent_wav(audio, duration=1.0, sample_rate=16000)

    s = settings.model_copy(
        update={
            "stt_model": "gpt-4o-transcribe-diarize",
            "resume_cache": False,
            "language": "en",
        }
    )
    t = OpenAITranscriber(s)
    # Size-based limit alone would allow a single ~2988s upload (the failing case).
    size_only_limit = 2988.0
    expected_chunk = 1400.0 * 0.95
    piece_durations: list[float] = []

    def _fake_one_upload(job: _UploadJob, on_status, speaker_order=None):
        piece_durations.append(job.duration)
        assert job.duration <= expected_chunk
        return (
            [
                Segment(
                    start=job.time_offset,
                    end=job.time_offset + 0.5,
                    text=f"at-{job.time_offset:.0f}",
                    words=None,
                    speaker="Speaker 1",
                )
            ],
            {
                "language": "en",
                "language_probability": 1.0,
                "duration": job.duration,
                "partial": False,
            },
        )

    with (
        patch("tapeback._stt_openai._wav_duration", return_value=3000.0),
        patch("tapeback._stt_openai._max_chunk_seconds", return_value=size_only_limit),
        patch("tapeback._stt_openai._slice_wav"),
        patch.object(OpenAITranscriber, "_transcribe_one_upload", side_effect=_fake_one_upload),
    ):
        segments, info = t.transcribe(audio)

    assert piece_durations == [expected_chunk, expected_chunk, 3000.0 - 2 * expected_chunk]
    assert len(segments) == 3
    assert info["duration"] == 3000.0


def test_openai_transcriber_auto_language_locks_then_parallels(monkeypatch, settings, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    audio = tmp_path / "long.wav"
    create_silent_wav(audio, duration=1.0, sample_rate=16000)

    s = settings.model_copy(
        update={
            "stt_model": "whisper-1",
            "stt_concurrency": 4,
            "resume_cache": False,
            "language": "auto",
        }
    )
    t = OpenAITranscriber(s)
    seen_languages: list[str | None] = []

    def _fake_one_upload(job: _UploadJob, on_status):
        seen_languages.append(job.language)
        lang = job.language or "ru"
        return (
            [
                Segment(
                    start=job.time_offset,
                    end=job.time_offset + 0.5,
                    text=f"chunk-{job.time_offset:.0f}",
                    words=None,
                )
            ],
            {
                "language": lang,
                "language_probability": 1.0,
                "duration": job.duration,
                "partial": False,
            },
        )

    with (
        patch("tapeback._stt_openai._wav_duration", return_value=200.0),
        patch("tapeback._stt_openai._max_chunk_seconds", return_value=100.0),
        patch("tapeback._stt_openai._slice_wav"),
        patch.object(OpenAITranscriber, "_transcribe_one_upload", side_effect=_fake_one_upload),
    ):
        segments, info = t.transcribe(audio)

    assert seen_languages[0] is None
    assert seen_languages[1:] == ["ru"]
    assert [seg.text for seg in segments] == ["chunk-0", "chunk-100"]
    assert info["language"] == "ru"


def test_openai_transcribe_stereo_language_and_speaker(monkeypatch, settings, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mic = tmp_path / "mic.wav"
    monitor = tmp_path / "monitor.wav"
    create_silent_wav(mic, duration=0.5, sample_rate=16000)
    create_silent_wav(monitor, duration=0.5, sample_rate=16000)

    s = settings.model_copy(
        update={
            "stt_model": "whisper-1",
            "resume_cache": False,
            "language": "auto",
        }
    )
    t = OpenAITranscriber(s)
    order: list[str] = []

    def _fake_transcribe(
        audio_path: Path,
        *,
        stage: str = "transcribe",
        on_status=None,
        language_override: str | None = None,
    ):
        order.append(stage)
        if stage == "transcribe monitor":
            return (
                [Segment(start=0.0, end=0.5, text="hello from them", words=None)],
                {
                    "language": "ru",
                    "language_probability": 1.0,
                    "duration": 0.5,
                    "partial": False,
                },
            )
        assert language_override == "ru"
        return (
            [Segment(start=0.0, end=0.4, text="hello from me", words=None)],
            {
                "language": "ru",
                "language_probability": 1.0,
                "duration": 0.5,
                "partial": False,
            },
        )

    with patch.object(OpenAITranscriber, "transcribe", side_effect=_fake_transcribe):
        mic_segs, mon_segs, info = t.transcribe_stereo(mic, monitor)

    assert order == ["transcribe monitor", "transcribe mic"]
    assert mon_segs[0].text == "hello from them"
    assert mic_segs[0].text == "hello from me"
    assert mic_segs[0].speaker == "You"
    assert info["language"] == "ru"
    assert info["partial"] is False


def test_load_transcriber_selects_openai(monkeypatch, settings):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    s = settings.model_copy(update={"stt_backend": "openai"})
    t = load_transcriber(s)
    assert isinstance(t, OpenAITranscriber)
