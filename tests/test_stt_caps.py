"""Remote STT capability matrix — local diarize policy per backend/model."""

import pytest

from tapeback._stt_caps import (
    allows_local_diarize,
    openai_capabilities_for,
)
from tapeback.remote_stt import load_remote_transcriber
from tapeback.settings import Settings


def test_whisper_1_allows_local_diarize():
    caps = openai_capabilities_for("whisper-1")
    assert caps.word_timestamps is True
    assert caps.remote_diarize is False
    assert caps.allows_local_diarize is True
    assert allows_local_diarize("openai", "whisper-1") is True


def test_gpt_transcribe_skips_local_diarize():
    caps = openai_capabilities_for("gpt-transcribe")
    assert caps.word_timestamps is False
    assert caps.remote_diarize is False
    assert caps.modern_context is True
    assert caps.allows_local_diarize is False
    assert allows_local_diarize("openai", "gpt-transcribe") is False


def test_gpt_diarize_skips_local_diarize():
    caps = openai_capabilities_for("gpt-4o-transcribe-diarize")
    assert caps.remote_diarize is True
    assert caps.allows_local_diarize is False
    assert caps.max_upload_seconds == 1400.0
    assert allows_local_diarize("openai", "gpt-4o-transcribe-diarize") is False


def test_whisper_1_has_no_duration_cap():
    caps = openai_capabilities_for("whisper-1")
    assert caps.max_upload_seconds is None


def test_gpt_transcribe_has_duration_cap():
    caps = openai_capabilities_for("gpt-transcribe")
    assert caps.max_upload_seconds == 1500.0


def test_local_backend_always_allows_diarize():
    assert allows_local_diarize("local", "gpt-transcribe") is True


def test_unknown_remote_backend_skips_local_diarize():
    assert allows_local_diarize("future-vendor", "any-model") is False


def test_load_remote_transcriber_rejects_unknown(tmp_vault):
    s = Settings(vault_path=tmp_vault, stt_backend="openai")
    # Bypass Literal validation by copying with an unsupported id.
    s = s.model_copy(update={"stt_backend": "acme"})  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="Unsupported remote STT backend"):
        load_remote_transcriber(s)
