"""Regression: TAPEBACK_WHISPER_MODEL must not leak into remote STT backends."""

import pytest

from tapeback.settings import Settings


def test_whisper_model_ignored_for_openai_backend(monkeypatch, vault_env):
    """WHISPER_MODEL + openai backend (no STT_MODEL) must resolve to whisper-1."""
    monkeypatch.delenv("TAPEBACK_STT_MODEL", raising=False)
    monkeypatch.setenv("TAPEBACK_WHISPER_MODEL", "large-v3-turbo")
    monkeypatch.setenv("TAPEBACK_STT_BACKEND", "openai")

    with pytest.warns(FutureWarning, match="TAPEBACK_WHISPER_MODEL"):
        s = Settings()
    assert s.stt_backend == "openai"
    assert s.stt_model == "whisper-1"
