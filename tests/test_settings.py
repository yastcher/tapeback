from pathlib import Path

import pytest
from pydantic import ValidationError

from tapeback.settings import Settings


def test_settings_from_env(monkeypatch, vault_env):
    """Settings should parse values from environment variables."""
    monkeypatch.setenv("TAPEBACK_WHISPER_MODEL", "tiny")
    monkeypatch.setenv("TAPEBACK_LANGUAGE", "ru")

    s = Settings()
    assert s.vault_path == vault_env
    assert s.whisper_model == "tiny"
    assert s.language == "ru"


def test_settings_vault_path_default(monkeypatch, tmp_path):
    """vault_path should default to ~/tapeback when not set."""
    monkeypatch.delenv("TAPEBACK_VAULT_PATH", raising=False)
    # Use empty dir so no .env is found
    monkeypatch.chdir(tmp_path)

    s = Settings(_env_file=str(tmp_path / "nonexistent.env"))  # type: ignore
    assert s.vault_path == Path.home() / "tapeback"


def test_settings_defaults(tmp_vault):
    """Default settings should match expected values."""
    s = Settings(vault_path=tmp_vault)
    assert s.whisper_model == "large-v3-turbo"
    assert s.language == "auto"
    assert s.device == "cuda"
    assert s.compute_type == "auto"
    assert s.beam_size == 5
    assert s.vad_filter is True
    assert s.monitor_source == "auto"
    assert s.mic_source == "auto"
    assert s.sample_rate == 48000
    assert s.meetings_dir == "meetings"
    assert s.attachments_dir == "attachments/audio"
    assert s.diarize is True
    assert s.max_speakers is None


def test_settings_hf_token_from_env(monkeypatch, vault_env):
    """HF token should be parsed from environment variable."""
    monkeypatch.setenv("TAPEBACK_HF_TOKEN", "hf_test_token_123")

    s = Settings()
    assert s.hf_token.get_secret_value() == "hf_test_token_123"


def test_secrets_not_leaked_in_repr_or_dump(monkeypatch, vault_env):
    """HF token and LLM API key must not appear in repr/str/model_dump output."""
    monkeypatch.setenv("TAPEBACK_HF_TOKEN", "hf_supersecret_abc")
    monkeypatch.setenv("TAPEBACK_LLM_API_KEY", "sk-ant-supersecret-xyz")

    s = Settings()

    assert "hf_supersecret_abc" not in repr(s)
    assert "hf_supersecret_abc" not in str(s)
    assert "hf_supersecret_abc" not in s.model_dump_json()
    assert "sk-ant-supersecret-xyz" not in repr(s)
    assert "sk-ant-supersecret-xyz" not in str(s)
    assert "sk-ant-supersecret-xyz" not in s.model_dump_json()

    assert s.hf_token.get_secret_value() == "hf_supersecret_abc"
    assert s.llm_api_key.get_secret_value() == "sk-ant-supersecret-xyz"


def test_settings_live_defaults(tmp_vault):
    """Live transcription is opt-in: default off, can be enabled via env."""
    s = Settings(vault_path=tmp_vault)
    assert s.live is False
    assert s.live_interval == 60
    assert s.live_overlap == 2.0
    assert s.live_min_chunk == 5.0


def test_settings_live_from_env(monkeypatch, vault_env):
    """Live settings should be configurable via environment variables."""
    monkeypatch.setenv("TAPEBACK_LIVE", "true")
    monkeypatch.setenv("TAPEBACK_LIVE_INTERVAL", "30")
    monkeypatch.setenv("TAPEBACK_LIVE_OVERLAP", "3.0")
    monkeypatch.setenv("TAPEBACK_LIVE_MIN_CHUNK", "10.0")

    s = Settings()
    assert s.live is True
    assert s.live_interval == 30
    assert s.live_overlap == 3.0
    assert s.live_min_chunk == 10.0


# --- Validation ---


def test_spectral_merge_threshold_out_of_range_rejected(tmp_vault):
    with pytest.raises(ValidationError):
        Settings(vault_path=tmp_vault, spectral_merge_threshold=1.5)
    with pytest.raises(ValidationError):
        Settings(vault_path=tmp_vault, spectral_merge_threshold=-0.1)


def test_clustering_threshold_out_of_range_rejected(tmp_vault):
    with pytest.raises(ValidationError):
        Settings(vault_path=tmp_vault, clustering_threshold=1.5)
    with pytest.raises(ValidationError):
        Settings(vault_path=tmp_vault, clustering_threshold=-0.1)


def test_negative_pause_threshold_rejected(tmp_vault):
    with pytest.raises(ValidationError):
        Settings(vault_path=tmp_vault, pause_threshold=-1.0)


def test_live_min_chunk_gt_interval_rejected(tmp_vault):
    """live_min_chunk > live_interval would starve the live transcription loop."""
    with pytest.raises(ValidationError, match="live_min_chunk"):
        Settings(vault_path=tmp_vault, live=True, live_interval=10, live_min_chunk=30.0)


def test_live_min_chunk_gt_interval_allowed_when_live_disabled(tmp_vault):
    """The chunk/interval check only applies when live transcription is enabled."""
    s = Settings(vault_path=tmp_vault, live=False, live_interval=10, live_min_chunk=30.0)
    assert s.live is False
