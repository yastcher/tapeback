from pathlib import Path

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
    assert s.hf_token == "hf_test_token_123"


def test_settings_live_defaults(tmp_vault):
    """Live transcription settings should have correct defaults."""
    s = Settings(vault_path=tmp_vault)
    assert s.live is True
    assert s.live_interval == 60
    assert s.live_overlap == 2.0
    assert s.live_min_chunk == 5.0


def test_settings_live_from_env(monkeypatch, vault_env):
    """Live settings should be configurable via environment variables."""
    monkeypatch.setenv("TAPEBACK_LIVE", "false")
    monkeypatch.setenv("TAPEBACK_LIVE_INTERVAL", "30")
    monkeypatch.setenv("TAPEBACK_LIVE_OVERLAP", "3.0")
    monkeypatch.setenv("TAPEBACK_LIVE_MIN_CHUNK", "10.0")

    s = Settings()
    assert s.live is False
    assert s.live_interval == 30
    assert s.live_overlap == 3.0
    assert s.live_min_chunk == 10.0
