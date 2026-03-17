import pytest

from meetrec.settings import Settings, get_settings


def test_settings_from_env(monkeypatch, tmp_path):
    """Settings should parse values from environment variables."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("MEETREC_VAULT_PATH", str(vault))
    monkeypatch.setenv("MEETREC_WHISPER_MODEL", "tiny")
    monkeypatch.setenv("MEETREC_LANGUAGE", "ru")

    s = Settings()
    assert s.vault_path == vault
    assert s.whisper_model == "tiny"
    assert s.language == "ru"


def test_settings_vault_path_required(monkeypatch, tmp_path):
    """get_settings should exit with clear message if MEETREC_VAULT_PATH is not set."""
    monkeypatch.delenv("MEETREC_VAULT_PATH", raising=False)
    # Change to a temp dir without .env file
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        get_settings()

    assert "MEETREC_VAULT_PATH" in str(exc_info.value)


def test_settings_defaults(tmp_vault):
    """Default settings should match expected values."""
    s = Settings(vault_path=tmp_vault)
    assert s.whisper_model == "large-v3-turbo"
    assert s.language == "en"
    assert s.device == "cuda"
    assert s.compute_type == "float16"
    assert s.beam_size == 5
    assert s.vad_filter is True
    assert s.monitor_source == "auto"
    assert s.mic_source == "auto"
    assert s.sample_rate == 48000
    assert s.meetings_dir == "meetings"
    assert s.attachments_dir == "attachments/audio"
    assert s.diarize is True
    assert s.max_speakers is None


def test_settings_hf_token_from_env(monkeypatch, tmp_path):
    """HF token should be parsed from environment variable."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("MEETREC_VAULT_PATH", str(vault))
    monkeypatch.setenv("MEETREC_HF_TOKEN", "hf_test_token_123")

    s = Settings()
    assert s.hf_token == "hf_test_token_123"
