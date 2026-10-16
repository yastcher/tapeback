"""Regression: a `~` in a path setting was taken literally.

README's quick start writes `TAPEBACK_VAULT_PATH=~/Documents/obsidian/vault` in single
quotes, and neither a .env file nor pydantic expands it: the vault became a relative
path, a directory literally named `~` inside wherever tapeback was started.
"""

from tapeback.settings import Settings


def test_a_tilde_in_a_path_setting_is_the_home_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TAPEBACK_VAULT_PATH", "~/vault")
    monkeypatch.setenv("TAPEBACK_RUN_LOG_DIR", "~/runs")
    monkeypatch.setenv("TAPEBACK_RESUME_CACHE_DIR", "~/resume")
    monkeypatch.setenv("TAPEBACK_SESSIONS_DIR", "~/sessions")

    settings = Settings()

    assert settings.vault_path == tmp_path / "vault"
    assert settings.run_log_dir == tmp_path / "runs"
    assert settings.resume_cache_dir == tmp_path / "resume"
    assert settings.sessions_dir == tmp_path / "sessions"


def test_a_path_without_a_tilde_is_kept_as_given(tmp_path, monkeypatch):
    monkeypatch.setenv("TAPEBACK_VAULT_PATH", str(tmp_path / "vault"))

    settings = Settings()

    assert settings.vault_path == tmp_path / "vault"
    assert settings.sessions_dir is None
