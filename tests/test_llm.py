"""LLM transport: keys, the provider table, the fallback chain, retries.

Mocked only at the SDK boundary (anthropic.Anthropic / openai.OpenAI).
"""

from typing import get_args
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from tapeback._llm import ENDPOINT, LOCAL, PROVIDERS, api_key, call, has_api_key, provider_chain
from tapeback.settings import LLMProvider, Settings
from tests.fixtures import (
    VALID_LLM_RESPONSE,
    HttpError,
    clear_all_provider_env_vars,
    mock_anthropic_response,
)


def test_the_table_covers_every_provider_settings_accepts():
    assert set(PROVIDERS) | {ENDPOINT, LOCAL} == set(get_args(LLMProvider.__value__))


# --- API key resolution ---


def test_api_key_resolution_tapeback_env(tmp_vault, monkeypatch):
    """TAPEBACK_LLM_API_KEY takes priority over provider-specific env vars."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "provider-key")
    settings = Settings(vault_path=tmp_vault, llm_api_key=SecretStr("tapeback-key"))

    assert api_key("anthropic", settings) == "tapeback-key"


@pytest.mark.parametrize(
    ("provider", "env_var"),
    [
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("groq", "GROQ_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("qwen", "DASHSCOPE_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
    ],
)
def test_api_key_resolution_provider_env(tmp_vault, monkeypatch, provider, env_var):
    """Falls back to provider-specific env var when TAPEBACK_LLM_API_KEY is empty."""
    monkeypatch.setenv(env_var, "test-key")
    settings = Settings(vault_path=tmp_vault, llm_provider=provider, llm_api_key=SecretStr(""))

    assert api_key(provider, settings) == "test-key"
    assert has_api_key(settings) is True


def test_no_key_anywhere_is_empty(tmp_vault, monkeypatch):
    clear_all_provider_env_vars(monkeypatch)
    settings = Settings(vault_path=tmp_vault, llm_api_key=SecretStr(""))

    assert api_key("anthropic", settings) == ""
    assert has_api_key(settings) is False


# --- Provider chain building ---


@pytest.mark.parametrize(
    ("provider", "default_model"),
    [
        ("anthropic", "claude-sonnet-5-5"),
        ("groq", "openai/gpt-oss-120b"),
        ("openrouter", "nvidia/nemotron-3-super-120b-a12b:free"),
        ("gemini", "gemini-3.8-flash"),
        ("deepseek", "deepseek-flash"),
        ("qwen", "qwen-plus"),
        ("openai", "gpt-6-luna"),
    ],
)
def test_each_provider_has_a_default_model(tmp_vault, monkeypatch, provider, default_model):
    clear_all_provider_env_vars(monkeypatch)
    settings = Settings(
        vault_path=tmp_vault, llm_provider=provider, llm_api_key=SecretStr("k"), llm_model=""
    )

    assert provider_chain(settings) == [(provider, "k", default_model)]


def test_build_provider_chain_primary_first(tmp_vault, monkeypatch):
    """Primary provider appears first in chain."""
    clear_all_provider_env_vars(monkeypatch)
    settings = Settings(
        vault_path=tmp_vault, llm_provider="groq", llm_api_key=SecretStr("main-key"), llm_model=""
    )

    chain = provider_chain(settings)

    assert chain == [("groq", "main-key", "openai/gpt-oss-120b")]


def test_build_provider_chain_includes_fallbacks(tmp_vault, monkeypatch):
    """Chain includes all providers with available API keys."""
    clear_all_provider_env_vars(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    settings = Settings(
        vault_path=tmp_vault, llm_provider="anthropic", llm_api_key=SecretStr("ant-key")
    )

    chain = provider_chain(settings)

    providers = [p for p, _, _ in chain]
    assert providers == ["anthropic", "groq", "gemini"]


def test_build_provider_chain_skips_providers_without_keys(tmp_vault, monkeypatch):
    """Providers without API keys are excluded from chain."""
    clear_all_provider_env_vars(monkeypatch)
    settings = Settings(vault_path=tmp_vault, llm_provider="anthropic", llm_api_key=SecretStr(""))

    chain = provider_chain(settings)

    assert chain == []


def test_build_provider_chain_explicit_model_for_primary_only(tmp_vault, monkeypatch):
    """Explicit model setting applies only to primary provider."""
    clear_all_provider_env_vars(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    settings = Settings(
        vault_path=tmp_vault,
        llm_provider="anthropic",
        llm_api_key=SecretStr("ant-key"),
        llm_model="claude-opus-4-20250514",
    )

    chain = provider_chain(settings)

    assert chain[0] == ("anthropic", "ant-key", "claude-opus-4-20250514")
    assert chain[1] == ("groq", "groq-key", "openai/gpt-oss-120b")


# --- Calls: retry and fallback ---


def test_provider_retry_and_fallback_flow(tmp_vault, monkeypatch):
    """Rate limit retry, provider fallback, all-fail, empty chain."""
    clear_all_provider_env_vars(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    settings = Settings(
        vault_path=tmp_vault, llm_provider="anthropic", llm_api_key=SecretStr("ant-key")
    )

    # 429 on anthropic → retry with backoff → succeed on second attempt
    with (
        patch("anthropic.Anthropic") as mock_ant,
        patch("tapeback._llm.time.sleep") as mock_sleep,
    ):
        mock_ant.return_value.messages.create.side_effect = [
            HttpError("rate limited", 429),
            mock_anthropic_response(VALID_LLM_RESPONSE),
        ]
        result = call("system", "user", settings)
    assert "Discussed the project plan" in result
    mock_sleep.assert_called_once_with(5)

    # Primary fails (500, non-retryable) → fallback to groq (openai-compatible)
    groq_response = MagicMock()
    groq_response.choices = [MagicMock(message=MagicMock(content="fallback ok"))]
    with (
        patch("anthropic.Anthropic") as mock_ant,
        patch("openai.OpenAI") as mock_oai,
    ):
        mock_ant.return_value.messages.create.side_effect = HttpError("server error", 500)
        mock_oai.return_value.chat.completions.create.return_value = groq_response
        result = call("system", "user", settings)
    assert result == "fallback ok"

    # All providers fail → raises last exception
    with (
        patch("anthropic.Anthropic") as mock_ant,
        patch("openai.OpenAI") as mock_oai,
        pytest.raises(Exception, match="groq error"),
    ):
        mock_ant.return_value.messages.create.side_effect = HttpError("anthropic error", 500)
        mock_oai.return_value.chat.completions.create.side_effect = HttpError("groq error", 500)
        call("system", "user", settings)

    # No providers at all → RuntimeError
    clear_all_provider_env_vars(monkeypatch)
    no_key_settings = Settings(vault_path=tmp_vault, llm_api_key=SecretStr(""))
    with pytest.raises(RuntimeError, match="No LLM providers available"):
        call("system", "user", no_key_settings)
