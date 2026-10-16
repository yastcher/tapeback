"""The openai_compatible provider: summaries through an endpoint the user names.

The endpoint is a real HTTP server on 127.0.0.1 (`llm_endpoint`); what is asserted is
what reached it, byte for byte. Validation that would otherwise send text somewhere
else is tested on `endpoint_url` alone.
"""

import json
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from tapeback._llm import ENDPOINT, call, endpoint_url, provider_chain
from tapeback.cli import cli
from tapeback.settings import Settings
from tapeback.summarizer import maybe_summarize, summarize
from tests.fixtures import SAMPLE_MD


def _settings(tmp_vault, url: str, **overrides) -> Settings:
    return Settings(
        vault_path=tmp_vault,
        llm_provider="openai_compatible",
        llm_base_url=url,
        llm_model="qwen3.5-4b",
        **overrides,
    )


def test_a_summary_goes_through_the_endpoint(tmp_vault, llm_endpoint, capsys):
    summary = summarize("We ship on Friday.", _settings(tmp_vault, llm_endpoint.url))

    assert summary.brief == "Discussed the project plan and assigned tasks."
    [request] = llm_endpoint.requests
    assert (request.method, request.path) == ("POST", "/v1/chat/completions")
    body = json.loads(request.body)
    assert body["model"] == "qwen3.5-4b"
    assert body["messages"][1] == {"role": "user", "content": "We ship on Friday."}
    assert "Warning" not in capsys.readouterr().err


def test_the_endpoint_is_asked_for_one_answer_not_a_stream(tmp_vault, llm_endpoint):
    """An API may document streaming as its default (anyapi.ai's docs say both); tapeback
    reads one JSON answer, so the request says so."""
    call("system", "transcript", _settings(tmp_vault, llm_endpoint.url))

    [request] = llm_endpoint.requests
    assert json.loads(request.body)["stream"] is False


def test_no_cloud_key_crosses_the_wire(tmp_vault, llm_endpoint, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-cloud-key-must-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-cloud-key-must-not-leak")

    call("system", "transcript", _settings(tmp_vault, llm_endpoint.url))

    sent = b"".join(request.raw for request in llm_endpoint.requests)
    assert b"cloud-key-must-not-leak" not in sent


def test_the_endpoint_key_comes_from_tapeback_llm_api_key(tmp_vault, llm_endpoint):
    settings = _settings(tmp_vault, llm_endpoint.url, llm_api_key=SecretStr("local-secret"))

    call("system", "transcript", settings)

    [request] = llm_endpoint.requests
    assert b"Bearer local-secret" in request.headers


def test_a_failing_endpoint_never_falls_back_to_the_cloud(tmp_vault, llm_endpoint, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-cloud")
    llm_endpoint.status = 500

    with patch("anthropic.Anthropic") as anthropic_client, pytest.raises(Exception, match="500"):
        call("system", "transcript", _settings(tmp_vault, llm_endpoint.url))

    anthropic_client.assert_not_called()
    # Asked once: an SDK retry would make a local model read the whole transcript again.
    assert len(llm_endpoint.requests) == 1


def test_a_cloud_provider_never_falls_back_to_the_endpoint(tmp_vault, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-cloud")
    settings = Settings(
        vault_path=tmp_vault,
        llm_provider="anthropic",
        llm_api_key=SecretStr("sk-ant"),
        llm_base_url="http://127.0.0.1:8080/v1",
    )

    assert [provider for provider, _, _ in provider_chain(settings)] == ["anthropic", "groq"]
    assert ENDPOINT == "openai_compatible"


def test_the_masked_text_is_what_crosses_the_wire(tmp_vault, llm_endpoint):
    settings = _settings(tmp_vault, llm_endpoint.url, mask_pii=True, mask_terms="Alice")

    summarize("Alice ships it on Friday.", settings)

    [request] = llm_endpoint.requests
    assert b"Alice" not in request.raw
    assert b"[TERM_1] ships it on Friday." in request.body


def test_a_misconfigured_endpoint_skips_the_summary_and_says_why(tmp_vault, capsys):
    note = tmp_vault / "meetings" / "2026-10-16_10-00-00.md"
    note.parent.mkdir(parents=True)
    note.write_text(SAMPLE_MD)

    maybe_summarize(note, _settings(tmp_vault, ""))

    assert note.read_text() == SAMPLE_MD
    assert "TAPEBACK_LLM_BASE_URL is not set" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("url", "model", "error"),
    [
        ("", "qwen", "TAPEBACK_LLM_BASE_URL is not set"),
        ("http://127.0.0.1:8080/v1", "", "TAPEBACK_LLM_MODEL is not set"),
        ("127.0.0.1:8080/v1", "qwen", r"not an http\(s\) address"),
    ],
)
def test_an_incomplete_endpoint_is_refused(tmp_vault, url, model, error):
    settings = _settings(tmp_vault, url).model_copy(update={"llm_model": model})

    with pytest.raises(RuntimeError, match=error):
        endpoint_url(settings)


def test_credentials_in_the_address_are_refused_without_echoing_them(tmp_vault):
    settings = _settings(tmp_vault, "https://user:hunter2@llm.example.com/v1")

    with pytest.raises(RuntimeError) as refused:
        endpoint_url(settings)

    assert "llm.example.com" in str(refused.value)
    assert "hunter2" not in str(refused.value)


@pytest.mark.parametrize(
    ("url", "warned"),
    [
        ("http://llm.example.com/v1", True),
        ("https://llm.example.com/v1", False),
        ("http://127.0.0.1:8080/v1", False),
        ("http://localhost:11434/v1", False),
        ("http://[::1]:8080/v1", False),
    ],
)
def test_plain_http_off_this_machine_warns(tmp_vault, capsys, url, warned):
    assert endpoint_url(_settings(tmp_vault, url)) == url

    assert ("plain http" in capsys.readouterr().err) is warned


def test_status_shows_the_endpoint_host_and_port_only(runner, real_env_files):
    user_config, _ = real_env_files
    user_config.write_text(
        "TAPEBACK_LLM_PROVIDER=openai_compatible\n"
        "TAPEBACK_LLM_BASE_URL=http://user:hunter2@127.0.0.1:8080/v1?token=abc\n"
        "TAPEBACK_LLM_MODEL=qwen3.5-4b\n"
    )

    result = runner.invoke(cli, ["status"])

    assert result.exit_code == 0, result.output
    assert "LLM: openai_compatible at 127.0.0.1:8080, model qwen3.5-4b" in result.output
    assert "hunter2" not in result.output
    assert "token=abc" not in result.output


def test_setup_asks_no_key_of_an_endpoint(runner, real_env_files):
    """A local server needs no key, so its absence is no reason for a hint."""
    user_config, _ = real_env_files
    user_config.write_text(
        "TAPEBACK_LLM_PROVIDER=openai_compatible\n"
        "TAPEBACK_LLM_BASE_URL=http://127.0.0.1:8080/v1\n"
        "TAPEBACK_LLM_MODEL=qwen3.5-4b\n"
        "TAPEBACK_HF_TOKEN=hf_fake\n"
    )

    result = runner.invoke(cli, ["setup"])

    assert result.exit_code == 0, result.output
    assert not any(line.startswith("Hint:") for line in result.output.splitlines())
