"""Summarizer tests — markdown formatting, injection, extraction, LLM mocking."""

import json
from unittest.mock import patch

import pytest

from meetrec.models import ActionItem, Summary
from meetrec.settings import Settings
from meetrec.summarizer import (
    _get_model,
    _resolve_api_key,
    extract_transcript_from_markdown,
    format_summary_markdown,
    inject_summary_into_markdown,
    summarize,
)

SAMPLE_MD = """\
---
date: 2026-03-20
time: "14:30"
duration: "00:10:00"
language: en
---

# Meeting 2026-03-20 14:30

[00:00:01] **You:** Hello there.
[00:00:05] **Speaker 1:** Hi, let's discuss the plan.
"""

SAMPLE_MD_WITH_SUMMARY = """\
---
date: 2026-03-20
time: "14:30"
duration: "00:10:00"
language: en
---

## Summary

Old summary here.

---

# Meeting 2026-03-20 14:30

[00:00:01] **You:** Hello there.
"""

VALID_LLM_RESPONSE = json.dumps(
    {
        "brief": "Discussed the project plan and assigned tasks.",
        "action_items": [
            {"assignee": "You", "action": "Send the report", "deadline": "Friday"},
            {"assignee": "Speaker 1", "action": "Review the code", "deadline": None},
        ],
        "key_decisions": ["Use PostgreSQL instead of MongoDB"],
        "is_trivial": False,
    }
)

TRIVIAL_LLM_RESPONSE = json.dumps(
    {
        "brief": "Short status sync, no substantive topics discussed.",
        "action_items": [],
        "key_decisions": [],
        "is_trivial": True,
    }
)


@pytest.fixture
def summarize_settings(tmp_vault):
    return Settings(
        vault_path=tmp_vault,
        llm_provider="anthropic",
        llm_api_key="sk-ant-test-key",
    )


# --- format_summary_markdown ---


def test_format_summary_markdown_full():
    """Summary with action items and decisions → correct markdown sections."""
    summary = Summary(
        brief="Discussed the project plan.",
        action_items=[
            ActionItem(assignee="You", action="Send the report", deadline="Friday"),
            ActionItem(assignee="Speaker 1", action="Review the code"),
        ],
        key_decisions=["Use PostgreSQL instead of MongoDB"],
    )

    result = format_summary_markdown(summary)

    assert "## Summary" in result
    assert "Discussed the project plan." in result
    assert "### Action Items" in result
    assert "- [ ] **You:** Send the report by Friday" in result
    assert "- [ ] **Speaker 1:** Review the code" in result
    assert "### Key Decisions" in result
    assert "- Use PostgreSQL instead of MongoDB" in result
    assert result.strip().endswith("---")


def test_format_summary_markdown_trivial():
    """Trivial meeting → no Action Items/Decisions sections."""
    summary = Summary(
        brief="Short sync, nothing important.",
        is_trivial=True,
    )

    result = format_summary_markdown(summary)

    assert "## Summary" in result
    assert "Short sync, nothing important." in result
    assert "### Action Items" not in result
    assert "### Key Decisions" not in result


# --- inject_summary_into_markdown ---


def test_inject_summary_into_markdown_new():
    """Inject summary into file without existing summary."""
    summary_md = "\n## Summary\n\nTest summary.\n\n---\n"

    result = inject_summary_into_markdown(SAMPLE_MD, summary_md)

    assert "## Summary" in result
    assert "Test summary." in result
    # Frontmatter preserved
    assert result.startswith("---\n")
    assert "date: 2026-03-20" in result
    # Transcript preserved
    assert "# Meeting 2026-03-20 14:30" in result
    assert "Hello there." in result
    # Summary is between frontmatter and transcript
    fm_end = result.index("---\n", 4) + 4
    summary_pos = result.index("## Summary")
    transcript_pos = result.index("# Meeting")
    assert fm_end <= summary_pos < transcript_pos


def test_inject_summary_into_markdown_replace():
    """Replace existing summary, keep transcript intact."""
    new_summary_md = "\n## Summary\n\nNew summary.\n\n---\n"

    result = inject_summary_into_markdown(SAMPLE_MD_WITH_SUMMARY, new_summary_md)

    assert "New summary." in result
    assert "Old summary here." not in result
    assert "# Meeting 2026-03-20 14:30" in result
    assert "Hello there." in result


# --- extract_transcript_from_markdown ---


def test_extract_transcript_from_markdown():
    """Extract transcript from file with frontmatter + summary."""
    result = extract_transcript_from_markdown(SAMPLE_MD_WITH_SUMMARY)

    assert result.startswith("# Meeting 2026-03-20 14:30")
    assert "Hello there." in result
    assert "date: 2026-03-20" not in result
    assert "Old summary" not in result


def test_extract_transcript_no_summary():
    """Extract transcript from file without summary."""
    result = extract_transcript_from_markdown(SAMPLE_MD)

    assert result.startswith("# Meeting 2026-03-20 14:30")
    assert "Hello there." in result


# --- summarize ---


def test_summarize_calls_llm_with_correct_prompt(summarize_settings):
    """summarize should pass system prompt and transcript to _call_llm."""
    with patch("meetrec.summarizer._call_llm", return_value=VALID_LLM_RESPONSE) as mock_call:
        summarize("Some transcript text", summarize_settings)

    mock_call.assert_called_once()
    system_prompt = mock_call.call_args[0][0]
    user_message = mock_call.call_args[0][1]
    assert "JSON" in system_prompt
    assert "action_items" in system_prompt
    assert user_message == "Some transcript text"


def test_summarize_parses_valid_json(summarize_settings):
    """Valid JSON response → correct Summary dataclass."""
    with patch("meetrec.summarizer._call_llm", return_value=VALID_LLM_RESPONSE):
        result = summarize("transcript", summarize_settings)

    assert result.brief == "Discussed the project plan and assigned tasks."
    assert len(result.action_items) == 2
    assert result.action_items[0].assignee == "You"
    assert result.action_items[0].deadline == "Friday"
    assert result.action_items[1].deadline is None
    assert result.key_decisions == ["Use PostgreSQL instead of MongoDB"]
    assert result.is_trivial is False


def test_summarize_retries_on_invalid_json(summarize_settings):
    """First call returns garbage, second returns valid JSON → success."""
    with patch(
        "meetrec.summarizer._call_llm",
        side_effect=["not json at all", VALID_LLM_RESPONSE],
    ):
        result = summarize("transcript", summarize_settings)

    assert result.brief == "Discussed the project plan and assigned tasks."


def test_summarize_raises_on_persistent_invalid_json(summarize_settings):
    """Both calls return garbage → RuntimeError."""
    with (
        patch("meetrec.summarizer._call_llm", return_value="still not json"),
        pytest.raises(RuntimeError, match="Failed to parse"),
    ):
        summarize("transcript", summarize_settings)


# --- Retry on rate limit ---


def test_call_llm_retries_on_429(summarize_settings):
    """429 error → retry with backoff, succeed on second attempt."""
    rate_limit_exc = Exception("rate limited")
    rate_limit_exc.status_code = 429

    with (
        patch("meetrec.summarizer._call_llm_once", side_effect=[rate_limit_exc, "ok"]) as mock,
        patch("meetrec.summarizer.time.sleep") as mock_sleep,
    ):
        from meetrec.summarizer import _call_llm

        result = _call_llm("system", "user", summarize_settings)

    assert result == "ok"
    assert mock.call_count == 2
    mock_sleep.assert_called_once_with(5)


def test_call_llm_gives_up_after_max_retries(summarize_settings):
    """Persistent 429 → raises after _MAX_RETRIES attempts."""
    rate_limit_exc = Exception("rate limited")
    rate_limit_exc.status_code = 429

    with (
        patch("meetrec.summarizer._call_llm_once", side_effect=rate_limit_exc),
        patch("meetrec.summarizer.time.sleep"),
        pytest.raises(Exception, match="rate limited"),
    ):
        from meetrec.summarizer import _call_llm

        _call_llm("system", "user", summarize_settings)


def test_call_llm_no_retry_on_non_429(summarize_settings):
    """Non-retryable error (e.g. 401) → raised immediately, no retry."""
    auth_exc = Exception("unauthorized")
    auth_exc.status_code = 401

    with (
        patch("meetrec.summarizer._call_llm_once", side_effect=auth_exc),
        patch("meetrec.summarizer.time.sleep") as mock_sleep,
        pytest.raises(Exception, match="unauthorized"),
    ):
        from meetrec.summarizer import _call_llm

        _call_llm("system", "user", summarize_settings)

    mock_sleep.assert_not_called()


# --- API key resolution ---


def test_api_key_resolution_meetrec_env(tmp_vault, monkeypatch):
    """MEETREC_LLM_API_KEY takes priority over provider-specific env vars."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "provider-key")
    settings = Settings(vault_path=tmp_vault, llm_api_key="meetrec-key")

    assert _resolve_api_key(settings) == "meetrec-key"


@pytest.mark.parametrize(
    "provider,env_var",
    [
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("groq", "GROQ_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("qwen", "DASHSCOPE_API_KEY"),
    ],
)
def test_api_key_resolution_provider_env(tmp_vault, monkeypatch, provider, env_var):
    """Falls back to provider-specific env var when MEETREC_LLM_API_KEY is empty."""
    monkeypatch.setenv(env_var, "test-key")
    settings = Settings(vault_path=tmp_vault, llm_provider=provider, llm_api_key="")

    assert _resolve_api_key(settings) == "test-key"


def test_api_key_missing_raises(tmp_vault, monkeypatch):
    """No key anywhere → RuntimeError with instructions."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(vault_path=tmp_vault, llm_api_key="")

    with pytest.raises(RuntimeError, match="No API key"):
        _resolve_api_key(settings)


# --- Model defaults ---


@pytest.mark.parametrize(
    "provider,expected_model",
    [
        ("anthropic", "claude-sonnet-4-20250514"),
        ("openai", "gpt-4o"),
        ("groq", "llama-3.3-70b-versatile"),
        ("gemini", "gemini-2.0-flash"),
        ("openrouter", "google/gemma-3-27b-it:free"),
        ("deepseek", "deepseek-chat"),
        ("qwen", "qwen-turbo"),
    ],
)
def test_get_model_defaults(tmp_vault, provider, expected_model):
    """Each provider has a sensible default model."""
    settings = Settings(vault_path=tmp_vault, llm_provider=provider, llm_model="")

    assert _get_model(settings) == expected_model


def test_get_model_explicit_override(tmp_vault):
    """Explicit llm_model overrides provider default."""
    settings = Settings(vault_path=tmp_vault, llm_provider="groq", llm_model="custom-model")

    assert _get_model(settings) == "custom-model"
