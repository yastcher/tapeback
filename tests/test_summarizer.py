"""Summarizer tests — markdown formatting, injection, extraction, LLM integration flows."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from tapeback.models import ActionItem, Summary
from tapeback.settings import Settings
from tapeback.summarizer import (
    extract_transcript_from_markdown,
    format_summary_markdown,
    inject_summary_into_markdown,
    maybe_summarize,
    summarize,
)
from tests.fixtures import (
    SAMPLE_MD,
    SAMPLE_MD_WITH_SUMMARY,
    VALID_LLM_RESPONSE,
    clear_all_provider_env_vars,
    mock_anthropic_response,
)

# A diarized note as tapeback wrote it before one transcript per note: the raw
# You/Other transcript, then the diarized copy of the same text.
TWO_SECTION_NOTE = """\
---
date: 2026-08-06
---

# Meeting 2026-08-06 12:00

**Duration:** 15s | **Language:** en

---

## Transcript

[00:00:00] **You:** Ship it on Friday.

[00:00:05] **Other:** Then tests come first.

---

## Diarized Transcript

[00:00:00] **You:** Ship it on Friday.

[00:00:05] **Speaker 1:** Then tests come first.
"""

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


def test_inject_summary_no_frontmatter():
    """No frontmatter → summary prepended to content."""
    content = "# Meeting 2026-03-20\n\n[00:00:01] **You:** Hello.\n"
    summary_md = "\n## Summary\n\nBrief.\n\n---\n"

    result = inject_summary_into_markdown(content, summary_md)

    assert result.startswith(summary_md)
    assert "Hello." in result


def test_inject_summary_frontmatter_no_transcript_header():
    """Frontmatter present but no '# Meeting' header → summary inserted after frontmatter."""
    content = "---\ndate: 2026-03-20\n---\n\nSome notes without header.\n"
    summary_md = "\n## Summary\n\nBrief.\n\n---\n"

    result = inject_summary_into_markdown(content, summary_md)

    assert "## Summary" in result
    assert "Some notes without header." in result


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


def test_extract_transcript_no_meeting_header():
    """No '# Meeting' header → empty string."""
    result = extract_transcript_from_markdown("---\ndate: 2026-03-20\n---\n\nJust notes.\n")

    assert result == ""


def test_extract_transcript_drops_the_raw_half_of_an_older_note():
    """An older note printed the transcript twice; only the diarized copy is kept."""
    result = extract_transcript_from_markdown(TWO_SECTION_NOTE)

    assert result == (
        "# Meeting 2026-08-06 12:00\n\n"
        "**Duration:** 15s | **Language:** en\n\n"
        "---\n\n"
        "[00:00:00] **You:** Ship it on Friday.\n\n"
        "[00:00:05] **Speaker 1:** Then tests come first.\n"
    )


# --- Integration flow tests ---
# Mock only at SDK boundary (anthropic.Anthropic / openai.OpenAI).
# Exercises the full chain: _llm.provider_chain → _llm.call → summarize → parse →
# format → inject. The transport alone is in tests/test_llm.py.


def test_summarize_to_markdown_flow(summarize_settings):
    """Full flow: transcript → LLM (anthropic SDK) → parse → format → inject."""
    with patch("anthropic.Anthropic") as mock_cls:
        mock_cls.return_value.messages.create.return_value = mock_anthropic_response(
            VALID_LLM_RESPONSE
        )
        summary = summarize("Test transcript text", summarize_settings)

    # Summary parsed correctly from LLM response
    assert summary.brief == "Discussed the project plan and assigned tasks."
    assert len(summary.action_items) == 2
    assert summary.action_items[0].assignee == "You"
    assert summary.action_items[0].deadline == "Friday"
    assert summary.action_items[1].deadline is None
    assert summary.key_decisions == ["Use PostgreSQL instead of MongoDB"]
    assert summary.is_trivial is False

    # Format → inject → verify full document
    md = format_summary_markdown(summary)
    assert "## Summary" in md
    assert "### Action Items" in md
    assert "- [ ] **You:** Send the report by Friday" in md

    doc = inject_summary_into_markdown(SAMPLE_MD, md)
    assert "## Summary" in doc
    assert "# Meeting 2026-03-20 14:30" in doc
    assert "Hello there." in doc


def test_summarize_openai_compatible_flow(tmp_vault, monkeypatch):
    """OpenAI-compatible provider flow: groq → openai SDK with custom base_url."""
    clear_all_provider_env_vars(monkeypatch)
    settings = Settings(
        vault_path=tmp_vault, llm_provider="groq", llm_api_key=SecretStr("gsk-test-key")
    )

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=VALID_LLM_RESPONSE))]

    with patch("openai.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.return_value = mock_response
        summary = summarize("Test transcript", settings)

    assert summary.brief == "Discussed the project plan and assigned tasks."
    mock_cls.assert_called_once_with(
        api_key="gsk-test-key",
        base_url="https://api.groq.com/openai/v1",
        timeout=600.0,
    )


def test_summarize_resilience_flow(summarize_settings):
    """Resilience: fenced JSON, invalid JSON retry, persistent failure."""
    # Fenced JSON — stripped and parsed correctly
    fenced = f"```json\n{VALID_LLM_RESPONSE}\n```"
    with patch("anthropic.Anthropic") as mock_cls:
        mock_cls.return_value.messages.create.return_value = mock_anthropic_response(fenced)
        summary = summarize("transcript", summarize_settings)
    assert summary.brief == "Discussed the project plan and assigned tasks."

    # Invalid JSON first call → retry with shorter prompt → success
    with patch("anthropic.Anthropic") as mock_cls:
        mock_cls.return_value.messages.create.side_effect = [
            mock_anthropic_response("not json at all"),
            mock_anthropic_response(VALID_LLM_RESPONSE),
        ]
        summary = summarize("transcript", summarize_settings)
    assert summary.brief == "Discussed the project plan and assigned tasks."

    # Persistent invalid JSON → RuntimeError
    with (
        patch("anthropic.Anthropic") as mock_cls,
        pytest.raises(RuntimeError, match="Failed to parse"),
    ):
        mock_cls.return_value.messages.create.return_value = mock_anthropic_response(
            "still not json"
        )
        summarize("transcript", summarize_settings)


def test_maybe_summarize_end_to_end(tmp_vault, monkeypatch):
    """maybe_summarize: full flow including edge cases and happy path."""
    clear_all_provider_env_vars(monkeypatch)

    md_file = tmp_vault / "meetings" / "test.md"
    md_file.parent.mkdir(parents=True)
    md_file.write_text(SAMPLE_MD)

    settings_with_key = Settings(
        vault_path=tmp_vault,
        summarize=True,
        llm_provider="anthropic",
        llm_api_key=SecretStr("sk-test"),
    )

    # md_path=None → no-op
    maybe_summarize(None, settings_with_key)

    # summarize=False → no-op, file unchanged
    settings_off = Settings(vault_path=tmp_vault, summarize=False)
    maybe_summarize(md_file, settings_off)
    assert "## Summary" not in md_file.read_text()

    # No API key → warning, file unchanged
    settings_no_key = Settings(
        vault_path=tmp_vault, summarize=True, llm_provider="anthropic", llm_api_key=SecretStr("")
    )
    maybe_summarize(md_file, settings_no_key)
    assert "## Summary" not in md_file.read_text()

    # Empty transcript (no '# Meeting' header) → warning, file unchanged
    empty_md = tmp_vault / "meetings" / "empty.md"
    empty_md.write_text("---\ndate: 2026-03-20\n---\n\nNo transcript header here.\n")
    maybe_summarize(empty_md, settings_with_key)
    assert "## Summary" not in empty_md.read_text()

    # Happy path: file gets summary injected
    with patch("anthropic.Anthropic") as mock_cls:
        mock_cls.return_value.messages.create.return_value = mock_anthropic_response(
            VALID_LLM_RESPONSE
        )
        maybe_summarize(md_file, settings_with_key)
    content = md_file.read_text()
    assert "## Summary" in content
    assert "Discussed the project plan" in content
    assert "# Meeting 2026-03-20 14:30" in content

    # LLM failure → warning, transcript intact
    md_file.write_text(SAMPLE_MD)
    with patch("anthropic.Anthropic") as mock_cls:
        mock_cls.return_value.messages.create.side_effect = RuntimeError("API down")
        maybe_summarize(md_file, settings_with_key)
    assert "## Summary" not in md_file.read_text()
    assert "Hello there." in md_file.read_text()


def test_summary_of_an_older_note_sends_its_transcript_once(tmp_vault, summarize_settings):
    md_file = tmp_vault / "meetings" / "2026-08-06_12-00-00.md"
    md_file.parent.mkdir(parents=True)
    md_file.write_text(TWO_SECTION_NOTE)

    with patch("anthropic.Anthropic") as mock_cls:
        create = mock_cls.return_value.messages.create
        create.return_value = mock_anthropic_response(VALID_LLM_RESPONSE)
        maybe_summarize(md_file, summarize_settings.model_copy(update={"summarize": True}))

    sent = str(create.call_args)
    assert sent.count("Ship it on Friday.") == 1
    assert "**Other:**" not in sent
    assert "**Speaker 1:** Then tests come first." in sent
