"""Regression: a local model summarised a Russian meeting in English.

The system prompt is English and asked for "the same language as the transcript"; a
4B model followed the language of the instructions. The note knows its language
(`language: ru`, from Whisper), so the prompt now names it. Asserted on the request
that reached the endpoint.
"""

import json

from tapeback.cli import cli
from tapeback.settings import Settings
from tapeback.summarizer import maybe_summarize
from tests.fixtures import SAMPLE_MD

NAMED = (
    "- The transcript is in ru (ISO 639-1): write the brief, the actions and the "
    "decisions in that language, whatever the language of these instructions"
)


def _note(tmp_vault, md: str):
    note = tmp_vault / "meetings" / "2026-10-16_10-00-00.md"
    note.parent.mkdir(parents=True)
    note.write_text(md)
    return note


def _system_prompt(endpoint) -> str:
    [request] = endpoint.requests
    return json.loads(request.body)["messages"][0]["content"]


def test_the_notes_language_is_named_in_the_prompt(tmp_vault, llm_endpoint):
    note = _note(tmp_vault, SAMPLE_MD.replace("language: en", "language: ru"))
    settings = Settings(
        vault_path=tmp_vault,
        llm_provider="openai_compatible",
        llm_base_url=llm_endpoint.url,
        llm_model="model",
    )

    maybe_summarize(note, settings)

    assert _system_prompt(llm_endpoint).endswith(NAMED)


def test_a_note_without_a_language_keeps_the_general_rule(tmp_vault, llm_endpoint):
    note = _note(tmp_vault, SAMPLE_MD.replace("language: en\n", ""))
    settings = Settings(
        vault_path=tmp_vault,
        llm_provider="openai_compatible",
        llm_base_url=llm_endpoint.url,
        llm_model="model",
    )

    maybe_summarize(note, settings)

    prompt = _system_prompt(llm_endpoint)
    assert "ISO 639-1" not in prompt
    assert "- Write in the same language as the transcript" in prompt


def test_tapeback_summarize_names_the_language_too(runner, real_env_files, tmp_vault, llm_endpoint):
    user_config, _ = real_env_files
    user_config.write_text(
        f"TAPEBACK_VAULT_PATH={tmp_vault}\n"
        "TAPEBACK_LLM_PROVIDER=openai_compatible\n"
        f"TAPEBACK_LLM_BASE_URL={llm_endpoint.url}\n"
        "TAPEBACK_LLM_MODEL=model\n"
    )
    note = _note(tmp_vault, SAMPLE_MD.replace("language: en", "language: ru"))

    result = runner.invoke(cli, ["summarize", str(note)])

    assert result.exit_code == 0, result.output
    assert _system_prompt(llm_endpoint).endswith(NAMED)
