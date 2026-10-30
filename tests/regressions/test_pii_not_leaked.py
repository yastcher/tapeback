"""Regression: with TAPEBACK_MASK_PII on, no raw PII reaches any provider.

The unit tests in test_mask.py pin the masker itself. These pin the wiring, on the three
paths that are easy to leave unmasked: the first request, the JSON retry, and the second
provider in the fallback chain — which is precisely the path that hands the same
transcript to a different company. The last test pins the opposite direction: with
masking off, the request must be byte-identical to what it was before the feature.
"""

from unittest.mock import MagicMock, patch

from pydantic import SecretStr

from tapeback.settings import Settings
from tapeback.summarizer import summarize
from tests.fixtures import clear_all_provider_env_vars, mock_anthropic_response

_TRANSCRIPT = "Ivan said write to ivan.petrov@example.com or call +7 900 123-45-67 today."

_MASKED_RESPONSE = """{
  "brief": "Contact [EMAIL_1] or [PHONE_1].",
  "action_items": [{"assignee": "You", "action": "email [EMAIL_1]", "deadline": null}],
  "key_decisions": ["reach out on [PHONE_1]"],
  "is_trivial": false
}"""

_PLAIN_RESPONSE = """{
  "brief": "Short meeting.",
  "action_items": [],
  "key_decisions": [],
  "is_trivial": false
}"""


def _anthropic_prompt(mock_cls) -> str:
    return str(mock_cls.return_value.messages.create.call_args.kwargs["messages"][0]["content"])


def test_no_raw_pii_reaches_the_provider(summarize_settings):
    settings = summarize_settings.model_copy(update={"mask_pii": True})

    with patch("anthropic.Anthropic") as mock_cls:
        mock_cls.return_value.messages.create.return_value = mock_anthropic_response(
            _MASKED_RESPONSE
        )
        summary = summarize(_TRANSCRIPT, settings)

    sent = _anthropic_prompt(mock_cls)
    assert "ivan.petrov@example.com" not in sent
    assert "+7 900 123-45-67" not in sent
    assert sent == "Ivan said write to [EMAIL_1] or call [PHONE_1] today."

    # The vault gets the real values back.
    assert summary.brief == "Contact ivan.petrov@example.com or +7 900 123-45-67."
    assert summary.action_items[0].action == "email ivan.petrov@example.com"
    assert summary.key_decisions == ["reach out on +7 900 123-45-67"]


def test_listed_terms_do_not_reach_the_provider(summarize_settings):
    """Names are what people say aloud, so the term list is the part that carries."""
    settings = summarize_settings.model_copy(update={"mask_pii": True, "mask_terms": "Ivan"})
    response = _MASKED_RESPONSE.replace("Contact", "[TERM_1] contacts").replace(
        '"deadline": null', '"deadline": "before [TERM_1] leaves"'
    )

    with patch("anthropic.Anthropic") as mock_cls:
        mock_cls.return_value.messages.create.return_value = mock_anthropic_response(response)
        summary = summarize(_TRANSCRIPT, settings)

    sent = _anthropic_prompt(mock_cls)
    assert "Ivan" not in sent
    assert sent == "[TERM_1] said write to [EMAIL_1] or call [PHONE_1] today."
    assert summary.brief == "Ivan contacts ivan.petrov@example.com or +7 900 123-45-67."
    assert summary.action_items[0].deadline == "before Ivan leaves"


def test_retry_prompt_is_masked_too(summarize_settings):
    settings = summarize_settings.model_copy(update={"mask_pii": True})

    with patch("anthropic.Anthropic") as mock_cls:
        mock_cls.return_value.messages.create.side_effect = [
            mock_anthropic_response("not json at all"),
            mock_anthropic_response(_MASKED_RESPONSE),
        ]
        summary = summarize(_TRANSCRIPT, settings)

    calls = mock_cls.return_value.messages.create.call_args_list
    sent = [str(call.kwargs["messages"][0]["content"]) for call in calls]
    assert sent == ["Ivan said write to [EMAIL_1] or call [PHONE_1] today."] * 2
    assert summary.brief == "Contact ivan.petrov@example.com or +7 900 123-45-67."


def test_fallback_provider_also_gets_masked_text(tmp_vault, monkeypatch):
    clear_all_provider_env_vars(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fallback-key")
    settings = Settings(
        vault_path=tmp_vault,
        llm_provider="anthropic",
        llm_api_key=SecretStr("sk-ant-test-key"),
        mask_pii=True,
    )

    oai_response = MagicMock()
    oai_response.choices = [MagicMock(message=MagicMock(content=_MASKED_RESPONSE))]

    with patch("anthropic.Anthropic") as ant_cls, patch("openai.OpenAI") as oai_cls:
        ant_cls.return_value.messages.create.side_effect = RuntimeError("provider down")
        oai_cls.return_value.chat.completions.create.return_value = oai_response
        summary = summarize(_TRANSCRIPT, settings)

    messages = oai_cls.return_value.chat.completions.create.call_args.kwargs["messages"]
    sent = str(messages[1]["content"])
    assert sent == "Ivan said write to [EMAIL_1] or call [PHONE_1] today."
    assert summary.brief == "Contact ivan.petrov@example.com or +7 900 123-45-67."


def test_masking_off_sends_the_transcript_unchanged(summarize_settings):
    assert summarize_settings.mask_pii is False

    with patch("anthropic.Anthropic") as mock_cls:
        mock_cls.return_value.messages.create.return_value = mock_anthropic_response(
            _PLAIN_RESPONSE
        )
        summary = summarize(_TRANSCRIPT, summarize_settings)

    assert _anthropic_prompt(mock_cls) == _TRANSCRIPT
    assert summary.brief == "Short meeting."


def test_unresolved_placeholder_is_reported(summarize_settings, capsys):
    """The model invented an index — the value cannot be restored, so say so."""
    settings = summarize_settings.model_copy(update={"mask_pii": True})
    # Two of them, so the list in the warning has a separator.
    invented = _MASKED_RESPONSE.replace("[PHONE_1]", "[PHONE_9]").replace(
        "email [EMAIL_1]", "email [EMAIL_2]"
    )

    with patch("anthropic.Anthropic") as mock_cls:
        mock_cls.return_value.messages.create.return_value = mock_anthropic_response(invented)
        summary = summarize(_TRANSCRIPT, settings)

    assert summary.key_decisions == ["reach out on [PHONE_9]"]
    assert capsys.readouterr().err == (
        "Warning: summary keeps unresolved placeholders ([EMAIL_2], [PHONE_9]) — the model "
        "altered them, so those values could not be restored.\n"
    )
