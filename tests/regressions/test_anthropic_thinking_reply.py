"""Regression: a summary through anthropic failed when the model thought first.

Current Claude models think adaptively, so a reply can open with a thinking block and
carry the text in the next one. The transport read the first block only and raised
"Unexpected anthropic response block type".
"""

from unittest.mock import MagicMock, patch

import pytest
from anthropic.types import ThinkingBlock
from pydantic import SecretStr

from tapeback._llm import call
from tapeback.settings import Settings
from tests.fixtures import VALID_LLM_RESPONSE, clear_all_provider_env_vars, mock_anthropic_response


def test_a_reply_that_thinks_first_still_gives_its_text(tmp_vault, monkeypatch):
    clear_all_provider_env_vars(monkeypatch)
    settings = Settings(vault_path=tmp_vault, llm_api_key=SecretStr("ant-key"))

    with patch("anthropic.Anthropic") as client:
        client.return_value.messages.create.return_value = mock_anthropic_response(
            VALID_LLM_RESPONSE, thinking="Weighing the decisions first."
        )
        assert call("system", "user", settings) == VALID_LLM_RESPONSE


def test_a_reply_without_text_says_what_it_held(tmp_vault, monkeypatch):
    clear_all_provider_env_vars(monkeypatch)
    settings = Settings(vault_path=tmp_vault, llm_api_key=SecretStr("ant-key"))
    reply = MagicMock(content=[ThinkingBlock(type="thinking", thinking="...", signature="s")])

    with patch("anthropic.Anthropic") as client, pytest.raises(RuntimeError) as failed:
        client.return_value.messages.create.return_value = reply
        call("system", "user", settings)

    assert str(failed.value) == "anthropic reply holds no text block, only: thinking"
