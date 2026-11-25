"""Names in a summary request: the one who recorded, and the known voices tapeback wrote in.

Asserted on what crossed the wire (`llm_endpoint`), not on an object a test built. The
names are made up.
"""

import json

from tapeback.settings import Settings
from tapeback.summarizer import masked_preview, summarize

TRANSCRIPT = (
    "[00:00:00] **Mira:** We ship on Friday.\n"
    "[00:00:03] **Ann:** I will test it on Monday.\n"
    "[00:00:06] **Speaker 1:** Fine.\n"
)
MASKED = (
    "[00:00:00] **[TERM_1]:** We ship on Friday.\n"
    "[00:00:03] **[TERM_2]:** I will test it on Monday.\n"
    "[00:00:06] **Speaker 1:** Fine.\n"
)


def _settings(tmp_vault, url: str, **overrides) -> Settings:
    return Settings(
        vault_path=tmp_vault,
        llm_provider="openai_compatible",
        llm_base_url=url,
        llm_model="qwen3.5-4b",
        **overrides,
    )


def _sent(endpoint) -> tuple[str, str]:
    [request] = endpoint.requests
    system, user = json.loads(request.body)["messages"]
    return system["content"], user["content"]


def test_the_prompt_names_who_recorded_by_their_own_name(tmp_vault, llm_endpoint):
    summarize(TRANSCRIPT, _settings(tmp_vault, llm_endpoint.url, your_name="Mira"))

    prompt, transcript = _sent(llm_endpoint)
    assert '"Mira" is the person who recorded the meeting.' in prompt
    assert transcript == TRANSCRIPT


def test_unless_set_the_one_who_recorded_is_you(tmp_vault, llm_endpoint):
    summarize("[00:00:00] **You:** We ship on Friday.\n", _settings(tmp_vault, llm_endpoint.url))

    prompt, _ = _sent(llm_endpoint)
    assert '"You" is the person who recorded the meeting.' in prompt
    assert "(e.g. 'You', 'Speaker 1')" in prompt


def test_with_masking_no_name_tapeback_wrote_in_crosses_the_wire(tmp_vault, llm_endpoint):
    summarize(TRANSCRIPT, _settings(tmp_vault, llm_endpoint.url, your_name="Mira", mask_pii=True))

    [request] = llm_endpoint.requests
    assert b"Mira" not in request.raw
    assert b"Ann" not in request.raw
    prompt, transcript = _sent(llm_endpoint)
    assert transcript == MASKED
    assert '"[TERM_1]" is the person who recorded the meeting.' in prompt


def test_your_name_is_masked_in_the_prompt_when_you_never_spoke(tmp_vault, llm_endpoint):
    summarize(
        "[00:00:00] **Ann:** Hello.\n",
        _settings(tmp_vault, llm_endpoint.url, your_name="Mira", mask_pii=True),
    )

    [request] = llm_endpoint.requests
    assert b"Mira" not in request.raw
    prompt, transcript = _sent(llm_endpoint)
    assert transcript == "[00:00:00] **[TERM_1]:** Hello.\n"
    assert '"[TERM_2]" is the person who recorded the meeting.' in prompt


def test_the_preview_masks_the_names_as_the_request_would(tmp_vault):
    settings = _settings(tmp_vault, "", your_name="Mira", mask_pii=True)

    assert masked_preview(TRANSCRIPT, settings) == (MASKED, {"TERM": 2})
