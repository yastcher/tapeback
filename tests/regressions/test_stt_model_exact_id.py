"""Regression: OpenAI model ids were matched case-insensitively.

The allowlist compared `model.strip().lower()`, but the id goes to OpenAI exactly as
written in TAPEBACK_STT_MODEL. So `Whisper-1` passed validation at settings load and
was sent on as `Whisper-1` — the allowlist exists to stop a mistyped id at load.
"""

import pytest
from pydantic import ValidationError

from tapeback.settings import Settings


@pytest.mark.parametrize("model", ["Whisper-1", "WHISPER-1", "Gpt-Transcribe"])
def test_a_model_id_in_another_case_is_refused_at_load(model, tmp_path):
    with pytest.raises(ValidationError, match=rf"Unsupported OpenAI STT model '{model}'"):
        Settings(vault_path=tmp_path, stt_backend="openai", stt_model=model)


@pytest.mark.parametrize("model", ["whisper-1", "gpt-transcribe", "gpt-4o-transcribe-diarize"])
def test_the_documented_ids_still_load(model, tmp_path):
    assert Settings(vault_path=tmp_path, stt_backend="openai", stt_model=model).stt_model == model
