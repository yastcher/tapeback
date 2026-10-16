"""Regression: a text-only remote model repeated the user's words once per pause.

gpt-transcribe returns one segment per upload, with no word timestamps. Pause
splitting cut the mic's segment at every pause of a second or more and gave each
piece the whole text, so the note carried the user's words once per pause.

A segment without word timestamps is no longer split. Whether it is the user's or
headphone bleed is still judged stretch by stretch between the pauses.
"""

import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from tapeback.pipeline import process_file
from tapeback.settings import Settings
from tests.fixtures import create_stereo_wav_segments


def _response(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text, languages=[SimpleNamespace(code="en")])


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_text_only_model_writes_the_mic_text_once(tmp_path):
    audio = tmp_path / "meeting.wav"
    # The mic (left) speaks, pauses while the call (right) speaks, and speaks again.
    create_stereo_wav_segments(
        audio, 48000, [(1.5, 0.8, 0.003), (1.5, 0.003, 0.8), (1.5, 0.8, 0.003)]
    )
    client = MagicMock()
    # The monitor channel is uploaded first, then the mic.
    client.audio.transcriptions.create.side_effect = [
        _response("Good morning, everyone."),
        _response("Budget first. Then hiring."),
    ]
    settings = Settings(
        vault_path=tmp_path,
        stt_backend="openai",
        stt_model="gpt-transcribe",
        stt_api_key=SecretStr("sk-test"),
        language="en",
        resume_cache=False,
    )

    with patch("openai.OpenAI", return_value=client):
        note = process_file(audio, settings, diarize=True, do_summarize=False).read_text()

    assert [line for line in note.splitlines() if line.startswith("[")] == [
        "[00:00:00] **You:** Budget first. Then hiring.",
        "[00:00:00] **Other:** Good morning, everyone.",
    ]
