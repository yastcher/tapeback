"""Regression: two people on a call came out as one speaker of the note.

The spectral merge joined pyannote's speakers whose long-term spectra were alike, and a
call's codec makes every voice's spectrum alike: on recorded calls it joined voices whose
embeddings were far apart, and the note showed a single remote speaker. Here both remote
voices are one synthetic tone, spectrally identical, while pyannote's own embeddings say
they are two people.
"""

from unittest.mock import patch

import pytest

from tapeback.cli import cli
from tests.fake_pyannote import pyannote_output
from tests.fixtures import mock_whisper_by_channel, requires_pyannote

pytestmark = pytest.mark.usefixtures("cpu_only")

SESSION = "2026-10-06_10-30-00"


@requires_pyannote
def test_two_voices_pyannote_tells_apart_stay_two_speakers(
    runner, vault_env, fake_parecord, ctrl_c_while_recording, monkeypatch
):
    monkeypatch.setenv("TAPEBACK_HF_TOKEN", "hf_fake")
    meeting = mock_whisper_by_channel(
        mic_segments=[(0.0, 2.0, "I will send the report.")],
        monitor_segments=[
            (
                2.0,
                4.0,
                "Thanks a lot. See you Friday.",
                [
                    (2.0, 2.4, "Thanks"),
                    (2.4, 2.6, "a"),
                    (2.6, 3.0, "lot."),
                    (3.0, 3.3, "See"),
                    (3.3, 3.6, "you"),
                    (3.6, 4.0, "Friday."),
                ],
            )
        ],
    )
    # A second each: a shorter turn would join the speech around it.
    output = pyannote_output(
        [(2.0, 3.0, "SPEAKER_00"), (3.0, 4.0, "SPEAKER_01")],
        {"SPEAKER_00": [1.0, 0.0, 0.0], "SPEAKER_01": [0.0, 1.0, 0.0]},
    )

    with (
        patch("tapeback.transcriber.WhisperModel", return_value=meeting),
        patch("pyannote.audio.Pipeline.from_pretrained", return_value=lambda *_a, **_k: output),
    ):
        result = runner.invoke(cli, ["start", SESSION, "--no-summarize", "--no-live"])

    assert result.exit_code == 0, result.output + repr(result.exception)
    note = (vault_env / "meetings" / f"{SESSION}.md").read_text()
    assert note.endswith(
        "---\n\n"
        "[00:00:00] **You:** I will send the report.\n\n"
        "[00:00:02] **Speaker 1:** Thanks a lot.\n\n"
        "[00:00:03] **Speaker 2:** See you Friday.\n"
    )
