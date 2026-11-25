"""Names in the note, recorded the way a user records: `tapeback start`, Ctrl+C, a note.

The mic side under the user's own name, and a remote voice the profiles know under its
person's. Only what sits outside tapeback is faked: `parecord`, the Whisper model, the
user's Ctrl+C, and pyannote's output. The names are made up.
"""

import json
from unittest.mock import patch

import pytest

from tapeback import pipeline
from tapeback.cli import cli
from tapeback.models import Diarization, DiarizationSegment, VoiceProfile, VoiceSample
from tapeback.voice_profiles import profiles_path, save_profiles
from tests.fake_pyannote import pyannote_output
from tests.fixtures import mock_whisper_by_channel, requires_pyannote

pytestmark = pytest.mark.usefixtures("cpu_only")

SESSION = "2026-10-06_10-30-00"

A = (1.0, 0.0, 0.0)
B = (0.0, 1.0, 0.0)


def _profile(name: str, vector: tuple[float, ...]) -> VoiceProfile:
    sample = VoiceSample(vector=vector, session="2026-01-05_10-00-00", speaker="SPEAKER_00")
    return VoiceProfile(id=f"id-{name}", name=name, samples=(sample,))


def _turns(*labels: str) -> list[DiarizationSegment]:
    return [
        DiarizationSegment(speaker=label, start=float(i), end=float(i) + 1.0)
        for i, label in enumerate(labels)
    ]


def _write(data: object) -> None:
    path = profiles_path()
    path.parent.mkdir(parents=True, mode=0o700)
    path.write_text(json.dumps(data))


def test_the_mic_side_goes_by_your_name(
    runner, vault_env, fake_parecord, ctrl_c_while_recording, monkeypatch
):
    monkeypatch.setenv("TAPEBACK_YOUR_NAME", "Mira")
    meeting = mock_whisper_by_channel(
        mic_segments=[(0.0, 2.0, "I will send the report.")],
        monitor_segments=[(2.0, 4.0, "Thanks, see you Friday.")],
    )

    with patch("tapeback.transcriber.WhisperModel", return_value=meeting):
        result = runner.invoke(
            cli, ["start", SESSION, "--no-diarize", "--no-summarize", "--no-live"]
        )

    assert result.exit_code == 0, result.output + repr(result.exception)
    note = (vault_env / "meetings" / f"{SESSION}.md").read_text()
    assert note.endswith(
        "[00:00:00] **Mira:** I will send the report.\n\n"
        "[00:00:02] **Other:** Thanks, see you Friday.\n"
    )


@requires_pyannote
def test_a_voice_the_profiles_know_is_named_and_the_rest_count_from_one(
    runner, vault_env, fake_parecord, ctrl_c_while_recording, monkeypatch
):
    monkeypatch.setenv("TAPEBACK_HF_TOKEN", "hf_fake")
    monkeypatch.setenv("TAPEBACK_YOUR_NAME", "Mira")
    ann = VoiceSample(vector=(1.0, 0.0, 0.0), session="2026-10-01_10-00-00", speaker="SPEAKER_01")
    save_profiles([VoiceProfile(id="ann", name="Ann", samples=(ann,))])
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
    # The first voice is a stranger's, the second Ann's in another call (0.95 to her sample).
    output = pyannote_output(
        [(2.0, 3.0, "SPEAKER_00"), (3.0, 4.0, "SPEAKER_01")],
        {"SPEAKER_00": [0.0, 1.0, 0.0], "SPEAKER_01": [0.95, 0.312, 0.0]},
    )

    with (
        patch("tapeback.transcriber.WhisperModel", return_value=meeting),
        patch("pyannote.audio.Pipeline.from_pretrained", return_value=lambda *_a, **_k: output),
    ):
        result = runner.invoke(cli, ["start", SESSION, "--no-summarize", "--no-live"])

    assert result.exit_code == 0, result.output + repr(result.exception)
    assert "Voice profiles: SPEAKER_00 0.00, SPEAKER_01 0.95 known" in result.output
    assert "Ann" not in result.output
    note = (vault_env / "meetings" / f"{SESSION}.md").read_text()
    assert note.endswith(
        "---\n\n"
        "[00:00:00] **Mira:** I will send the report.\n\n"
        "[00:00:02] **Speaker 1:** Thanks a lot.\n\n"
        "[00:00:03] **Ann:** See you Friday.\n"
    )


def test_the_pipeline_reports_matches_without_names():
    """The run record keeps status lines as they are, and is meant to be shareable."""
    save_profiles([_profile("Ann", A)])
    turns = _turns("SPEAKER_00", "SPEAKER_01")
    status: list[str] = []

    names = pipeline._known_speakers(
        turns, Diarization(segments=turns, voices={"SPEAKER_00": A, "SPEAKER_01": B}), status.append
    )

    assert names == {"SPEAKER_00": "Ann"}
    assert status == ["Voice profiles: SPEAKER_00 1.00 known, SPEAKER_01 0.00"]


def test_the_pipeline_goes_on_without_profiles_it_cannot_use():
    _write({"model": "other/embedder", "profiles": []})
    turns = _turns("SPEAKER_00")
    status: list[str] = []

    names = pipeline._known_speakers(
        turns, Diarization(segments=turns, voices={"SPEAKER_00": A}), status.append
    )

    assert names == {}
    assert status == [
        f"Warning: voice profiles not used: {profiles_path()} holds voices of other/embedder, "
        "not of pyannote/speaker-diarization-3.1"
    ]


def test_without_profiles_the_pipeline_says_nothing():
    turns = _turns("SPEAKER_00")
    status: list[str] = []

    names = pipeline._known_speakers(
        turns, Diarization(segments=turns, voices={"SPEAKER_00": A}), status.append
    )

    assert (names, status) == ({}, [])


def test_a_call_without_voices_reports_nothing():
    """A pipeline that gives no embeddings has nothing to compare: no empty status line."""
    save_profiles([_profile("Ann", A)])
    turns = _turns("SPEAKER_00")
    status: list[str] = []

    names = pipeline._known_speakers(turns, Diarization(segments=turns, voices={}), status.append)

    assert (names, status) == ({}, [])
