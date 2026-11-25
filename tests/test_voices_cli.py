"""`tapeback voices`: learn a voice from a note, list the people, forget one.

The loop a user walks: a call recorded with an unknown voice, `voices learn` naming it,
and the next call naming that voice by itself. Only what sits outside tapeback is faked:
`parecord`, the Whisper model, the user's Ctrl+C, pyannote's output. Names are made up.
"""

from unittest.mock import patch

import pytest

from tapeback.cli import cli
from tapeback.models import VoiceProfile, VoiceSample
from tapeback.voice_profiles import load_profiles, profiles_path, save_profiles
from tests.fake_pyannote import pyannote_output
from tests.fixtures import mock_whisper_by_channel, requires_pyannote

pytestmark = pytest.mark.usefixtures("cpu_only")

FIRST = "2026-10-06_10-30-00"
SECOND = "2026-10-13_10-30-00"
ANN = [0.95, 0.312, 0.0]  # Ann's voice as pyannote embeds it
STRANGER = [0.0, 1.0, 0.0]


def _sample(session: str) -> VoiceSample:
    return VoiceSample(vector=(1.0, 0.0, 0.0), session=session, speaker="SPEAKER_00")


def _meeting():
    return mock_whisper_by_channel(
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


def _pyannote(first_voice: list[float]):
    """Two remote speakers, a second each: the first voice given, then a stranger's."""
    output = pyannote_output(
        [(2.0, 3.0, "SPEAKER_00"), (3.0, 4.0, "SPEAKER_01")],
        {"SPEAKER_00": first_voice, "SPEAKER_01": STRANGER},
    )
    return patch("pyannote.audio.Pipeline.from_pretrained", return_value=lambda *_a, **_k: output)


def _record(runner, session: str, first_voice: list[float]):
    with (
        patch("tapeback.transcriber.WhisperModel", return_value=_meeting()),
        _pyannote(first_voice),
    ):
        result = runner.invoke(cli, ["start", session, "--no-summarize", "--no-live"])
    assert result.exit_code == 0, result.output + repr(result.exception)


@requires_pyannote
def test_a_voice_learned_from_one_note_is_named_in_the_next(
    runner, vault_env, fake_parecord, ctrl_c_while_recording, monkeypatch
):
    monkeypatch.setenv("TAPEBACK_HF_TOKEN", "hf_fake")
    _record(runner, FIRST, ANN)
    note = vault_env / "meetings" / f"{FIRST}.md"
    assert "[00:00:02] **Speaker 1:** Thanks a lot." in note.read_text()

    with _pyannote(ANN):
        learned = runner.invoke(cli, ["voices", "learn", str(note), "Speaker 1=Ann"])

    assert learned.exit_code == 0, learned.output + repr(learned.exception)
    assert "Learned Ann's voice." in learned.output
    assert "Not learned" not in learned.output  # your own side and Speaker 2 are not tried
    [ann] = load_profiles()
    assert (ann.name, [(s.session, s.speaker) for s in ann.samples]) == (
        "Ann",
        [(FIRST, "SPEAKER_00")],
    )
    assert "[00:00:02] **Ann:** Thanks a lot." in note.read_text()

    # The next call, processed from a recording: the fixtures stop one recording a test.
    recording = vault_env / "attachments" / "audio" / f"{FIRST}.wav"
    with patch("tapeback.transcriber.WhisperModel", return_value=_meeting()), _pyannote(ANN):
        result = runner.invoke(cli, ["process", str(recording), "--name", SECOND, "--no-summarize"])
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert (
        (vault_env / "meetings" / f"{SECOND}.md")
        .read_text()
        .endswith(
            "[00:00:00] **You:** I will send the report.\n\n"
            "[00:00:02] **Ann:** Thanks a lot.\n\n"
            "[00:00:03] **Speaker 1:** See you Friday.\n"
        )
    )


@requires_pyannote
def test_a_learn_that_finds_no_voice_fails_and_keeps_nothing(
    runner, vault_env, fake_parecord, ctrl_c_while_recording, monkeypatch
):
    monkeypatch.setenv("TAPEBACK_HF_TOKEN", "hf_fake")
    _record(runner, FIRST, ANN)
    note = vault_env / "meetings" / f"{FIRST}.md"
    # This time pyannote gives the voices no embedding, as a legacy pipeline does.
    silent = pyannote_output(
        [(2.0, 3.0, "SPEAKER_00"), (3.0, 4.0, "SPEAKER_01")],
        {"SPEAKER_00": [0.0, 0.0, 0.0], "SPEAKER_01": [0.0, 0.0, 0.0]},
    )

    with patch("pyannote.audio.Pipeline.from_pretrained", return_value=lambda *_a, **_k: silent):
        result = runner.invoke(cli, ["voices", "learn", str(note), "Speaker 1=Ann"])

    assert result.exit_code == 1
    assert "Not learned, Ann: pyannote gave their voice no embedding." in result.output
    assert "No voice learned." in result.output
    assert not profiles_path().exists()


@requires_pyannote
def test_the_names_a_note_already_has_are_learned_without_renames(
    runner, vault_env, fake_parecord, ctrl_c_while_recording, monkeypatch
):
    monkeypatch.setenv("TAPEBACK_HF_TOKEN", "hf_fake")
    _record(runner, FIRST, ANN)
    note = vault_env / "meetings" / f"{FIRST}.md"
    note.write_text(note.read_text().replace("**Speaker 2:**", "**Boris:**"))

    with _pyannote(ANN):
        learned = runner.invoke(cli, ["voices", "learn", str(note)])

    assert learned.exit_code == 0, learned.output + repr(learned.exception)
    assert [(p.name, p.samples[0].speaker) for p in load_profiles()] == [("Boris", "SPEAKER_01")]


def _note(vault, *, audio: bool = True) -> str:
    note = vault / "meetings" / f"{FIRST}.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    field = f'audio: "[[attachments/audio/{FIRST}.wav]]"\n' if audio else ""
    note.write_text(f"---\n{field}---\n\n[00:00:02] **Speaker 1:** Thanks.\n")
    return str(note)


@pytest.mark.parametrize(
    ("pair", "error"),
    [
        ("Speaker 1", "'Speaker 1' is not LABEL=NAME, as 'Speaker 2=Anna'."),
        ("Speaker 1=", "'Speaker 1=' is not LABEL=NAME, as 'Speaker 2=Anna'."),
        ("You=Ann", "'You' is your side of the call, which is not learned: set "),
        ("Speaker 3=Ann", "'Speaker 3' speaks nowhere in the note."),
        ("Speaker 1=Speaker 4", "'Speaker 4' cannot name a speaker: it is one of tapeback's"),
    ],
)
def test_a_rename_that_cannot_stand_is_refused_before_anything_runs(runner, vault_env, pair, error):
    result = runner.invoke(cli, ["voices", "learn", _note(vault_env), pair])

    assert result.exit_code == 1
    assert error in result.output
    assert not profiles_path().exists()


def test_a_note_without_its_recording_in_the_vault_learns_nothing(runner, vault_env):
    result = runner.invoke(cli, ["voices", "learn", _note(vault_env, audio=False)])

    assert result.exit_code == 1
    assert f"names no recording in {vault_env}." in result.output


def test_learning_without_a_token_says_what_it_needs(runner, vault_env):
    recording = vault_env / "attachments" / "audio" / f"{FIRST}.wav"
    recording.parent.mkdir(parents=True)
    recording.write_bytes(b"RIFF")

    result = runner.invoke(cli, ["voices", "learn", _note(vault_env)])

    assert result.exit_code == 1
    assert "Learning a voice diarizes the recording: set TAPEBACK_HF_TOKEN." in result.output


def test_the_list_names_each_person_and_where_their_voice_was_heard(runner):
    save_profiles(
        [
            VoiceProfile(id="1", name="Ann", samples=(_sample(SECOND), _sample(FIRST))),
            VoiceProfile(id="2", name="Boris", samples=(_sample(FIRST),)),
        ]
    )

    result = runner.invoke(cli, ["voices", "list"])

    assert result.exit_code == 0
    assert result.output == (f"Ann: heard in {FIRST}, {SECOND}\nBoris: heard in {FIRST}\n")


def test_without_profiles_the_list_says_how_to_make_one(runner):
    result = runner.invoke(cli, ["voices", "list"])

    assert (result.exit_code, result.output) == (
        0,
        "No voice profiles yet: see `tapeback voices learn --help`.\n",
    )


def test_a_forgotten_voice_leaves_the_machine(runner):
    save_profiles(
        [
            VoiceProfile(id="1", name="Ann", samples=(_sample(FIRST),)),
            VoiceProfile(id="2", name="Boris", samples=(_sample(FIRST),)),
        ]
    )

    result = runner.invoke(cli, ["voices", "forget", "Ann"])

    assert (result.exit_code, result.output) == (0, "Forgot Ann.\n")
    assert [p.name for p in load_profiles()] == ["Boris"]
    assert "Ann" not in profiles_path().read_text()


def test_forgetting_someone_unknown_changes_nothing(runner):
    save_profiles([VoiceProfile(id="2", name="Boris", samples=(_sample(FIRST),))])

    result = runner.invoke(cli, ["voices", "forget", "Ann"])

    assert result.exit_code == 1
    assert "No voice profile is named 'Ann'." in result.output
    assert [p.name for p in load_profiles()] == ["Boris"]
