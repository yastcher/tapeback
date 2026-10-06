"""A meeting recorded the way a user records one: `tapeback start`, Ctrl+C, a note.

Everything inside tapeback is real — the Recorder, the session directory and its
privacy check, the stop signals, ffmpeg, the pipeline, the vault. Only what sits
outside it is faked, at its boundary: `parecord` (a script on PATH writing real WAVs),
the Whisper model (answering per channel), and the user's Ctrl+C. The checks this
replaces mocked tapeback's own parts, so no test ever ran this path end to end.

Writing it exposed a stop that always waited out its five-second kill timeout. A flow
cannot see a wait — spawning ffmpeg afterwards reaps the leftover processes anyway —
so that check lives in tests/regressions/test_stop_reaps_recorders.py.
"""

import os
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tapeback.cli import cli
from tests.fixtures import mock_whisper_by_channel, process_state

pytestmark = pytest.mark.usefixtures("cpu_only")

SESSION = "2026-10-06_10-30-00"
START = ["start", SESSION, "--no-diarize", "--no-summarize", "--no-live"]


def _meeting():
    return mock_whisper_by_channel(
        mic_segments=[(0.0, 2.0, "I will send the report.")],
        monitor_segments=[(2.0, 4.0, "Thanks, see you Friday.")],
    )


def _sessions_root() -> Path:
    return Path(os.environ["XDG_STATE_HOME"]) / "tapeback" / "sessions"


def test_start_then_ctrl_c_records_transcribes_and_cleans_up(
    runner, vault_env, fake_parecord, ctrl_c_while_recording
):
    with patch("tapeback.transcriber.WhisperModel", return_value=_meeting()):
        result = runner.invoke(cli, START)

    assert result.exit_code == 0, result.output + repr(result.exception)

    # Both channels were recorded into one private directory under the XDG state dir.
    recorders = fake_parecord()
    assert sorted(call.device for call in recorders) == ["fake.mic", "fake.monitor"]
    assert [call.dir_mode for call in recorders] == ["700", "700"]
    # Nothing is left recording once the meeting is processed.
    assert [process_state(call.pid) for call in recorders] == [None, None]

    # The note says who said what, and when.
    note = (vault_env / "meetings" / f"{SESSION}.md").read_text()
    assert "date: 2026-10-06\n" in note
    assert 'time: "10:30"\n' in note
    assert 'duration: "00:00:04"\n' in note
    assert note.endswith(
        "[00:00:00] **You:** I will send the report.\n\n"
        "[00:00:02] **Other:** Thanks, see you Friday.\n"
    )

    # The audio is in the vault once, as the stereo merge of both channels.
    assert [p.name for p in (vault_env / "attachments" / "audio").iterdir()] == [f"{SESSION}.wav"]
    with wave.open(str(vault_env / "attachments" / "audio" / f"{SESSION}.wav"), "rb") as wf:
        assert (wf.getnchannels(), wf.getframerate(), wf.getnframes()) == (2, 48000, 192000)

    # Processed, so nothing is left to recover and nothing is still recording.
    assert list(_sessions_root().iterdir()) == []
    assert not (Path(os.environ["XDG_STATE_HOME"]) / "tapeback" / "session.json").exists()


def test_failed_processing_keeps_the_recording_for_recovery(
    runner, vault_env, fake_parecord, ctrl_c_while_recording
):
    """Sessions outlive a reboot so that a meeting whose processing failed can still be
    processed. That promise is only worth something if a failure leaves the audio."""
    model = MagicMock()
    model.transcribe.side_effect = RuntimeError("decoder exploded")

    with patch("tapeback.transcriber.WhisperModel", return_value=model):
        result = runner.invoke(cli, START)

    assert result.exit_code == 1
    assert str(result.exception) == "decoder exploded"
    session = _sessions_root() / SESSION
    assert sorted(p.name for p in session.iterdir() if p.name.endswith(".wav")) == [
        "mic.wav",
        "mic_16k.wav",
        "monitor.wav",
        "monitor_16k.wav",
        "stereo.wav",
    ]
    assert [process_state(call.pid) for call in fake_parecord()] == [None, None]
    assert not (vault_env / "meetings").exists()


@pytest.mark.usefixtures("ctrl_c_while_recording")
@pytest.mark.parametrize("mode", [0o700, 0o755], ids=["private dir records", "shared dir refused"])
def test_start_records_only_into_a_directory_no_one_else_can_reach(
    runner, vault_env, fake_parecord, monkeypatch, tmp_path, mode
):
    """The same command on the same directory, one permission bit apart: the refusal is
    the privacy check and nothing else, and it happens before anything is recorded."""
    chosen = tmp_path / "sessions"
    chosen.mkdir()
    chosen.chmod(mode)
    monkeypatch.setenv("TAPEBACK_SESSIONS_DIR", str(chosen))

    with patch("tapeback.transcriber.WhisperModel", return_value=_meeting()):
        result = runner.invoke(cli, START)

    if mode == 0o700:
        assert result.exit_code == 0, result.output + repr(result.exception)
        assert [call.dir_mode for call in fake_parecord()] == ["700", "700"]
        assert (vault_env / "meetings" / f"{SESSION}.md").exists()
    else:
        assert result.exit_code == 1
        assert str(result.exception) == (
            f"Refusing to use {chosen}: reachable by other users (mode 0755, expected 0700). "
            f"If it is yours, run: chmod 700 {chosen}"
        )
        assert fake_parecord() == []
        assert list(chosen.iterdir()) == []
