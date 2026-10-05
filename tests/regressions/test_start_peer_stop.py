"""Regression: start must not crash when peer `tapeback stop` already finished."""

from unittest.mock import MagicMock, patch

from tapeback.cli import cli
from tapeback.recorder import NoActiveRecording
from tapeback.settings import Settings


def test_start_exits_cleanly_when_peer_stop_finished(runner, vault_env):
    """If `tapeback stop` already cleared the session, start must not crash."""
    recorder = MagicMock()
    recorder.start.return_value = "peer-stopped"
    # Wait loop sees active once, then peer stop cleared the session.
    recorder.is_recording.side_effect = [True, False, False]

    with (
        patch("tapeback.cli.get_settings", return_value=Settings(vault_path=vault_env)),
        patch("tapeback.cli.Recorder", return_value=recorder),
        patch("tapeback.cli.detect_devices", return_value=("mon", "mic")),
        patch("tapeback.pipeline.stop_and_process") as mock_stop,
        patch("time.sleep"),
    ):
        result = runner.invoke(cli, ["start", "peer-stopped", "--no-summarize"])

    assert result.exit_code == 0, result.output + str(result.exception or "")
    mock_stop.assert_not_called()


def test_start_exits_cleanly_when_stop_races_after_is_recording(runner, vault_env):
    """is_recording stays True; stop_and_process raises NoActiveRecording → exit 0."""
    recorder = MagicMock()
    recorder.start.return_value = "race-window"
    # Wait loop exits on Ctrl+C path: is_recording True once then False ends loop;
    # post-loop pre-check sees True (still looks active), then stop races.
    recorder.is_recording.side_effect = [True, False, True]

    with (
        patch("tapeback.cli.get_settings", return_value=Settings(vault_path=vault_env)),
        patch("tapeback.cli.Recorder", return_value=recorder),
        patch("tapeback.cli.detect_devices", return_value=("mon", "mic")),
        patch(
            "tapeback.pipeline.stop_and_process",
            side_effect=NoActiveRecording("No recording in progress."),
        ) as mock_stop,
        patch("time.sleep"),
    ):
        result = runner.invoke(cli, ["start", "race-window", "--no-summarize"])

    assert result.exit_code == 0, result.output + str(result.exception or "")
    assert result.exception is None
    mock_stop.assert_called_once()
