"""Regression: start must not crash when peer `tapeback stop` already finished."""

from unittest.mock import MagicMock, patch

from tapeback.cli import cli
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
