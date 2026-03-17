import json
import signal
from unittest.mock import MagicMock, patch

from meetrec.recorder import detect_devices
from meetrec.settings import Settings
from tests.fixtures import create_session_file


def test_detect_devices_auto(settings):
    """Auto-detect should parse pactl JSON output."""
    pactl_output = json.dumps(
        {
            "default_sink_name": "alsa_output.pci-0000_00_1f.3.analog-stereo",
            "default_source_name": "alsa_input.pci-0000_00_1f.3.analog-stereo",
        }
    )

    with (
        patch("meetrec.recorder.shutil.which", return_value="/usr/bin/pactl"),
        patch("meetrec.recorder.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(stdout=pactl_output, returncode=0)

        monitor, mic = detect_devices(settings)

    assert monitor == "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
    assert mic == "alsa_input.pci-0000_00_1f.3.analog-stereo"


def test_detect_devices_explicit(tmp_vault):
    """Explicit device names should be returned without calling pactl."""
    s = Settings(
        vault_path=tmp_vault,
        monitor_source="my_monitor",
        mic_source="my_mic",
    )

    with patch("meetrec.recorder.subprocess.run") as mock_run:
        monitor, mic = detect_devices(s)
        mock_run.assert_not_called()

    assert monitor == "my_monitor"
    assert mic == "my_mic"


def test_start_creates_session_file(recorder, settings, session_file):
    """start() should create session.json with PIDs and paths."""
    with (
        patch("meetrec.recorder.detect_devices", return_value=("monitor_dev", "mic_dev")),
        patch("meetrec.recorder.shutil.which", return_value="/usr/bin/parecord"),
        patch("meetrec.recorder.subprocess.Popen") as mock_popen,
    ):
        proc_mock = MagicMock()
        proc_mock.pid = 12345
        mock_popen.return_value = proc_mock

        session_name = recorder.start(settings, session_name="test_session")

    assert session_name == "test_session"
    assert session_file.exists()

    data = json.loads(session_file.read_text())
    assert data["session_name"] == "test_session"
    assert data["pid_monitor"] == 12345
    assert data["pid_mic"] == 12345
    assert "started_at" in data
    assert "monitor_path" in data
    assert "mic_path" in data


def test_stop_sends_sigterm(recorder, session_file):
    """stop() should send SIGTERM to both processes."""
    create_session_file(session_file)

    killed = []

    def mock_kill(pid, sig):
        killed.append((pid, sig))
        if sig == 0:
            raise ProcessLookupError
        if sig == signal.SIGTERM:
            return  # SIGTERM accepted

    with patch("meetrec.recorder.os.kill", side_effect=mock_kill):
        _monitor_path, _mic_path = recorder.stop()

    # SIGTERM sent to both
    assert (99998, signal.SIGTERM) in killed
    assert (99999, signal.SIGTERM) in killed
    assert not session_file.exists()
