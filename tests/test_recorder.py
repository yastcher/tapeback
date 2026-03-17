import json
import os
import signal
from unittest.mock import MagicMock, patch

import pytest

from meetrec.recorder import Recorder, detect_devices
from meetrec.settings import Settings


@pytest.fixture
def recorder(tmp_path):
    """Recorder with temporary state directory."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    return Recorder(state_dir=state_dir)


@pytest.fixture
def session_file(recorder):
    """Path to session file."""
    return recorder.session_file


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
    session_data = {
        "pid_monitor": 99998,
        "pid_mic": 99999,
        "session_name": "test_session",
        "monitor_path": "/tmp/meetrec/test_session/monitor.wav",
        "mic_path": "/tmp/meetrec/test_session/mic.wav",
        "started_at": "2026-03-17T14:30:00",
    }
    session_file.write_text(json.dumps(session_data))

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


def test_stop_without_start_raises(recorder):
    """stop() without active recording should raise RuntimeError."""
    with pytest.raises(RuntimeError, match="No recording in progress"):
        recorder.stop()


def test_start_while_recording_raises(recorder, settings, session_file):
    """start() while already recording should raise RuntimeError."""
    session_data = {
        "pid_monitor": os.getpid(),  # Use own PID so it appears alive
        "pid_mic": os.getpid(),
        "session_name": "existing",
        "monitor_path": "/tmp/meetrec/existing/monitor.wav",
        "mic_path": "/tmp/meetrec/existing/mic.wav",
        "started_at": "2026-03-17T14:30:00",
    }
    session_file.write_text(json.dumps(session_data))

    with pytest.raises(RuntimeError, match="already in progress"):
        recorder.start(settings)


def test_parecord_not_found(recorder, settings):
    """Should give clear error when parecord is not installed."""
    with (
        patch("meetrec.recorder.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="parecord not found"),
    ):
        recorder.start(settings)
