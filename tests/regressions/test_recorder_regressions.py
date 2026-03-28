"""Regression tests for recorder bugs."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from tapeback.recorder import detect_devices
from tests.fixtures import create_session_file


def test_detect_devices_auto_legacy_keys(settings):
    """Auto-detect should also work with legacy pactl keys (default_sink/default_source).

    Bug: PulseAudio <17 uses 'default_sink' instead of 'default_sink_name'.
    """
    pactl_output = json.dumps(
        {
            "default_sink": "alsa_output.usb-stereo",
            "default_source": "alsa_input.usb-stereo",
        }
    )

    with (
        patch("tapeback.recorder.shutil.which", return_value="/usr/bin/pactl"),
        patch("tapeback.recorder._probe_source", return_value=False),
        patch("tapeback.recorder.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(stdout=pactl_output, returncode=0)

        monitor, mic = detect_devices(settings)

    assert monitor == "alsa_output.usb-stereo.monitor"
    assert mic == "alsa_input.usb-stereo"


def test_stop_without_start_raises(recorder):
    """stop() without active recording should raise RuntimeError.

    Bug: stop() crashed with unclear error instead of clean message.
    """
    with pytest.raises(RuntimeError, match="No recording in progress"):
        recorder.stop()


def test_start_while_recording_raises(recorder, settings, session_file):
    """start() while already recording should raise RuntimeError.

    Bug: starting a second recording corrupted the session file.
    """
    create_session_file(
        session_file,
        pid_monitor=os.getpid(),
        pid_mic=os.getpid(),
        session_name="existing",
    )

    with pytest.raises(RuntimeError, match="already in progress"):
        recorder.start(settings)


def test_parecord_not_found(recorder, settings):
    """Should give clear error when parecord is not installed.

    Bug: cryptic subprocess error instead of user-friendly message.
    """
    with (
        patch("tapeback.recorder.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="parecord not found"),
    ):
        recorder.start(settings)
