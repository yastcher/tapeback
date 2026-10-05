"""Regression: in-progress recordings must not sit in a world-writable directory.

Bug: sessions were written to /tmp/tapeback, created with
`mkdir(parents=True, exist_ok=True, mode=0o700)`. The mode applies only when mkdir
actually creates the directory, and nothing verified one that already existed — so any
other local user could pre-create /tmp/tapeback (or point it at a directory of their
own) and then read every meeting recording: raw microphone and system audio, captured
before any of the masking this project does later.
"""

import json
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from tapeback.recorder import Recorder, session_dir, session_root
from tapeback.settings import Settings


def test_session_root_lives_under_xdg_runtime_dir(tmp_path, monkeypatch):
    """/run/user/$UID is created by logind as 0700 and owned by the user, so a
    directory under it is private by construction rather than by convention."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))

    assert session_root() == tmp_path / "tapeback"


def test_start_records_into_the_private_root(tmp_path, monkeypatch):
    """The call site, not just the helper: a real start() must place both channels
    under the private root and nowhere near the old shared /tmp/tapeback."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    recorder = Recorder(state_dir=tmp_path / "state")

    with (
        patch("tapeback.recorder.detect_devices", return_value=("mon_dev", "mic_dev")),
        patch("tapeback.recorder.shutil.which", return_value="/usr/bin/parecord"),
        patch("tapeback.recorder.subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 4242
        recorder.start(Settings(vault_path=tmp_path / "vault"), session_name="meeting")

    # Read the session file directly: get_session_info() first checks the recording is
    # alive, and the mocked PID is not a live process.
    session = json.loads(recorder.session_file.read_text())
    assert session["mic_path"] == str(tmp_path / "tapeback" / "meeting" / "mic.wav")
    assert session["monitor_path"] == str(tmp_path / "tapeback" / "meeting" / "monitor.wav")
    # The literal old path is the point of this assertion — S108 is what the fix removed.
    assert not Path("/tmp/tapeback/meeting").exists()  # noqa: S108


def test_session_root_is_created_private(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))

    assert stat.S_IMODE(session_root().stat().st_mode) == 0o700


def test_falls_back_to_the_user_cache_without_xdg(tmp_path, monkeypatch):
    """No XDG_RUNTIME_DIR (ssh without logind, containers, cron) — the home directory
    is user-owned for the same reason, so it is the safe fallback."""
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert session_root() == tmp_path / ".cache" / "tapeback" / "sessions"


@pytest.mark.parametrize("mode", [0o700, 0o600, 0o500])
def test_accepts_a_directory_only_the_owner_can_reach(mode, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    existing = tmp_path / "tapeback"
    existing.mkdir()
    existing.chmod(mode)

    assert session_root() == existing


@pytest.mark.parametrize("mode", [0o701, 0o710, 0o750, 0o777])
def test_refuses_a_directory_other_users_can_reach(mode, tmp_path, monkeypatch):
    """0o700 passes and one bit more anywhere fails — the exact boundary the check
    draws. Refusing beats repairing: a chmod cannot undo the window in which someone
    else already held the directory open."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    existing = tmp_path / "tapeback"
    existing.mkdir()
    existing.chmod(mode)

    with pytest.raises(RuntimeError, match="other users"):
        session_root()


def test_refuses_a_symlink(tmp_path, monkeypatch):
    """mkdir(exist_ok=True) follows a symlink to a directory and reports success, so
    the check has to lstat rather than stat."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    (tmp_path / "tapeback").symlink_to(elsewhere)

    with pytest.raises(RuntimeError, match="symlink"):
        session_root()


def test_session_dir_hangs_off_the_root(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))

    assert session_dir("meeting-1") == tmp_path / "tapeback" / "meeting-1"


def test_session_dir_rejects_a_traversing_name(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="Invalid session name"):
        session_dir("../escape")
