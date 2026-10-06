"""Regression: where in-progress recordings live.

Bug 1: sessions were written to /tmp/tapeback, created with
`mkdir(parents=True, exist_ok=True, mode=0o700)`. The mode applies only when mkdir
actually creates the directory, and nothing verified one that already existed — so any
other local user could pre-create /tmp/tapeback (or point it at a directory of their
own) and then read every meeting recording: raw microphone and system audio, captured
before any of the masking this project does later.

Bug 2 (issue #13): a tmpfs cannot hold a long meeting. /tmp carries a per-user quota
on systemd 258+, and $XDG_RUNTIME_DIR — tried as the fix for bug 1 — is a tmpfs of
10% of RAM (780 MB on an 8 GB laptop), while an hour of recording needs about 1.6 GB
once the merge and the 16 kHz copies land next to the raw channels. Both are also
emptied at reboot or logout, taking an interrupted meeting with them.
"""

import stat
from pathlib import Path

import pytest

from tapeback.recorder import Recorder, session_dir, session_root
from tapeback.settings import Settings


def _settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(vault_path=tmp_path / "vault", **overrides)


def test_sessions_live_in_the_xdg_state_dir_never_the_runtime_dir(tmp_path, monkeypatch):
    """On disk and under the user's home: private by construction, no tmpfs quota,
    and still there after a reboot."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    root = session_root(_settings(tmp_path))

    assert root == tmp_path / "state" / "tapeback" / "sessions"
    assert not (tmp_path / "run").exists()


def test_without_xdg_state_home_the_default_is_under_home(tmp_path, monkeypatch):
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert session_root(_settings(tmp_path)) == tmp_path / ".local/state/tapeback/sessions"


def test_a_relative_xdg_state_home_is_ignored(tmp_path, monkeypatch):
    """The XDG spec: a relative path in these variables is invalid and must be ignored —
    otherwise recordings would land wherever the command happened to be run from."""
    monkeypatch.setenv("XDG_STATE_HOME", "relative/state")
    monkeypatch.setenv("HOME", str(tmp_path))

    assert session_root(_settings(tmp_path)) == tmp_path / ".local/state/tapeback/sessions"


def test_the_sessions_dir_setting_overrides_the_default(tmp_path):
    """Issue #13: a user whose home cannot hold the recording points it elsewhere."""
    chosen = tmp_path / "big-disk" / "tapeback"

    root = session_root(_settings(tmp_path, sessions_dir=chosen))

    assert root == chosen
    assert stat.S_IMODE(chosen.stat().st_mode) == 0o700


def test_session_state_follows_xdg_state_home_too(tmp_path, monkeypatch):
    """session.json and the recordings it points at share one base, so a user who moves
    XDG_STATE_HOME does not end up with state in one place and audio in another."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    assert Recorder().session_file == tmp_path / "state" / "tapeback" / "session.json"


def test_session_root_is_created_private(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    assert stat.S_IMODE(session_root(_settings(tmp_path)).stat().st_mode) == 0o700


@pytest.mark.parametrize("mode", [0o700, 0o600, 0o500])
def test_accepts_a_directory_only_the_owner_can_reach(mode, tmp_path):
    existing = tmp_path / "sessions"
    existing.mkdir()
    existing.chmod(mode)

    assert session_root(_settings(tmp_path, sessions_dir=existing)) == existing


@pytest.mark.parametrize("mode", [0o701, 0o710, 0o750, 0o777])
def test_refuses_a_directory_other_users_can_reach(mode, tmp_path):
    """0o700 passes and one bit more anywhere fails — the exact boundary the check
    draws. Refusing beats repairing: a chmod cannot undo the window in which someone
    else already held the directory open. The message says how to fix it, since with
    TAPEBACK_SESSIONS_DIR the directory is often one the user made themselves."""
    existing = tmp_path / "sessions"
    existing.mkdir()
    existing.chmod(mode)

    with pytest.raises(RuntimeError, match=rf"other users.*chmod 700 {existing}"):
        session_root(_settings(tmp_path, sessions_dir=existing))


def test_refuses_a_symlink(tmp_path):
    """mkdir(exist_ok=True) follows a symlink to a directory and reports success, so
    the check has to lstat rather than stat."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    link = tmp_path / "sessions"
    link.symlink_to(elsewhere)

    with pytest.raises(RuntimeError, match="symlink"):
        session_root(_settings(tmp_path, sessions_dir=link))


def test_session_dir_hangs_off_the_root(tmp_path):
    root = tmp_path / "sessions"

    assert session_dir(_settings(tmp_path, sessions_dir=root), "meeting-1") == root / "meeting-1"


def test_session_dir_rejects_a_traversing_name(tmp_path):
    with pytest.raises(ValueError, match="Invalid session name"):
        session_dir(_settings(tmp_path, sessions_dir=tmp_path / "sessions"), "../escape")
