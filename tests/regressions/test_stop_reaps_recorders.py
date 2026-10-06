"""Regression: stopping a recording waited out the whole kill timeout, every time.

Bug: Recorder.stop() sent SIGTERM to the two parecord processes and then polled
os.kill(pid, 0) until they were gone. A process that has exited stays a zombie until
its parent reaps it, and kill(pid, 0) still succeeds on a zombie. `tapeback start`
(Ctrl+C) and the tray stop processes they spawned themselves and never reap, and
`tapeback stop` from another terminal looks at the zombies of the `start` process —
so every stop waited the full 5 s, then sent SIGKILL to processes already dead. The
same check made is_recording() blind to a parecord that died mid-meeting.

Real processes, not mocks: the bug lives in how the kernel reports them.
"""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from tapeback.recorder import Recorder, _terminate_process, _wait_and_kill
from tests.fixtures import process_state

# Bounds a wait for a state the kernel reaches within milliseconds; never asserted on.
_STATE_WAIT_SECONDS = 2.0


def _start_like_recorder() -> int:
    """A child started the way Recorder.start() starts parecord: the Popen is dropped."""
    return subprocess.Popen(["sleep", "100"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE).pid


def _wait_for_state(pid: int, state: str) -> None:
    deadline = time.monotonic() + _STATE_WAIT_SECONDS
    while process_state(pid) != state and time.monotonic() < deadline:
        time.sleep(0.01)
    assert process_state(pid) == state


def _wait_for_command(pid: int, command: str) -> None:
    comm = Path(f"/proc/{pid}/comm")
    deadline = time.monotonic() + _STATE_WAIT_SECONDS
    while comm.read_text().strip() != command and time.monotonic() < deadline:
        time.sleep(0.01)
    assert comm.read_text().strip() == command


def _session(recorder: Recorder, tmp_path: Path, monitor_pid: int, mic_pid: int) -> None:
    recorder.session_file.write_text(
        json.dumps(
            {
                "pid_monitor": monitor_pid,
                "pid_mic": mic_pid,
                "session_name": "meeting",
                "monitor_path": str(tmp_path / "monitor.wav"),
                "mic_path": str(tmp_path / "mic.wav"),
                "started_at": "2026-10-06T10:00:00+03:00",
            }
        )
    )


def test_stop_reaps_the_recorders_it_started(tmp_path):
    """The Ctrl+C and tray path: the stopping process is the parent."""
    pids = [_start_like_recorder(), _start_like_recorder()]
    recorder = Recorder(state_dir=tmp_path)
    _session(recorder, tmp_path, *pids)

    recorder.stop()

    # Reaped, not left behind as zombies: their /proc entries are gone.
    assert [process_state(pid) for pid in pids] == [None, None]


def test_a_zombie_left_for_another_process_reads_as_stopped(tmp_path):
    """The `tapeback stop` path: the recorders belong to the `start` process, so this
    one cannot reap them — an exited one must still read as stopped."""
    parent = subprocess.Popen(
        ["bash", "-c", "sleep 100 & echo $!; exec sleep 100"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert parent.stdout is not None
        orphan = int(parent.stdout.readline())
        # Until bash has exec'd into `sleep` it reaps its own children, and the orphan
        # would vanish instead of becoming a zombie.
        _wait_for_command(parent.pid, "sleep")
        os.kill(orphan, signal.SIGTERM)
        # Its parent is now a plain `sleep`, which never reaps it.
        _wait_for_state(orphan, "Z")
        recorder = Recorder(state_dir=tmp_path)
        _session(recorder, tmp_path, orphan, orphan)

        assert recorder.is_recording() is False
    finally:
        parent.kill()
        parent.wait()


def test_is_recording_notices_a_recorder_that_died(tmp_path):
    """A parecord that exits mid-meeting (device gone) is a zombie of the `start`
    process, and must not keep the recording looking alive."""
    alive = _start_like_recorder()
    dead = _start_like_recorder()
    os.kill(dead, signal.SIGTERM)
    _wait_for_state(dead, "Z")
    recorder = Recorder(state_dir=tmp_path)
    _session(recorder, tmp_path, alive, dead)
    try:
        assert recorder.is_recording() is False
    finally:
        os.kill(alive, signal.SIGKILL)
        os.waitpid(alive, 0)


def test_a_recorder_that_ignores_sigterm_is_killed_and_reaped():
    """The last resort must not leave a zombie behind either. `exec` keeps the ignored
    SIGTERM, so the recorder outlives the polite stop and needs SIGKILL."""
    pid = subprocess.Popen(
        ["bash", "-c", "trap '' TERM; exec sleep 100"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    ).pid
    _wait_for_state(pid, "S")
    _terminate_process(pid)

    _wait_and_kill([pid], timeout=0.2)

    assert process_state(pid) is None
