"""Regression: parecord's stderr went into a pipe nobody read.

Recorder.start() gave each parecord `stderr=PIPE` and dropped the Popen. Nothing ever
read that pipe: 64 KiB of messages would have blocked the recorder mid-meeting, the
starter's exit would have killed it with SIGPIPE at its next message, and whatever
parecord said about a failure was lost. Each recorder now writes its stderr to a log
of its own in the private session directory.
"""

import os
import stat
import time

from tapeback.recorder import Recorder, _spawn_recorder, session_dir
from tapeback.settings import Settings

# Bounds the wait for both fake recorders to start; never asserted on.
_RECORDERS_START_SECONDS = 5.0


def test_each_recorder_writes_its_stderr_to_a_private_log(tmp_path, fake_parecord):
    settings = Settings(vault_path=tmp_path / "vault")
    recorder = Recorder(state_dir=tmp_path / "state")

    name = recorder.start(settings)
    try:
        deadline = time.monotonic() + _RECORDERS_START_SECONDS
        while len(fake_parecord()) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        recorder.stop()

    directory = session_dir(settings, name)
    assert (directory / "monitor.log").read_text() == "fake parecord recording fake.monitor\n"
    assert (directory / "mic.log").read_text() == "fake parecord recording fake.mic\n"
    assert stat.S_IMODE((directory / "monitor.log").stat().st_mode) == 0o600
    assert stat.S_IMODE((directory / "mic.log").stat().st_mode) == 0o600


def test_a_recorder_gets_the_signal_defaults_popen_would_give_it(tmp_path):
    """Python ignores SIGPIPE in itself, and an ignored signal survives exec: without
    the defaults back, a recorder would see EPIPE where Popen's child died. bash reports
    an inherited ignore through `trap -p`."""
    log = tmp_path / "log"

    pid = _spawn_recorder(["bash", "-c", "trap -p PIPE XFSZ >&2"], log)
    os.waitpid(pid, 0)

    assert log.read_text() == ""
