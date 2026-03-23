import contextlib
import datetime
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import TypedDict

from meetrec.settings import Settings


class SessionData(TypedDict):
    pid_monitor: int
    pid_mic: int
    session_name: str
    monitor_path: str
    mic_path: str
    started_at: str


_DEFAULT_STATE_DIR = Path.home() / ".local" / "state" / "meetrec"


def detect_devices(settings: Settings) -> tuple[str, str]:
    """Return (monitor_source, mic_source).

    If settings.monitor_source == "auto":
        Use @DEFAULT_MONITOR@ (PulseAudio/PipeWire dynamic reference that
        follows the current default sink — survives device switches).
        Falls back to pactl info -> default_sink + ".monitor" when
        @DEFAULT_MONITOR@ is not supported.
    If settings.mic_source == "auto":
        Use @DEFAULT_SOURCE@ (follows the current default source).
        Falls back to pactl info -> default_source.

    Raises RuntimeError if pactl is not available or devices not found.
    """
    monitor = settings.monitor_source
    mic = settings.mic_source

    if monitor == "auto" or mic == "auto":
        if not shutil.which("pactl"):
            raise RuntimeError("pactl not found. Install: sudo apt install pulseaudio-utils")

        # Try dynamic references first (PipeWire / PulseAudio 14+).
        # They follow the current default device, so recording survives
        # hot-switching between speakers, headphones, etc.
        if monitor == "auto":
            if _probe_source("@DEFAULT_MONITOR@"):
                monitor = "@DEFAULT_MONITOR@"
            else:
                monitor = _resolve_monitor_via_pactl()

        if mic == "auto":
            if _probe_source("@DEFAULT_SOURCE@"):
                mic = "@DEFAULT_SOURCE@"
            else:
                mic = _resolve_source_via_pactl()

    return monitor, mic


def _probe_source(source_name: str) -> bool:
    """Return True if parecord can open the given source (quick 0.2s test)."""
    if not shutil.which("parecord"):
        return False
    try:
        proc = subprocess.Popen(
            [
                "parecord",
                f"--device={source_name}",
                "--format=s16le",
                "--rate=16000",
                "--channels=1",
                "/dev/null",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.2)
        proc.terminate()
        proc.wait(timeout=2)
        # If parecord ran for 0.2s without crashing, the source exists
        return proc.returncode is not None
    except Exception:
        return False


def _resolve_monitor_via_pactl() -> str:
    """Resolve monitor source from pactl info (legacy fallback)."""
    result = subprocess.run(
        ["pactl", "--format=json", "info"],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    default_sink = info.get("default_sink_name") or info.get("default_sink", "")
    if not default_sink:
        raise RuntimeError(
            "No default sink found. Run 'pactl list sources short' to check devices."
        )
    return f"{default_sink}.monitor"


def _resolve_source_via_pactl() -> str:
    """Resolve default source from pactl info (legacy fallback)."""
    result = subprocess.run(
        ["pactl", "--format=json", "info"],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    default_source = info.get("default_source_name") or info.get("default_source", "")
    if not default_source:
        raise RuntimeError(
            "No default source found. Run 'pactl list sources short' to check devices."
        )
    return default_source


def _terminate_process(pid: int) -> None:
    """Send SIGTERM to a single process, ignore if already dead."""
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)


def _wait_and_kill(pids: list[int], timeout: float = 5.0) -> None:
    """Wait for processes to exit, then SIGKILL survivors."""
    alive = set(pids)
    deadline = time.monotonic() + timeout

    while alive and time.monotonic() < deadline:
        for pid in list(alive):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                alive.discard(pid)
        if alive:
            time.sleep(0.1)

    for pid in alive:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


class Recorder:
    def __init__(self, state_dir: Path | None = None) -> None:
        self._state_dir = state_dir or _DEFAULT_STATE_DIR
        self._session_file = self._state_dir / "session.json"

    @property
    def session_file(self) -> Path:
        return self._session_file

    def start(self, settings: Settings, session_name: str | None = None) -> str:
        """Start two parecord subprocesses for monitor and mic recording.

        Creates temp directory /tmp/meetrec/{session_name}/ with monitor.wav and mic.wav.
        Saves state to session.json. Returns session_name.
        """
        if self.is_recording():
            raise RuntimeError(
                "Recording already in progress. Run 'meetrec stop' to finish it first."
            )

        if not shutil.which("parecord"):
            raise RuntimeError("parecord not found. Install: sudo apt install pulseaudio-utils")

        monitor_source, mic_source = detect_devices(settings)

        if session_name is None:
            session_name = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d_%H-%M-%S")
        elif not re.match(r"^[\w-]+$", session_name):
            raise ValueError(
                f"Invalid session name: {session_name!r}. "
                "Only alphanumerics, dashes, and underscores are allowed."
            )

        base_dir = Path("/tmp/meetrec")
        base_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp_dir = base_dir / session_name
        tmp_dir.mkdir(exist_ok=True, mode=0o700)

        monitor_path = tmp_dir / "monitor.wav"
        mic_path = tmp_dir / "mic.wav"

        base_cmd = [
            "parecord",
            "--format=s16le",
            f"--rate={settings.sample_rate}",
            "--channels=1",
            "--file-format=wav",
        ]

        monitor_proc = subprocess.Popen(
            [*base_cmd, f"--device={monitor_source}", str(monitor_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        mic_proc = subprocess.Popen(
            [*base_cmd, f"--device={mic_source}", str(mic_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # Save session state
        self._state_dir.mkdir(parents=True, exist_ok=True)
        session_data = {
            "pid_monitor": monitor_proc.pid,
            "pid_mic": mic_proc.pid,
            "session_name": session_name,
            "monitor_path": str(monitor_path),
            "mic_path": str(mic_path),
            "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        self._session_file.write_text(json.dumps(session_data, indent=2))

        return session_name

    def stop(self) -> tuple[Path, Path]:
        """Stop both subprocesses (SIGTERM, then SIGKILL after 5 sec).

        Returns paths to (monitor.wav, mic.wav).
        Removes session.json.
        """
        if not self._session_file.exists():
            raise RuntimeError("No recording in progress.")

        session: SessionData = json.loads(self._session_file.read_text())
        pids = [session["pid_monitor"], session["pid_mic"]]

        # Send SIGTERM to both
        for pid in pids:
            _terminate_process(pid)

        # Wait, then force-kill survivors
        _wait_and_kill(pids)

        self._session_file.unlink()

        return Path(session["monitor_path"]), Path(session["mic_path"])

    def is_recording(self) -> bool:
        """Check if recording is active (session.json exists and processes are alive)."""
        if not self._session_file.exists():
            return False

        try:
            session: SessionData = json.loads(self._session_file.read_text())
        except json.JSONDecodeError, KeyError:
            return False

        for key in ("pid_monitor", "pid_mic"):
            try:
                os.kill(session[key], 0)
            except ProcessLookupError:
                # Process is dead — clean up stale session
                self._session_file.unlink(missing_ok=True)
                return False

        return True

    def get_session_info(self) -> SessionData | None:
        """Return session info dict if recording, else None."""
        if not self.is_recording():
            return None
        return json.loads(self._session_file.read_text())
