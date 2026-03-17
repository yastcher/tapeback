import contextlib
import datetime
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from meetrec.settings import Settings

_DEFAULT_STATE_DIR = Path.home() / ".local" / "state" / "meetrec"


def detect_devices(settings: Settings) -> tuple[str, str]:
    """Return (monitor_source, mic_source).

    If settings.monitor_source == "auto":
        pactl --format=json info -> default_sink + ".monitor"
    If settings.mic_source == "auto":
        pactl --format=json info -> default_source

    Raises RuntimeError if pactl is not available or devices not found.
    """
    monitor = settings.monitor_source
    mic = settings.mic_source

    if monitor == "auto" or mic == "auto":
        if not shutil.which("pactl"):
            raise RuntimeError("pactl not found. Install: sudo apt install pulseaudio-utils")

        result = subprocess.run(
            ["pactl", "--format=json", "info"],
            capture_output=True,
            text=True,
            check=True,
        )
        info = json.loads(result.stdout)

        if monitor == "auto":
            default_sink = info.get("default_sink_name") or info.get("default_sink", "")
            if not default_sink:
                raise RuntimeError(
                    "No default sink found. Run 'pactl list sources short' to check devices."
                )
            monitor = f"{default_sink}.monitor"

        if mic == "auto":
            default_source = info.get("default_source_name") or info.get("default_source", "")
            if not default_source:
                raise RuntimeError(
                    "No default source found. Run 'pactl list sources short' to check devices."
                )
            mic = default_source

    return monitor, mic


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

        tmp_dir = Path("/tmp/meetrec") / session_name
        tmp_dir.mkdir(parents=True, exist_ok=True)

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

        session = json.loads(self._session_file.read_text())
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
            session = json.loads(self._session_file.read_text())
        except (json.JSONDecodeError, KeyError):
            return False

        for key in ("pid_monitor", "pid_mic"):
            try:
                os.kill(session[key], 0)
            except ProcessLookupError:
                # Process is dead — clean up stale session
                self._session_file.unlink(missing_ok=True)
                return False

        return True

    def get_session_info(self) -> dict | None:
        """Return session info dict if recording, else None."""
        if not self.is_recording():
            return None
        return json.loads(self._session_file.read_text())
