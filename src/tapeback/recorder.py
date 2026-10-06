import contextlib
import datetime
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import time
from pathlib import Path
from typing import TypedDict

from tapeback import const
from tapeback.settings import Settings


class NoActiveRecording(RuntimeError):
    """Raised when stop() is called with no session in progress."""


class SessionData(TypedDict):
    pid_monitor: int
    pid_mic: int
    session_name: str
    monitor_path: str
    mic_path: str
    started_at: str


_SESSION_NAME_RE = re.compile(r"^[\w-]+$")

_SESSIONS_DIR_NAME = "sessions"


def _state_dir() -> Path:
    """tapeback's XDG state directory — ~/.local/state/tapeback unless XDG_STATE_HOME
    says otherwise. Holds session.json and, under `sessions/`, the recordings it points
    at. A relative XDG_STATE_HOME is ignored, as the XDG spec requires: it would put the
    state wherever the command happened to be run from."""
    xdg_state_home = os.environ.get("XDG_STATE_HOME", "")
    base = Path(xdg_state_home)
    if not base.is_absolute():
        base = Path.home() / ".local" / "state"
    return base / "tapeback"


def _ensure_private_dir(path: Path) -> None:
    """Create the directory as 0700, or refuse one that is not privately ours.

    `mkdir(mode=...)` sets the mode only when it actually creates the directory, so an
    existing one has to be checked rather than trusted. lstat, not stat: mkdir with
    exist_ok follows a symlink to a directory and reports success, and a symlink is
    exactly what an attacker would plant. Refuse rather than chmod — repairing the mode
    cannot undo the window in which someone else already held the directory open.
    """
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise RuntimeError(f"Refusing to use {path}: it is a symlink, not a directory")
    if info.st_uid != os.geteuid():
        raise RuntimeError(f"Refusing to use {path}: owned by uid {info.st_uid}, not this user")
    if info.st_mode & 0o077:
        raise RuntimeError(
            f"Refusing to use {path}: reachable by other users "
            f"(mode {stat.S_IMODE(info.st_mode):04o}, expected 0700). "
            f"If it is yours, run: chmod 700 {path}"
        )


def session_root(settings: Settings) -> Path:
    """Directory holding recordings that are still in progress. Created if missing.

    `TAPEBACK_SESSIONS_DIR` when set, otherwise `sessions/` in the XDG state directory
    (~/.local/state/tapeback/sessions). Wherever it is, it is verified private before
    use. Rejected locations, and why:

    - /tmp/tapeback: /tmp is world-writable, so any other local user could pre-create
      the path and quietly collect every meeting recording, raw microphone and system
      audio both, captured long before any masking applies.
    - $XDG_RUNTIME_DIR: private, but a tmpfs of 10% of RAM — 780 MB on an 8 GB laptop,
      while an hour of recording needs about 1.6 GB once the merge and the 16 kHz copies
      land next to the raw channels. The XDG spec itself asks applications not to put
      large files there. /tmp has the same problem on systemd 258+, where it is a tmpfs
      with a per-user quota (issue #13).
    - Both of those are emptied at reboot or logout. A recording that was interrupted
      before it was transcribed should survive that, so it can still be processed.

    The state directory is on disk under the user's home: private by construction, as
    the runtime directory was, without its size limit or its lifetime. A session is
    removed once it has been processed; one that failed stays to be recovered.
    """
    root = settings.sessions_dir or _state_dir() / _SESSIONS_DIR_NAME
    _ensure_private_dir(root)
    return root


def session_dir(settings: Settings, session_name: str) -> Path:
    """Directory holding one session's raw channels. Not created here — `start` does
    that; callers that only need the path (live transcription, status messages) should
    not have a side effect."""
    validate_session_name(session_name)
    return session_root(settings) / session_name


def validate_session_name(session_name: str) -> None:
    """Reject session names that could traverse the filesystem.

    Only alphanumerics, dashes, and underscores are allowed — this keeps
    names safe to use as path components in the vault.
    """
    if not _SESSION_NAME_RE.match(session_name):
        raise ValueError(
            f"Invalid session name: {session_name!r}. "
            "Only alphanumerics, dashes, and underscores are allowed."
        )


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
            if _probe_source(const.PA_DEFAULT_MONITOR):
                monitor = const.PA_DEFAULT_MONITOR
            else:
                monitor = _resolve_monitor_via_pactl()

        if mic == "auto":
            if _probe_source(const.PA_DEFAULT_SOURCE):
                mic = const.PA_DEFAULT_SOURCE
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
    info: dict[str, str] = json.loads(result.stdout)
    default_sink = info.get("default_sink_name") or info.get("default_sink", "")
    if not default_sink:
        raise RuntimeError(
            "No default sink found. Run 'pactl list sources short' to check devices."
        )
    return f"{default_sink}{const.PA_MONITOR_SUFFIX}"


def _resolve_source_via_pactl() -> str:
    """Resolve default source from pactl info (legacy fallback)."""
    result = subprocess.run(
        ["pactl", "--format=json", "info"],
        capture_output=True,
        text=True,
        check=True,
    )
    info: dict[str, str] = json.loads(result.stdout)
    default_source = info.get("default_source_name") or info.get("default_source", "")
    if not default_source:
        raise RuntimeError(
            "No default source found. Run 'pactl list sources short' to check devices."
        )
    return default_source


def _process_running(pid: int) -> bool:
    """Whether a recorder process is still running.

    kill(pid, 0) alone cannot tell: a process that has exited stays a zombie until its
    parent reaps it, and kill still succeeds on a zombie. `tapeback start` and the tray
    spawn parecord themselves and never wait() on it, and `tapeback stop` runs in another
    process altogether — so a stopped recorder kept reading as alive, and every stop
    waited out the whole kill timeout before SIGKILLing processes already dead.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    if not _is_zombie(pid):
        return True
    # Exited. Reap it if it is ours; another process's zombie is its parent's to collect.
    _reap(pid, os.WNOHANG)
    return False


def _reap(pid: int, flags: int) -> None:
    """Collect an exited child of this process, so it does not linger as a zombie.
    A pid that is not our child raises ChildProcessError, which is the expected case for
    `tapeback stop` run against another terminal's recording."""
    with contextlib.suppress(ChildProcessError):
        os.waitpid(pid, flags)


def _is_zombie(pid: int) -> bool:
    """An exited process another parent has not reaped yet. Reads /proc: Linux only,
    as tapeback is."""
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    # The state follows the command name, which is parenthesised and may hold spaces.
    return stat_line.rsplit(")", 1)[1].split()[0] == "Z"


def _terminate_process(pid: int) -> None:
    """Send SIGTERM to a single process, ignore if already dead."""
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)


def _wait_and_kill(pids: list[int], timeout: float = 5.0) -> None:
    """Wait for processes to exit, then SIGKILL survivors."""
    alive = set(pids)
    deadline = time.monotonic() + timeout

    while alive and time.monotonic() < deadline:
        alive = {pid for pid in alive if _process_running(pid)}
        if alive:
            time.sleep(0.1)

    for pid in alive:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
            # SIGKILL cannot be ignored, so waiting for our own child cannot hang.
            _reap(pid, 0)


class Recorder:
    def __init__(self, state_dir: Path | None = None) -> None:
        self._state_dir = state_dir or _state_dir()
        self._session_file = self._state_dir / const.FILE_SESSION

    @property
    def session_file(self) -> Path:
        return self._session_file

    def start(self, settings: Settings, session_name: str | None = None) -> str:
        """Start two parecord subprocesses for monitor and mic recording.

        Creates a private session directory (see `session_root`) holding monitor.wav
        and mic.wav.
        Saves state to session.json. Returns session_name.
        """
        if self.is_recording():
            raise RuntimeError(
                "Recording already in progress. Run 'tapeback stop' to finish it first."
            )

        if not shutil.which("parecord"):
            raise RuntimeError("parecord not found. Install: sudo apt install pulseaudio-utils")

        monitor_source, mic_source = detect_devices(settings)

        if session_name is None:
            session_name = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d_%H-%M-%S")
        else:
            validate_session_name(session_name)

        tmp_dir = session_dir(settings, session_name)
        _ensure_private_dir(tmp_dir)

        monitor_path = tmp_dir / const.FILE_MONITOR
        mic_path = tmp_dir / const.FILE_MIC

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
            raise NoActiveRecording("No recording in progress.")

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
        except (json.JSONDecodeError, KeyError):
            return False

        for key in ("pid_monitor", "pid_mic"):
            if not _process_running(session[key]):
                # Process is dead — clean up stale session
                self._session_file.unlink(missing_ok=True)
                return False

        return True

    def get_session_info(self) -> SessionData | None:
        """Return session info dict if recording, else None."""
        if not self.is_recording():
            return None
        data: SessionData = json.loads(self._session_file.read_text())
        return data
