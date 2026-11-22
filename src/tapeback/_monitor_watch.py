"""Watching the monitor while a meeting records, so a side the recording misses is noticed.

The monitor records one output: the default one, unless TAPEBACK_MONITOR_SOURCE names
another. A meeting app set to play to another output leaves it digitally silent: the other
side is never recorded, and nothing said so until the note showed one side of the
conversation.

Two signs, combined. Digital silence alone is not enough: before a call starts, and in some
apps while nobody talks, the monitor is silent too. An app playing to another output
while the recorded one stays silent is the case itself, so it is checked first and warned
about early; silence with no such app only after a longer while, and more cautiously.
"""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tapeback import const
from tapeback.settings import Settings

# A pactl that hangs must not stall the recording loop that asks it.
PACTL_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class Stream:
    """An application's playback stream, and the output it plays to."""

    index: int
    app: str
    sink: str


def _pactl_json(*args: str) -> Any:
    """`pactl -f json <args>` parsed; None when pactl is missing, fails or hangs.

    The C locale: under another, pactl's JSON encoder complains about every non-ASCII
    character on stderr.
    """
    if not shutil.which("pactl"):
        return None
    try:
        result = subprocess.run(
            ["pactl", "-f", "json", *args],
            capture_output=True,
            text=True,
            timeout=PACTL_TIMEOUT_SECONDS,
            env={**os.environ, "LC_ALL": "C"},
            check=True,
        )
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _objects(*args: str) -> list[dict[str, Any]]:
    """The objects in the list `pactl -f json <args>` answers; none for another shape.

    The watch runs inside the recording loop, so nothing pactl answers may raise there.
    """
    reply = _pactl_json(*args)
    if not isinstance(reply, list):
        return []
    return [entry for entry in reply if isinstance(entry, dict)]


def _index(entry: dict[str, Any], key: str) -> int | None:
    value = entry.get(key)
    return value if isinstance(value, int) else None


def _names(*args: str) -> dict[int, str]:
    """Index to name, for the outputs or sources pactl lists."""
    return {
        index: entry["name"]
        for entry in _objects(*args)
        if (index := _index(entry, "index")) is not None and isinstance(entry.get("name"), str)
    }


def _properties(entry: dict[str, Any]) -> dict[str, Any]:
    properties = entry.get("properties")
    return properties if isinstance(properties, dict) else {}


def recorded_sink(monitor_path: Path) -> str | None:
    """The output whose monitor the recording into monitor_path takes, as the stream says.

    Asked of the recording itself, not of the default output: the default may have changed
    since the recording started. parecord names its stream after the file it writes. None
    when the stream is not found or records something other than an output's monitor.
    """
    for entry in _objects("list", "source-outputs"):
        source = _index(entry, "source")
        if source is None or _properties(entry).get("media.name") != str(monitor_path):
            continue
        name = _names("list", "short", "sources").get(source, "")
        if name.endswith(const.PA_MONITOR_SUFFIX):
            return name.removesuffix(const.PA_MONITOR_SUFFIX)
    return None


def streams_playing_elsewhere(sink: str) -> list[Stream]:
    """The application streams that play, not paused, to a named output other than sink."""
    outputs = _names("list", "short", "sinks")
    streams = []
    for entry in _objects("list", "sink-inputs"):
        index, output = _index(entry, "index"), _index(entry, "sink")
        if index is None or output is None or entry.get("corked"):
            continue
        name = outputs.get(output)
        if name is None or name == sink:
            continue
        app = _properties(entry).get("application.name")
        streams.append(
            Stream(index=index, app=app if isinstance(app, str) else "an application", sink=name)
        )
    return streams


class MonitorWatch:
    """Reads what the monitor recorded since the last poll, and says when to worry."""

    def __init__(self, monitor_path: Path, settings: Settings) -> None:
        self._path = monitor_path
        self._bytes_per_second = settings.sample_rate * const.BYTES_PER_SAMPLE
        self._check_after = settings.monitor_silence_check_seconds
        self._warn_after = settings.monitor_silence_warn_seconds
        self._read = 0
        # Bytes of digital silence since the last sound. The file's WAV header counts as
        # sound, so the first run starts right after it.
        self._silent = 0
        self._next_check = self._check_after
        self._warned_elsewhere = False
        self._warned_silent = False

    def poll(self) -> str | None:
        """A warning when the monitor's silence calls for one, each kind once a silence.

        The app playing elsewhere is looked for again every check period: a call often
        starts after the recording, and its app may begin playing elsewhere minutes in,
        after the cautious warning was already given.
        """
        chunk = self._new_bytes()
        sound = chunk.rstrip(b"\0")
        if sound:
            self._silent = len(chunk) - len(sound)
            self._next_check = self._check_after
            self._warned_elsewhere = self._warned_silent = False
        else:
            self._silent += len(chunk)
        seconds = self._silent / self._bytes_per_second
        if self._check_after and not self._warned_elsewhere and seconds >= self._next_check:
            self._next_check = seconds + self._check_after
            warning = self._elsewhere(seconds)
            if warning:
                self._warned_elsewhere = True
                return warning
        if (
            self._warn_after
            and not (self._warned_silent or self._warned_elsewhere)
            and seconds >= self._warn_after
        ):
            self._warned_silent = True
            sink = recorded_sink(self._path)
            return (
                f"Warning: the monitor has recorded nothing for {seconds / 60:.0f} min. If "
                f"the other side is talking, the meeting app plays to an output tapeback does "
                f"not record{f' (it records {sink})' if sink else ''}."
            )
        return None

    def _elsewhere(self, seconds: float) -> str | None:
        sink = recorded_sink(self._path)
        if sink is None:
            return None
        streams = streams_playing_elsewhere(sink)
        if not streams:
            return None
        lines = [
            f"Warning: the monitor has recorded silence for {seconds:.0f} s, and the other "
            f"side is not being recorded: tapeback records {sink}, but"
        ]
        lines += [
            f"  {stream.app} plays to {stream.sink}. To record it, move it: "
            f"pactl move-sink-input {stream.index} {sink}"
            for stream in streams
        ]
        lines.append("  (or pick that output in the app's audio settings)")
        return "\n".join(lines)

    def _new_bytes(self) -> bytes:
        try:
            with self._path.open("rb") as f:
                f.seek(self._read)
                chunk = f.read()
        except OSError:
            return b""
        self._read += len(chunk)
        return chunk
