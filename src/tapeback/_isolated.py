"""Parent side of out-of-process transcription.

Spawns `tapeback._worker`, streams its events back, and converts them into the same
`(segments, info)` a direct `Transcriber.transcribe` call returns — so callers do not
have to know which side of a process boundary the work happened on.

Why this exists is in `_worker`'s docstring: a CUDA out-of-memory permanently leaks
VRAM inside the process that hit it, and the only reliable way to give that memory
back is for the process to end.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tapeback._worker import (
    EVENT_ERROR,
    EVENT_INFO,
    EVENT_SEGMENT,
    EVENT_STATUS,
    WORKER_SETTINGS,
)
from tapeback.models import Segment, Word
from tapeback.settings import Settings

# How long to wait for a worker to exit after we ask it to stop, before killing it.
WORKER_SHUTDOWN_TIMEOUT_SEC = 10.0


class WorkerFailed(RuntimeError):
    """The worker exited without delivering a result."""


def job_settings(settings: Settings) -> dict[str, Any]:
    """The subset of settings the worker needs, as JSON-safe values.

    An explicit allow-list rather than a full dump: `Settings` carries `hf_token` and
    `llm_api_key`, and a process that only transcribes audio has no use for either.
    """
    payload: dict[str, Any] = {}
    for name in WORKER_SETTINGS:
        value = getattr(settings, name)
        payload[name] = list(value) if isinstance(value, tuple) else value
    # Load-bearing: the worker builds a Transcriber of its own, and inheriting this
    # would have it spawn another worker, and that one another, without end.
    payload["isolate_transcription"] = False
    # The parent stores the result; a child writing it too would be a second, racier
    # writer of the same file for no benefit.
    payload["resume_cache"] = False
    return payload


def _to_segment(data: dict[str, Any]) -> Segment:
    words = None
    if data.get("words"):
        words = [
            Word(start=w["start"], end=w["end"], word=w["word"], probability=w["probability"])
            for w in data["words"]
        ]
    return Segment(
        start=data["start"],
        end=data["end"],
        text=data["text"],
        words=words,
        speaker=data.get("speaker"),
    )


def transcribe_isolated(
    audio_path: Path,
    settings: Settings,
    *,
    stage: str = "transcribe",
    on_status: Callable[[str], None],
    language_override: str | None = None,
) -> tuple[list[Segment], dict[str, Any]]:
    """Transcribe in a child process, returning what it delivered.

    Segments that arrived before a crash are kept: each is a complete line, so a
    worker killed by the OOM killer — or by the user — still leaves usable work
    behind. That is reported as a partial result rather than raised as an error,
    matching what an in-process interrupt does.
    """
    job = json.dumps(
        {
            "settings": job_settings(settings),
            "audio_path": str(audio_path),
            "stage": stage,
            "language_override": language_override,
        }
    )

    segments: list[Segment] = []
    info: dict[str, Any] = {}
    error: str | None = None

    process = subprocess.Popen(
        [sys.executable, "-m", "tapeback._worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stdin, stdout = process.stdin, process.stdout
    if stdin is None or stdout is None:  # pragma: no cover — both were opened as PIPEs
        _stop(process)
        raise WorkerFailed("transcription worker pipes were not created")

    try:
        stdin.write(job)
        stdin.close()

        for line in stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                # Anything the worker's dependencies printed to stdout. Not ours to
                # interpret, but not worth failing over either.
                continue
            kind = event.get("type")
            if kind == EVENT_STATUS:
                on_status(event["message"])
            elif kind == EVENT_SEGMENT:
                segments.append(_to_segment(event["data"]))
            elif kind == EVENT_INFO:
                info = event["data"]
            elif kind == EVENT_ERROR:
                error = event["message"]
    except KeyboardInterrupt:
        _stop(process)
        on_status(f"Interrupted — keeping the {len(segments)} segments the worker sent.")
        info = dict(info)
        info["partial"] = True
        return segments, info
    finally:
        _stop(process)
        stdout.close()

    if info:
        return segments, info

    # No info event means the worker never finished. Segments already received are
    # still worth keeping — this is the out-of-memory path the isolation exists for.
    if segments:
        on_status(
            f"Worker stopped early ({error or f'exit code {process.returncode}'}) — "
            f"keeping the {len(segments)} segments it produced."
        )
        return segments, {"partial": True, "language": "", "duration": 0.0}

    raise WorkerFailed(error or f"transcription worker exited with code {process.returncode}")


def _stop(process: subprocess.Popen[str]) -> None:
    """Ask the worker to exit, then insist. Returning its VRAM is the whole point."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=WORKER_SHUTDOWN_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
