"""Out-of-process transcription worker.

Run as ``python -m tapeback._worker``. Reads one JSON job from stdin, writes
newline-delimited JSON events to stdout, exits.

The point is memory, not parallelism. A CUDA out-of-memory during model load leaks
its allocation on ctranslate2's C++ side for the life of the process — measured, free
VRAM went 3674 MiB to 95 MiB and never came back, which then starved diarization too.
Nothing reachable from Python can release it. A separate process can simply die, and
the kernel reclaims the device memory with it.

The protocol is deliberately plain text so the parent can keep whatever arrived before
a crash: every segment is a complete line, so a worker killed mid-run still leaves the
parent holding everything it had decoded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Settings the worker needs. An explicit list, not the whole model: credentials have no
# business crossing into a process that only transcribes audio.
WORKER_SETTINGS = (
    "stt_backend",
    "stt_model",
    "language",
    "device",
    "compute_type",
    "beam_size",
    "temperature",
    "batch_size",
    "hotwords",
    "vad_filter",
    "chunk_length",
    "condition_on_previous_text",
    "no_speech_threshold",
    "language_detection_segments",
    "multilingual",
    "hallucination_silence_threshold",
    "min_free_vram_mib",
    "thermal_clamp_check",
    "thermal_clamp_wait",
    "thermal_clamp_cpu_fallback",
)

EVENT_STATUS = "status"
EVENT_SEGMENT = "segment"
EVENT_INFO = "info"
EVENT_ERROR = "error"


def emit(stream: Any, event: str, **payload: Any) -> None:
    """Write one event as a self-contained line and flush it.

    Flushing per event is the whole contract: a line the parent has read is a line it
    keeps even if this process is killed a moment later.
    """
    stream.write(json.dumps({"type": event, **payload}, ensure_ascii=False) + "\n")
    stream.flush()


def _segment_payload(segment: Any) -> dict[str, Any]:
    words = None
    if segment.words:
        words = [
            {"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
            for w in segment.words
        ]
    return {
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "speaker": segment.speaker,
        "words": words,
    }


def run_job(job: dict[str, Any], stdout: Any) -> int:
    """Transcribe one file, streaming events to ``stdout``. Returns an exit code."""
    # Imported here, not at module scope: the parent spawns this module for its side
    # effect of isolating a ~10s ML import that must not run when the job is invalid.
    from tapeback.settings import Settings  # noqa: PLC0415
    from tapeback.transcriber import Transcriber  # noqa: PLC0415

    settings = Settings(**job["settings"])
    transcriber = Transcriber(settings)
    emit(stdout, EVENT_STATUS, message=transcriber.describe())

    segments, info = transcriber.transcribe(
        Path(job["audio_path"]),
        stage=job.get("stage", "transcribe"),
        on_status=lambda message: emit(stdout, EVENT_STATUS, message=message),
        language_override=job.get("language_override"),
    )
    for segment in segments:
        emit(stdout, EVENT_SEGMENT, data=_segment_payload(segment))
    emit(stdout, EVENT_INFO, data=info)
    return 0


def main() -> int:
    raw = sys.stdin.read()
    try:
        job = json.loads(raw)
    except ValueError as exc:
        emit(sys.stdout, EVENT_ERROR, message=f"invalid job: {exc}")
        return 2
    try:
        return run_job(job, sys.stdout)
    except KeyboardInterrupt:
        # The parent already has every segment emitted so far; say so and leave.
        emit(sys.stdout, EVENT_STATUS, message="Worker interrupted")
        return 130
    except BaseException as exc:  # the process boundary: report, never traceback
        emit(sys.stdout, EVENT_ERROR, message=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
