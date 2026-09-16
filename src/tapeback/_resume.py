"""Reuse a channel that was already transcribed, so a re-run does not start from zero.

An interrupted stereo run used to redo everything. The monitor channel of a 31-minute
recording takes minutes even after the speed work, and repeating it because the *other*
channel was interrupted is pure waste.

**Granularity is a whole channel, deliberately.** Resuming part-way through one would
mean handing faster-whisper the remaining span via `clip_timestamps`, and its own
documentation says "vad_filter will be ignored if clip_timestamps is used". VAD is load
bearing here — it is half of why hallucinations on silence went away — so trading it for
a faster resume is a bad deal. That leaves the honest limitation: an interrupt during the
first channel has nothing to reuse, while one during the second saves the first.

A cached entry is only valid for the exact audio and the exact settings that produced it,
so the key covers both. Anything that changes what Whisper outputs invalidates it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tapeback.models import Segment, Word
from tapeback.settings import Settings

# Settings that change what Whisper produces. A cached channel is only reusable when
# every one of these matches, so adding a knob that affects output means adding it here.
OUTPUT_AFFECTING_SETTINGS = (
    "stt_backend",
    "stt_model",
    "device",
    "compute_type",
    "language",
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
)

# Keep the directory bounded; entries are cheap but not free.
MAX_RESUME_ENTRIES = 50


def default_resume_dir() -> Path:
    """XDG data directory for resumable channel results."""
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg_data_home) if xdg_data_home else Path.home() / ".local" / "share"
    return base / "tapeback" / "resume"


@dataclass(frozen=True)
class ResumeKey:
    """Identifies one (audio, settings, channel) combination."""

    digest: str

    @property
    def filename(self) -> str:
        return f"{self.digest}.json"


def resume_key(audio_path: Path, settings: Settings, stage: str) -> ResumeKey | None:
    """Fingerprint the inputs. None when the audio cannot be described.

    Identity is path + size + mtime rather than a content hash: hashing a 400 MB WAV
    on every run would cost more than it saves, and these files are written once.
    """
    try:
        stat = audio_path.stat()
    except OSError:
        return None
    parts = [str(audio_path.resolve()), str(stat.st_size), str(stat.st_mtime_ns), stage]
    parts += [f"{name}={getattr(settings, name)!r}" for name in OUTPUT_AFFECTING_SETTINGS]
    return ResumeKey(hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:32])


def _to_payload(segments: list[Segment], info: dict[str, Any]) -> dict[str, Any]:
    return {
        "info": info,
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "speaker": s.speaker,
                "words": None
                if s.words is None
                else [
                    {"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
                    for w in s.words
                ],
            }
            for s in segments
        ],
    }


def _from_payload(payload: dict[str, Any]) -> tuple[list[Segment], dict[str, Any]]:
    segments = [
        Segment(
            start=s["start"],
            end=s["end"],
            text=s["text"],
            speaker=s.get("speaker"),
            words=None
            if s.get("words") is None
            else [
                Word(start=w["start"], end=w["end"], word=w["word"], probability=w["probability"])
                for w in s["words"]
            ],
        )
        for s in payload["segments"]
    ]
    return segments, payload["info"]


def load(key: ResumeKey, directory: Path) -> tuple[list[Segment], dict[str, Any]] | None:
    """Return a previously stored channel, or None. Never raises on bad cache data."""
    path = directory / key.filename
    try:
        payload = json.loads(path.read_text())
        return _from_payload(payload)
    except (OSError, ValueError, KeyError, TypeError):
        # A corrupt or half-written entry is not worth a failed run; redo the work.
        return None


def store(
    key: ResumeKey,
    directory: Path,
    segments: list[Segment],
    info: dict[str, Any],
) -> Path | None:
    """Persist a completed channel. Returns the path, or None if it could not be written.

    Failing to write a cache entry must never fail the run that produced it.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / key.filename
        path.write_text(json.dumps(_to_payload(segments, info), ensure_ascii=False))
        _prune(directory)
    except OSError:
        return None
    return path


def _prune(directory: Path, keep: int = MAX_RESUME_ENTRIES) -> None:
    """Drop the least recently modified entries so the directory stays bounded."""
    entries = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if len(entries) <= keep:
        return
    for stale in entries[: len(entries) - keep]:
        stale.unlink(missing_ok=True)


def resume_dir(settings: Settings) -> Path:
    return settings.resume_cache_dir or default_resume_dir()
