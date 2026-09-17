"""Chunk planning, resume, and sequential/parallel upload for OpenAI STT."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tapeback import _resume
from tapeback._stt_media import _ChunkPlan, _slice_wav, _UploadJob
from tapeback.models import Segment

if TYPE_CHECKING:
    from tapeback._stt_openai import OpenAITranscriber

# Internal info key: raw API speaker ids in first-seen order across diarize chunks.
SPEAKER_ORDER_KEY = "remote_speaker_order"


def threadsafe_status(
    on_status: Callable[[str], None], lock: threading.Lock
) -> Callable[[str], None]:
    """Wrap a status sink so concurrent chunk uploads do not interleave prints."""

    def _report(message: str) -> None:
        with lock:
            on_status(message)

    return _report


def build_chunk_jobs(audio_path: Path, tmp_dir: Path, plan: _ChunkPlan) -> list[_UploadJob]:
    n_chunks = int(plan.duration // plan.chunk_limit) + (
        1 if plan.duration % plan.chunk_limit else 0
    )
    jobs: list[_UploadJob] = []
    for i in range(n_chunks):
        start = i * plan.chunk_limit
        piece_dur = min(plan.chunk_limit, plan.duration - start)
        if piece_dur <= 0:
            break
        slice_path = tmp_dir / f"slice_{i}.wav"
        mp3_path = tmp_dir / f"slice_{i}.mp3"
        _slice_wav(audio_path, slice_path, start, piece_dur, timeout=plan.ffmpeg_timeout)
        jobs.append(
            _UploadJob(
                wav_path=slice_path,
                mp3_path=mp3_path,
                time_offset=start,
                duration=piece_dur,
                language=plan.language,
                stage=plan.stage,
                chunk_label=f"{i + 1}/{n_chunks}",
                chunk_index=i,
                source_audio=audio_path,
            )
        )
    return jobs


def load_chunk(
    tx: OpenAITranscriber,
    job: _UploadJob,
    on_status: Callable[[str], None],
) -> tuple[list[Segment], dict[str, Any]] | None:
    key = tx._chunk_key(job)
    if key is None:
        return None
    cached = _resume.load(key, _resume.resume_dir(tx._settings))
    if cached is None:
        return None
    label = f" chunk {job.chunk_label}" if job.chunk_label else ""
    on_status(f"OpenAI STT ({tx._model}): reusing {job.stage}{label} from an earlier run.")
    return cached


def store_chunk(
    tx: OpenAITranscriber,
    job: _UploadJob,
    segments: list[Segment],
    info: dict[str, Any],
) -> None:
    key = tx._chunk_key(job)
    if key is None or info.get("partial") or not segments:
        return
    _resume.store(key, _resume.resume_dir(tx._settings), segments, info)


def apply_speaker_order(speaker_order: list[str], info: dict[str, Any]) -> None:
    raw = info.get(SPEAKER_ORDER_KEY)
    if not isinstance(raw, list):
        return
    speaker_order.clear()
    speaker_order.extend(str(item) for item in raw)


def upload_or_resume(
    tx: OpenAITranscriber,
    job: _UploadJob,
    on_status: Callable[[str], None],
) -> tuple[list[Segment], dict[str, Any]]:
    cached = load_chunk(tx, job, on_status)
    if cached is not None:
        return cached
    piece_segs, piece_info = tx._transcribe_one_upload(job, on_status)
    store_chunk(tx, job, piece_segs, piece_info)
    return piece_segs, piece_info


def transcribe_chunks_sequential(
    tx: OpenAITranscriber,
    jobs: list[_UploadJob],
    *,
    duration: float,
    on_status: Callable[[str], None],
) -> tuple[list[Segment], dict[str, Any]]:
    """Upload slices one-by-one, sharing speaker-label order across chunks."""
    if not jobs:
        return [], {
            "language": "unknown",
            "language_probability": 0.0,
            "duration": duration,
            "partial": False,
        }
    on_status(
        f"OpenAI STT ({tx._model}): uploading {len(jobs)} chunk(s) sequentially (remote diarize)..."
    )
    speaker_order: list[str] = []
    segments: list[Segment] = []
    info: dict[str, Any] = {
        "language": jobs[0].language or "unknown",
        "language_probability": 1.0,
        "duration": duration,
        "partial": False,
    }
    language = jobs[0].language
    for job in jobs:
        if language is not None and job.language is None:
            job = job._replace(language=language)
        cached = load_chunk(tx, job, on_status)
        if cached is not None:
            piece_segs, piece_info = cached
            apply_speaker_order(speaker_order, piece_info)
        else:
            piece_segs, piece_info = tx._transcribe_one_upload(
                job, on_status, speaker_order=speaker_order
            )
            piece_info = dict(piece_info)
            piece_info[SPEAKER_ORDER_KEY] = list(speaker_order)
            store_chunk(tx, job, piece_segs, piece_info)
        segments.extend(piece_segs)
        info = piece_info
        info["duration"] = duration
        if language is None and piece_info.get("language"):
            language = str(piece_info["language"])
    return segments, info


def transcribe_chunks_parallel(
    tx: OpenAITranscriber,
    jobs: list[_UploadJob],
    *,
    duration: float,
    on_status: Callable[[str], None],
) -> tuple[list[Segment], dict[str, Any]]:
    if not jobs:
        return [], {
            "language": "unknown",
            "language_probability": 0.0,
            "duration": duration,
            "partial": False,
        }

    leading: list[tuple[float, list[Segment], dict[str, Any]]] = []
    remaining = jobs
    if jobs[0].language is None and len(jobs) > 1:
        on_status(f"OpenAI STT ({tx._model}): detecting language on chunk 1/{len(jobs)}...")
        first_segs, first_info = upload_or_resume(tx, jobs[0], on_status)
        detected = first_info.get("language")
        language = str(detected) if detected else None
        leading.append((jobs[0].time_offset, first_segs, first_info))
        remaining = [job._replace(language=language) for job in jobs[1:]]

    if not remaining:
        info = dict(leading[0][2])
        info["duration"] = duration
        return list(leading[0][1]), info

    workers = min(tx._settings.stt_concurrency, len(remaining))
    on_status(
        f"OpenAI STT ({tx._model}): uploading {len(remaining)} chunk(s) "
        f"(up to {workers} in parallel)..."
    )
    status_lock = threading.Lock()
    report = threadsafe_status(on_status, status_lock)
    results: list[tuple[float, list[Segment], dict[str, Any]]] = list(leading)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(upload_or_resume, tx, job, report): job for job in remaining}
        for fut in as_completed(futures):
            job = futures[fut]
            piece_segs, piece_info = fut.result()
            results.append((job.time_offset, piece_segs, piece_info))

    results.sort(key=lambda item: item[0])
    segments: list[Segment] = []
    for _offset, piece_segs, _piece_info in results:
        segments.extend(piece_segs)
    info = dict(results[-1][2])
    info["duration"] = duration
    return segments, info
