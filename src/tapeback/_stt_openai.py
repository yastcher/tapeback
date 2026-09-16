"""OpenAI remote STT backend (`TAPEBACK_STT_BACKEND=openai`).

One implementation of the remote STT seam in ``remote_stt``. Uploads meeting
audio to OpenAI — see README privacy notes. Requires `tapeback[stt]` (or
`tapeback[llm]`, which also installs the openai SDK) and `TAPEBACK_STT_API_KEY`
(or `OPENAI_API_KEY`).
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tapeback import _resume, const
from tapeback._stt_caps import openai_capabilities_for
from tapeback._stt_media import (
    _MAX_UPLOAD,
    _encode_mp3,
    _max_chunk_seconds,
    _slice_wav,
    _UploadJob,
    _wav_duration,
)
from tapeback._stt_openai_fmt import (
    _info_from_response,
    _keywords_from_hotwords,
    _segments_from_diarized,
    _segments_from_text_response,
    _segments_from_verbose,
)
from tapeback._timing import stage_timer
from tapeback.models import Segment
from tapeback.settings import Settings


def _threadsafe_status(
    on_status: Callable[[str], None], lock: threading.Lock
) -> Callable[[str], None]:
    """Wrap a status sink so concurrent chunk uploads do not interleave prints."""

    def _report(message: str) -> None:
        with lock:
            on_status(message)

    return _report


def _noop_status(_message: str) -> None:
    """Default status sink when callers pass none."""


def _resolve_api_key(settings: Settings) -> str:
    """Resolve remote STT key: TAPEBACK_STT_API_KEY, else OPENAI_API_KEY, else LLM key."""
    key = settings.stt_api_key.get_secret_value()
    if key:
        return key
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key
    if settings.llm_provider == "openai":
        key = settings.llm_api_key.get_secret_value()
        if key:
            return key
    raise RuntimeError(
        "OpenAI transcription requires TAPEBACK_STT_API_KEY "
        "(or OPENAI_API_KEY, or TAPEBACK_LLM_API_KEY with TAPEBACK_LLM_PROVIDER=openai)."
    )


class OpenAITranscriber:
    """Transcriber that sends audio to OpenAI instead of running Whisper locally."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_key = _resolve_api_key(settings)
        self._model = settings.stt_model
        self._caps = openai_capabilities_for(self._model)

    def describe(self) -> str:
        if self._caps.remote_diarize:
            detail = "remote diarize"
        elif self._caps.word_timestamps:
            detail = "timestamps; local diarize ok"
        else:
            detail = "text; local diarize skipped"
        return f"OpenAI STT: {self._model} ({detail})"

    def transcribe(
        self,
        audio_path: Path,
        *,
        stage: str = "transcribe",
        on_status: Callable[[str], None] = _noop_status,
        language_override: str | None = None,
    ) -> tuple[list[Segment], dict[str, str | float | bool]]:
        """Transcribe one audio file via OpenAI Audio Transcriptions."""
        key = self._resume_key(audio_path, stage)
        if key is not None:
            cached = _resume.load(key, _resume.resume_dir(self._settings))
            if cached is not None:
                on_status(f"Reusing the '{stage}' result from an earlier run.")
                return cached

        configured = self._settings.language
        language = configured if configured != "auto" else language_override

        on_status(f"OpenAI STT ({self._model}): preparing upload for {stage}...")
        try:
            segments, info = self._transcribe_file(
                audio_path, language=language, on_status=on_status, stage=stage
            )
        except KeyboardInterrupt:
            # Nothing partial from a remote call — treat as empty partial like local.
            on_status("OpenAI transcription interrupted.")
            info = {
                "language": language or "unknown",
                "language_probability": 0.0,
                "duration": 0.0,
                "partial": True,
            }
            return [], info

        if not segments:
            print("Warning: No speech detected in audio", file=sys.stderr)

        self._store_resume(key, segments, info)
        return segments, info

    def transcribe_stereo(
        self,
        mic_16k: Path,
        monitor_16k: Path,
        *,
        on_status: Callable[[str], None] = _noop_status,
    ) -> tuple[list[Segment], list[Segment], dict[str, str | float | bool]]:
        """Transcribe both channels separately (same contract as local Transcriber)."""
        with stage_timer("transcribe monitor", on_status):
            monitor_segments, monitor_info = self.transcribe(
                monitor_16k, stage="transcribe monitor", on_status=on_status
            )

        detected = monitor_info.get("language")
        mic_language = str(detected) if detected else None

        mic_segments: list[Segment] = []
        mic_info: dict[str, str | float | bool] = {}
        if monitor_info.get("partial"):
            on_status("Skipping the mic channel — transcription was interrupted.")
        else:
            with stage_timer("transcribe mic", on_status):
                mic_segments, mic_info = self.transcribe(
                    mic_16k,
                    stage="transcribe mic",
                    on_status=on_status,
                    language_override=mic_language,
                )

        mic_segments = [
            Segment(
                start=s.start,
                end=s.end,
                text=s.text,
                words=s.words,
                speaker=const.SPEAKER_YOU,
            )
            for s in mic_segments
        ]

        mic_speech = sum(s.end - s.start for s in mic_segments)
        monitor_speech = sum(s.end - s.start for s in monitor_segments)
        info = mic_info if mic_speech >= monitor_speech else monitor_info
        info = dict(info)
        info["partial"] = bool(monitor_info.get("partial") or mic_info.get("partial"))
        return mic_segments, monitor_segments, info

    def _resume_key(self, audio_path: Path, stage: str) -> _resume.ResumeKey | None:
        if not self._settings.resume_cache:
            return None
        return _resume.resume_key(audio_path, self._settings, stage)

    def _chunk_limit_seconds(self) -> float:
        """Longest slice that fits under both size and model duration caps."""
        size_limit = _max_chunk_seconds()
        duration_cap = self._caps.max_upload_seconds
        if duration_cap is None:
            return size_limit
        return min(size_limit, duration_cap * const.STT_REMOTE_UPLOAD_MARGIN)

    def _store_resume(
        self,
        key: _resume.ResumeKey | None,
        segments: list[Segment],
        info: dict[str, Any],
    ) -> None:
        if key is None or info.get("partial") or not segments:
            return
        _resume.store(key, _resume.resume_dir(self._settings), segments, info)

    def _transcribe_file(
        self,
        audio_path: Path,
        *,
        language: str | None,
        on_status: Callable[[str], None],
        stage: str,
    ) -> tuple[list[Segment], dict[str, str | float | bool]]:
        duration = _wav_duration(audio_path)
        chunk_limit = self._chunk_limit_seconds()

        with tempfile.TemporaryDirectory(prefix="tapeback-openai-stt-") as tmp:
            tmp_dir = Path(tmp)
            if duration <= chunk_limit:
                return self._transcribe_one_upload(
                    _UploadJob(
                        wav_path=audio_path,
                        mp3_path=tmp_dir / "upload.mp3",
                        time_offset=0.0,
                        duration=duration,
                        language=language,
                        stage=stage,
                        chunk_label=None,
                    ),
                    on_status,
                )

            # Long meeting: slice under the upload cap. Parallel uploads are fine for
            # whisper-1 / gpt-transcribe; remote diarize stays sequential so speaker
            # labels stay consistent across slices.
            jobs = self._build_chunk_jobs(
                audio_path,
                tmp_dir,
                duration=duration,
                chunk_limit=chunk_limit,
                language=language,
                stage=stage,
            )
            if self._caps.remote_diarize:
                return self._transcribe_chunks_sequential(
                    jobs, duration=duration, on_status=on_status
                )
            return self._transcribe_chunks_parallel(jobs, duration=duration, on_status=on_status)

    def _build_chunk_jobs(
        self,
        audio_path: Path,
        tmp_dir: Path,
        *,
        duration: float,
        chunk_limit: float,
        language: str | None,
        stage: str,
    ) -> list[_UploadJob]:
        n_chunks = int(duration // chunk_limit) + (1 if duration % chunk_limit else 0)
        jobs: list[_UploadJob] = []
        for i in range(n_chunks):
            start = i * chunk_limit
            piece_dur = min(chunk_limit, duration - start)
            if piece_dur <= 0:
                break
            slice_path = tmp_dir / f"slice_{i}.wav"
            mp3_path = tmp_dir / f"slice_{i}.mp3"
            _slice_wav(audio_path, slice_path, start, piece_dur)
            jobs.append(
                _UploadJob(
                    wav_path=slice_path,
                    mp3_path=mp3_path,
                    time_offset=start,
                    duration=piece_dur,
                    language=language,
                    stage=stage,
                    chunk_label=f"{i + 1}/{n_chunks}",
                )
            )
        return jobs

    def _transcribe_chunks_sequential(
        self,
        jobs: list[_UploadJob],
        *,
        duration: float,
        on_status: Callable[[str], None],
    ) -> tuple[list[Segment], dict[str, str | float | bool]]:
        """Upload slices one-by-one, sharing speaker-label order across chunks."""
        if not jobs:
            return [], {
                "language": "unknown",
                "language_probability": 0.0,
                "duration": duration,
                "partial": False,
            }
        on_status(
            f"OpenAI STT ({self._model}): uploading {len(jobs)} chunk(s) sequentially "
            "(remote diarize)..."
        )
        speaker_order: list[str] = []
        segments: list[Segment] = []
        info: dict[str, str | float | bool] = {
            "language": jobs[0].language or "unknown",
            "language_probability": 1.0,
            "duration": duration,
            "partial": False,
        }
        language = jobs[0].language
        for job in jobs:
            if language is not None and job.language is None:
                job = job._replace(language=language)
            piece_segs, piece_info = self._transcribe_one_upload(
                job, on_status, speaker_order=speaker_order
            )
            segments.extend(piece_segs)
            info = piece_info
            info["duration"] = duration
            if language is None and piece_info.get("language"):
                language = str(piece_info["language"])
        return segments, info

    def _transcribe_chunks_parallel(
        self,
        jobs: list[_UploadJob],
        *,
        duration: float,
        on_status: Callable[[str], None],
    ) -> tuple[list[Segment], dict[str, str | float | bool]]:
        if not jobs:
            return [], {
                "language": "unknown",
                "language_probability": 0.0,
                "duration": duration,
                "partial": False,
            }

        # Auto language: lock detection from the first chunk, then parallelize the rest
        # with that language so chunks do not disagree.
        leading: list[tuple[float, list[Segment], dict[str, str | float | bool]]] = []
        remaining = jobs
        if jobs[0].language is None and len(jobs) > 1:
            on_status(f"OpenAI STT ({self._model}): detecting language on chunk 1/{len(jobs)}...")
            first_segs, first_info = self._transcribe_one_upload(jobs[0], on_status)
            detected = first_info.get("language")
            language = str(detected) if detected else None
            leading.append((jobs[0].time_offset, first_segs, first_info))
            remaining = [job._replace(language=language) for job in jobs[1:]]

        if not remaining:
            info = dict(leading[0][2])
            info["duration"] = duration
            return list(leading[0][1]), info

        workers = min(self._settings.stt_concurrency, len(remaining))
        on_status(
            f"OpenAI STT ({self._model}): uploading {len(remaining)} chunk(s) "
            f"(up to {workers} in parallel)..."
        )
        status_lock = threading.Lock()
        report = _threadsafe_status(on_status, status_lock)
        results: list[tuple[float, list[Segment], dict[str, str | float | bool]]] = list(leading)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._transcribe_one_upload, job, report): job for job in remaining
            }
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

    def _transcribe_one_upload(
        self,
        job: _UploadJob,
        on_status: Callable[[str], None],
        *,
        speaker_order: list[str] | None = None,
    ) -> tuple[list[Segment], dict[str, str | float | bool]]:
        _encode_mp3(job.wav_path, job.mp3_path)
        size = job.mp3_path.stat().st_size
        if size > _MAX_UPLOAD:
            raise RuntimeError(
                f"Encoded upload is {size} bytes, over the OpenAI limit of {_MAX_UPLOAD}. "
                "Lower TAPEBACK settings that lengthen audio, or split the recording."
            )
        label = f" chunk {job.chunk_label}" if job.chunk_label else ""
        on_status(
            f"OpenAI STT ({self._model}): uploading {job.stage}{label} ({size // 1024} KiB)..."
        )
        response = self._call_api(job.mp3_path, language=job.language, duration=job.duration)
        on_status(f"OpenAI STT ({self._model}): received {job.stage}{label}")

        if self._caps.remote_diarize:
            segments = _segments_from_diarized(
                response, job.time_offset, speaker_order=speaker_order
            )
        elif self._caps.word_timestamps:
            segments = _segments_from_verbose(response, job.time_offset)
        else:
            segments = _segments_from_text_response(response, job.time_offset, job.duration)
        info = _info_from_response(response, job.duration, language_fallback=job.language)
        return segments, info

    def _call_api(self, mp3_path: Path, *, language: str | None, duration: float) -> Any:
        try:
            import openai
        except ImportError:
            raise RuntimeError(
                'openai package not installed. Install with: uv pip install "tapeback[stt]"'
            ) from None

        client = openai.OpenAI(api_key=self._api_key)
        kwargs: dict[str, Any] = {"model": self._model}
        extra_body: dict[str, Any] = {}

        if self._caps.remote_diarize:
            kwargs["response_format"] = "diarized_json"
            # Required for recordings longer than 30s per OpenAI's diarize guide.
            if duration > const.OPENAI_DIARIZE_CHUNKING_SECONDS:
                kwargs["chunking_strategy"] = "auto"
        elif self._caps.word_timestamps:
            if language:
                kwargs["language"] = language
            if self._settings.temperature:
                kwargs["temperature"] = float(self._settings.temperature[0])
            if self._settings.hotwords:
                kwargs["prompt"] = self._settings.hotwords
            kwargs["response_format"] = "verbose_json"
            kwargs["timestamp_granularities"] = ["word", "segment"]
        else:
            # gpt-transcribe family: languages[] replaces language; keywords for terms.
            if language:
                extra_body["languages"] = [language]
            if self._settings.hotwords:
                keywords = _keywords_from_hotwords(self._settings.hotwords)
                if keywords:
                    extra_body["keywords"] = keywords
                kwargs["prompt"] = self._settings.hotwords

        create_kwargs = dict(kwargs)
        if extra_body:
            create_kwargs["extra_body"] = extra_body

        with mp3_path.open("rb") as audio_file:
            return client.audio.transcriptions.create(file=audio_file, **create_kwargs)
