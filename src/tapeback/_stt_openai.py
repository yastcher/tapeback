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
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tapeback import _resume, const
from tapeback._stt_caps import openai_capabilities_for
from tapeback._stt_media import (
    _MAX_UPLOAD,
    _ChunkPlan,
    _encode_mp3,
    _max_chunk_seconds,
    _UploadJob,
    _wav_duration,
)
from tapeback._stt_openai_chunks import (
    SPEAKER_ORDER_KEY,
    build_chunk_jobs,
    transcribe_chunks_parallel,
    transcribe_chunks_sequential,
)
from tapeback._stt_openai_fmt import (
    _info_from_response,
    _keywords_from_hotwords,
    _segments_from_diarized,
    _segments_from_text_response,
    _segments_from_verbose,
)
from tapeback._stt_retry import RetryPolicy, call_with_retry
from tapeback._timing import stage_timer
from tapeback.models import Segment
from tapeback.settings import Settings


def _noop_status(_message: str) -> None:
    """Default status sink when callers pass none."""


def _public_info(info: dict[str, Any]) -> dict[str, str | float | bool]:
    """Drop internal resume fields before returning channel-level info."""
    out = dict(info)
    out.pop(SPEAKER_ORDER_KEY, None)
    return out  # type: ignore[return-value]


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
        self._client: Any | None = None

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
                return cached[0], _public_info(cached[1])

        configured = self._settings.language
        language = configured if configured != "auto" else language_override

        on_status(f"OpenAI STT ({self._model}): preparing upload for {stage}...")
        try:
            segments, info = self._transcribe_file(
                audio_path, language=language, on_status=on_status, stage=stage
            )
        except KeyboardInterrupt:
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

        info = _public_info(info)
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

    def _chunk_key(self, job: _UploadJob) -> _resume.ResumeKey | None:
        if not self._settings.resume_cache:
            return None
        return _resume.chunk_resume_key(
            job.source_audio,
            self._settings,
            job.stage,
            chunk_index=job.chunk_index,
            time_offset=job.time_offset,
            piece_duration=job.duration,
        )

    def _soft_target_seconds(self) -> float | None:
        if self._caps.remote_diarize:
            return const.OPENAI_DIARIZE_TARGET_UPLOAD_SECONDS
        if self._caps.max_upload_seconds is not None:
            return const.OPENAI_GPT_TRANSCRIBE_TARGET_UPLOAD_SECONDS
        return None

    def _chunk_limit_seconds(self) -> float:
        """Longest slice under size, hard API duration, and soft target caps."""
        limits = [_max_chunk_seconds()]
        duration_cap = self._caps.max_upload_seconds
        if duration_cap is not None:
            limits.append(duration_cap * const.STT_REMOTE_UPLOAD_MARGIN)
        soft = self._soft_target_seconds()
        if soft is not None:
            limits.append(soft)
        return min(limits)

    def _store_resume(
        self,
        key: _resume.ResumeKey | None,
        segments: list[Segment],
        info: dict[str, Any],
    ) -> None:
        if key is None or info.get("partial") or not segments:
            return
        _resume.store(key, _resume.resume_dir(self._settings), segments, info)

    def _openai_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import httpx
            import openai
        except ImportError:
            raise RuntimeError(
                'openai package not installed. Install with: uv pip install "tapeback[stt]"'
            ) from None
        timeout = float(self._settings.stt_timeout)
        self._client = openai.OpenAI(
            api_key=self._api_key,
            timeout=httpx.Timeout(
                connect=const.STT_REMOTE_CONNECT_TIMEOUT,
                read=timeout,
                write=timeout,
                pool=timeout,
            ),
            max_retries=0,
        )
        return self._client

    def _transcribe_file(
        self,
        audio_path: Path,
        *,
        language: str | None,
        on_status: Callable[[str], None],
        stage: str,
    ) -> tuple[list[Segment], dict[str, Any]]:
        duration = _wav_duration(audio_path)
        chunk_limit = self._chunk_limit_seconds()
        ffmpeg_timeout = float(self._settings.stt_ffmpeg_timeout)

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
                        chunk_index=0,
                        source_audio=audio_path,
                    ),
                    on_status,
                )

            jobs = build_chunk_jobs(
                audio_path,
                tmp_dir,
                _ChunkPlan(
                    duration=duration,
                    chunk_limit=chunk_limit,
                    language=language,
                    stage=stage,
                    ffmpeg_timeout=ffmpeg_timeout,
                ),
            )
            if self._caps.remote_diarize:
                return transcribe_chunks_sequential(
                    self, jobs, duration=duration, on_status=on_status
                )
            return transcribe_chunks_parallel(self, jobs, duration=duration, on_status=on_status)

    def _transcribe_one_upload(
        self,
        job: _UploadJob,
        on_status: Callable[[str], None],
        *,
        speaker_order: list[str] | None = None,
    ) -> tuple[list[Segment], dict[str, Any]]:
        ffmpeg_timeout = float(self._settings.stt_ffmpeg_timeout)
        _encode_mp3(job.wav_path, job.mp3_path, timeout=ffmpeg_timeout)
        size = job.mp3_path.stat().st_size
        if size > _MAX_UPLOAD:
            raise RuntimeError(
                f"Encoded upload is {size} bytes, over the OpenAI limit of {_MAX_UPLOAD}. "
                "Lower TAPEBACK settings that lengthen audio, or split the recording."
            )
        label = f" chunk {job.chunk_label}" if job.chunk_label else ""
        status_label = f"{job.stage}{label}"
        on_status(f"OpenAI STT ({self._model}): uploading {status_label} ({size // 1024} KiB)...")
        response = call_with_retry(
            lambda: self._call_api(job.mp3_path, language=job.language, duration=job.duration),
            policy=RetryPolicy(
                max_retries=self._settings.stt_max_retries,
                base_delay=float(self._settings.stt_retry_base_delay),
                heartbeat_seconds=float(self._settings.stt_heartbeat_seconds),
                delay_cap=const.STT_REMOTE_RETRY_DELAY_CAP,
            ),
            on_status=on_status,
            label=status_label,
        )
        on_status(f"OpenAI STT ({self._model}): received {status_label}")

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
        client = self._openai_client()
        kwargs: dict[str, Any] = {"model": self._model}
        extra_body: dict[str, Any] = {}

        if self._caps.remote_diarize:
            kwargs["response_format"] = "diarized_json"
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
