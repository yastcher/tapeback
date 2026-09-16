import locale
import os
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from faster_whisper import BatchedInferencePipeline, WhisperModel
from huggingface_hub.errors import LocalEntryNotFoundError

from tapeback import _resume, const
from tapeback._gpu import (
    free_gpu_memory,
    get_free_vram_mib,
    is_cuda_error,
    preload_cuda_libs,
    wait_for_clamp_release,
)
from tapeback._isolated import transcribe_isolated
from tapeback._timing import ProgressReporter, stage_timer
from tapeback.models import Segment, Word
from tapeback.settings import Settings

# Work around PyAV bug: Cython directive c_string_encoding=ascii cannot handle
# non-ASCII error messages from strerror_r() on non-English locales (e.g. Russian).
# glibc's strerror_r() uses dcgettext() which respects LANGUAGE env var for fallback
# translations. Setting both env var AND C locale is required — env var alone does not
# change the already-initialized C locale after Python startup.
# See: https://github.com/PyAV-Org/PyAV — setup.py c_string_encoding directive.
os.environ["LC_MESSAGES"] = "C"
locale.setlocale(locale.LC_MESSAGES, "C")


# Compute types ctranslate2 cannot run on CPU. Requesting one there raises
# ValueError rather than degrading, so a device fallback has to translate it.
_CUDA_ONLY_COMPUTE_TYPES = frozenset({"float16", "int8_float16", "bfloat16", "int8_bfloat16"})


def _resolve_compute_type(compute_type: str, device: str) -> str:
    """Resolve the compute type for the device we ended up on.

    - auto + cuda → int8_float16
    - auto + cpu  → int8
    - an explicit CUDA-only type on CPU → int8, because ctranslate2 raises otherwise
    - any other explicit value passes through.

    The CPU translation matters because the device is now chosen at runtime: a card that
    is thermally clamped or out of VRAM sends us to the CPU carrying whatever
    TAPEBACK_COMPUTE_TYPE was set for the GPU. Without this, that combination died with
    "Requested int8_float16 compute type, but the target device or backend do not
    support efficient int8_float16 computation" — a crash instead of a fallback.

    int8_float16 rather than float16 because it is faster *and* smaller, which is not
    the usual trade-off. Measured on a GTX 1650 Ti with large-v3-turbo, same 90 s clip,
    twice each: float16 3.90x real time and 2139 MiB, int8_float16 **14.16x and
    1115 MiB**. Quality does not pay for it — decoding the same audio both ways gave
    near-identical text with single-word differences in both directions, and across the
    benchmark grid int8_float16 had the lower share of low-confidence words.

    The likely reason is hardware: this is a Turing part without tensor cores, so fp16
    gets no acceleration while int8 uses the integer datapath. That is a hypothesis;
    the measurements are not. ctranslate2 falls back on its own if a device does not
    support the requested type, so this stays safe on other GPUs.
    """
    if compute_type == "auto":
        return "int8_float16" if device == "cuda" else "int8"
    if device != "cuda" and compute_type in _CUDA_ONLY_COMPUTE_TYPES:
        print(
            f"Warning: compute type {compute_type} is GPU-only; using int8 on {device}.",
            file=sys.stderr,
        )
        return "int8"
    return compute_type


def _noop_status(_message: str) -> None:
    """Default status sink — used when transcribe_stereo gets no reporter."""


def _enough_vram(settings: Settings) -> bool:
    """False if the card plainly cannot hold a model right now.

    This is prevention, not optimisation. A CUDA out-of-memory during model load
    **leaks the allocation**: ctranslate2 builds the model on the C++ side, and when the
    load fails partway the object is never destroyed and never reaches Python, so there
    is no handle to release. Measured, free VRAM went 3674 MiB -> 95 MiB and stayed
    there for the life of the process; neither dropping the exception's traceback nor
    `CT2_CUDA_ALLOCATOR=cuda_malloc_async` recovers it. Everything afterwards — the
    diarizer's own VRAM check included — then finds an empty card.

    So the only reliable fix is to not attempt a load that cannot fit. The floor is
    deliberately crude: the smallest configuration measured here (large-v3-turbo in
    int8_float16) needs ~1115 MiB, so anything under the threshold cannot work at all.
    Sizing per model would need a table that goes stale; this catches the case that
    actually recurs, which is a card already occupied by something else.
    """
    free_mib = get_free_vram_mib()
    if free_mib is None or free_mib >= settings.min_free_vram_mib:
        return True
    print(
        f"Warning: only {free_mib} MiB VRAM free, below the "
        f"{settings.min_free_vram_mib} MiB needed to load a model — using CPU. "
        "A previous CUDA out-of-memory leaks VRAM until the process restarts.",
        file=sys.stderr,
    )
    return False


def _resolve_device(settings: Settings) -> str:
    """Pick the device to actually run on, avoiding a thermally clamped GPU.

    On a laptop sharing one heatsink between CPU and GPU, the controller responds to a
    hot *system* by cutting the GPU's power budget — measured here as 50 W dropping to
    5 W with clocks pinned at 300 MHz while the GPU itself sat at 74 C and the CPU
    package at 93 C. It releases only on idle, and after sustained load it stayed
    latched for over 900 s.

    Transcribing on a card in that state is strictly worse than using the CPU: measured
    on the same clip, CPU 2.39x real time against 0.31x clamped, i.e. the CPU is ~8x
    faster. Waiting it out is what turned a fifteen-minute job into a multi-hour one.
    """
    if settings.device != "cuda":
        return settings.device
    if not _enough_vram(settings):
        return "cpu"
    if not settings.thermal_clamp_check:
        return settings.device

    # Checking is always cheap; waiting is the part that has to be justified. With a
    # zero wait this is a single query, which is what makes returning to the GPU work:
    # the decision is retaken for every stage, so a clamp that clears between channels
    # is picked up at the next one instead of stranding the whole run on the CPU.
    #
    # Reported, not silent: a wait can legitimately last minutes, and a process that
    # prints nothing while it does is indistinguishable from one that has hung.
    if wait_for_clamp_release(
        settings.thermal_clamp_wait,
        report=lambda message: print(message, file=sys.stderr),
    ):
        return "cuda"
    if not settings.thermal_clamp_cpu_fallback:
        print(
            "Warning: GPU is thermally clamped; transcription will be very slow. "
            "Set TAPEBACK_THERMAL_CLAMP_CPU_FALLBACK=true to use the CPU instead.",
            file=sys.stderr,
        )
        return "cuda"
    print(
        "Warning: GPU is thermally clamped and did not release — transcribing on CPU, "
        "which is faster in this state. Let the machine idle to clear it.",
        file=sys.stderr,
    )
    return "cpu"


# Parameters tapeback configures that BatchedInferencePipeline silently drops.
# Verified against faster-whisper 1.2.1's own "Unused Arguments" docstring; the
# temperature entry is separate because it is not ignored outright — only the
# first value of the ladder is used, which disables the anti-hallucination retries.
BATCHED_IGNORED_SETTINGS = (
    "no_speech_threshold",
    "condition_on_previous_text",
    "hallucination_silence_threshold",
)


def _batched_warning(settings: Settings) -> str | None:
    """Warn if batching would silently drop anti-hallucination settings.

    Enabling batching quietly reverts several deliberate choices, and the run
    otherwise looks identical — so the user must be told which ones, rather than
    discovering it in a transcript full of repeats.
    """
    dropped = [name for name in BATCHED_IGNORED_SETTINGS if getattr(settings, name) is not None]
    if len(settings.temperature) > 1:
        dropped.append("temperature (only the first value is used)")
    if not dropped:
        return None
    return (
        f"Warning: TAPEBACK_BATCH_SIZE={settings.batch_size} enables batched inference, "
        f"which ignores: {', '.join(dropped)}. "
        "These are anti-hallucination settings; expect more repeats on quiet channels."
    )


class Transcriber:
    def __init__(self, settings: Settings) -> None:
        """Initialize faster-whisper model.

        Falls back from CUDA to CPU if CUDA is not available.
        First run downloads the model automatically.
        """
        self._settings = settings
        self._isolated = settings.isolate_transcription
        if self._isolated:
            # No model here: the child process owns it, so a CUDA out-of-memory takes
            # the child down instead of leaking this process's VRAM permanently.
            # Device resolution belongs there too — doing it twice would wait out the
            # thermal clamp twice.
            self._device = settings.device
            self._compute_type = settings.compute_type
            self._model = None
            self._batched = None
            return
        self._device = _resolve_device(settings)
        if self._device == "cuda":
            # Make ctranslate2 (CUDA 12) find cuBLAS/cuDNN on CUDA 13 systems.
            preload_cuda_libs()
        self._compute_type = _resolve_compute_type(settings.compute_type, self._device)
        self._model = self._load_model(self._device, self._compute_type)
        self._batched = self._wrap_batched(self._model)
        if self._batched is not None:
            warning = _batched_warning(settings)
            if warning is not None:
                print(warning, file=sys.stderr)

    def describe(self) -> str:
        """Human-readable record of where the model actually landed.

        Until now the only device-related output was a warning on CPU fallback,
        which scrolls past between other status lines — so a run that silently
        dropped to CPU (roughly an order of magnitude slower) looked exactly like
        a healthy one. Reporting the resolved device positively makes that
        distinguishable without reproducing the run.
        """
        if self._isolated:
            # The child resolves the real device and reports it as a status event once
            # it has one; there is nothing truthful to say about it from here yet.
            return f"Whisper: {self._settings.stt_model} in an isolated worker"
        batched = f", batch_size={self._settings.batch_size}" if self._batched else ""
        return (
            f"Whisper: {self._settings.stt_model} on {self._device}/{self._compute_type}{batched}"
        )

    def _wrap_batched(self, model: WhisperModel) -> BatchedInferencePipeline | None:
        """Wrap the model for batched inference when batch_size > 0 (faster on GPU)."""
        if self._settings.batch_size > 0:
            return BatchedInferencePipeline(model=model)
        return None

    def _new_model(self, device: str, compute_type: str) -> WhisperModel:
        """Instantiate WhisperModel, preferring the local cache.

        faster-whisper otherwise queries HuggingFace for model metadata on every
        start, adding latency and hanging when offline. Try the cache first and
        download only when the model isn't present yet (first run).
        """
        try:
            return WhisperModel(
                self._settings.stt_model,
                device=device,
                compute_type=compute_type,
                local_files_only=True,
            )
        except LocalEntryNotFoundError:
            return WhisperModel(
                self._settings.stt_model,
                device=device,
                compute_type=compute_type,
                local_files_only=False,
            )

    def _release_gpu_model(self) -> None:
        """Drop every reference to the GPU model, then ask for the memory back.

        Order matters. Assigning the replacement over `self._model` would keep the
        failed GPU model alive for as long as the new one takes to build, and after an
        out-of-memory failure there is by definition no room for both. Observed without
        this: free VRAM went 3674 MiB -> 95 MiB and stayed there, so the diarizer's own
        VRAM check then sent it to CPU as well — one failure degraded the whole run.
        """
        self._model = None
        self._batched = None
        free_gpu_memory()

    def _load_model(self, device: str, compute_type: str) -> WhisperModel:
        """Load WhisperModel, falling back to CPU on CUDA errors."""
        try:
            return self._new_model(device, compute_type)
        except RuntimeError as exc:
            if device != "cuda" or not is_cuda_error(exc):
                raise
            message = str(exc)
            # The exception's traceback holds the frame the failed model was built in,
            # which keeps its allocation reachable. Break that before retrying, or the
            # CPU model is constructed while the dead GPU one still occupies VRAM.
            exc.__traceback__ = None
            free_gpu_memory()
            print(
                f"Warning: CUDA not available at load time, falling back to CPU: {message}",
                file=sys.stderr,
            )
            self._device = "cpu"
            self._compute_type = "int8"
            return self._new_model("cpu", "int8")

    def _fallback_to_cpu(self, exc: Exception) -> None:
        """Recreate model on CPU after a CUDA runtime failure.

        The real error is printed (not just "CUDA runtime error") so the user can
        tell an out-of-memory failure from a cuDNN/driver problem.
        """
        print(
            f"Warning: CUDA runtime error, falling back to CPU: {exc}",
            file=sys.stderr,
        )
        exc.__traceback__ = None
        self._release_gpu_model()
        self._device = "cpu"
        self._compute_type = "int8"
        self._model = self._new_model("cpu", "int8")
        self._batched = self._wrap_batched(self._model)

    def transcribe(
        self,
        audio_path: Path,
        *,
        stage: str = "transcribe",
        on_status: Callable[[str], None] = _noop_status,
        language_override: str | None = None,
    ) -> tuple[list[Segment], dict[str, str | float | bool]]:
        """Transcribe audio file.

        Returns (list of Segments, info dict with language/duration/etc).
        Falls back to CPU if CUDA fails — either when calling transcribe()
        (eager language detection raises before yielding) or while iterating
        the segment generator.

        Progress is reported through ``on_status`` as the segment generator is
        consumed, so a long run shows movement instead of a single opening line.
        """
        # Checked before the isolation branch so a cache hit costs no process at all.
        key = self._resume_key(audio_path, stage)
        if key is not None:
            cached = _resume.load(key, _resume.resume_dir(self._settings))
            if cached is not None:
                on_status(f"Reusing the '{stage}' result from an earlier run.")
                return cached

        if self._isolated:
            segments, info = transcribe_isolated(
                audio_path,
                self._settings,
                stage=stage,
                on_status=on_status,
                language_override=language_override,
            )
            self._store_resume(key, segments, info)
            return segments, info

        # "auto" → None lets faster-whisper auto-detect language. An override wins over
        # "auto" but never over an explicitly configured language.
        configured = self._settings.language
        language = configured if configured != "auto" else language_override or None

        segments: list[Segment] = []
        try:
            segments_iter, info = self._invoke_transcribe(audio_path, language)
            segments, interrupted = self._collect_segments(
                segments_iter, stage, info.duration, on_status
            )
        except RuntimeError as exc:
            if self._device != "cuda" or not is_cuda_error(exc):
                raise
            self._fallback_to_cpu(exc)
            segments_iter, info = self._invoke_transcribe(audio_path, language)
            segments, interrupted = self._collect_segments(
                segments_iter, stage, info.duration, on_status
            )

        if not segments and not interrupted:
            print("Warning: No speech detected in audio", file=sys.stderr)

        info_dict: dict[str, str | float | bool] = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "partial": interrupted,
        }
        self._store_resume(key, segments, info_dict)

        return segments, info_dict

    def _resume_key(self, audio_path: Path, stage: str) -> _resume.ResumeKey | None:
        if not self._settings.resume_cache:
            return None
        return _resume.resume_key(audio_path, self._settings, stage)

    def _store_resume(
        self,
        key: _resume.ResumeKey | None,
        segments: list[Segment],
        info: dict[str, Any],
    ) -> None:
        """Cache a channel, but only a complete one.

        Storing a partial result would make the next run reuse the truncated version
        and call it done — the opposite of what resuming is for.
        """
        if key is None or info.get("partial") or not segments:
            return
        _resume.store(key, _resume.resume_dir(self._settings), segments, info)

    def _invoke_transcribe(self, audio_path: Path, language: str | None) -> tuple[Any, Any]:
        """Single point that calls into faster-whisper with the configured args.

        Uses the batched pipeline when batch_size > 0, otherwise the plain model.
        """
        kwargs: dict[str, Any] = {
            "language": language,
            "beam_size": self._settings.beam_size,
            "temperature": self._settings.temperature,
            "vad_filter": self._settings.vad_filter,
            "chunk_length": self._settings.chunk_length,
            "word_timestamps": True,
            "condition_on_previous_text": self._settings.condition_on_previous_text,
            "no_speech_threshold": self._settings.no_speech_threshold,
            "multilingual": self._settings.multilingual,
            "language_detection_segments": self._settings.language_detection_segments,
            "hallucination_silence_threshold": self._settings.hallucination_silence_threshold,
        }
        # Only pass hotwords when set: faster-whisper tokenises the string into a
        # decoder bias, and an empty one would still cost that work per window.
        if self._settings.hotwords:
            kwargs["hotwords"] = self._settings.hotwords
        if self._batched is not None:
            kwargs["batch_size"] = self._settings.batch_size
            return self._batched.transcribe(str(audio_path), **kwargs)
        if self._model is None:
            # Only reachable if a CPU fallback itself failed after the GPU model was
            # released. Say so plainly rather than raising AttributeError on None.
            raise RuntimeError("No Whisper model is loaded — the CPU fallback failed")
        return self._model.transcribe(str(audio_path), **kwargs)

    def transcribe_stereo(
        self,
        mic_16k: Path,
        monitor_16k: Path,
        *,
        on_status: Callable[[str], None] = _noop_status,
    ) -> tuple[list[Segment], list[Segment], dict[str, str | float | bool]]:
        """Transcribe both channels separately.

        Returns (mic_segments, monitor_segments, info).
        mic_segments get speaker="You" automatically.
        info from the channel with more total speech duration.

        Each channel is timed separately via on_status so the cost of the mic
        pass (mostly silence while the user listens) is visible on its own.

        The monitor channel goes first so its detected language can be reused for the
        mic. Both channels are one conversation, but the mic is gated to near silence
        while the user listens, leaving auto-detection almost nothing to work from — it
        guessed wrong often enough to produce notes labelled `language: en` whose text
        was Russian.
        """
        with stage_timer("transcribe monitor", on_status):
            monitor_segments, monitor_info = self.transcribe(
                monitor_16k, stage="transcribe monitor", on_status=on_status
            )

        detected = monitor_info.get("language")
        mic_language = str(detected) if detected else None

        mic_segments: list[Segment] = []
        mic_info: dict[str, str | float | bool] = {}
        if monitor_info.get("partial"):
            # Ctrl+C means stop, not "stop this channel". Starting the second one would
            # make the user interrupt twice; the monitor's work is already kept.
            on_status("Skipping the mic channel — transcription was interrupted.")
        else:
            self._pace(on_status)
            with stage_timer("transcribe mic", on_status):
                mic_segments, mic_info = self.transcribe(
                    mic_16k,
                    stage="transcribe mic",
                    on_status=on_status,
                    language_override=mic_language,
                )

        # Assign speaker="You" to mic segments
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

        # Pick info from channel with more speech
        mic_speech = sum(s.end - s.start for s in mic_segments)
        monitor_speech = sum(s.end - s.start for s in monitor_segments)
        info = mic_info if mic_speech >= monitor_speech else monitor_info
        # Partiality belongs to the run, not to whichever channel happened to be
        # picked for its language — a transcript missing one channel is partial.
        info = dict(info)
        info["partial"] = bool(monitor_info.get("partial") or mic_info.get("partial"))

        return mic_segments, monitor_segments, info

    def _pace(self, on_status: Callable[[str], None]) -> None:
        """Idle between stages so the chassis sheds heat instead of latching the clamp.

        Cheaper than recovering from a clamp: once latched it needs minutes of idle,
        and after sustained load it stayed latched past 900 s. Off by default because
        it costs wall-clock on a machine that cools adequately.
        """
        pause = self._settings.stage_pause_seconds
        if pause <= 0 or self._device != "cuda":
            return
        on_status(f"Pausing {pause:.0f}s to let the GPU cool...")
        time.sleep(pause)

    @staticmethod
    def _collect_segments(
        segments_iter: Iterable[Any],
        stage: str,
        total_duration: float,
        on_status: Callable[[str], None],
    ) -> tuple[list[Segment], bool]:
        """Iterate over faster-whisper segments and convert to dataclasses.

        Returns (segments, interrupted). On Ctrl+C the segments decoded so far are
        kept and returned rather than discarded: transcription can run for hours, and
        throwing all of it away on an interrupt is how sixteen recordings ended up
        with no transcript at all. The interrupt is not re-raised — the caller writes
        what it has — but it is reported, so a later one still stops the process.

        Reports throttled progress as it goes: faster-whisper's own
        ``log_progress`` writes a tqdm bar straight to the terminal, bypassing
        ``on_status``, so it would never reach the tray log.
        """
        progress = ProgressReporter(stage, total_duration, on_status)
        segments: list[Segment] = []
        try:
            return Transcriber._convert_segments(segments_iter, segments, progress), False
        except KeyboardInterrupt:
            on_status(
                f"Interrupted during '{stage}' — keeping the {len(segments)} "
                "segments decoded so far."
            )
            return segments, True

    @staticmethod
    def _convert_segments(
        segments_iter: Iterable[Any],
        segments: list[Segment],
        progress: ProgressReporter,
    ) -> list[Segment]:
        """Drain the generator into ``segments`` (shared so a caller keeps partials)."""
        for seg in segments_iter:
            progress.update(seg.end)
            words: list[Word] | None = None
            if seg.words:
                words = [
                    Word(
                        start=w.start,
                        end=w.end,
                        word=w.word,
                        probability=w.probability,
                    )
                    for w in seg.words
                ]

            segments.append(
                Segment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    words=words,
                )
            )

        return segments
