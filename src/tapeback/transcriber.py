import locale
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from tapeback import const
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


def _resolve_compute_type(compute_type: str, device: str) -> str:
    """Resolve 'auto' compute type based on device.

    - auto + cuda → float16 (works on 4 GB cards; large-v3-turbo fits in ~1.5 GiB)
    - auto + cpu  → int8
    - explicit value passes through.

    Users on memory-tight GPUs can pin TAPEBACK_COMPUTE_TYPE=int8 explicitly.
    """
    if compute_type != "auto":
        return compute_type
    if device == "cuda":
        return "float16"
    return "int8"


class Transcriber:
    def __init__(self, settings: Settings) -> None:
        """Initialize faster-whisper model.

        Falls back from CUDA to CPU if CUDA is not available.
        First run downloads the model automatically.
        """
        self._settings = settings
        self._device = settings.device
        compute_type = _resolve_compute_type(settings.compute_type, settings.device)
        self._model = self._load_model(settings.device, compute_type)

    def _load_model(self, device: str, compute_type: str) -> WhisperModel:
        """Load WhisperModel, falling back to CPU on CUDA errors."""
        try:
            return WhisperModel(
                self._settings.whisper_model,
                device=device,
                compute_type=compute_type,
            )
        except RuntimeError:
            if device == "cuda":
                print(
                    "Warning: CUDA not available at load time, falling back to CPU",
                    file=sys.stderr,
                )
                self._device = "cpu"
                return WhisperModel(
                    self._settings.whisper_model,
                    device="cpu",
                    compute_type="int8",
                )
            raise

    def _fallback_to_cpu(self) -> None:
        """Recreate model on CPU after a CUDA runtime failure."""
        print(
            "Warning: CUDA runtime error, falling back to CPU",
            file=sys.stderr,
        )
        self._device = "cpu"
        self._model = WhisperModel(
            self._settings.whisper_model,
            device="cpu",
            compute_type="int8",
        )

    def transcribe(self, audio_path: Path) -> tuple[list[Segment], dict[str, str | float]]:
        """Transcribe audio file.

        Returns (list of Segments, info dict with language/duration/etc).
        Falls back to CPU if CUDA fails — either when calling transcribe()
        (eager language detection raises before yielding) or while iterating
        the segment generator.
        """
        # "auto" → None lets faster-whisper auto-detect language
        language = self._settings.language if self._settings.language != "auto" else None

        segments: list[Segment] = []
        try:
            segments_iter, info = self._invoke_transcribe(audio_path, language)
            segments = self._collect_segments(segments_iter)
        except RuntimeError:
            if self._device != "cuda":
                raise
            self._fallback_to_cpu()
            segments_iter, info = self._invoke_transcribe(audio_path, language)
            segments = self._collect_segments(segments_iter)

        if not segments:
            print("Warning: No speech detected in audio", file=sys.stderr)

        info_dict: dict[str, str | float] = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
        }

        return segments, info_dict

    def _invoke_transcribe(self, audio_path: Path, language: str | None) -> tuple[Any, Any]:
        """Single point that calls into faster-whisper with the configured args."""
        return self._model.transcribe(
            str(audio_path),
            language=language,
            beam_size=self._settings.beam_size,
            vad_filter=self._settings.vad_filter,
            chunk_length=self._settings.chunk_length,
            word_timestamps=True,
            condition_on_previous_text=self._settings.condition_on_previous_text,
            no_speech_threshold=self._settings.no_speech_threshold,
        )

    def transcribe_stereo(
        self, mic_16k: Path, monitor_16k: Path
    ) -> tuple[list[Segment], list[Segment], dict[str, str | float]]:
        """Transcribe both channels separately.

        Returns (mic_segments, monitor_segments, info).
        mic_segments get speaker="You" automatically.
        info from the channel with more total speech duration.
        """
        mic_segments, mic_info = self.transcribe(mic_16k)
        monitor_segments, monitor_info = self.transcribe(monitor_16k)

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

        return mic_segments, monitor_segments, info

    @staticmethod
    def _collect_segments(segments_iter: Iterable[Any]) -> list[Segment]:
        """Iterate over faster-whisper segments and convert to dataclasses."""
        segments: list[Segment] = []
        for seg in segments_iter:
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
