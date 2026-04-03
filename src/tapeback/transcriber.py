import locale
import os
import subprocess
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

# Free VRAM below this threshold triggers int8 quantization instead of float16.
# 4 GiB leaves no headroom for inference allocations with large-v3-turbo in float16.
VRAM_INT8_THRESHOLD_MIB = 4096


def _get_free_vram_mib() -> int | None:
    """Get free GPU VRAM in MiB via nvidia-smi. Returns None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0])
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def _resolve_compute_type(compute_type: str, device: str) -> str:
    """Resolve 'auto' compute type based on available VRAM.

    - auto + cuda: float16 if enough VRAM, otherwise int8
    - auto + cpu: int8
    - explicit value: pass through as-is
    """
    if compute_type != "auto":
        return compute_type
    if device != "cuda":
        return "int8"

    free_vram = _get_free_vram_mib()
    if free_vram is None:
        return "float16"

    if free_vram < VRAM_INT8_THRESHOLD_MIB:
        print(
            f"Auto compute type: int8 (free VRAM {free_vram} MiB < {VRAM_INT8_THRESHOLD_MIB} MiB)",
            file=sys.stderr,
        )
        return "int8"

    return "float16"


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
        Falls back to CPU if CUDA fails during inference.
        """
        # "auto" → None lets faster-whisper auto-detect language
        language = self._settings.language if self._settings.language != "auto" else None

        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=language,
            beam_size=self._settings.beam_size,
            vad_filter=self._settings.vad_filter,
            chunk_length=self._settings.chunk_length,
            word_timestamps=True,
            condition_on_previous_text=self._settings.condition_on_previous_text,
        )

        segments: list[Segment] = []
        try:
            segments = self._collect_segments(segments_iter)
        except RuntimeError:
            if self._device == "cuda":
                self._fallback_to_cpu()
                segments_iter, info = self._model.transcribe(
                    str(audio_path),
                    language=self._settings.language,
                    beam_size=self._settings.beam_size,
                    vad_filter=self._settings.vad_filter,
                    chunk_length=self._settings.chunk_length,
                    word_timestamps=True,
                    condition_on_previous_text=self._settings.condition_on_previous_text,
                )
                segments = self._collect_segments(segments_iter)
            else:
                raise

        if not segments:
            print("Warning: No speech detected in audio", file=sys.stderr)

        info_dict: dict[str, str | float] = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
        }

        return segments, info_dict

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
