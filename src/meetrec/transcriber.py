import sys
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from meetrec.settings import Settings


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class Segment:
    start: float  # seconds
    end: float  # seconds
    text: str
    words: list[Word] | None = None
    speaker: str | None = None  # None in phase 1, filled in phase 2


class Transcriber:
    def __init__(self, settings: Settings) -> None:
        """Initialize faster-whisper model.

        Falls back from CUDA to CPU if CUDA is not available.
        First run downloads the model automatically.
        """
        self._settings = settings
        self._device = settings.device
        self._model = self._load_model(settings.device, settings.compute_type)

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

    def transcribe(self, audio_path: Path) -> tuple[list[Segment], dict]:
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
            word_timestamps=True,
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
                    word_timestamps=True,
                )
                segments = self._collect_segments(segments_iter)
            else:
                raise

        if not segments:
            print("Warning: No speech detected in audio", file=sys.stderr)

        info_dict = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
        }

        return segments, info_dict

    @staticmethod
    def _collect_segments(segments_iter) -> list[Segment]:
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
