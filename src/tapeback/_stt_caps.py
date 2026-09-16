"""Remote STT model capabilities — behaviour derived from backend + model id.

Local transcription always allows pyannote afterward. Remote backends declare
what their models provide (timestamps, remote speaker labels, duration caps, …).
Today only the OpenAI backend is implemented; its model ids drive the matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

from tapeback import const

# OpenAI Audio model ids (TAPEBACK_STT_BACKEND=openai).
OPENAI_MODEL_WHISPER_1 = "whisper-1"
OPENAI_MODEL_GPT_TRANSCRIBE = "gpt-transcribe"
OPENAI_MODEL_GPT_DIARIZE = "gpt-4o-transcribe-diarize"


@dataclass(frozen=True, slots=True)
class RemoteSttCapabilities:
    """What a remote STT model can do for tapeback's pipeline."""

    # Word/segment timings (pause splits, assign_speakers overlap).
    word_timestamps: bool
    # Backend returns speaker-labeled segments.
    remote_diarize: bool
    # Modern context API: languages[] + keywords[] (not language=).
    modern_context: bool
    # Max seconds per upload the API accepts (None = size limit only).
    max_upload_seconds: float | None

    @property
    def allows_local_diarize(self) -> bool:
        """Whether local pyannote should run after this model.

        Remote diarize already labels speakers. Text-only models lack timings, so
        overlapping pyannote onto one blob per chunk is not useful — skip it.
        """
        if self.remote_diarize:
            return False
        return self.word_timestamps


def openai_capabilities_for(model: str) -> RemoteSttCapabilities:
    """Derive capabilities from an OpenAI Audio Transcriptions model id."""
    name = model.strip().lower()
    if name == OPENAI_MODEL_WHISPER_1:
        return RemoteSttCapabilities(
            word_timestamps=True,
            remote_diarize=False,
            modern_context=False,
            max_upload_seconds=None,
        )
    if name == OPENAI_MODEL_GPT_DIARIZE or name.endswith("-diarize"):
        return RemoteSttCapabilities(
            word_timestamps=False,
            remote_diarize=True,
            modern_context=False,
            max_upload_seconds=const.OPENAI_DIARIZE_MAX_UPLOAD_SECONDS,
        )
    # gpt-transcribe, gpt-4o-transcribe, gpt-4o-mini-transcribe, …
    return RemoteSttCapabilities(
        word_timestamps=False,
        remote_diarize=False,
        modern_context=True,
        max_upload_seconds=const.OPENAI_GPT_TRANSCRIBE_MAX_UPLOAD_SECONDS,
    )


def allows_local_diarize(stt_backend: str, stt_model: str) -> bool:
    """Whether the pipeline should attempt local pyannote after transcription."""
    if stt_backend == "local":
        return True
    if stt_backend == "openai":
        return openai_capabilities_for(stt_model).allows_local_diarize
    # Unknown remote backend: do not assume local diarize is useful.
    return False
