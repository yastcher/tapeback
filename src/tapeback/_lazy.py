"""Lazy loaders for heavy ML dependencies.

Loading `tapeback.transcriber` drags in `faster_whisper` and `torch`
(~10 seconds cold-start). Keep the import inside the call so that
`tapeback --help` and `tapeback status` stay fast. Remote STT backends
(see ``remote_stt``) only need their vendor SDK, so they stay on a
separate branch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from tapeback.models import Segment

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tapeback.settings import Settings


class TranscriberLike(Protocol):
    """Shared surface used by pipeline / live for local and remote STT backends."""

    def describe(self) -> str: ...

    def transcribe(
        self,
        audio_path: Path,
        *,
        stage: str = "transcribe",
        on_status: Callable[[str], None] = ...,
        language_override: str | None = None,
    ) -> tuple[list[Segment], dict[str, str | float | bool]]: ...

    def transcribe_stereo(
        self,
        mic_16k: Path,
        monitor_16k: Path,
        *,
        on_status: Callable[[str], None] = ...,
    ) -> tuple[list[Segment], list[Segment], dict[str, str | float | bool]]: ...


def load_transcriber(settings: Settings) -> TranscriberLike:
    """Import and instantiate the configured STT backend on demand."""
    if settings.stt_backend != "local":
        from tapeback.remote_stt import load_remote_transcriber  # noqa: PLC0415

        return load_remote_transcriber(settings)

    from tapeback.transcriber import Transcriber  # noqa: PLC0415 — 10s ML import, must stay lazy

    return Transcriber(settings)
