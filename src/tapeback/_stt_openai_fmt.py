"""Map OpenAI Audio Transcriptions API responses to tapeback Segments."""

from __future__ import annotations

from typing import Any

from tapeback import const
from tapeback.models import Segment, Word


def _word_from_api(item: Any) -> Word | None:
    """Map one OpenAI word object to Word, or None if timings are missing."""
    start = getattr(item, "start", None)
    end = getattr(item, "end", None)
    text = getattr(item, "word", None)
    if (start is None or end is None or text is None) and isinstance(item, dict):
        start = item.get("start")
        end = item.get("end")
        text = item.get("word")
    if start is None or end is None or not text:
        return None
    # whisper-1 verbose_json does not include per-word confidence; treat as certain
    # so formatter italics stay reserved for local Whisper's real probabilities.
    return Word(start=float(start), end=float(end), word=str(text), probability=1.0)


def _attr(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _segments_from_verbose(response: Any, time_offset: float) -> list[Segment]:
    """Convert whisper-1 verbose_json segments (+ optional words) to domain Segments."""
    raw_segments = _attr(response, "segments")
    if not raw_segments:
        text = (_attr(response, "text") or "").strip()
        if not text:
            return []
        duration = float(_attr(response, "duration") or 0.0)
        return [
            Segment(
                start=time_offset,
                end=time_offset + duration,
                text=text,
                words=None,
            )
        ]

    out: list[Segment] = []
    for seg in raw_segments:
        start = float(_attr(seg, "start") or 0.0)
        end = float(_attr(seg, "end") or start)
        text = str(_attr(seg, "text") or "").strip()
        raw_words = _attr(seg, "words")
        if not text:
            continue
        words: list[Word] | None = None
        if raw_words:
            mapped = [_word_from_api(w) for w in raw_words]
            words = [w for w in mapped if w is not None] or None
            if words is not None:
                words = [
                    Word(
                        start=w.start + time_offset,
                        end=w.end + time_offset,
                        word=w.word,
                        probability=w.probability,
                    )
                    for w in words
                ]
        out.append(
            Segment(
                start=start + time_offset,
                end=end + time_offset,
                text=text,
                words=words,
            )
        )
    return out


def _segments_from_text_response(
    response: Any, time_offset: float, duration: float
) -> list[Segment]:
    """Map a text-only transcription (gpt-transcribe etc.) to one Segment."""
    text = (_attr(response, "text") or "").strip()
    if not text:
        return []
    return [Segment(start=time_offset, end=time_offset + duration, text=text, words=None)]


def _display_speaker(raw: str, order: list[str]) -> str:
    """Map an API speaker id to Speaker N by first-seen order."""
    if raw not in order:
        order.append(raw)
    return const.SPEAKER_LABEL_FMT.format(order.index(raw) + 1)


def _segments_from_diarized(
    response: Any,
    time_offset: float,
    *,
    speaker_order: list[str] | None = None,
) -> list[Segment]:
    """Map diarized_json segments (gpt-4o-transcribe-diarize) to labeled Segments."""
    order = speaker_order if speaker_order is not None else []
    raw_segments = _attr(response, "segments") or []
    out: list[Segment] = []
    for seg in raw_segments:
        text = str(_attr(seg, "text") or "").strip()
        if not text:
            continue
        start = float(_attr(seg, "start") or 0.0)
        end = float(_attr(seg, "end") or start)
        raw_speaker = _attr(seg, "speaker")
        speaker = _display_speaker(str(raw_speaker), order) if raw_speaker else None
        out.append(
            Segment(
                start=start + time_offset,
                end=end + time_offset,
                text=text,
                words=None,
                speaker=speaker,
            )
        )
    if out:
        return out
    # Fallback: some SDK shapes only expose the joined text.
    return _segments_from_text_response(
        response, time_offset, float(_attr(response, "duration") or 0.0)
    )


def _language_from_response(response: Any, fallback: str | None) -> str:
    """Prefer modern languages[{code}], then legacy language, then fallback."""
    languages = _attr(response, "languages")
    if languages:
        first = languages[0]
        code = _attr(first, "code") if not isinstance(first, str) else first
        if code:
            return str(code)
    language = _attr(response, "language")
    if language:
        return str(language)
    return fallback or "unknown"


def _info_from_response(
    response: Any, duration: float, *, language_fallback: str | None
) -> dict[str, str | float | bool]:
    return {
        "language": _language_from_response(response, language_fallback),
        "language_probability": 1.0,
        "duration": duration,
        "partial": False,
    }


def _keywords_from_hotwords(hotwords: str) -> list[str]:
    """Split the glossary string into OpenAI keywords[] entries."""
    return [part.strip() for part in hotwords.split(",") if part.strip()]
