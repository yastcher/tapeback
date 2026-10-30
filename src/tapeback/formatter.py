from dataclasses import dataclass, replace

from tapeback import const
from tapeback._quality import has_speech, strip_hallucinations
from tapeback.models import Segment

# Words with probability below this are marked as uncertain (italic in markdown).
# 0.35 is tuned for multilingual speech: English loanwords inside Russian sentences
# (code-switching) often come back with 0.3-0.5 probability even when correct.
WORD_LOW_CONFIDENCE = 0.35

# Upper bound on how much audio one merged block may span, in seconds.
# Merging is driven by speaker identity and pause length, neither of which bounds the
# result: a long uninterrupted monologue merged into a single block, so a 31-minute
# recording rendered with its last timecode at [00:00:45]. The text was intact but
# unnavigable. 60s keeps a block readable while still giving a timecode a minute.
MAX_BLOCK_SECONDS = 60.0


def _hms(seconds: float) -> tuple[int, int, int]:
    """Whole hours, minutes and seconds of a time, which is never negative here."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return hours, minutes, secs


def _format_timecode(seconds: float) -> str:
    """Format seconds as [HH:MM:SS]."""
    return f"[{_format_duration_hms(seconds)}]"


def _format_duration_human(seconds: float) -> str:
    """Format duration as human-readable string (e.g. '1h 23m 45s')."""
    hours, minutes, secs = _hms(seconds)

    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _format_duration_hms(seconds: float) -> str:
    """Format duration as HH:MM:SS."""
    hours, minutes, secs = _hms(seconds)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _merge_consecutive_speakers(
    segments: list[Segment],
    pause_threshold: float = 1.0,
    max_block_seconds: float = MAX_BLOCK_SECONDS,
) -> list[tuple[float, str | None, str]]:
    """Merge consecutive segments from the same speaker into one block.

    Segments from the same speaker are NOT merged when the gap between them
    exceeds pause_threshold — this preserves intentional pauses within a
    single speaker's speech.

    A block is also closed once it spans max_block_seconds, so an uninterrupted
    monologue still gets timecodes to navigate by.

    Returns list of (start_time, speaker, merged_text).
    """
    if not segments:
        return []

    merged: list[tuple[float, str | None, str]] = []
    current_start = segments[0].start
    current_end = segments[0].end
    current_speaker = segments[0].speaker
    current_texts = [segments[0].text]

    for seg in segments[1:]:
        gap = seg.start - current_end
        too_long = seg.end - current_start >= max_block_seconds
        if seg.speaker == current_speaker and gap < pause_threshold and not too_long:
            current_texts.append(seg.text)
            current_end = seg.end
        else:
            merged.append((current_start, current_speaker, " ".join(current_texts)))
            current_start = seg.start
            current_end = seg.end
            current_speaker = seg.speaker
            current_texts = [seg.text]

    merged.append((current_start, current_speaker, " ".join(current_texts)))
    return merged


def _mark_low_confidence_words(segment: Segment) -> Segment:
    """Create a new Segment with low-confidence words marked in italic.

    Consecutive low-confidence words are grouped into a single italic span:
    ``*Sorry could you* repeat`` instead of ``*Sorry* *could* *you* repeat``.
    """
    if not segment.words:
        return segment

    parts: list[str] = []
    low_group: list[str] = []

    for word in segment.words:
        text = word.word.strip()
        if not text:
            continue
        if word.probability < WORD_LOW_CONFIDENCE:
            low_group.append(text)
        else:
            if low_group:
                parts.append(f"*{' '.join(low_group)}*")
                low_group = []
            parts.append(text)

    if low_group:
        parts.append(f"*{' '.join(low_group)}*")

    if not parts:
        return segment

    return Segment(
        start=segment.start,
        end=segment.end,
        text=" ".join(parts),
        words=segment.words,
        speaker=segment.speaker,
    )


def _strip_hallucinated_text(segment: Segment) -> Segment:
    """Drop subtitle-corpus phrases from a segment's text.

    A cleaned segment loses its words: the italic pass rebuilds the text from them and
    would put the phrase back, so its unsure words go unmarked. Rendering is the last
    stage, and nothing after it needs their timings.
    """
    cleaned = strip_hallucinations(segment.text)
    if cleaned == segment.text:
        return segment
    return replace(segment, text=cleaned, words=None)


def _is_vad_artifact(segment: Segment) -> bool:
    """Shorter than MIN_SEGMENT_DURATION as Whisper returned it.

    A piece diarization rebuilt from a segment's words is measured by that segment:
    the words of a longer segment can span less than a second.
    """
    duration = max(segment.end - segment.start, segment.source_duration or 0.0)
    return duration < const.MIN_SEGMENT_DURATION


def _format_segments_block(segments: list[Segment]) -> list[str]:
    """Format a list of segments into timecoded markdown lines.

    Low-confidence words (probability < WORD_LOW_CONFIDENCE) are marked with *italics*.
    """
    kept = [s for s in segments if not _is_vad_artifact(s)]
    kept = [_strip_hallucinated_text(s) for s in kept]
    kept = [s for s in kept if has_speech(s.text)]
    kept = [_mark_low_confidence_words(s) for s in kept]
    merged = _merge_consecutive_speakers(kept)

    lines: list[str] = []
    for start_time, speaker, text in merged:
        timecode = _format_timecode(start_time)

        if speaker:
            lines.append(f"{timecode} **{speaker}:** {text}")
        else:
            lines.append(f"{timecode} {text}")
        lines.append("")

    return lines


@dataclass(frozen=True)
class TranscriptMeta:
    """Everything the note needs that is not the transcript itself.

    Grouped rather than passed as loose arguments: these all describe the run, they
    all come from the same place in the pipeline, and the parameter list had grown
    past the point where call sites stayed readable.
    """

    session_name: str
    audio_rel_path: str
    duration_seconds: float
    language: str
    # True when transcription was interrupted and the note covers only part of the
    # recording. A partial transcript that looks complete is worse than none, because
    # nothing prompts a re-run.
    partial: bool = False


def _session_date_time(session_name: str) -> tuple[str, str]:
    """The date and HH:MM of a session named YYYY-MM-DD_HH-MM-SS; 00:00 if it has no time."""
    parts = session_name.split("_")
    time_str = parts[1].replace("-", ":") if len(parts) > 1 else "00:00"
    return parts[0], ":".join(time_str.split(":")[:2])


def format_markdown(segments: list[Segment], meta: TranscriptMeta) -> str:
    """Generate markdown with YAML front matter and one transcript.

    Segments Whisper returned shorter than 1 second are filtered out (VAD artifacts).
    Each segment starts with [HH:MM:SS] timecode.
    """
    date_str, time_display = _session_date_time(meta.session_name)

    duration_hms = _format_duration_hms(meta.duration_seconds)
    duration_human = _format_duration_human(meta.duration_seconds)

    lines = [
        "---",
        f"date: {date_str}",
        f'time: "{time_display}"',
        f'duration: "{duration_hms}"',
        f"language: {meta.language}",
        f'audio: "[[{meta.audio_rel_path}]]"',
        "tags:",
        "  - meeting",
        "  - transcript",
    ]
    if meta.partial:
        # Both machine- and human-visible: a partial transcript that looks complete is
        # worse than no transcript, because nothing prompts you to re-run it.
        lines.append("  - partial")
        lines.append("partial: true")
    lines += [
        "---",
        "",
        f"# Meeting {date_str} {time_display}",
        "",
        f"**Duration:** {duration_human} | **Language:** {meta.language}",
        "",
    ]
    if meta.partial:
        lines.append(
            "> [!warning] Interrupted — this transcript covers only part of the "
            "recording. Re-run to complete it."
        )
        lines.append("")
    lines += ["---", ""]
    lines.extend(_format_segments_block(segments))

    return "\n".join(lines)


def format_live_markdown(
    segments: list[Segment],
    session_name: str,
    language: str,
) -> str:
    """Generate a simplified live markdown transcript (no duration).

    Updated atomically during recording so the user can open it mid-meeting.
    Replaced by the final polished transcript after recording stops.
    """
    date_str, time_display = _session_date_time(session_name)

    lines = [
        f"# Live Transcript {date_str} {time_display}",
        "",
        f"**Language:** {language} | **Status:** recording in progress",
        "",
        "---",
        "",
    ]

    if segments:
        lines.extend(_format_segments_block(segments))
    else:
        lines.append("*Waiting for first transcription cycle...*")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Live preview. Final transcript with diarization will replace this file.*")
    lines.append("")

    return "\n".join(lines)
