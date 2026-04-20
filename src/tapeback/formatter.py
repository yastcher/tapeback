from tapeback import const
from tapeback.models import Segment

# Words with probability below this are marked as uncertain (italic in markdown).
# 0.35 is tuned for multilingual speech: English loanwords inside Russian sentences
# (code-switching) often come back with 0.3-0.5 probability even when correct.
WORD_LOW_CONFIDENCE = 0.35


def _format_timecode(seconds: float) -> str:
    """Format seconds as [HH:MM:SS]."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def _format_duration_human(seconds: float) -> str:
    """Format duration as human-readable string (e.g. '1h 23m 45s')."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _format_duration_hms(seconds: float) -> str:
    """Format duration as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _merge_consecutive_speakers(
    segments: list[Segment],
    pause_threshold: float = 1.0,
) -> list[tuple[float, str | None, str]]:
    """Merge consecutive segments from the same speaker into one block.

    Segments from the same speaker are NOT merged when the gap between them
    exceeds pause_threshold — this preserves intentional pauses within a
    single speaker's speech.

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
        if seg.speaker == current_speaker and gap < pause_threshold:
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


def _format_segments_block(segments: list[Segment]) -> list[str]:
    """Format a list of segments into timecoded markdown lines.

    Low-confidence words (probability < 0.5) are marked with *italics*.
    """
    long_enough = [s for s in segments if s.end - s.start >= const.MIN_SEGMENT_DURATION]
    long_enough = [_mark_low_confidence_words(s) for s in long_enough]
    merged = _merge_consecutive_speakers(long_enough)

    lines: list[str] = []
    for start_time, speaker, text in merged:
        timecode = _format_timecode(start_time)

        if speaker:
            lines.append(f"{timecode} **{speaker}:** {text}")
        else:
            lines.append(f"{timecode} {text}")
        lines.append("")

    return lines


def format_markdown(
    segments: list[Segment],
    session_name: str,
    audio_rel_path: str,
    duration_seconds: float,
    language: str,
    raw_segments: list[Segment] | None = None,
) -> str:
    """Generate markdown with YAML front matter.

    Segments shorter than 1 second are filtered out (VAD artifacts).
    Each segment starts with [HH:MM:SS] timecode.

    When raw_segments is provided, outputs two sections:
    - "## Transcript" with raw (pre-diarization) segments
    - "## Diarized Transcript" with diarized segments
    """
    # Parse date and time from session name (format: YYYY-MM-DD_HH-MM-SS)
    parts = session_name.split("_")
    date_str = parts[0] if parts else session_name
    time_str = parts[1].replace("-", ":") if len(parts) > 1 else "00:00"
    # Only HH:MM for display
    time_display = ":".join(time_str.split(":")[:2])

    duration_hms = _format_duration_hms(duration_seconds)
    duration_human = _format_duration_human(duration_seconds)

    lines = [
        "---",
        f"date: {date_str}",
        f'time: "{time_display}"',
        f'duration: "{duration_hms}"',
        f"language: {language}",
        f'audio: "[[{audio_rel_path}]]"',
        "tags:",
        "  - meeting",
        "  - transcript",
        "---",
        "",
        f"# Meeting {date_str} {time_display}",
        "",
        f"**Duration:** {duration_human} | **Language:** {language}",
        "",
        "---",
        "",
    ]

    if raw_segments is not None:
        lines.append("## Transcript")
        lines.append("")
        lines.extend(_format_segments_block(raw_segments))
        lines.append("---")
        lines.append("")
        lines.append("## Diarized Transcript")
        lines.append("")
        lines.extend(_format_segments_block(segments))
    else:
        lines.extend(_format_segments_block(segments))

    return "\n".join(lines)


def format_live_markdown(
    segments: list[Segment],
    session_name: str,
    language: str,
) -> str:
    """Generate a simplified live markdown transcript (no duration, no raw_segments).

    Updated atomically during recording so the user can open it mid-meeting.
    Replaced by the final polished transcript after recording stops.
    """
    parts = session_name.split("_")
    date_str = parts[0] if parts else session_name
    time_str = parts[1].replace("-", ":") if len(parts) > 1 else "00:00"
    time_display = ":".join(time_str.split(":")[:2])

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
