import shutil
from pathlib import Path

from meetrec.settings import Settings
from meetrec.transcriber import Segment


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


def format_markdown(
    segments: list[Segment],
    session_name: str,
    audio_rel_path: str,
    duration_seconds: float,
    language: str,
) -> str:
    """Generate markdown with YAML front matter.

    Segments shorter than 1 second are filtered out (VAD artifacts).
    Each segment starts with [HH:MM:SS] timecode.
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

    for segment in segments:
        # Skip segments shorter than 1 second (VAD artifacts)
        if segment.end - segment.start < 1.0:
            continue

        timecode = _format_timecode(segment.start)

        # Include speaker label if present (phase 2)
        if segment.speaker:
            lines.append(f"{timecode} **{segment.speaker}:** {segment.text}")
        else:
            lines.append(f"{timecode} {segment.text}")
        lines.append("")

    return "\n".join(lines)


def _unique_path(path: Path) -> Path:
    """Return a unique path by adding _1, _2, etc. suffix if file exists."""
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    counter = 1
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def save_audio_to_vault(
    audio_path: Path,
    settings: Settings,
    session_name: str,
) -> Path:
    """Copy audio file to Obsidian vault attachments directory.

    Creates {vault}/{attachments_dir}/ if missing.
    Does not overwrite existing files — adds _1, _2, etc. suffix.
    Returns path to the saved audio file.
    """
    attachments_dir = settings.vault_path / settings.attachments_dir
    attachments_dir.mkdir(parents=True, exist_ok=True)

    audio_dest = _unique_path(attachments_dir / f"{session_name}.wav")
    shutil.copy2(audio_path, audio_dest)

    return audio_dest


def save_markdown_to_vault(
    markdown: str,
    settings: Settings,
    session_name: str,
) -> Path:
    """Write markdown transcript to Obsidian vault meetings directory.

    Creates {vault}/{meetings_dir}/ if missing.
    Does not overwrite existing files — adds _1, _2, etc. suffix.
    Returns path to the markdown file.
    """
    meetings_dir = settings.vault_path / settings.meetings_dir
    meetings_dir.mkdir(parents=True, exist_ok=True)

    md_dest = _unique_path(meetings_dir / f"{session_name}.md")
    md_dest.write_text(markdown, encoding="utf-8")

    return md_dest


def save_to_vault(
    markdown: str,
    stereo_wav: Path,
    settings: Settings,
    session_name: str,
) -> Path:
    """Save markdown and audio to Obsidian vault (legacy convenience wrapper).

    Does not overwrite existing files — adds _1, _2, etc. suffix.
    """
    save_audio_to_vault(stereo_wav, settings, session_name)
    return save_markdown_to_vault(markdown, settings, session_name)
