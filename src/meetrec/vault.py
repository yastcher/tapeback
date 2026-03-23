"""Obsidian vault I/O — save audio and markdown to vault directories."""

import shutil
from pathlib import Path

from meetrec.settings import Settings


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
