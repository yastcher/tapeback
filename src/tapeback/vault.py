"""Obsidian vault I/O — save audio and markdown to vault directories."""

import os
import shutil
from pathlib import Path

from tapeback.recorder import validate_session_name
from tapeback.settings import Settings


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


def _ensure_within_vault(path: Path, vault_path: Path) -> None:
    """Defence-in-depth: reject paths that resolve outside the vault root."""
    if not path.resolve().is_relative_to(vault_path.resolve()):
        raise ValueError(f"Refusing to write outside vault: {path}")


def _atomic_write(path: Path, content: str) -> None:
    """Write file atomically via temp + rename so readers never see a partial file."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def save_audio_to_vault(
    audio_path: Path,
    settings: Settings,
    session_name: str,
) -> Path:
    """Copy audio file to Obsidian vault attachments directory.

    Creates {vault}/{attachments_dir}/ if missing.
    Does not overwrite existing files — adds _1, _2, etc. suffix.
    Returns path to the saved audio file.

    A file that already lives in the attachments directory is returned as-is. Copying
    it would add a `_1.wav` beside itself on every re-run, because the vault never
    overwrites — which is how roughly 2.2 GB of a 5.5 GB audio directory turned out to
    be duplicates of recordings that were simply transcribed more than once.
    """
    validate_session_name(session_name)
    attachments_dir = settings.vault_path / settings.attachments_dir

    resolved = audio_path.resolve()
    if attachments_dir.exists() and resolved.parent == attachments_dir.resolve():
        return audio_path

    attachments_dir.mkdir(parents=True, exist_ok=True)

    audio_dest = _unique_path(attachments_dir / f"{session_name}.wav")
    _ensure_within_vault(audio_dest, settings.vault_path)
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
    validate_session_name(session_name)
    meetings_dir = settings.vault_path / settings.meetings_dir
    meetings_dir.mkdir(parents=True, exist_ok=True)

    md_dest = _unique_path(meetings_dir / f"{session_name}.md")
    _ensure_within_vault(md_dest, settings.vault_path)
    _atomic_write(md_dest, markdown)

    return md_dest


def replace_note(path: Path, markdown: str, settings: Settings) -> None:
    """Write a note over itself, whole: a reader sees the old note or the new one."""
    _ensure_within_vault(path, settings.vault_path)
    _atomic_write(path, markdown)


def save_live_markdown(markdown: str, settings: Settings, session_name: str) -> Path:
    """Write live transcript to vault, overwriting on each update."""
    validate_session_name(session_name)
    meetings_dir = settings.vault_path / settings.meetings_dir
    meetings_dir.mkdir(parents=True, exist_ok=True)

    md_path = meetings_dir / f"{session_name}_live.md"
    _ensure_within_vault(md_path, settings.vault_path)
    _atomic_write(md_path, markdown)
    return md_path


def remove_live_markdown(settings: Settings, session_name: str) -> None:
    """Remove the live transcript file (superseded by final version)."""
    md_path = settings.vault_path / settings.meetings_dir / f"{session_name}_live.md"
    md_path.unlink(missing_ok=True)


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
