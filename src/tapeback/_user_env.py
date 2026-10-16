"""The user's settings file: where `tapeback setup` writes, and where a value comes from.

Settings reads the environment first, then its .env files, a later file winning over an
earlier one, then its defaults. The user's own file is the first of those files, so a
value written there can still lose to the environment or to the .env of the directory
tapeback runs in; `origin` says which.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values, set_key

from tapeback.settings import Settings

ENVIRONMENT = "environment"
DEFAULT = "default"


@dataclass(frozen=True)
class Origin:
    """Where the value Settings uses for a key comes from, and what that overrides."""

    source: str  # ENVIRONMENT, a file as display() shows it, or DEFAULT
    overrides: tuple[str, ...]


def env_files() -> list[Path]:
    """Settings' .env files in the order it reads them; the user's own comes first."""
    files = Settings.model_config.get("env_file")
    if files is None:
        return []
    if isinstance(files, str | Path):
        return [Path(files)]
    return [Path(file) for file in files]


def user_file() -> Path:
    return env_files()[0]


def display(path: Path) -> str:
    """A file as the user knows it: the working directory's one reads ./.env."""
    return str(path) if path.is_absolute() else f"./{path}"


def origin(key: str) -> Origin:
    """Where the value Settings uses for key comes from.

    Names are matched without regard to case, as Settings matches them.
    """
    sources: list[tuple[str, set[str]]] = [(ENVIRONMENT, {name.lower() for name in os.environ})]
    seen: set[Path] = set()
    for path in reversed(env_files()):
        if not path.is_file() or path.resolve() in seen:
            continue
        seen.add(path.resolve())
        sources.append((display(path), {name.lower() for name in dotenv_values(path)}))

    setters = [source for source, names in sources if key.lower() in names]
    if not setters:
        return Origin(DEFAULT, ())
    return Origin(setters[0], tuple(setters[1:]))


def write(values: dict[str, str]) -> list[str]:
    """Set values in the user's file and return the keys whose line changed.

    Other lines stay as they were, the file keeps its mode, and a symlink keeps
    pointing where it did. A new file is created 0600 in a 0700 directory. A value the
    file already holds is not written again. Raises OSError when the file cannot be
    written.
    """
    path = user_file()
    current = dotenv_values(path) if path.is_file() else {}
    changed = [key for key, value in values.items() if current.get(key) != value]
    if changed:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    for key in changed:
        set_key(path, key, values[key], quote_mode="never", follow_symlinks=True)
    return changed
