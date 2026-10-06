#!/usr/bin/env python3
"""Close `## [Unreleased]` under a version, and open a fresh one above it.

    python3 scripts/changelog_release.py 0.10.0 [--allow-empty]
    python3 scripts/changelog_release.py 0.10.0 --check     # refuse early, write nothing

The closed section is what `scripts/release_from_tag.py` publishes as the GitHub
release notes, so the hint comments that sit under `[Unreleased]` must not travel
into it: they are an instruction to whoever writes next, and a shipped section has
no next.

`--check` exists so the caller can fail before it edits any manifest: an empty
section stops the release, and stopping it after the manifests have been rewritten
is a worse place to stop.
"""

import argparse
import datetime
import pathlib
import re
import sys

UNRELEASED = "## [Unreleased]"
SCAFFOLD = (
    "<!-- New entries go HERE, via scripts/changelog_add.py — never under a dated release. -->",
    "<!-- Subsections in order: Security / Added / Changed / Fixed / Removed / Docs. -->",
)

_SECTION = re.compile(r"^## \[(?P<version>[^\]]+)\]")
_COMMENT = re.compile(r"^\s*<!--.*-->\s*$")


def _is_scaffolding(line: str) -> bool:
    """A line that carries no entry: blank, or one of the hint comments."""
    return not line.strip() or bool(_COMMENT.match(line))


def close_unreleased(
    text: str, *, version: str, today: datetime.date, allow_empty: bool = False
) -> str:
    """Name the open section, and open the next one above it."""
    lines = text.splitlines()
    if any((m := _SECTION.match(line)) and m["version"] == version for line in lines):
        raise SystemExit(f"CHANGELOG.md already has a section for {version}")

    try:
        top = next(i for i, line in enumerate(lines) if line.rstrip() == UNRELEASED)
    except StopIteration:
        raise SystemExit(
            f"CHANGELOG.md has no `{UNRELEASED}` section. It is opened by the previous "
            "release, so this means the last tag was cut by hand instead of through "
            "scripts/release.sh."
        ) from None

    following = next(
        (i for i, line in enumerate(lines) if i > top and _SECTION.match(line)), len(lines)
    )
    body = lines[top + 1 : following]
    if not allow_empty and all(_is_scaffolding(line) for line in body):
        raise SystemExit(
            f"the `{UNRELEASED}` section is empty — nothing to release. "
            "Add entries with scripts/changelog_add.py, or pass --allow-empty."
        )

    # The closed section keeps its entries and loses the scaffolding around
    # them; the fresh one gets the scaffolding and nothing else.
    kept = [line for line in body if not _COMMENT.match(line)]
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()

    opened = [UNRELEASED, "", *SCAFFOLD, ""]
    # `--allow-empty` leaves nothing to carry over, and a heading followed by two
    # blank lines is just untidy.
    closed = [f"## [{version}] — {today.isoformat()}", "", *([*kept, ""] if kept else [])]
    return "\n".join(lines[:top] + opened + closed + lines[following:]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--check", action="store_true", help="validate only, write nothing")
    args = parser.parse_args()

    # Relative to this file, not to `git rev-parse --show-toplevel`: a copy of the
    # tree inside the work tree would send a dry run at the real CHANGELOG.
    path = pathlib.Path(__file__).resolve().parents[1] / "CHANGELOG.md"
    result = close_unreleased(
        path.read_text(encoding="utf-8"),
        version=args.version,
        # The local calendar date: a release cut after midnight is dated the day the
        # person cutting it sees. Spelled through UTC because a bare `now()` is not
        # tz-aware.
        today=datetime.datetime.now(datetime.UTC).astimezone().date(),
        allow_empty=args.allow_empty,
    )
    if args.check:
        sys.stderr.write(f"{UNRELEASED} can be closed as {args.version}\n")
        return
    path.write_text(result, encoding="utf-8")
    sys.stderr.write(f"{UNRELEASED} -> [{args.version}]; a fresh one is open above it\n")


if __name__ == "__main__":
    main()
