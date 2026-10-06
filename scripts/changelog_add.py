#!/usr/bin/env python3
"""Append a CHANGELOG entry to `## [Unreleased]` — never to a released section.

Adding an entry by hand means locating a subsection by its heading and deciding
whether the top section is still open. Both are easy to get wrong and neither is
visible in review: a duplicate `### Added` reads exactly like a correct one, and
an entry under a released version reads as if that version gained features after
it shipped. Choosing the version number in a feature branch is worse still: two
branches that each opened `## [0.9.9]` with their own date merged into a conflict
over a number neither of them had the right to pick.

So there is nothing left to locate or decide: entries go into the FIRST section of
the file, which between releases is always `## [Unreleased]`; a release
(`scripts/release.sh`) names it and opens a fresh one above. The subsection is
created when missing, in the documented order.

    scripts/changelog_add.py Fixed "- **Topic.** What was wrong, what is right now."
    scripts/changelog_add.py Docs --file entry.md
    scripts/changelog_add.py Fixed --top "- **Topic.** ..."   # above the infrastructure lines

The text is passed through verbatim and must already be a markdown bullet: the
entry is prose someone wrote on purpose, and reformatting it here would be a
second opinion nobody asked for. Everything already in the file is left alone —
the CHANGELOG is append-only.
"""

import argparse
import pathlib
import re
import subprocess
import sys

# The order AGENTS.md documents: Security first, because it is the entry a user
# most needs to see. A new subsection is inserted so the section stays in it.
SUBSECTIONS = ("Security", "Added", "Changed", "Fixed", "Removed", "Docs")

ROOT = pathlib.Path(__file__).resolve().parents[1]

_SECTION = re.compile(r"^## \[(?P<version>[^\]]+)\]")
_SUBSECTION = re.compile(r"^### (?P<name>.+?)\s*$")


def released_versions() -> set[str]:
    """Versions a tag has already closed. Their sections are history, not a draft."""
    try:
        out = subprocess.run(
            ["git", "tag", "--list", "v*"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    return {line.strip().removeprefix("v") for line in out.splitlines() if line.strip()}


def add_entry(
    text: str, *, subsection: str, entry: str, released: set[str], first: bool = False
) -> str:
    """Put `entry` in `subsection` of the topmost release section.

    At the end, unless `first` — entries are ordered by user impact, so an
    infrastructure line lands correctly by being appended while a user-facing one
    under existing infrastructure lines has to be raised.

    `released` is injected rather than read here so the decision stays a pure
    function of the file and the tags — and so its refusal can be tested without
    a repository that happens to carry the right tag.
    """
    lines = text.splitlines()
    sections = [(i, m) for i, line in enumerate(lines) if (m := _SECTION.match(line))]
    if not sections:
        raise SystemExit("CHANGELOG.md has no release sections at all")
    top, heading = sections[0]
    following = sections[1][0] if len(sections) > 1 else len(lines)

    version = heading.group("version")
    if version in released:
        raise SystemExit(
            f"the top section is [{version}], which tag v{version} has already closed. "
            "A release opens the next section itself, so this means a tag was made by "
            "hand instead of through scripts/release.sh. Fix the release, do not edit "
            "the CHANGELOG around it."
        )

    body = lines[top + 1 : following]
    present = [
        (i, m.group("name")) for i, line in enumerate(body) if (m := _SUBSECTION.match(line))
    ]
    bullet = entry.rstrip("\n")

    # This file writes the bullets right under their heading, as one tight list: a
    # blank line between two bullets would make the list loose, which markdown
    # renders differently.
    for i, name in present:
        if name == subsection:
            if first:
                body.insert(i + 1, bullet)
            else:
                # The end of this subsection: the next heading, or the end of the
                # section — minus the blank lines that separate them.
                end = next((j for j, _ in present if j > i), len(body))
                while end > i + 1 and not body[end - 1].strip():
                    end -= 1
                body.insert(end, bullet)
            break
    else:
        # Missing: insert it after the last subsection that outranks it, before
        # the first one it outranks, or at the end if it outranks nothing there.
        rank = SUBSECTIONS.index(subsection)
        earlier = [
            i for i, name in present if name in SUBSECTIONS and SUBSECTIONS.index(name) < rank
        ]
        at = len(body)
        if earlier:
            at = next((j for j, _ in present if j > earlier[-1]), len(body))
        elif present:
            at = present[0][0]
        while at > 0 and not body[at - 1].strip():
            at -= 1
        body[at:at] = ["", f"### {subsection}", bullet]

    return "\n".join(lines[: top + 1] + body + lines[following:]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subsection", choices=SUBSECTIONS)
    parser.add_argument("entry", nargs="?", default="", help="the markdown bullet, verbatim")
    parser.add_argument("--file", help="read the bullet from this file instead")
    parser.add_argument(
        "--top",
        action="store_true",
        help="put it first in the subsection — for a user-facing entry above infrastructure ones",
    )
    args = parser.parse_args()

    entry = pathlib.Path(args.file).read_text(encoding="utf-8") if args.file else args.entry
    if not entry.strip():
        raise SystemExit("nothing to add")

    # Relative to this file, for the reason spelled out in changelog_release.py.
    path = ROOT / "CHANGELOG.md"
    path.write_text(
        add_entry(
            path.read_text(encoding="utf-8"),
            subsection=args.subsection,
            entry=entry,
            released=released_versions(),
            first=args.top,
        ),
        encoding="utf-8",
    )
    sys.stderr.write(f"added to ### {args.subsection}\n")


if __name__ == "__main__":
    main()
