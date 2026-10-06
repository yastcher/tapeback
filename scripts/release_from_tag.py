#!/usr/bin/env python3
"""Check that a `vX.Y.Z` tag agrees with the tree, and print its release notes.

The tag is the release decision, and therefore the last moment anything can be
checked for free. `publish.yml` runs this before anything is built or uploaded:

* **the tag must name the version `pyproject.toml` declares.** The tag is the one
  copy no tool writes — a person types it — so it is the copy worth verifying.
* **`uv.lock` must agree.** It records the project's own version, and CI installs
  with `uv sync --locked`, which refuses a stale lockfile — better here, with a
  plain message, than as a failed install halfway through the publish job.
* **every PKGBUILD must agree.** `scripts/bump-version.sh` writes them; a release
  that skipped it ships AUR packages pointing at the previous tarball.
* **README's install commands must agree.** They download a release by name, so a
  stale one installs an old version for everyone who copies them — they said 0.9.5
  through three more releases.
* **the CHANGELOG must have a non-empty section for it.** That section IS the
  release notes, and notes written after the release never get written.

    python3 scripts/release_from_tag.py v0.10.0
    python3 scripts/release_from_tag.py v0.10.0 --notes-file notes.md

Exits non-zero with a plain message on any mismatch. Runs on bare python3: no
project dependencies, so it can run before `uv sync`.
"""

import argparse
import pathlib
import re
import sys
import tomllib

_TAG = re.compile(r"^v(\d+\.\d+\.\d+)$")
_LOCKED = re.compile(r'^name = "tapeback"\nversion = "(?P<version>[^"]+)"', re.M)
_PKGVER = re.compile(r"^pkgver=(?P<version>\S+)$", re.M)
# The version in README's install commands: the release URL and the package file names.
# The same two patterns scripts/bump-version.sh rewrites.
_README_VERSION = re.compile(
    r"releases/download/v(?P<url>\d+\.\d+\.\d+)/|tapeback(?:-[a-z]+)?_(?P<file>\d+\.\d+\.\d+)_"
)


def version_of_tag(tag: str) -> str:
    """`v0.10.0` -> `0.10.0`. Anything else is not a release tag and must not publish."""
    match = _TAG.match(tag)
    if match is None:
        raise SystemExit(f"not a release tag: {tag!r} (expected vX.Y.Z)")
    return match.group(1)


def declared_version(root: pathlib.Path) -> str:
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def locked_version(root: pathlib.Path) -> str:
    """The version `uv.lock` records for tapeback itself — anchored on the package
    name, so a dependency that happens to sit on the same number is never read."""
    match = _LOCKED.search((root / "uv.lock").read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit("no tapeback version in uv.lock")
    return match.group("version")


def pkgbuild_versions(root: pathlib.Path) -> dict[str, str]:
    """`pkgver` of every PKGBUILD in packaging/, keyed by its path relative to root."""
    found = {}
    for path in sorted([root / "packaging" / "PKGBUILD", *root.glob("packaging/*/PKGBUILD")]):
        if not path.exists():
            continue
        match = _PKGVER.search(path.read_text(encoding="utf-8"))
        if match is None:
            raise SystemExit(f"no pkgver in {path.relative_to(root)}")
        found[str(path.relative_to(root))] = match.group("version")
    return found


def readme_versions(root: pathlib.Path) -> set[str]:
    """Every version README's install commands name; empty if it names none."""
    text = (root / "README.md").read_text(encoding="utf-8")
    return {m["url"] or m["file"] for m in _README_VERSION.finditer(text)}


def changelog_notes(root: pathlib.Path, version: str) -> str:
    """The `## [X.Y.Z]` section, verbatim, without its heading or date."""
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = re.compile(rf"^## \[{re.escape(version)}\].*$", re.M)
    start = heading.search(text)
    if start is None:
        raise SystemExit(
            f"CHANGELOG.md has no section for {version} — "
            "cut it with scripts/release.sh instead of tagging by hand"
        )
    rest = text[start.end() :]
    end = re.search(r"^## ", rest, re.M)
    body = (rest[: end.start()] if end else rest).strip()
    if not body:
        raise SystemExit(f"the CHANGELOG section for {version} is empty")
    return body


def check(root: pathlib.Path, tag: str) -> str:
    """Every mismatch between the tag and the tree, refused; the release notes otherwise."""
    version = version_of_tag(tag)
    declared = declared_version(root)
    if declared != version:
        raise SystemExit(
            f"tag {tag} disagrees with pyproject.toml ({declared}) — "
            "cut releases with scripts/release.sh, not by hand"
        )
    locked = locked_version(root)
    if locked != version:
        raise SystemExit(f"tag {tag} disagrees with uv.lock ({locked}) — run `uv lock`")
    stale = {path: found for path, found in pkgbuild_versions(root).items() if found != version}
    readme = readme_versions(root)
    if readme - {version}:
        stale["README.md"] = ", ".join(sorted(readme))
    if stale:
        listed = ", ".join(f"{path} ({found})" for path, found in stale.items())
        raise SystemExit(f"tag {tag} disagrees with {listed} — run scripts/bump-version.sh")
    return changelog_notes(root, version)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("--notes-file", default="")
    parser.add_argument("--root", default="")
    args = parser.parse_args()

    root = pathlib.Path(args.root) if args.root else pathlib.Path(__file__).resolve().parents[1]
    notes = check(root, args.tag)
    if args.notes_file:
        pathlib.Path(args.notes_file).write_text(notes + "\n", encoding="utf-8")
    else:
        sys.stdout.write(notes + "\n")


if __name__ == "__main__":
    main()
