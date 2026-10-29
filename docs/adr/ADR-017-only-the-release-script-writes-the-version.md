# ADR-017 — Only the release script writes the version

**Status:** accepted 2026-10-05 (`scripts/release.sh`, `scripts/bump-version.sh`,
`scripts/changelog_add.py`, `scripts/release_from_tag.py`,
`tests/test_release_scripts.py`).

**Context.** The version lives in `pyproject.toml`, `uv.lock`, every PKGBUILD and
README's install commands. Two branches that each opened `## [0.9.9]` with their own
date merged into a conflict over a number neither had the right to pick; README's
commands named 0.9.5 through 0.9.8.

**Decision.**
- Between releases the top CHANGELOG section is `## [Unreleased]`; entries land through
  `scripts/changelog_add.py`.
- A branch never touches the version. `scripts/release.sh` writes it into every copy at
  once, closes `[Unreleased]`, commits, pushes and tags.
- The tag is checked against every copy before anything is published.

**Why.** A number claimed by a branch is a merge conflict waiting; a copy updated by
hand is the one forgotten.

**Consequences.** A new copy of the version must be added to both `bump-version.sh` and
`release_from_tag.py`; a test runs them against each other.
