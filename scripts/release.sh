#!/usr/bin/env bash
# Cut a release: name the unreleased section, commit, push, tag. The tag is what
# publishes — `.github/workflows/publish.yml` builds and uploads to PyPI and GitHub.
#
#     scripts/release.sh patch      # 0.9.8 -> 0.9.9
#     scripts/release.sh minor      # 0.9.8 -> 0.10.0
#     scripts/release.sh major      # 0.9.8 -> 1.0.0
#
# **The part is chosen HERE, at the release, because only here is it knowable.**
# Entries accumulate under `## [Unreleased]`, which claims no number; this command
# turns that heading into the version and opens a fresh one above it. A feature
# branch never touches the version.
#
# Run by the maintainer, on main, after the PRs are merged. It is the one thing
# that pushes to main directly: the release commit is mechanical, and there is
# nothing in it to review that the tagged PRs did not already show.
#
# One command, because the ORDER is the whole point and git cannot enforce it: the
# branch must reach the remote before the tag does, or the release names a commit
# main does not contain. Every step is safe to retry — a failure leaves the tree in
# a state this same command can run from, or finishes with a plain `git push`.
set -euo pipefail

PART="${1:-patch}"
case "$PART" in
  patch | minor | major) ;;
  *)
    echo "usage: $0 patch|minor|major   (what this release is)" >&2
    exit 2
    ;;
esac

cd "$(git rev-parse --show-toplevel)"

# A dirty tree means the tag would name a state nobody can check out again.
if [ -n "$(git status --porcelain)" ]; then
  echo "working tree is dirty — commit first" >&2
  exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" != "main" ]; then
  echo "releases are cut from main, not '$BRANCH' — merge the PR first" >&2
  exit 1
fi

# Everything being released must already be on the remote branch: `git push <tag>`
# carries the objects, so a tag on local-only history publishes perfectly well
# while main silently lacks the code that was released.
if ! git merge-base --is-ancestor HEAD "origin/$BRANCH" 2>/dev/null; then
  echo "HEAD is not on origin/$BRANCH — push the branch first, then release" >&2
  echo "(compared against the local remote-tracking ref; 'git fetch' if it is stale)" >&2
  exit 1
fi

# Only a tree `scripts/pre_release_qa.sh` passed may be tagged: the stamp holds the
# tree hash. SKIP_PRERELEASE_QA=1 is for a hotfix that cannot wait for the run; the
# tag still has the publish workflow's lint, types and tests behind it.
stamp="$(git rev-parse --path-format=absolute --git-path pre_release_qa.ok)"
tree="$(git rev-parse 'HEAD^{tree}')"
if [ "${SKIP_PRERELEASE_QA:-}" = "1" ]; then
  echo "SKIP_PRERELEASE_QA=1: releasing without the pre-release run" >&2
elif [ "$(cat "$stamp" 2>/dev/null)" != "$tree" ]; then
  echo "no green pre-release run for this tree — run scripts/pre_release_qa.sh first" >&2
  exit 1
fi

# Bump every manifest, rename `## [Unreleased]`, open a fresh one. Refuses an
# empty section, because the section IS the release notes.
RELEASING=1 scripts/bump-version.sh "$PART"

VERSION="$(grep -m1 -E '^version = "' pyproject.toml | sed -E 's/^version = "(.*)"/\1/')"
TAG="v$VERSION"

if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  echo "$TAG already exists — tags are immutable, bump again instead" >&2
  exit 1
fi

git add -A
git commit -q -m "chore(release): $VERSION"

# The same check the publish workflow runs on the tag, asked here so a bad section
# costs a second instead of a failed pipeline. Its output is the release notes, so
# reading them now IS the review.
echo "── release notes for $TAG ─────────────────────────────────────"
python3 scripts/release_from_tag.py "$TAG"
echo "───────────────────────────────────────────────────────────────"

# Branch before tag, for the reason above. If this push fails nothing has been
# published yet, and the fix is a plain `git push` — the commit is already made.
git push origin "$BRANCH"
git tag "$TAG"
git push origin "$TAG"

echo
echo "released $TAG · the publish workflow builds it · [Unreleased] is open again"
echo "once the GitHub release is up: scripts/aur-publish.sh $VERSION"
