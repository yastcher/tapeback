#!/usr/bin/env bash
#
# Bump the project version everywhere it is written and cut the CHANGELOG section.
#
#   scripts/bump-version.sh <patch|minor|major|X.Y.Z> [--allow-empty]
#
# What it does:
#   1. Reads the current version from pyproject.toml — the single source of truth.
#   2. Writes the new one into pyproject.toml, uv.lock, every PKGBUILD in packaging/
#      and README's install commands. The .deb configs read $VERSION at build time
#      and need nothing.
#   3. Rewrites the CHANGELOG: `## [Unreleased]` becomes `## [X.Y.Z] — YYYY-MM-DD`
#      and a fresh empty `## [Unreleased]` goes on top for the next cycle.
#
# It does NOT commit, tag or push — review the diff first. As part of a release it
# is called by `scripts/release.sh`, which does those steps in the right order.
set -euo pipefail

cd "$(dirname "$0")/.."

LEVEL="${1:-}"
ALLOW_EMPTY="${2:-}"

if [[ -z "$LEVEL" ]]; then
  echo "Usage: $0 <patch|minor|major|X.Y.Z> [--allow-empty]" >&2
  exit 1
fi

# Only the first `version = "..."` — that is the [project] one; a pin further
# down must not be mistaken for it.
CURRENT="$(grep -m1 -E '^version = "' pyproject.toml | sed -E 's/^version = "(.*)"/\1/')"
if [[ ! "$CURRENT" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Cannot read a X.Y.Z version from pyproject.toml (got '$CURRENT')" >&2
  exit 1
fi

case "$LEVEL" in
  major | minor | patch)
    IFS=. read -r MA MI PA <<<"$CURRENT"
    case "$LEVEL" in
      major) MA=$((MA + 1)); MI=0; PA=0 ;;
      minor) MI=$((MI + 1)); PA=0 ;;
      patch) PA=$((PA + 1)) ;;
    esac
    NEW="$MA.$MI.$PA"
    ;;
  *)
    if [[ ! "$LEVEL" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      echo "Invalid version/level: '$LEVEL' (use patch|minor|major or X.Y.Z)" >&2
      exit 1
    fi
    NEW="$LEVEL"
    ;;
esac

echo "Version: $CURRENT → $NEW"

# Refuse before touching anything: a missing or empty unreleased section stops the
# release, and stopping it after the manifests have been rewritten is a worse place
# to stop. Same code that does the rewrite below, in --check mode.
python3 scripts/changelog_release.py "$NEW" ${ALLOW_EMPTY:+"$ALLOW_EMPTY"} --check

awk -v new="$NEW" '
  !done && /^version = "/ { print "version = \"" new "\""; done = 1; next }
  { print }
' pyproject.toml > pyproject.toml.tmp && mv pyproject.toml.tmp pyproject.toml
echo "  pyproject.toml → $NEW"

# uv.lock records the project's own version, and CI installs with
# `uv sync --locked`, which refuses a stale one. Edited in place rather than
# through `uv lock`, which would also re-resolve every dependency: anchored on the
# package NAME and applied to the line after it, so a dependency sitting on the
# same number is never touched.
sed -i '/^name = "tapeback"$/{n;s/^version = ".*"$/version = "'"$NEW"'"/}' uv.lock
echo "  uv.lock → $NEW"

# Tracked PKGBUILDs only: packaging/src and packaging/pkg are makepkg's build
# directories and hold copies from whatever release was built there last.
while IFS= read -r pkgbuild; do
  sed -i "s/^pkgver=.*/pkgver=$NEW/" "$pkgbuild"
  echo "  $pkgbuild → $NEW"
done < <(git ls-files -- 'packaging/PKGBUILD' 'packaging/*/PKGBUILD')

# README's install commands download a release by name. The two patterns are the ones
# scripts/release_from_tag.py checks: the release URL and the package file names.
sed -i -E \
  -e "s#(releases/download/v)[0-9]+\.[0-9]+\.[0-9]+/#\1$NEW/#g" \
  -e "s#(tapeback(-[a-z]+)?_)[0-9]+\.[0-9]+\.[0-9]+_#\1${NEW}_#g" \
  README.md
echo "  README.md → $NEW"

python3 scripts/changelog_release.py "$NEW" ${ALLOW_EMPTY:+"$ALLOW_EMPTY"}

# `release.sh` performs these steps itself; printing them there would tell the
# reader to run commands that have already run.
if [[ -z "${RELEASING:-}" ]]; then
  cat <<EOF

Done — a dry run. To publish, revert it and run scripts/release.sh, which does
this bump itself and then commits, pushes and tags in the order publishing needs.
EOF
fi
