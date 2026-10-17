#!/usr/bin/env bash
# Publish every AUR package after a GitHub release: packaging/PKGBUILD (tapeback) and
# each packaging/<name>/PKGBUILD, with the .install file it names. A package whose
# files AUR already holds is skipped, so a rerun for the same version publishes only
# what is new: a package added after the release, say.
# Usage: ./scripts/aur-publish.sh 0.8.1
#
# Prerequisites:
#   - SSH key added to AUR profile
#   - GitHub tag v$VERSION pushed (tarball must exist)
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.8.1"
    exit 1
fi

VERSION="$1"
TARBALL_URL="https://github.com/yastcher/tapeback/archive/v$VERSION.tar.gz"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Tests point this at local repositories.
AUR_REMOTE="${AUR_REMOTE:-ssh://aur@aur.archlinux.org}"
AUR_DIR="$(mktemp -d)"
trap 'rm -rf "$AUR_DIR"' EXIT

# Verify tarball exists
echo "Checking tarball at $TARBALL_URL..."
HTTP_CODE=$(curl -sL -o /dev/null -w '%{http_code}' "$TARBALL_URL")
if [ "$HTTP_CODE" != "200" ]; then
    echo "Error: tarball returned HTTP $HTTP_CODE. Did you push the v$VERSION tag?"
    exit 1
fi

# Compute sha256sum
echo "Computing sha256sum..."
SHA256=$(curl -sL "$TARBALL_URL" | sha256sum | cut -d' ' -f1)
echo "  $SHA256"

for pkgbuild in "$REPO_ROOT/packaging/PKGBUILD" "$REPO_ROOT"/packaging/*/PKGBUILD; do
    pkg="$(sed -n 's/^pkgname=//p' "$pkgbuild")"
    hook="$(sed -n 's/^install=//p' "$pkgbuild")"
    echo ""
    echo "==> Updating $pkg..."
    git clone "$AUR_REMOTE/$pkg.git" "$AUR_DIR/$pkg"
    cp "$pkgbuild" "$AUR_DIR/$pkg/PKGBUILD"
    if [ -n "$hook" ]; then
        cp "$(dirname "$pkgbuild")/$hook" "$AUR_DIR/$pkg/"
    fi
    # The release tarball is the base package's source; the companions have none.
    if [ "$pkgbuild" = "$REPO_ROOT/packaging/PKGBUILD" ]; then
        sed -i "s/sha256sums=('SKIP')/sha256sums=('$SHA256')/" "$AUR_DIR/$pkg/PKGBUILD"
    fi
    (
        cd "$AUR_DIR/$pkg"
        makepkg --printsrcinfo > .SRCINFO
        git add PKGBUILD .SRCINFO ${hook:+"$hook"}
        if git diff --cached --quiet; then
            echo "    AUR already has $pkg in this state, skipped"
        else
            git commit -m "Update to $VERSION"
            # AUR takes master only, and an empty repository has no upstream yet.
            git push origin HEAD:master
        fi
    )
done

echo ""
echo "All AUR packages published for $VERSION."
