#!/usr/bin/env bash
# Update AUR packages after a GitHub release.
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
AUR_DIR="/tmp/aur-tapeback-release"

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

rm -rf "$AUR_DIR"
mkdir -p "$AUR_DIR"

# --- tapeback (main package) ---
echo ""
echo "==> Updating tapeback..."
git clone ssh://aur@aur.archlinux.org/tapeback.git "$AUR_DIR/tapeback"
cp "$REPO_ROOT/packaging/PKGBUILD" "$AUR_DIR/tapeback/PKGBUILD"
sed -i "s/sha256sums=('SKIP')/sha256sums=('$SHA256')/" "$AUR_DIR/tapeback/PKGBUILD"
(cd "$AUR_DIR/tapeback" && makepkg --printsrcinfo > .SRCINFO)
(cd "$AUR_DIR/tapeback" && git add PKGBUILD .SRCINFO && git commit -m "Update to $VERSION" && git push)

# --- tapeback-llm ---
echo ""
echo "==> Updating tapeback-llm..."
git clone ssh://aur@aur.archlinux.org/tapeback-llm.git "$AUR_DIR/tapeback-llm"
cp "$REPO_ROOT/packaging/tapeback-llm/PKGBUILD" "$AUR_DIR/tapeback-llm/PKGBUILD"
cp "$REPO_ROOT/packaging/tapeback-llm/tapeback-llm.install" "$AUR_DIR/tapeback-llm/"
(cd "$AUR_DIR/tapeback-llm" && makepkg --printsrcinfo > .SRCINFO)
(cd "$AUR_DIR/tapeback-llm" && git add PKGBUILD .SRCINFO tapeback-llm.install && git commit -m "Update to $VERSION" && git push)

# --- tapeback-diarize ---
echo ""
echo "==> Updating tapeback-diarize..."
git clone ssh://aur@aur.archlinux.org/tapeback-diarize.git "$AUR_DIR/tapeback-diarize"
cp "$REPO_ROOT/packaging/tapeback-diarize/PKGBUILD" "$AUR_DIR/tapeback-diarize/PKGBUILD"
cp "$REPO_ROOT/packaging/tapeback-diarize/tapeback-diarize.install" "$AUR_DIR/tapeback-diarize/"
(cd "$AUR_DIR/tapeback-diarize" && makepkg --printsrcinfo > .SRCINFO)
(cd "$AUR_DIR/tapeback-diarize" && git add PKGBUILD .SRCINFO tapeback-diarize.install && git commit -m "Update to $VERSION" && git push)

# --- tapeback-tray ---
echo ""
echo "==> Updating tapeback-tray..."
git clone ssh://aur@aur.archlinux.org/tapeback-tray.git "$AUR_DIR/tapeback-tray"
cp "$REPO_ROOT/packaging/tapeback-tray/PKGBUILD" "$AUR_DIR/tapeback-tray/PKGBUILD"
cp "$REPO_ROOT/packaging/tapeback-tray/tapeback-tray.install" "$AUR_DIR/tapeback-tray/"
(cd "$AUR_DIR/tapeback-tray" && makepkg --printsrcinfo > .SRCINFO)
(cd "$AUR_DIR/tapeback-tray" && git add PKGBUILD .SRCINFO tapeback-tray.install && git commit -m "Update to $VERSION" && git push)

# Cleanup
rm -rf "$AUR_DIR"

echo ""
echo "All AUR packages updated to $VERSION."
