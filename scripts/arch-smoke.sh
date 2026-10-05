#!/usr/bin/env bash
# Build the AUR package from THIS tree and run it, inside a clean Arch container.
#
#     docker run --rm -v "$PWD:/src:ro" archlinux:latest /src/scripts/arch-smoke.sh
#
# Called by `.github/workflows/arch-e2e.yml` and by `scripts/gate.sh`, so the two
# run one check, not two copies of it. Covers what the .deb smoke cannot:
# packaging/PKGBUILD, and tapeback on Arch's system Python rather than a bundled one.
set -euo pipefail

pacman -Syu --noconfirm --needed base-devel git sudo >/dev/null

# makepkg refuses to run as root.
useradd -m builder
echo 'builder ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/builder

work=/home/builder/pkg
install -d "$work"
cp /src/packaging/PKGBUILD "$work/"
pkgver=$(sed -n 's/^pkgver=//p' "$work/PKGBUILD")

# The PKGBUILD downloads the tarball of tag v$pkgver. makepkg uses a source file
# that already sits next to the PKGBUILD instead of downloading it, so this builds
# the tree under test — not the last release. The working tree, not HEAD: the gate
# runs before the commit. Tracked and new files, never ignored ones (.venv, dist);
# a tracked file deleted in the working tree is left out. safe.directory: /src
# belongs to the host's user, not to root in here.
cd /src
git -c safe.directory=/src ls-files -z --cached --others --exclude-standard \
  | while IFS= read -r -d '' file; do [ -e "$file" ] && printf '%s\0' "$file"; done \
  | tar --null -T - -czf "$work/tapeback-$pkgver.tar.gz" --transform "s,^,tapeback-$pkgver/,"
chown -R builder "$work"

cd "$work"
sudo -u builder makepkg -si --noconfirm

tapeback --version
tapeback status
