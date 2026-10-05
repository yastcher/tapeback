#!/usr/bin/env bash
# Build the AUR package from THIS tree and run it, inside a clean Arch container.
#
#     docker run --rm -v "$PWD:/src:ro" archlinux:latest /src/scripts/arch-smoke.sh
#
# Called by `.github/workflows/arch-e2e.yml` and by `scripts/pre_release_qa.sh`, so
# the two run one check, not two copies of it. Covers what the .deb smoke cannot:
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
# the tree under test — not the last release. safe.directory: /src belongs to the
# host's user, not to root in here.
git -c safe.directory=/src -C /src archive --prefix="tapeback-$pkgver/" \
  -o "$work/tapeback-$pkgver.tar.gz" HEAD
chown -R builder "$work"

cd "$work"
sudo -u builder makepkg -si --noconfirm

tapeback --version
tapeback status
