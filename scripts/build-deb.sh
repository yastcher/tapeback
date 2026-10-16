#!/usr/bin/env bash
# Build Debian/Ubuntu .deb packages for tapeback via nfpm.
#
# The .deb bundles a standalone Python interpreter (from astral-sh's
# python-build-standalone, downloaded directly by tag) so the package does
# NOT depend on the target system having any specific Python version.
# This avoids the python3.13 / python3.14 minor-version coupling that bites us
# because faster-whisper / ctranslate2 / pyav wheels include compiled .so files
# tagged for a specific cpython minor.
#
# Layout inside the .deb:
#   /opt/tapeback/python/         standalone CPython (~75 MB, RPATH-relative)
#   /opt/tapeback/venv/           virtualenv with tapeback + deps installed
#   /usr/bin/tapeback             wrapper that exec's /opt/tapeback/venv/bin/tapeback
#   /opt/tapeback/llama.cpp/      tapeback-llm-local: llama.cpp's server, linked into venv/bin
#
# Usage:
#   ./scripts/build-deb.sh                 # install tapeback from PyPI
#   ./scripts/build-deb.sh dist/*.whl      # install from a local wheel (faster, no network)
#
# Smoke-test the result:
#   docker run --rm -v $PWD/dist:/dist ubuntu:26.04 bash -c \
#     'apt-get update -qq && apt-get install -y -qq /dist/tapeback_*.deb && tapeback --version'
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

VERSION="$(grep -m1 '^version' pyproject.toml | cut -d'"' -f2)"
if [ -z "$VERSION" ]; then
    echo "Error: could not read version from pyproject.toml" >&2
    exit 1
fi
export VERSION

# Pinned python-build-standalone release. Bump deliberately when a new release
# brings security fixes or a desired patch level. Lives outside any external
# resolver so the build is byte-deterministic.
# Browse: https://github.com/astral-sh/python-build-standalone/releases
PYBS_DATE="${PYBS_DATE:-20260510}"
PYBS_VERSION="${PYBS_VERSION:-3.13.13}"
PYBS_TARBALL="cpython-${PYBS_VERSION}+${PYBS_DATE}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
PYBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PYBS_DATE}/${PYBS_TARBALL}"

# Pinned llama.cpp release for tapeback-llm-local. The Vulkan build: it runs on Ubuntu
# 22.04+ and Debian 12+, and reaches an NVIDIA GPU through the driver's Vulkan ICD; the
# CUDA builds need Ubuntu 24.04+ and weigh 25 times more. llama.cpp publishes no
# SHA256SUMS; the checksum is the asset digest GitHub's release API reports.
# Browse: https://github.com/ggml-org/llama.cpp/releases
LLAMA_CPP_BUILD="${LLAMA_CPP_BUILD:-b11669}"
LLAMA_CPP_SHA256="${LLAMA_CPP_SHA256:-e8709e5ba6c1b093dea7f3de57f4aba842f9a188fba6b465782d65d6fcfa4d9f}"
LLAMA_CPP_TARBALL="llama-${LLAMA_CPP_BUILD}-bin-ubuntu-vulkan-x64.tar.gz"
LLAMA_CPP_URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP_BUILD}/${LLAMA_CPP_TARBALL}"

BUILD_ROOT="$REPO_ROOT/build/opt/tapeback"
LLAMA_DEST="$REPO_ROOT/build/llama.cpp"
PY_DEST="$BUILD_ROOT/python"
VENV_DEST="$BUILD_ROOT/venv"
DIST_DIR="$REPO_ROOT/dist"

if ! command -v nfpm >/dev/null 2>&1; then
    echo "Error: nfpm is not installed. See https://nfpm.goreleaser.com/install/" >&2
    exit 1
fi

WHEEL_SPEC="${1:-tapeback==$VERSION}"

echo "==> Building tapeback $VERSION .deb packages"
echo "    Python:     cpython-${PYBS_VERSION}+${PYBS_DATE} (python-build-standalone)"
echo "    llama.cpp:  ${LLAMA_CPP_BUILD} (Vulkan)"
echo "    Wheel/spec: $WHEEL_SPEC"

# Clean previous venv build but keep dist/ (wheel artifact may live there)
rm -rf "$REPO_ROOT/build"
mkdir -p "$BUILD_ROOT" "$DIST_DIR"

echo "==> Downloading and extracting standalone Python"
# Tarball extracts to ./python/{bin,lib,include,share} — directly into BUILD_ROOT.
curl -fsSL "$PYBS_URL" | tar -xz -C "$BUILD_ROOT"
if [ ! -x "$PY_DEST/bin/python3.13" ]; then
    echo "Error: $PY_DEST/bin/python3.13 missing after extract." >&2
    echo "       Tarball $PYBS_TARBALL may have unexpected layout." >&2
    ls -la "$BUILD_ROOT" || true
    exit 1
fi
echo "    Extracted: $("$PY_DEST/bin/python3.13" --version) at $PY_DEST"

# Trim headers — not needed at runtime, ~5 MB. Stdlib + libpython stay
# (they're what makes this self-contained).
rm -rf "$PY_DEST/include"
find "$PY_DEST" -type d -name __pycache__ -prune -exec rm -rf {} +

echo "==> Creating venv at $VENV_DEST"
"$PY_DEST/bin/python3.13" -m venv "$VENV_DEST"
"$VENV_DEST/bin/pip" install --quiet --upgrade pip
"$VENV_DEST/bin/pip" install --quiet "$WHEEL_SPEC"

echo "==> Rewriting paths for /opt/tapeback install location"
# Replace venv's python symlinks with relative ones so they resolve correctly
# from either the build dir or /opt/tapeback after install.
ln -sf "../../python/bin/python3.13" "$VENV_DEST/bin/python"
ln -sf "python" "$VENV_DEST/bin/python3"
ln -sf "python" "$VENV_DEST/bin/python3.13"

# Rewrite shebangs in venv/bin/* from build path to /opt/tapeback/venv/bin/python.
find "$VENV_DEST/bin" -type f -exec \
    sed -i "1s|^#!.*/build/opt/tapeback/venv/bin/python[^ ]*|#!/opt/tapeback/venv/bin/python|" {} \;

# pyvenv.cfg points at the python that built it — rewrite to bundled path.
sed -i "s|$PY_DEST|/opt/tapeback/python|g" "$VENV_DEST/pyvenv.cfg"

echo "==> Trimming bytecode caches"
find "$VENV_DEST" -type d -name __pycache__ -prune -exec rm -rf {} +

echo "==> Downloading llama.cpp ${LLAMA_CPP_BUILD} for tapeback-llm-local"
mkdir -p "$LLAMA_DEST"
curl -fsSL -o "$REPO_ROOT/build/$LLAMA_CPP_TARBALL" "$LLAMA_CPP_URL"
echo "$LLAMA_CPP_SHA256  $REPO_ROOT/build/$LLAMA_CPP_TARBALL" | sha256sum --check --quiet
# One flat directory: the server, its libraries and the backends it loads at run time,
# all found through RUNPATH=$ORIGIN.
tar -xzf "$REPO_ROOT/build/$LLAMA_CPP_TARBALL" -C "$LLAMA_DEST" --strip-components=1
rm "$REPO_ROOT/build/$LLAMA_CPP_TARBALL"
if [ ! -x "$LLAMA_DEST/llama-server" ] || [ ! -f "$LLAMA_DEST/LICENSE" ]; then
    echo "Error: llama-server or its LICENSE missing after extract." >&2
    ls -la "$LLAMA_DEST" || true
    exit 1
fi

echo "==> Running nfpm"
for cfg in \
    packaging/deb/nfpm-tapeback.yaml \
    packaging/deb/nfpm-tapeback-diarize.yaml \
    packaging/deb/nfpm-tapeback-cuda.yaml \
    packaging/deb/nfpm-tapeback-llm.yaml \
    packaging/deb/nfpm-tapeback-llm-local.yaml \
    packaging/deb/nfpm-tapeback-tray.yaml
do
    echo "    - $cfg"
    nfpm package -f "$cfg" -p deb -t "$DIST_DIR"
done

echo ""
echo "Done. Artifacts in $DIST_DIR:"
ls -1 "$DIST_DIR"/tapeback*.deb 2>/dev/null || true
