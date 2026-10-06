#!/usr/bin/env bash
# Build the wheel and the .debs from THIS tree, and install them in clean containers.
#
#     scripts/deb-smoke.sh
#
# The local twin of `.github/workflows/deb-e2e.yml`, on the same images
# (`tests/test_release_scripts.py` fails when the two lists differ). Called by
# `scripts/gate.sh`. Needs docker and nfpm.
set -euo pipefail

cd "$(dirname "$0")/.."

# The images `.github/workflows/deb-e2e.yml` installs into.
SMOKE_IMAGES=(ubuntu:26.10 ubuntu:26.04 ubuntu:24.04 ubuntu:22.04 debian:13 debian:12)

version=$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
# dist/ keeps whatever earlier builds left, and the install below goes by glob.
rm -f dist/tapeback-*.whl dist/tapeback-*.tar.gz dist/*.deb
uv build
scripts/build-deb.sh "dist/tapeback-$version-py3-none-any.whl"

for image in "${SMOKE_IMAGES[@]}"; do
  printf '── %s\n' "$image"
  docker run --rm -v "$PWD/dist:/dist:ro" "$image" bash -c '
    set -e
    apt-get update -qq
    apt-get install -y -qq /dist/tapeback_*.deb >/dev/null
    tapeback --version
    tapeback status
  '
done
