#!/usr/bin/env bash
# Everything slower than a pull request can afford, run once before a release.
# `scripts/release.sh` refuses to tag a tree this script has not passed.
#
#   scripts/pre_release_qa.sh
#
# Cheapest first, so a failure costs as little as possible:
#   1. the gate        — scripts/gate.sh, the CI job step for step
#   2. e2e quality     — tests/test_e2e_quality.py on real audio and real models.
#                        Needs the recordings in tests/data/, HF_TOKEN for
#                        pyannote, and ideally a GPU; takes minutes.
#   3. packages        — uv build, scripts/build-deb.sh
#   4. .deb smoke      — the deb-e2e workflow's install test, on the same images
#   5. Arch smoke      — scripts/arch-smoke.sh, the arch-e2e workflow's test
#
# Needs docker and nfpm. Writes the tree hash into the git directory on success;
# `release.sh` compares it with the tree it is about to tag.
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"

# The images `.github/workflows/deb-e2e.yml` installs into.
SMOKE_IMAGES=(ubuntu:26.10 ubuntu:26.04 ubuntu:24.04 ubuntu:22.04 debian:13 debian:12)

stamp=$(git rev-parse --path-format=absolute --git-path pre_release_qa.ok)
tree=$(git rev-parse 'HEAD^{tree}')

if [ -n "$(git status --porcelain)" ]; then
  echo "pre_release_qa: the tree is dirty — a stamp would name a state nobody can check out" >&2
  exit 1
fi
rm -f "$stamp"

step() { printf '\n\033[1m── %s\033[0m\n' "$1"; }

step "gate"
scripts/gate.sh

step "e2e quality"
# Every e2e test skips itself when something it needs is missing, and a run made
# only of skips exits 0. Here a skip is a failure: the report is read for the
# count, not the exit code alone.
report=$(mktemp)
trap 'rm -f "$report"' EXIT
TAPEBACK_RUN_E2E=1 uv run pytest tests/test_e2e_quality.py --no-cov -rs --junitxml="$report"
python3 - "$report" <<'EOF'
import sys
import xml.etree.ElementTree as ET

suite = ET.parse(sys.argv[1]).getroot()
suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
tests, skipped = int(suite.get("tests", 0)), int(suite.get("skipped", 0))
if tests == 0 or skipped:
    sys.exit(f"pre_release_qa: e2e ran {tests - skipped} of {tests} tests — see the skip reasons above")
EOF

step "packages"
version=$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
# dist/ keeps whatever earlier builds left, and the smoke test installs by glob.
rm -f dist/tapeback-*.whl dist/tapeback-*.tar.gz dist/*.deb
uv build
./scripts/build-deb.sh "dist/tapeback-$version-py3-none-any.whl"

step ".deb smoke"
for image in "${SMOKE_IMAGES[@]}"; do
  printf '%s\n' "$image"
  docker run --rm -v "$root/dist:/dist:ro" "$image" bash -c '
    set -e
    apt-get update -qq
    apt-get install -y -qq /dist/tapeback_*.deb >/dev/null
    tapeback --version
    tapeback status
  '
done

step "Arch smoke"
docker run --rm -v "$root:/src:ro" archlinux:latest /src/scripts/arch-smoke.sh

echo "$tree" > "$stamp"
printf '\n\033[32mpre_release_qa: passed for tree %s\033[0m\n' "$tree"
