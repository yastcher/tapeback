#!/usr/bin/env bash
# Everything slower than a pull request can afford, run once before a release.
# `scripts/release.sh` refuses to tag a tree this script has not passed.
#
#   scripts/pre_release_qa.sh
#
#   1. the gate        — scripts/gate.sh --all: the CI checks, then every packaging
#                        stand (.deb on the deb-e2e images, the Arch package)
#                        whatever the diff — on main there is no diff to go by
#   2. e2e quality     — tests/test_e2e_quality.py on real audio and real models.
#                        Needs the recordings in tests/data/, HF_TOKEN for
#                        pyannote, and ideally a GPU; takes minutes.
#
# Needs docker and nfpm. Writes the tree hash into the git directory on success;
# `release.sh` compares it with the tree it is about to tag.
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"

stamp=$(git rev-parse --path-format=absolute --git-path pre_release_qa.ok)
tree=$(git rev-parse 'HEAD^{tree}')

if [ -n "$(git status --porcelain)" ]; then
  echo "pre_release_qa: the tree is dirty — a stamp would name a state nobody can check out" >&2
  exit 1
fi
rm -f "$stamp"

step() { printf '\n\033[1m── %s\033[0m\n' "$1"; }

step "gate"
scripts/gate.sh --all

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

echo "$tree" > "$stamp"
printf '\n\033[32mpre_release_qa: passed for tree %s\033[0m\n' "$tree"
