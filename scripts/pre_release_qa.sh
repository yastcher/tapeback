#!/usr/bin/env bash
# Everything slower than a pull request can afford, run once before a release.
# `scripts/release.sh` refuses to tag a tree this script has not passed.
#
#   scripts/pre_release_qa.sh
#
#   1. the gate        — scripts/gate.sh --all: the CI checks, then every packaging
#                        stand (.deb on the deb-e2e images, the Arch package)
#                        whatever the diff — on main there is no diff to go by
#   2. CRAP            — scripts/crap.py over the coverage the gate's pytest wrote
#   3. e2e quality     — tests/test_e2e_quality.py on real audio and real models.
#                        Needs the recordings in tests/data/, HF_TOKEN for
#                        pyannote, and ideally a GPU; takes minutes.
#   4. mutation        — mutmut over the slice in [tool.mutmut], from scratch, in a
#                        memory-capped scope; about an hour with two workers
#
# Needs docker, nfpm and a systemd user session. Writes the tree hash into the git
# directory on success; `release.sh` compares it with the tree it is about to tag.
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

# Scratch under the repository's tmp/, never /tmp: on some machines /tmp is a small
# tmpfs with a per-user quota.
mkdir -p tmp

step "CRAP"
# Complexity meeting missing tests, per function. The limit sat just above the worst
# function when it was set, and only comes down: lower it as that function improves,
# never raise it to let a change in.
uv run coverage json -q -o tmp/coverage.json
uv run python scripts/crap.py tmp/coverage.json --max 29

step "e2e quality"
# Every e2e test skips itself when something it needs is missing, and a run made
# only of skips exits 0. Here a skip is a failure: the report is read for the
# count, not the exit code alone.
report=$(mktemp tmp/pre_release_qa.XXXXXX)
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

step "mutation"
# Every function of the slice in [tool.mutmut], changed in small ways, against the tests
# that cover it (ADR-019); survivors and untested mutants only come down. From scratch:
# mutmut keeps a copied file unmutated when only the slice changed. Capped in memory: a
# mutant can make code allocate without end (one that dropped a loop's step passed 2 GB
# on a minute of audio), and the cap kills that worker, counted as died, instead of the
# desktop. The cap allows 1.2 GB for mutmut and its fork server and 0.5 GB per worker;
# with two workers the run peaked at 1.5 GB.
children=${MUTMUT_CHILDREN:-2}
if ! [[ $children =~ ^[1-9][0-9]*$ ]]; then
  echo "pre_release_qa: MUTMUT_CHILDREN must be a positive whole number, not '$children'" >&2
  exit 1
fi
rm -rf mutants
systemd-run --user --scope --quiet -p MemoryMax=$((1200 + 500 * children))M \
  -p MemorySwapMax=0 -p OOMPolicy=continue uv run mutmut run --max-children "$children"
uv run mutmut export-cicd-stats
uv run python scripts/mutation_ratchet.py mutants/mutmut-cicd-stats.json \
  --survived-max 23 --no-tests-max 0

echo "$tree" > "$stamp"
printf '\n\033[32mpre_release_qa: passed for tree %s\033[0m\n' "$tree"
