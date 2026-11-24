#!/usr/bin/env bash
# Mutation testing (ADR-019): every function of the slice in [tool.mutmut], changed in
# small ways, against the tests that cover it.
#
#   scripts/mutation.sh                             # the slice, then the ratchet
#   scripts/mutation.sh 'tapeback.speaker_merge.*'  # only the mutants these names match
#
# Always from scratch: mutmut keeps a copied file unmutated when only the slice changed,
# and keeps in its copy a test the tree has since deleted, whose import then fails the
# collection of every test. A name selects mutants as `mutmut run` does (fnmatch over
# `module.x_function__mutmut_N`); such a run lists the selected mutants the tests did
# not kill, fails when there is one, and skips the ratchet, which counts the whole slice.
#
# Capped in memory: a mutant can make code allocate without end (one that dropped a
# loop's step passed 2 GB on a minute of audio), and the cap kills that worker, counted
# as died, instead of the desktop. The cap allows 1.2 GB for mutmut and its fork server
# and 0.5 GB per worker; with two workers the run peaked at 1.5 GB.
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"

children=${MUTMUT_CHILDREN:-2}
if ! [[ $children =~ ^[1-9][0-9]*$ ]]; then
  echo "mutation: MUTMUT_CHILDREN must be a positive whole number, not '$children'" >&2
  exit 1
fi
rm -rf mutants
systemd-run --user --scope --quiet -p MemoryMax=$((1200 + 500 * children))M \
  -p MemorySwapMax=0 -p OOMPolicy=continue uv run mutmut run --max-children "$children" "$@"
uv run mutmut export-cicd-stats
if [ $# -gt 0 ]; then
  results=$(uv run mutmut results)
  # The rest of the slice is listed as not checked: it was not run.
  missed=$(grep -v ': not checked$' <<<"$results" || true)
  if [ -n "$missed" ]; then
    echo "$missed"
    exit 1
  fi
  echo "mutation: the tests killed every selected mutant"
  exit 0
fi
# Survivors and untested mutants only come down.
uv run python scripts/mutation_ratchet.py mutants/mutmut-cicd-stats.json \
  --survived-max 23 --no-tests-max 0
