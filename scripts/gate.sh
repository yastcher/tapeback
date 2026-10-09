#!/usr/bin/env bash
# The gate CI runs, run here first.
#
#     scripts/gate.sh          # what CI would run on this branch
#     scripts/gate.sh --all    # every stand regardless of the diff (pre_release_qa.sh)
#
# Every step below is a step of the `lint-and-test` job in `.github/workflows/ci.yml`,
# in the same order and with the same command — `tests/test_gate.py` fails when the
# two drift apart. One deliberate difference: CI installs with `uv sync --locked`,
# which here would also strip whatever extras the developer has installed, so the
# gate asks the only question that flag answers — is uv.lock current? — with
# `uv lock --locked`, which changes nothing.
#
# The packaging stands — `.github/workflows/deb-e2e.yml` and `arch-e2e.yml` — run on
# a pull request only when it touches their `paths:`. The gate asks the branch the
# same question (committed or not, against origin/main) with the same lists, and the
# test holds those lists to the workflows. They need docker and nfpm and take minutes.
#
# The verdict is the exit code and the last line, never the output in between:
# reading "563 passed" is not the same as reading the coverage gate that refused
# the run two lines further down. Nothing here greps output.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The `paths:` of the packaging workflows, plus the local script each stand runs.
DEB_PATHS=("packaging/**" "scripts/build-deb.sh" "pyproject.toml" "src/**" "tests/smoke/**"
  ".github/workflows/deb-e2e.yml" "scripts/deb-smoke.sh")
ARCH_PATHS=("packaging/**" "pyproject.toml" "src/**" "scripts/arch-smoke.sh" "tests/smoke/**"
  ".github/workflows/arch-e2e.yml")

all_stands=false
case "$*" in
  "") ;;
  --all) all_stands=true ;;
  *)
    printf 'usage: %s [--all]   (--all: every stand, whatever the diff)\n' "${0##*/}" >&2
    exit 2
    ;;
esac

# What the branch changes against origin/main, the working tree included: the gate
# runs before the commit. Without a merge base nothing can be ruled out, so every
# stand runs.
changed=()
if base="$(git -C "$root" merge-base HEAD origin/main 2>/dev/null)"; then
  mapfile -t changed < <(
    git -C "$root" diff --name-only --no-renames "$base"
    git -C "$root" ls-files --others --exclude-standard
  )
else
  all_stands=true
fi

# touches PATTERN... — does the branch change a file one of these globs matches?
touches() {
  [ "$all_stands" = true ] && return 0
  local file pattern
  for file in "${changed[@]}"; do
    for pattern in "$@"; do
      # shellcheck disable=SC2053  # the right side is a glob on purpose, as in `paths:`
      [[ $file == $pattern ]] && return 0
    done
  done
  return 1
}

failed=()

# Colour for a terminal only: the gate is also run into a log file, read later.
bold='' green='' red='' reset=''
if [ -t 1 ]; then
  bold=$'\033[1m' green=$'\033[32m' red=$'\033[31m' reset=$'\033[0m'
fi

step() {
  local name="$1"
  shift
  printf '\n%s── %s%s\n' "$bold" "$name" "$reset"
  if (cd "$root" && "$@"); then
    printf '%sok%s — %s\n' "$green" "$reset" "$name"
  else
    printf '%sFAILED%s — %s\n' "$red" "$reset" "$name"
    failed+=("$name")
  fi
}

step "workflow pins"  python3 scripts/check-workflow-pins.py
step "lockfile"       uv lock --locked
step "ruff check"     uv run ruff check
step "ruff format"    uv run ruff format --check
step "ty"             uv run ty check
step "pytest"         uv run pytest --tb=short

if touches "${DEB_PATHS[@]}"; then
  step ".deb smoke"   scripts/deb-smoke.sh
fi
if touches "${ARCH_PATHS[@]}"; then
  step "Arch smoke"   docker run --rm -v "$root:/src:ro" archlinux:latest /src/scripts/arch-smoke.sh
fi

printf '\n'
if [ ${#failed[@]} -eq 0 ]; then
  printf '%sall gates passed%s\n' "$green" "$reset"
  exit 0
fi
printf '%sfailed:%s\n' "$red" "$reset"
printf '  %s\n' "${failed[@]}"
exit 1
