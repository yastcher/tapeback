# Contributing

Thanks for taking the time. tapeback records meeting audio on Linux, transcribes it
locally with faster-whisper, and writes markdown to an Obsidian vault.

## Setup

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/), and `ffmpeg` on `PATH`.

```bash
git clone https://github.com/yastcher/tapeback
cd tapeback
uv sync --group dev
```

The dev group pulls in the optional extras (pyannote, torch, anthropic, openai,
dbus-next) so the whole suite runs without further setup. Recording itself needs
PulseAudio or PipeWire, but no test records anything.

## Before opening a pull request

Fix what the tools can fix, then run the gate — the CI checks, step for step:

```bash
uv run ruff check --fix
uv run ruff format
scripts/gate.sh
```

The gate has passed only when its last line reads `all gates passed`. Coverage is
enforced at 90% by `pyproject.toml`, so new code needs tests. When the branch touches
`src/`, `packaging/` or `pyproject.toml`, the gate also builds and installs the `.deb`
and Arch packages in containers, as CI does — that needs docker and
[nfpm](https://nfpm.goreleaser.com/), and some minutes.

## CHANGELOG and the version

- **Add an entry, never a version.** Entries go under `## [Unreleased]`, through the
  script rather than by hand:

  ```bash
  python3 scripts/changelog_add.py Fixed "- **Topic.** What was wrong, what is right now."
  ```

  Subsections are Security / Added / Changed / Fixed / Removed / Docs; changes to
  documentation alone go under `Docs`.
- **Do not touch the version** — not in `pyproject.toml`, `uv.lock`, `packaging/` or README's
  install commands.
  The maintainer's `scripts/release.sh` writes it everywhere at release time and turns
  `[Unreleased]` into the dated section; a PR that bumps it fails the tests.

## What reviewers look for

- **A bug fix starts with a failing test.** Write the test, watch it fail, then fix it.
  The test documents the exact failure and stops it coming back.
- **Tests assert exact values**, not ranges: `assert count == 2`, not `>= 1`. A loose
  assertion usually means the test is measuring the wrong thing.
- **Fixtures live in `tests/fixtures.py`**, never in a test file. Regression tests for a
  specific bug go in `tests/regressions/`.
- **No magic numbers.** Thresholds, limits and ratios belong in `settings.py` as a
  `TAPEBACK_` setting, or in `const.py` — not as an inline literal or a default argument.
- **No local imports inside functions**, except where an optional extra requires it
  (see `diarizer.py` and `summarizer.py` for the pattern and the `noqa` that documents it).
- Everything committed is in English — code, comments, logs, docs, commit messages.
- [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`,
  `docs:`, `refactor:`.

`AGENTS.md` holds the full working agreement for this repository. It is written for AI
assistants but describes the same rules a human reviewer applies.

## Configuration decisions are made from measurements

Anything touching transcription speed or quality — model, compute type, decoding
parameters, the glossary — is decided from `scripts/bench_transcribe.py`, which drives
the real `Transcriber` on real audio. Include its table in the PR. Several plausible
changes have been reverted after measurement contradicted them;
`docs/spec/BACKLOG.md` records which ones and why.

## CI on pull requests

Every commit pushed to a PR re-runs lint, format, types and tests, on Python 3.13 and
3.14. Changes to `src/`, `packaging/` or `pyproject.toml` additionally build the `.deb`
and install it in six Ubuntu and Debian containers, and build the AUR package in an
Arch container.

## Reporting a bug

Include the tapeback version, distro, whether audio came from PulseAudio or PipeWire,
and the GPU if transcription is involved. For a slow or stalled run, attach the run
record from `~/.local/share/tapeback/runs/` — it holds the settings the run used, its
status lines and its outcome, and never contains credentials.
