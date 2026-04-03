# CLAUDE.md

## Project

CLI tool for recording meeting audio (Google Meet, Zoom, Teams, Telegram — any platform) on Linux via PulseAudio/PipeWire system audio capture. Transcribes locally with faster-whisper, saves markdown to Obsidian vault.

Stack: Python 3.13+, click, faster-whisper, pydantic-settings, anthropic/openai, ffmpeg, parecord.

No web servers, databases, Docker.

## Safety rules

- Всегда сначала надо решить причину проблемы, а не следствие.
- Не подавлять warnings/errors/логи, не разобравшись в причине. Сначала спросить: это наш баг, баг зависимости, или информационное сообщение? Подавлять можно только осознанно и с объяснением почему.
- Перед планированием надо продумать оптимальную систему типов.
- Never delete or overwrite files without backup or user confirmation
- Never delete files not tracked in git. Run `git ls-files <path>` before removing any file. If untracked — ask user.
- Never simplify architecture by removing existing features unless explicitly asked.
- Any file with API keys, tokens or credentials is read-only.
- When fixing linter/import issues: fix one file at a time, run tests after each change.
- When renaming or refactoring across the project, grep for ALL old names (module, package, repo, env prefix, URLs) across the entire tree before considering the task done. Don't skip files that seem unimportant (PKGBUILD, .install, flake.nix, demo.tape, etc.).

## Architecture

- Source: `src/tapeback/` — cli.py, recorder.py, audio.py, transcriber.py, diarizer.py, channel.py, formatter.py, vault.py, summarizer.py, models.py, settings.py, const.py, tray.py, pipeline.py
- Constants: `src/tapeback/const.py` — import as `from tapeback import const`, use as `const.SPEAKER_YOU`
- Domain models (Segment, Word, DiarizationSegment, Summary, ActionItem) live in models.py — never in infrastructure modules
- Settings: pydantic-settings with `TAPEBACK_` prefix, env vars and `.env` only
- No config files (TOML, YAML) besides pyproject.toml
- Max 500 lines per file — decompose if exceeded

## Commands

- Lint: `uv run ruff check --fix`
- Format: `uv run ruff format`
- Type check: `uv run ty check`
- Test: `uv run pytest` (coverage ≥85% enforced via pyproject.toml)

## Code quality

- No magic numbers in logic. Thresholds, limits, sizes, ratios — all go into `settings.py` as named settings with `TAPEBACK_` env vars, or into `const.py` as module-level constants. Function parameter defaults are not a substitute for proper settings.
- Values used in multiple modules go into `const.py`. Values used only in one module stay as module-level constants in that module. Configurable values go into `settings.py`.
- No local imports inside functions. All imports at the top of the file.
  Local imports are only acceptable when explicitly required by documentation (e.g. circular dependency workarounds).

## Code style

Enforced by ruff. See pyproject.toml `[tool.ruff]` for full config.
Do not duplicate ruff rules here — if ruff can check it, ruff owns it.

## Testing

- pytest with mocks only at system boundaries (subprocess, file I/O)
- Audio tests with real ffmpeg marked `@pytest.mark.skipif(not shutil.which("ffmpeg"))`
- **All fixtures** in `tests/fixtures.py` (registered via `conftest.py`) — never define fixtures in test files
- **All imports at top of file** in tests — same rule as production code, no local imports
- WAV helpers in `tests/fixtures.py`
- E2E tests in `tests/test_e2e_quality.py` — run with `TAPEBACK_RUN_E2E=1`
- Regression tests (bug-fix) in `tests/regressions/`
- **Hardcode expected values in tests**: don't reuse the same constant in test and production code. If `const.SPEAKER_YOU = "You"`, the test should assert `== "You"`, not `== const.SPEAKER_YOU`.
- **Bug fix workflow**: every fix MUST start with a failing test that reproduces the bug.
  Write the test first, verify it fails, then apply the fix and verify the test passes.
  This prevents regressions and documents the exact failure scenario.

## Versioning & releases

- Semantic Versioning: MAJOR.MINOR.PATCH
- After a release tag is pushed, all subsequent changes MUST go into a new version.
  Never amend a released version — bump the version first, then make changes.
- CHANGELOG entries for released versions are immutable. Before writing to CHANGELOG.md, run `git tag --sort=-v:refname | head -5` — if the top section version ≤ latest tag, that section is frozen. Create a new patch version (e.g. 0.8.8 → 0.8.9) with today's date.
- Never use `[Unreleased]` — always assign the next concrete version number with today's date (e.g. `## [0.8.9] — 2026-04-02`).
- Order CHANGELOG entries by user impact: user-facing fixes first, infrastructure/internal changes last.
- Version is the single source of truth in `pyproject.toml`. All other files (PKGBUILDs) are updated via `scripts/release.sh <version>`.
- Release flow: bump version → update CHANGELOG → commit → tag → push → CI publishes to PyPI → update AUR
- AUR publishing is manual: clone AUR repo, copy PKGBUILD, generate `.SRCINFO`, compute sha256sum, push.
- PKGBUILD in this repo keeps `sha256sums=('SKIP')` — real checksum is set only in the AUR repo after the tarball is available.

## Git

- Conventional commits (feat:, fix:, docs:, refactoring:)
- Always PR, never push to main
- **Do not run git commit, checkout, reset, clean, stash, rebase** — these are blocked in settings.json. Ask user if needed.
- Max ~500 lines of diff per commit — stop and propose a commit before continuing
- Always work in the current branch — never switch branches

## Never do

- Never hardcode secrets, tokens, or passwords in code
- Never hardcode audio device names
- Never use absolute paths in code or configs

## Before finishing

0. `git diff --stat` — assess scope of changes
1. `uv run ruff check --fix`
2. `uv run ruff format`
3. `uv run mypy`
4. `uv run pytest`
5. Security review (see checklist below)
6. **Always update README.md** when features, settings, commands, or architecture change
7. **Always update CHANGELOG.md** — check `git tag` first; if top section is already released, bump patch version

Do not finish until lint, types, and tests pass.

## Security review checklist

Before completing any change, verify:

- **P0 (critical)**: No hardcoded secrets/tokens/passwords. No `shell=True` in subprocess. No user input in SQL/commands.
- **P1 (high)**: File paths validated. Temp files use restrictive permissions. API keys not leaked in logs/errors.
- **P2 (medium)**: Input validated at system boundaries. Error messages don't expose internals. Dependencies up to date.

## Gotchas

- Comments and logs in English
