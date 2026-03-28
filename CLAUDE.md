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

## Architecture

- Source: `src/tapeback/` — cli.py, recorder.py, audio.py, transcriber.py, diarizer.py, formatter.py, vault.py, summarizer.py, models.py, settings.py
- Domain models (Segment, Word, DiarizationSegment, Summary, ActionItem) live in models.py — never in infrastructure modules
- Settings: pydantic-settings with `TAPEBACK_` prefix, env vars and `.env` only
- No config files (TOML, YAML) besides pyproject.toml
- Max 500 lines per file — decompose if exceeded

## Commands

- Lint: `uv run ruff check --fix`
- Format: `uv run ruff format`
- Type check: `uv run ty check`
- Test: `uv run pytest` (coverage ≥85% enforced via pyproject.toml)

## Code style

Enforced by ruff. See pyproject.toml `[tool.ruff]` for full config.
Do not duplicate ruff rules here — if ruff can check it, ruff owns it.

- No local imports inside functions. All imports at the top of the file.
  Local imports are only acceptable when explicitly required by documentation (e.g. circular dependency workarounds).

## Testing

- pytest with mocks only at system boundaries (subprocess, file I/O)
- Audio tests with real ffmpeg marked `@pytest.mark.skipif(not shutil.which("ffmpeg"))`
- **All fixtures** in `tests/fixtures.py` (registered via `conftest.py`) — never define fixtures in test files
- **All imports at top of file** in tests — same rule as production code, no local imports
- WAV helpers in `tests/fixtures.py`
- E2E tests in `tests/test_e2e_quality.py` — run with `TAPEBACK_RUN_E2E=1`
- Regression tests (bug-fix) in `tests/regressions/`
- **Bug fix workflow**: every fix MUST start with a failing test that reproduces the bug.
  Write the test first, verify it fails, then apply the fix and verify the test passes.
  This prevents regressions and documents the exact failure scenario.

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
- Never add wav2vec2 or alternative diarization models — only pyannote
- Never add translation — future phase
- Never add web server, REST API

## Before finishing

0. `git diff --stat` — assess scope of changes
1. `uv run ruff check --fix`
2. `uv run ruff format`
3. `uv run mypy`
4. `uv run pytest`
5. Security review (see checklist below)
6. **Always update README.md** when features, settings, commands, or architecture change
7. **Always update CHANGELOG.md** — add entries under `## [Unreleased]` or current version

Do not finish until lint, types, and tests pass.

## Security review checklist

Before completing any change, verify:

- **P0 (critical)**: No hardcoded secrets/tokens/passwords. No `shell=True` in subprocess. No user input in SQL/commands.
- **P1 (high)**: File paths validated. Temp files use restrictive permissions. API keys not leaked in logs/errors.
- **P2 (medium)**: Input validated at system boundaries. Error messages don't expose internals. Dependencies up to date.

## Gotchas

- Comments and logs in English
