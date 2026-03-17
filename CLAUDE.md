# CLAUDE.md

## Project

CLI tool for recording meeting audio (Google Meet, Zoom, Teams, Telegram — any platform) on Linux via PulseAudio/PipeWire system audio capture. Transcribes locally with faster-whisper, saves markdown to Obsidian vault.

Stack: Python 3.11+, click, faster-whisper, pydantic-settings, ffmpeg, parecord.

No web servers, databases, Docker, GUI.

## Safety rules

- Never delete or overwrite files without backup or user confirmation
- Never delete files not tracked in git. Run `git ls-files <path>` before removing any file. If untracked — ask user.
- Never simplify architecture by removing existing features unless explicitly asked.
- Any file with API keys, tokens or credentials is read-only.
- When fixing linter/import issues: fix one file at a time, run tests after each change.

## Architecture

- Source: `src/meetrec/` — cli.py, recorder.py, audio.py, transcriber.py, formatter.py, settings.py
- Settings: pydantic-settings with `MEETREC_` prefix, env vars and `.env` only
- No config files (TOML, YAML) besides pyproject.toml
- Max 500 lines per file — decompose if exceeded

## Commands

- Lint: `uv run ruff check --fix`
- Format: `uv run ruff format`
- Test: `uv run pytest`

## Code style

Enforced by ruff. See pyproject.toml `[tool.ruff]` for full config.
Do not duplicate ruff rules here — if ruff can check it, ruff owns it.

## Testing

- pytest with mocks only at system boundaries (subprocess, file I/O)
- Audio tests with real ffmpeg marked `@pytest.mark.skipif(not shutil.which("ffmpeg"))`
- Fixtures in tests/conftest.py
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
- Never add summarization, translation, LLM calls — phase 3
- Never add GUI, web server, REST API

## Before finishing

0. `git diff --stat` — assess scope of changes
1. `uv run ruff check --fix`
2. `uv run ruff format`
3. `uv run pytest`
4. Security review: no hardcoded secrets, no injection, no unvalidated input
5. Update README.md if functionality changed

Do not finish until lint and tests pass.

## Gotchas

- Comments and logs in English
