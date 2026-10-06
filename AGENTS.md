## Project

CLI tool for recording meeting audio (Google Meet, Zoom, Teams, Telegram — any platform) on Linux via PulseAudio/PipeWire system audio capture. Transcribes locally with faster-whisper (optional OpenAI remote STT), saves markdown to Obsidian vault.

Stack: Python 3.13+, click, faster-whisper, pydantic-settings, anthropic/openai, ffmpeg, parecord.

No web servers, databases, Docker.

## Safety rules

- Always fix the cause of a problem, never the symptom.
- Never suppress warnings/errors/logs without understanding the cause first. Ask: is this our bug, a dependency bug, or an informational message? Suppress only deliberately, with an explanation of why.
- Design the optimal type system before planning the implementation.
- Never claim something doesn't exist without verifying first. Check the actual files/directories before making statements.
- Always propose solutions that make sense. No workarounds or hacks unless explicitly asked.
- Never delete or overwrite files without backup or user confirmation
- Never delete files not tracked in git. Run `git ls-files <path>` before removing any file. If untracked — ask user.
- Never simplify architecture by removing existing features unless explicitly asked.
- Any file with API keys, tokens or credentials is read-only.
- The LLM summarizer is the usual thing that leaves the machine, and a meeting transcript is personal data. `summarizer.summarize()` is the masking seam for *text* (`_mask.py`) — a new outbound *text* call must go through it, never past it. Masking is opt-in (`TAPEBACK_MASK_PII`) because it is the user's data and their call. Opt-in remote STT (`TAPEBACK_STT_BACKEND`, currently `openai`) is a second outbound path that uploads *audio*; masking cannot apply there — document it clearly and keep the default local.
- When fixing linter/import issues: fix one file at a time, run tests after each change.
- When renaming or refactoring across the project, grep for ALL old names (module, package, repo, env prefix, URLs) across the entire tree before considering the task done. Don't skip files that seem unimportant (PKGBUILD, .install, flake.nix, demo.tape, etc.).

## Architecture

- Source: `src/tapeback/` — cli.py, recorder.py, audio.py, channel.py, transcriber.py, remote_stt.py, diarizer.py, speaker_merge.py, formatter.py, vault.py, summarizer.py, glossary.py, live.py, tray.py, pipeline.py + models.py, settings.py, const.py
- Private helpers are `_`-prefixed: `_gpu.py` (nvidia-smi, thermal clamp, VRAM), `_worker.py` + `_isolated.py` (out-of-process transcription), `_resume.py` (reusing a finished channel or remote STT chunk), `_quality.py` (transcript metrics and the hallucination filter), `_mask.py` (PII masking at the LLM boundary), `_stt_openai.py` + `_stt_openai_chunks.py` + `_stt_openai_fmt.py` + `_stt_caps.py` + `_stt_media.py` + `_stt_retry.py` (remote STT: OpenAI backend, chunking/resume, response mapping, capabilities, ffmpeg upload helpers, retries/heartbeats), `_sni.py` + `_dbusmenu.py` + `_tray_env.py` (tray protocol), `_runlog.py`, `_timing.py`, `_lazy.py`. The prefix means "internal to tapeback", not "pure" — `_gpu.py` shells out and `_worker.py` spawns processes.
- Benchmarks live in `scripts/bench_transcribe.py` — it drives the real `Transcriber`, so it measures what ships. Configuration choices here are made from its table, not from reasoning; see `docs/spec/BACKLOG.md` for what that has already overturned.
- Constants: `src/tapeback/const.py` — import as `from tapeback import const`, use as `const.SPEAKER_YOU`
- Domain models (Segment, Word, DiarizationSegment, Summary, ActionItem) live in models.py — never in infrastructure modules
- Settings: pydantic-settings with `TAPEBACK_` prefix, env vars and `.env` only
- No config files (TOML, YAML) besides pyproject.toml
- Max 500 lines per file — decompose if exceeded

## Commands

- Lint: `uv run ruff check --fix`
- Format: `uv run ruff format`
- Type check: `uv run ty check`
- Test: `uv run pytest` (coverage ≥90% enforced via pyproject.toml)
- Gate (everything CI checks, one verdict): `scripts/gate.sh`; the packaging stands run when the diff touches their paths, `--all` forces them

## Code quality

- Prefer the simplest solution that works. Don't add layers (extra abstractions, design patterns, indirection) unless they solve a real, present problem. If a flat approach does the job — use it.
- The existing codebase is not a reference to copy from blindly. Question patterns — if existing code has an antipattern, write better code, don't propagate it.
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
- **Assert exact values, not ranges**: `assert count == 2`, not `assert count >= 1`. Weak assertions hide bugs. If an assertion has to be loose, the test is measuring the wrong thing — fix the seam instead (e.g. inject a clock) rather than weakening the assert.
- **Boundary values**: test the exact boundary (`==`), one below and one above. A `>=` in production code must have a test where left equals right.
- **Test both branches of conditionals**: if code has `if x: A else: B`, test both paths.
- **Isolation by construction, never by cleanup**: a test must not depend on leftover state from another test. Scope every assertion to what the test itself created. Nothing a test writes may land in the developer's home: the autouse `isolate_user_dirs` points `XDG_STATE_HOME` and `XDG_DATA_HOME` into the test's tmp dir.
- **Bug fix workflow**: every fix MUST start with a failing test that reproduces the bug.
  Write the test first, verify it fails, then apply the fix and verify the test passes.
  This prevents regressions and documents the exact failure scenario.
- **No test may reach a live vendor.** The autouse `isolate_settings_sources` fixture cuts off `.env`, every `TAPEBACK_*` variable, and every provider key in `summarizer._PROVIDER_ENV_VARS` — those are read under their own names (`ANTHROPIC_API_KEY`, ...), so a prefix sweep alone leaves them live. Drive the list off the production mapping, never a copy: a new provider must be isolated when it is added, not when someone remembers. A forgotten mock must then fail, not bill. Live vendors only under `TAPEBACK_RUN_E2E=1`.

## Versioning & releases

- Semantic Versioning: MAJOR.MINOR.PATCH
- **Between releases the top CHANGELOG section is always `## [Unreleased]`.** It claims no number, so a branch has nothing to decide — two branches that each opened `## [0.9.9]` with their own date once merged into a conflict over a number neither had the right to pick.
- **A branch never touches the version** — not `pyproject.toml`, not `uv.lock`, not `packaging/`. Only `scripts/release.sh` writes it, into every copy at once; `tests/test_release_scripts.py` fails a PR in which the copies disagree.
- **A CHANGELOG entry lands by command, not by editing the file:** `python3 scripts/changelog_add.py Fixed "- **Topic.** What changed."` (or `--file entry.md`). It writes into `[Unreleased]`, creates the subsection in the order Security / Added / Changed / Fixed / Removed / Docs, and refuses a section a tag has closed. What it cannot decide stays with you:
  - the text — passed verbatim, so hand it a finished markdown bullet (dry it in Chekhov style);
  - the subsection — documentation-only changes go under `### Docs`;
  - the order — by user impact. The script appends, so an infrastructure line lands last by itself; a user-facing one placed under existing infrastructure lines takes `--top`.
- Released sections are immutable. An outdated entry is superseded by a new one, never rewritten.
- **Release flow — maintainer only, on `main`, after the PRs are merged:** `scripts/pre_release_qa.sh` → `scripts/release.sh patch|minor|major` → the tag publishes to PyPI and GitHub (`publish.yml`) → `scripts/aur-publish.sh <version>`. `release.sh` refuses a tree `pre_release_qa.sh` has not stamped (`SKIP_PRERELEASE_QA=1` for a hotfix that cannot wait), bumps every copy of the version, closes `[Unreleased]` into `## [X.Y.Z] — <date>`, commits, pushes `main`, then tags. `publish.yml` checks the tag against every copy with `scripts/release_from_tag.py` before building, and publishes that section as the release notes. An agent never runs `release.sh` — it commits and pushes.
- `scripts/pre_release_qa.sh` runs `scripts/gate.sh --all` (every stand, whatever the diff), then the e2e quality suite (needs the `tests/data/` recordings and `HF_TOKEN` — a skipped test fails the run). Needs docker and nfpm.
- Bundled interpreters in distro packages come from a pinned tarball URL (`scripts/build-deb.sh`), not from a tool that fetches one (`uv python install`). The URL is deterministic and so is the archive layout; a tool's layout varies by its own version and by the runner's platform, which once put a broken python into the `.deb`.
- PKGBUILD in this repo keeps `sha256sums=('SKIP')` — the real checksum is set by `scripts/aur-publish.sh` in the AUR repo, once the tarball exists.

## Git

- Conventional commits (feat:, fix:, docs:, refactoring:)
- Always PR, never push to main. The one exception is `scripts/release.sh`, run by the maintainer: the release commit is mechanical and has nothing to review.
- **Do not run git commit, checkout, reset, clean, stash, rebase** — these are blocked in settings.json. Ask user if needed.
- Max ~500 lines of diff per commit — stop and propose a commit before continuing
- Always work in the current branch — never switch branches

## Never do

- Never hardcode secrets, tokens, or passwords in code
- Never hardcode audio device names
- Never use absolute paths in code or configs

## Before finishing

0. `git diff --stat` — assess scope of changes
1. `uv run ruff check --fix` and `uv run ruff format` — the fixers; the gate only checks.
2. **`scripts/gate.sh`** — CI step for step (`tests/test_gate.py` keeps the two in sync): the checks always, the `.deb` and Arch stands when the diff against `origin/main` touches the paths their workflows filter on — the same minutes CI would spend, spent before the push. The stands need docker and nfpm. The gate has passed only when its last line reads `all gates passed`; not "tests are green", not coverage read by eye.
3. Security review (see checklist below)
4. **Tech lead review**: re-read your own diff as a strict reviewer. Look for overengineering, antipatterns copied from existing code, unnecessary complexity, and assertions weakened to make a test pass. Fix what you find before finishing.
5. **Always update README.md** — re-read it and verify it still matches current functionality, settings, commands and architecture. It rots silently; check, don't assume.
6. **Always add a CHANGELOG entry** — under `[Unreleased]`, with `scripts/changelog_add.py`; never pick a version number.
7. **Propose a commit message** (Conventional Commits). `git commit` is blocked, so the user runs it — hand them the exact message. Split into several commits when the diff exceeds ~500 lines or mixes concerns (e.g. `docs:` separate from `feat:`).

**Run the gate bare, into a file:** `scripts/gate.sh > "$log" 2>&1`, then read the log in a separate call. A pipe destroys the exit code (`false | tail` returns 0), and a tail such as `; echo "EXIT=$?"` prints the right code while the call itself reports the 0 of `echo` — a background task then says "exit code 0" over a red gate.

Do not finish until the gate, security review, and tech lead review pass.

## Security review checklist

Before completing any change, verify:

- **P0 (critical)**: No hardcoded secrets/tokens/passwords. No `shell=True` in subprocess. No user input in SQL/commands.
- **P1 (high)**: File paths validated. Temp files use restrictive permissions. API keys not leaked in logs/errors.
- **P2 (medium)**: Input validated at system boundaries. Error messages don't expose internals. Dependencies up to date.

## Gotchas

- **Everything committed to git is in English** — code, comments, logs, README, CHANGELOG, AGENTS.md, specs in `docs/spec/`, commit messages. Chat replies to the user follow the user's language; files do not.
- At the end of each non-trivial session, suggest 1–3 items for .claude/insights-inbox.md
