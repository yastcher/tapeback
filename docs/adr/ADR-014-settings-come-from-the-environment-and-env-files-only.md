# ADR-014 — Settings come from the environment and .env files only; setup edits the user's file

**Status:** accepted 2026-03-17 (`settings.py`), setup on 2026-10-15 (`_user_env.py`,
`cli.py`: `setup`).

**Context.** Users configure tapeback once and rarely look at it again. Presets and
per-stage engines are planned.

**Decision.**
- Settings are `TAPEBACK_*` variables, read from the environment, `./.env` and
  `~/.config/tapeback/.env`, in that order of precedence. No TOML, YAML or other config
  files besides `pyproject.toml`.
- `tapeback setup` asks in a terminal with the current values as defaults, takes the
  same choices as flags, and only reports without a terminal. It edits the user's file
  in place with python-dotenv: other lines unchanged, mode kept, a new file 0600.
- Setup must not assume one engine per stage.

**Why.** One mechanism for scripts, systemd units and people; pydantic-settings already
validates every value at load. A second format would split where a value comes from.

**Consequences.** Presets, when built, are sets of `TAPEBACK_*` values in code or `.env`
fragments, not TOML. A provider's own key variable (`GEMINI_API_KEY`) is read from the
environment only, never from a file.
