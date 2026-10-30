# ADR-008 — Recordings in progress live in a private state directory, verified, never repaired

**Status:** accepted 2026-08-07 (`recorder.py`: `session_root`, `ensure_private_dir`;
`settings.py`: `sessions_dir`).

**Context.** Sessions were written to `/tmp/tapeback` with `mkdir(exist_ok=True,
mode=0o700)`, which applies the mode only when it creates the directory. Any local user
could create it first, or a symlink, and read both raw channels of every meeting,
upstream of every privacy control.

**Decision.**
- Sessions live in `sessions/` under the XDG state directory
  (`~/.local/state/tapeback`), or in `TAPEBACK_SESSIONS_DIR`.
- The directory is checked with `lstat`: owned by the effective uid, no group or other
  bits, not a symlink. A failing directory is refused, not repaired.

**Why.**
- `/tmp` is world-writable.
- `XDG_RUNTIME_DIR` is private but a tmpfs of 10% of RAM (780 MB on an 8 GB laptop)
  against about 1.6 GB per hour of recording, and it is emptied at logout, losing a
  meeting interrupted before transcription.
- `chmod` cannot undo the window in which someone else already held the directory open.

**Consequences.** A hardcoded `/tmp` path anywhere needs the same ownership check, not a
comment asserting one. Review: `docs/spec/done/2026-08-07-security-review.md`.
