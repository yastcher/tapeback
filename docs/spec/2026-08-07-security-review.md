# Security review — full repository, 2026-08-07

First full-repository security pass. Scope: all of `src/tapeback/` (30 modules),
`scripts/`, `.github/workflows/`, and packaging configuration. Reviewed at v0.9.8.

## Threat model

tapeback is a single-user desktop CLI. No server, no database, no network listener. The
inputs that matter:

| input | trust | where it lands |
|---|---|---|
| meeting audio captured from PulseAudio/PipeWire | the user's own, but the most sensitive artefact the tool holds | session directory, then the vault |
| audio file path given to `tapeback process` | the user's own | ffmpeg argv, vault filename |
| transcript text produced by Whisper | derived from audio, effectively attacker-influenced if a caller speaks | LLM request, vault markdown |
| LLM response | **untrusted** — comes from a third party | parsed to `Summary`, written into the vault |
| environment variables and CLI flags | trusted (the user's own shell) | settings |
| session/resume/run-log JSON in user-owned directories | trusted | control flow, cached transcripts |

The only egress is `summarizer.summarize()`. Everything else — recording, Whisper,
pyannote — is local.

## Finding 1 (High): recordings sat in a world-writable directory — FIXED

**Where:** `recorder.py` (`Recorder.start`), `const.TEMP_DIR = "/tmp/tapeback"`.

**What was wrong.** Sessions were written to a fixed path under `/tmp`:

```python
base_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
tmp_dir = base_dir / session_name
tmp_dir.mkdir(exist_ok=True, mode=0o700)
```

`mode=` applies only when `mkdir` actually creates the directory. With `exist_ok=True` a
pre-existing directory was accepted with no check of owner, mode, or whether it was a
symlink at all — and `grep` over `src/` found no `lstat`, `st_uid`, `st_mode`,
`is_symlink` or `geteuid` anywhere in the tree.

The project believed this was covered: `pyproject.toml` disabled ruff's `S108` with the
note "/tmp/tapeback is intentional, secured with 0700 permissions". The stated mitigation
did not hold for a directory somebody else created first.

**Exploit.** `/tmp` is sticky (01777), so any local user can create entries in it. On a
shared machine an attacker runs `mkdir -m 0777 /tmp/tapeback` (or points it at a
directory of their own with a symlink) before the victim's first recording. The victim
records; two `parecord` processes write `monitor.wav` — every other participant's speech
— and `mic.wav` — the victim's own microphone — straight into it. The attacker reads
whole meetings in raw audio, upstream of every privacy control the project has: the PII
masking added in 0.9.8 covers only text on its way to an LLM.

The symlink variant additionally gives an attacker-chosen destination for `mkdir` and the
subsequent writes, as the victim's user.

**Fix.** `recorder.session_root()` now resolves `XDG_RUNTIME_DIR` (`/run/user/$UID`),
which logind creates as 0700 owned by the user — private by construction rather than by
convention, and cleared at logout, which is the lifetime `/tmp` gave us. Without it
(ssh without logind, containers, cron) the fallback is `~/.cache/tapeback/sessions`,
user-owned for the same reason.

`_ensure_private_dir()` then verifies rather than trusts: `lstat` (not `stat`, because
`mkdir(exist_ok=True)` follows a symlink to a directory and reports success), owner must
be the effective uid, and no group or other permission bits. It refuses instead of
repairing — a `chmod` cannot undo the window in which someone else already held the
directory open.

Covered by `tests/regressions/test_session_dir_not_shared.py`, including the exact
permission boundary (0700 accepted, one bit more in any position refused) and an
end-to-end `start()` that asserts both channels land under the private root.

Tests themselves now pin `XDG_RUNTIME_DIR` to a temp directory in an autouse fixture —
before this change the suite was writing real directories into `/tmp/tapeback`.

## Reviewed and clean

Findings below this line were investigated and dismissed with a reason, so a later pass
does not have to re-derive them.

**Command injection.** No `shell=True` anywhere. Every subprocess (`ffmpeg`, `parecord`,
`pactl`, `nvidia-smi`, `sys.executable -m tapeback._worker`) is invoked with an argument
list. ffmpeg filter graphs are built from `const.py` values, never from user input.

**Path traversal.** `validate_session_name` uses an anchored `^[\w-]+$`. `vault.py`
additionally resolves each destination and rejects anything outside the vault root
(`_ensure_within_vault`), which is defence in depth rather than the only barrier. Covered
by `tests/regressions/test_path_traversal.py`.

**Deserialization.** Only `json.loads`. No `pickle`, `yaml.load`, `marshal`, `eval` or
`exec` in the tree. The worker protocol is newline-delimited JSON between a parent and a
child it spawned itself.

**Credential handling.** `hf_token` and `llm_api_key` are `SecretStr`. The two places
where settings leave the process both use explicit allow-lists rather than
`model_dump()`: `_runlog.RECORDED_SETTINGS` (16 fields, no credentials) and
`_isolated.job_settings`. Both carry a comment saying why.

**LLM boundary.** The provider response goes through `_strip_markdown_fences` →
`json.loads` → construction of `Summary`; the raw string is never written to the vault.
A parse failure reports at most 500 characters of the response, which with masking on is
the masked text.

**GitHub Actions.** The only `${{ }}` interpolation inside a `run:` block is
`github.base_ref` in `gemini-review.yml`, and the base branch exists in the base
repository, so an outside contributor cannot choose its name. The review workflow uses
`pull_request`, not `pull_request_target`, so fork code runs with a read-only token and
no secrets; the missing `GEMINI_API_KEY` is handled by an early exit.

**`scripts/check-workflow-pins.py`.** Host is hardcoded to `api.github.com`; owner, repo
and SHA come from a constrained regex. No SSRF surface.

## Not vulnerabilities, but worth knowing

- **The provider fallback chain hands the same transcript to a second company** when the
  first fails. Deliberate, documented in the README, and the reason masking exists — but
  it means "which provider saw my meeting" is not a single answer.
- **`^[\w-]+$` accepts a trailing newline**, because Python's `$` matches before one. The
  result is a filename ending in a newline, not a traversal. Not worth a change.
- **A filename beginning with `-` would be read as a flag by ffmpeg.** The path comes
  from the user's own command line, so there is no second party to exploit it.

## Follow-ups

- Re-run this pass whenever a new outbound call is added; `summarizer.summarize()` being
  the only egress is what keeps the analysis small.
- The `S108` justification in `pyproject.toml` was removed along with `const.TEMP_DIR`.
  If a hardcoded `/tmp` path is ever reintroduced, it needs the same ownership check, not
  a comment asserting one.
