# Release testing

CHANGELOG entries for tagged versions are immutable (see AGENTS.md). A broken
release forces a patch-version bump even if no functional change is intended.
This document describes what stands between a merged PR and a tag;
`scripts/release.sh` refuses to tag a tree `scripts/pre_release_qa.sh` has not passed.

## Why

`.deb` packages bundle a virtualenv with compiled wheels (faster-whisper,
ctranslate2, pyav) that are pinned to a specific Python minor. The bundled
standalone Python (from [python-build-standalone](https://github.com/astral-sh/python-build-standalone))
keeps us decoupled from the system Python, but it also means more moving parts
in the package that can drift silently. Hence the layered test plan below.

## Layered checks

Run them in order. Each layer is fast enough to be the default; the slower
layers below are only needed when the cheaper layers have surprises.

### 1. `scripts/pre_release_qa.sh` (run before every release)

Runs `scripts/gate.sh --all` — the CI checks, then every packaging stand whatever
the diff: the wheel and the .debs built and installed in clean containers on the
`deb-e2e` images (`scripts/deb-smoke.sh`), and the AUR package built and installed
in an Arch container (`scripts/arch-smoke.sh`) — then the e2e quality suite on real
recordings. In every container the installed package runs `tapeback --version`,
`tapeback status` and `tests/smoke/transcribe.sh`, which transcribes a two-voice
recording through the stereo pipeline with the `tiny` model and checks both
voices reach the note. The same stands run in the everyday gate whenever a change
touches what they cover.
This catches the most common failure modes: broken shebangs, wrong venv paths,
missing system dependencies, broken hooks — and a dependency release that breaks
transcription. CI's own runs install from `uv.lock`; the packages resolve their
dependencies when they are built, as users get them, so only the stands see such a
release. `--version` alone did not: the 0.9.9 stands passed with PyAV 19 inside.
On success it stamps the tree, and `scripts/release.sh` tags only a stamped tree.

Optional extras (these run pip install during postinst, ~30 s + network):

```bash
docker run --rm -v $PWD/dist:/dist ubuntu:26.04 bash -c '
    apt-get update -qq
    apt-get install -y -qq /dist/tapeback_*.deb /dist/tapeback-llm_*.deb
    /opt/tapeback/venv/bin/python -c "import anthropic, openai"
'
```

**Do not keep EOL releases in the image list** — their apt repositories are
removed, `apt-get update` fails, and the .deb dependency resolution can't
complete. An interim release lives nine months: add the new one when it ships and
drop the old one when it reaches EOL, in `deb-e2e.yml` and `scripts/deb-smoke.sh`
together (`tests/test_release_scripts.py` fails when the two lists differ).

### 2. CI gate on every PR (automatic)

`.github/workflows/deb-e2e.yml` runs the same docker smoke on a 6-image matrix
(Ubuntu 22.04 / 24.04 / 26.04 / 26.10, Debian 12 / 13) for any PR that touches
`packaging/`, `scripts/build-deb.sh`, `pyproject.toml`, `src/` or `tests/smoke/`.
`.github/workflows/arch-e2e.yml` builds `packaging/PKGBUILD` from the PR's tree in
an Arch container and runs it on the system Python (`scripts/arch-smoke.sh`). A
regression in either pipeline never reaches a release tag — the PR turns red
first. Both also run daily on `main`: a dependency release can break a version
that is already published, with no PR to turn red.

### 3. Manual acceptance (run once per minor, or when behavior changes)

On a fresh Ubuntu/Debian VM or real machine:

- `sudo apt install ./tapeback_*.deb` → `tapeback --version` prints the tag
- `tapeback start test-meeting` on a clip with known speakers → markdown
  written to vault, audio file linked
- `tapeback tray` on GNOME Wayland WITHOUT the AppIndicator extension → the
  warning hint is printed to stderr; pystray still starts (icon may be inert,
  that's expected)
- After `sudo apt install gnome-shell-extension-appindicator` + enabling the
  extension + re-login → tray icon menu actually responds
- `sudo apt install ./tapeback-llm_*.deb` → `/opt/tapeback/venv/bin/python -c
  "import anthropic, openai"` succeeds
- `sudo apt remove tapeback-llm` → anthropic/openai uninstalled from the venv;
  `sudo apt remove tapeback` removes everything

If any of these fails on the final tag → bump the patch version, fix, retag.
Don't amend the released tag.
