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

Runs the gate, the e2e quality suite on real recordings, then builds the wheel and
the .debs and installs them in clean containers — the same five images as the
`deb-e2e` workflow — running `tapeback --version` and `tapeback status` in each.
This catches the most common failure modes: broken shebangs, wrong venv paths,
missing system dependencies, broken hooks. On success it stamps the tree, and
`scripts/release.sh` tags only a stamped tree.

Optional extras (these run pip install during postinst, ~30 s + network):

```bash
docker run --rm -v $PWD/dist:/dist ubuntu:26.04 bash -c '
    apt-get update -qq
    apt-get install -y -qq /dist/tapeback_*.deb /dist/tapeback-llm_*.deb
    /opt/tapeback/venv/bin/python -c "import anthropic, openai"
'
```

**Do not use `ubuntu:24.10` or other EOL releases** — their apt repositories
are removed, `apt-get update` fails, and the .deb dependency resolution can't
complete. Stick to actively-supported releases: current LTS (26.04), previous
LTS (24.04), current interim (25.10 while supported), and current Debian stable
(13).

### 2. CI gate on every PR (automatic)

`.github/workflows/deb-e2e.yml` runs the same docker smoke on a 5-image matrix
(Ubuntu 22.04 / 24.04 / 26.04, Debian 12 / 13) for any PR that touches
`packaging/`, `scripts/build-deb.sh`, `pyproject.toml`, or `src/`. A regression
in the build pipeline never reaches a release tag — the PR turns red first.

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
