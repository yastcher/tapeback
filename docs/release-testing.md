# Release testing

CHANGELOG entries for tagged versions are immutable (see AGENTS.md). A broken
release forces a patch-version bump even if no functional change is intended.
This document describes what stands between a change and a tag;
`scripts/release.sh` refuses to tag a tree `scripts/pre_release_qa.sh` has not passed.

## Why

`.deb` packages bundle a virtualenv with compiled wheels (faster-whisper,
ctranslate2, pyav) that are pinned to a specific Python minor. The bundled
standalone Python (from [python-build-standalone](https://github.com/astral-sh/python-build-standalone))
keeps us decoupled from the system Python, but it also means more moving parts
in the package that can drift silently. Hence the layered test plan below.

## Layered checks

§1 runs before every release, and `scripts/release.sh` refuses a tree it has not
stamped; §2 runs by itself on every PR; §3 once per minor release, or when behaviour
changes.

### 1. `scripts/pre_release_qa.sh` (run before every release)

Runs `scripts/gate.sh --all` — the CI checks, then every packaging stand whatever
the diff: the wheel and the .debs built and installed in clean containers on the
`deb-e2e` images (`scripts/deb-smoke.sh`), and the AUR package built and installed
in an Arch container (`scripts/arch-smoke.sh`) — then the e2e quality suite on real
recordings. In every container the installed package runs `tapeback --version`,
`tapeback status` and `tests/smoke/transcribe.sh`, which transcribes a two-voice
recording through the stereo pipeline with the `tiny` model and checks both
voices reach the note. Then `tapeback-llm` and `tapeback-llm-local` are installed
and `tests/smoke/llm-local.sh` checks that tapeback finds `llama-server` (bundled
in the `.deb`, Arch's `llama-cpp` in the Arch container) and that it runs
`--version`. The same stands run in the everyday gate
whenever a change touches what they cover.
This catches the most common failure modes: broken shebangs, wrong venv paths,
missing system dependencies, broken hooks — and a dependency release that breaks
transcription. CI's own runs install from `uv.lock`; the packages resolve their
dependencies when they are built, as users get them, so only the stands see such a
release. `--version` alone did not: the 0.9.9 stands passed with PyAV 19 inside.

After the gate, two checks on the tests themselves (ADR-019):

- **CRAP** (`scripts/crap.py`): complexity squared times the uncovered share cubed, plus
  complexity, per function, from the coverage the gate's pytest wrote. A function over
  `--max` fails the run; the limit only comes down.
- **Mutation testing** (mutmut, the slice in `[tool.mutmut]`): every function of the
  slice is changed in small ways, and the tests must notice. It runs from scratch, two
  workers at a time (`MUTMUT_CHILDREN` sets more, and the cap grows with them), in a
  memory-capped systemd scope: a mutant can make code allocate without end, and the
  kernel then kills that worker instead of the desktop. mutmut calls such a mutant a
  segfault; the ratchet counts it as died, not survived. `scripts/mutation_ratchet.py`
  fails the run when more mutants survive, or go untested, than the maxima
  `pre_release_qa.sh` passes it, or when a mutant was left unchecked, and asks for the
  maxima to come down when fewer do. `mutmut results` lists the survivors and `mutmut show <name>` shows one.

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
drop the old one when it reaches EOL, in `deb-e2e.yml`, `scripts/deb-smoke.sh` and
the expected list in `tests/test_release_scripts.py` together (the test fails when
they differ).

### 2. CI gate on every PR (automatic)

`.github/workflows/deb-e2e.yml` runs the same docker smoke on a 6-image matrix
(Ubuntu 22.04 / 24.04 / 26.04 / 26.10, Debian 12 / 13) for any PR that touches
`packaging/`, `scripts/build-deb.sh`, `pyproject.toml`, `src/` or `tests/smoke/`.
`.github/workflows/arch-e2e.yml` builds `packaging/PKGBUILD`, then `tapeback-llm`
and `tapeback-llm-local`, from the PR's tree in an Arch container and runs them on
the system Python (`scripts/arch-smoke.sh`). A
regression in either pipeline never reaches a release tag — the PR turns red
first. Both also run daily on `main`: a dependency release can break a version
that is already published, with no PR to turn red.

### 3. Manual acceptance (run once per minor, or when behavior changes)

On a fresh Ubuntu/Debian VM or real machine:

- README's install, quick start and uninstall commands, copied as written, work on
  the released packages. No stand runs them: a package README offers that was never
  published, and uninstall lines that removed nothing, both shipped in 0.10.0.
- `sudo apt install ./tapeback_*.deb` → `tapeback --version` prints the tag
- `tapeback start test-meeting` on a clip with known speakers → markdown
  written to vault, audio file linked
- `tapeback tray` on GNOME Wayland WITHOUT the AppIndicator extension → the
  warning hint is printed to stderr; the tray still starts (the icon may be
  inert, that's expected)
- After `sudo apt install gnome-shell-extension-appindicator` + enabling the
  extension + re-login → tray icon menu actually responds
- `sudo apt install ./tapeback-llm_*.deb` → `/opt/tapeback/venv/bin/python -c
  "import anthropic, openai"` succeeds
- `sudo apt install ./tapeback-llm-local_*.deb` on a machine with a GPU →
  `TAPEBACK_LLM_PROVIDER=local tapeback summarize <note>` adds a summary, and
  `nvidia-smi` shows the VRAM free again afterwards. The stands only run
  `llama-server --version`: they have no GPU and no model. On Arch the same with
  `yay -S tapeback-llm-local ggml-cuda`, through Arch's `llama-cpp`.
- `sudo apt remove tapeback-llm` → anthropic/openai uninstalled from the venv;
  README's `dpkg -r` line removes the rest. `apt remove tapeback` alone leaves the
  extras installed: apt does not remove packages installed from files along with
  the package they depend on

If any of these fails on the final tag → bump the patch version, fix, retag.
Don't amend the released tag.
