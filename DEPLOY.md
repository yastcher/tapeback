# Development & deployment

The module map is in [AGENTS.md](AGENTS.md#architecture); setting up, the checks and the
gate are in [CONTRIBUTING.md](CONTRIBUTING.md). This file covers releases and packages.

## Release process

Semantic Versioning: MAJOR.MINOR.PATCH. The maintainer releases from `main` once the
PRs are merged; the rules are in AGENTS.md, "Versioning & releases", and what each
check covers is in [docs/release-testing.md](docs/release-testing.md).

```bash
scripts/pre_release_qa.sh       # every stand, then the e2e quality suite; stamps the tree
scripts/release.sh minor        # patch | minor | major: bumps every copy of the version,
                                # closes [Unreleased], commits, pushes main, then tags
# the tag publishes to PyPI and GitHub (publish.yml)
scripts/aur-publish.sh <version>  # once the GitHub release is up
```

`aur-publish.sh` publishes every PKGBUILD under `packaging/`:
- `tapeback` — base (recording + transcription)
- `tapeback-tray` — system tray icon (dbus-next)
- `tapeback-llm` — LLM summarization (anthropic + openai SDKs)
- `tapeback-llm-local` — summaries on this machine (Arch's `llama-cpp`)
- `tapeback-diarize` — speaker diarization (pyannote + PyTorch)
- `tapeback-cuda` — CUDA 12 cuBLAS/cuDNN for GPU transcription on CUDA 13 systems

A package AUR already holds in this state is skipped, so a rerun for the same version
publishes only what is new.

### AUR packages

Each companion is a meta-package. Most pip-install their Python extras into the
`/opt/tapeback/` venv through a `.install` hook; `tapeback-llm-local` depends on
Arch's `llama-cpp` instead. The base PKGBUILD keeps `sha256sums=('SKIP')`: its source
is the tag's GitHub tarball, and `aur-publish.sh` writes the real checksum into the
AUR copy only.

AUR publishing is manual: `aur-publish.sh` clones each AUR repo, copies the PKGBUILD
and the `.install` file it names, generates `.SRCINFO`, and pushes what changed.

## CI

GitHub Actions:
- **ci.yml**: on pushes to `main` and PRs into it: workflow pins, lint, format, types, tests
- **publish.yml**: on a tag: checks it against every copy of the version
  (`scripts/release_from_tag.py`), reruns the checks, publishes to PyPI, builds the
  `.deb`s and creates the GitHub Release
- **deb-e2e.yml**: builds the `.deb`s and installs them in Ubuntu and Debian containers
- **arch-e2e.yml**: builds the AUR packages and runs them in an Arch container
