# Development & deployment

## Architecture

```
src/tapeback/
  cli.py          Click CLI (start, stop, process, summarize, status, setup, tray)
  recorder.py     PulseAudio recording via parecord
  audio.py        ffmpeg audio processing (split channels, normalize, convert)
  transcriber.py  faster-whisper transcription
  diarizer.py     pyannote speaker diarization + spectral speaker merging
  formatter.py    Markdown generation (pure formatting, no I/O)
  vault.py        Obsidian vault file I/O
  tray.py         System tray icon (StatusNotifierItem over dbus-next, optional)
  summarizer.py   LLM summarization: prompts, masking, parsing
  _llm.py         LLM providers, keys, the fallback chain, SDK calls
  _llama_server.py  the llama-server a local summary starts and stops
  _device.py      the device Whisper and pyannote run on, chosen per stage
  _user_env.py    the user's settings file, as `tapeback setup` writes it
  pipeline.py     End-to-end recording → transcription → output pipeline
  models.py       Domain objects (Segment, Word, DiarizationSegment, Summary)
  settings.py     pydantic-settings configuration
```

## Dev setup

```bash
git clone https://github.com/yastcher/tapeback
cd tapeback
uv sync --group dev

uv run ruff check       # lint
uv run ruff format      # format
uv run ty check         # type check
uv run pytest           # test (coverage floor: fail_under in pyproject.toml)
```

## Release process

Semantic Versioning: MAJOR.MINOR.PATCH. The maintainer releases from `main` once the
PRs are merged; the rules are in AGENTS.md, "Versioning & releases".

```bash
scripts/pre_release_qa.sh       # every stand, then the e2e quality suite; stamps the tree
scripts/release.sh minor        # patch | minor | major: bumps every copy of the version,
                                # closes [Unreleased], commits, pushes main, then tags
# the tag publishes to PyPI and GitHub (publish.yml)
scripts/aur-publish.sh <version>  # once the PyPI publish has finished
```

`aur-publish.sh` updates the 5 AUR packages:
- `tapeback` — base (recording + transcription)
- `tapeback-tray` — system tray icon (dbus-next)
- `tapeback-llm` — LLM summarization (anthropic + openai SDKs)
- `tapeback-llm-local` — summaries on this machine (Arch's `llama-cpp`)
- `tapeback-diarize` — speaker diarization (pyannote + PyTorch)

### AUR packages

Each companion is a meta-package. Most pip-install their Python extras into the
`/opt/tapeback/` venv through a `.install` hook; `tapeback-llm-local` depends on
Arch's `llama-cpp` instead. PKGBUILDs live in
`packaging/` and keep `sha256sums=('SKIP')` — real checksums are set only in
the AUR repo after the PyPI tarball is available.

AUR publishing is manual: `aur-publish.sh` clones each AUR repo, copies
the PKGBUILD and its `.install`, if any, generates `.SRCINFO`, and pushes.

## CI

GitHub Actions:
- **ci.yml**: lint + type check + tests on every push/PR
- **publish.yml**: PyPI publish on tag push (Trusted Publisher), validates tag matches pyproject.toml version
- **deb-e2e.yml**: builds the `.deb`s and installs them in six Ubuntu and Debian containers
- **arch-e2e.yml**: builds the AUR packages and runs them in an Arch container
