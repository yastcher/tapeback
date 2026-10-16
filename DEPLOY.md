# Development & deployment

## Architecture

```
src/tapeback/
  cli.py          Click CLI (start, stop, process, summarize, status, tray)
  recorder.py     PulseAudio recording via parecord
  audio.py        ffmpeg audio processing (split channels, normalize, convert)
  transcriber.py  faster-whisper transcription
  diarizer.py     pyannote speaker diarization + spectral speaker merging
  formatter.py    Markdown generation (pure formatting, no I/O)
  vault.py        Obsidian vault file I/O
  tray.py         System tray icon (pystray, optional)
  summarizer.py   LLM summarization with multi-provider fallback
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
uv run pytest           # test (coverage >= 85%)
```

## Release process

Semantic Versioning: MAJOR.MINOR.PATCH.

```bash
# 1. Bump version in pyproject.toml + all PKGBUILDs
scripts/release.sh 0.9.0

# 2. Commit, tag, push
git add -A && git commit -m "release: 0.9.0"
git tag v0.9.0
git push && git push --tags
# CI publishes to PyPI automatically

# 3. Update AUR packages (after PyPI publish completes)
scripts/aur-publish.sh 0.9.0
```

`aur-publish.sh` updates the 5 AUR packages:
- `tapeback` — base (recording + transcription)
- `tapeback-tray` — system tray icon (pystray + Pillow)
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
