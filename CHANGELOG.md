# Changelog

All notable changes to this project will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] — 2026-03-27

### Added
- `scripts/release.sh` — version bump across pyproject.toml, PKGBUILD
- AUR packages: `echo-vault-diarize`, `echo-vault-llm` — optional extras as separate packages
- Version validation step in publish.yml — tag must match pyproject.toml
- Lint + tests run before PyPI publish

### Changed
- **CLI command renamed from `meetrec` to `echo-vault`** — entry point, help texts, temp dirs, state dir
- Speaker diarization (pyannote/torch) moved to optional dependency: `echo-vault[diarize]`
- LLM SDKs (anthropic/openai) moved to optional dependency: `echo-vault[llm]`
- Base install no longer requires PyTorch or LLM SDKs (~2 GB smaller)
- Monitor channel segments default to "Other" speaker when diarization is not available
- PKGBUILD rewritten with venv-based install (Python deps from PyPI, system deps from pacman)
- Nix flake: extras variants (`#llm`, `#diarize`, `#full`) via `nix run`

### Security
- GitHub Actions pinned to commit SHA (prevents supply-chain tag hijacking)
- CI workflow: explicit `permissions: contents: read`
- Publish workflow: type check step added, awk regex dots escaped in changelog extraction

## [0.7.0] — 2026-03-26

### Added
- LLM provider fallback chain — if primary provider fails, automatically tries next available provider
- Spectral similarity speaker merging — reduces pyannote over-segmentation of single speakers
- PyPI publishing via GitHub Actions (Trusted Publisher)
- GitHub Release automation with changelog extraction
- AUR PKGBUILD (`packaging/PKGBUILD`)
- Nix flake (`flake.nix`)
- VHS demo script (`packaging/demo.tape`)

### Changed
- License changed from proprietary to Apache-2.0
- Minimum Python version lowered from 3.14 to 3.13
- Type checker switched from mypy to ty (Astral)
- `DEFAULT_MODELS` moved to `settings.py` as single source of truth
- Default Gemini model updated to `gemini-2.5-flash` (2.0 deprecated)
- README rewritten with installation guide, configuration reference, CLI examples, roadmap
- CLI help texts expanded with examples and usage guidance

### Fixed
- Markdown fence stripping — LLM responses wrapped in ```json``` now parsed correctly

## [0.6.0] — 2026-03-23

### Added
- Free/cheap LLM providers: Groq, Gemini, OpenRouter, DeepSeek, Qwen
- Retry with exponential backoff on 429/529 rate limit errors (3 retries, 5s→10s→20s)
- `models.py` — domain objects extracted into dedicated module
- `vault.py` — Obsidian vault I/O separated from formatting
- Session name validation (alphanumerics, dashes, underscores only)
- Restrictive permissions (0700) on `/tmp/echo-vault` temp directories
- mypy strict mode
- Coverage threshold (85%) enforced in CI

### Changed
- `formatter.py` — pure formatting only, no I/O (moved to `vault.py`)
- `diarizer.py` no longer depends on `transcriber.py` — both import from `models.py`
- `summarizer.py` no longer defines domain models — imports from `models.py`
- Expanded ruff rules: W, G, PLC, PLR, S (security)

## [0.5.0] — 2026-03-22

### Added
- LLM summarization — brief summary, action items, key decisions
- `summarize` CLI command for re-summarizing existing transcripts
- `--no-summarize` flag for `start` and `process` commands
- Anthropic and OpenAI provider support
- Non-fatal summarization — transcript always saved first

### Changed
- Python 3.14 migration (PEP 758 exception syntax)
- Testing Trophy refactoring — more integration tests, fewer unit tests

### Fixed
- Audio device hot-switching — `@DEFAULT_MONITOR@`/`@DEFAULT_SOURCE@` follow device changes

## [0.4.0] — 2026-03-21

### Added
- Pause detection — split segments on word gaps >= threshold (configurable)
- CLI integration tests via CliRunner with mocked ML models
- Coverage reporting enabled by default in pytest

### Changed
- Strict typing refactoring across all modules
- Trophy testing approach — integration tests as primary coverage

### Fixed
- Segment splitting at actual silence gaps in raw mic audio

## [0.3.0] — 2026-03-18

### Added
- Stereo channel support — per-channel transcription (mic + monitor)
- Audio normalization (loudnorm) before transcription
- RMS-based crosstalk filtering — reject Whisper hallucinations on silent channels
- Channel-based speaker attribution (mic=You, monitor=Others)

### Fixed
- GPU memory management — free CUDA memory between transcription and diarization

## [0.2.0] — 2026-03-17

### Added
- Speaker diarization via pyannote — identifies "You" vs remote participants
- Speaker identification using mic/monitor RMS energy ratio
- Dual-channel recording (monitor + mic as separate WAV files)

## [0.1.0] — 2026-03-17

### Added
- Initial release
- Dual-channel audio recording via parecord (PulseAudio/PipeWire)
- Local transcription with faster-whisper (CUDA with CPU fallback)
- Markdown output with YAML frontmatter and `[HH:MM:SS]` timecodes
- Obsidian vault integration — saves audio + markdown
- pydantic-settings configuration with `MEETREC_` prefix
- `start`, `stop`, `process`, `status` CLI commands
