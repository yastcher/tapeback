# Changelog

All notable changes to this project will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.9] — 2026-09-16

### Added
- Optional remote speech-to-text backends (`TAPEBACK_STT_BACKEND`; default stays `local` / faster-whisper). OpenAI is the first remote backend (`openai`). New settings: `TAPEBACK_STT_MODEL` (local default `large-v3-turbo`, OpenAI default `whisper-1`; `TAPEBACK_WHISPER_MODEL` is deprecated), `TAPEBACK_STT_API_KEY` (preferred over `OPENAI_API_KEY` for STT), and `TAPEBACK_STT_CONCURRENCY`. Remote STT requires `tapeback[stt]` (openai SDK; `tapeback[llm]` also installs it) and uploads meeting audio off the machine — PII masking cannot apply there. For the OpenAI backend, model capabilities are derived automatically: `whisper-1` keeps timestamps + local pyannote; `gpt-transcribe` skips local diarization (≤1500s per upload); `gpt-4o-transcribe-diarize` returns remote speaker labels (≤1400s per upload). Long meetings are sliced under both the 25 MiB size cap and the per-model duration cap.

### Fixed
- `tapeback start` no longer crashes with `No recording in progress` when `tapeback stop` already finished the session from another terminal.
## [0.9.8] — 2026-08-05

### Added
- Optional PII masking for the LLM request (`TAPEBACK_MASK_PII`, off by default). Summarization is the only thing tapeback sends off the machine; with masking on, emails and phone numbers are replaced by `[EMAIL_1]` / `[PHONE_1]` placeholders before the transcript is sent — on the retry and on every fallback provider too — and the real values are restored in the summary written to the vault. The transcript on disk is never masked.
- `tapeback summarize --show-masked` prints the exact text that would go to the provider and stops — no request, no API key needed — with a tally of how many distinct values were masked. Without it "masking is on" is unfalsifiable from the outside, which for a privacy switch is the whole problem.
- `TAPEBACK_MASK_TERMS` masks names and other terms you list, case-insensitively and word-bounded, longest term first. This is the part that carries: across 56 real transcripts the built-in email and phone rules matched nothing, because what people say aloud in meetings is names. Matching is literal — every inflected form needs its own entry, since guessing them would silently rewrite unrelated words. Terms colliding with a transcript label (`You`, `Other`, `Speaker 1`) are refused with a warning.
- Progress during transcription: a percentage through the audio every 10 seconds, reported per channel (`transcribe mic: 40% (2:31 / 6:18)`). A long run now shows movement instead of a single opening line, and a channel stuck in a repeat loop is visible as a percentage that stops advancing.
- The resolved model, device and compute type are printed after the model loads (`Whisper: large-v3-turbo on cuda/float16`). A run that silently fell back to CPU used to be indistinguishable from a healthy GPU run; now it isn't. Shows `batch_size=N` when batching is enabled.
- Timings for two stages that previously ran untimed — reading the stereo channels and gating the mic. Their cost used to appear as unexplained dead time between stages.
- GPU telemetry during transcription (`TAPEBACK_GPU_TELEMETRY`, on by default): one line per stage with average/minimum SM clock, peak temperature, peak VRAM, and the share of samples where the card was thermally or power limited. On thermally constrained laptops a long run can spend most of its time clamped, which until now was indistinguishable from a slow model. Observation only — tapeback never changes clock or power caps. No-op without `nvidia-smi` or on CPU.
- Re-running an interrupted recording no longer redoes a channel it had already finished (`TAPEBACK_RESUME_CACHE`, on by default). A completed channel is cached against the audio and every setting that affects what Whisper outputs, so changing the model, glossary, language or any decoding parameter invalidates it. Partial channels are never cached — reusing a truncated one would make the next run call it complete. Resuming part-way *through* a channel was rejected deliberately: it would require `clip_timestamps`, which faster-whisper documents as disabling `vad_filter`, and VAD is half of why hallucinations on silence went away.
- Tests can no longer reach a live LLM vendor. Isolation swept `TAPEBACK_*` only, but provider keys are read under their own names (`ANTHROPIC_API_KEY`, `GROQ_API_KEY`, ...), so on a machine with any of them exported every test had a working provider chain and one forgotten mock would have billed the real API. Development-only; released packages are unaffected.
- Per-run JSON records (`TAPEBACK_RUN_LOG`, on by default) in `~/.local/share/tapeback/runs/`: the settings the run actually used, every status line, and the outcome (`completed` / `aborted` / `failed` with the error). Previously a run that failed or was interrupted left nothing behind, so there was no way to tell which configuration produced a given transcript — or why a recording never got one. Credentials are never recorded; the stored settings are an explicit allow-list.

### Changed
- **English technical terms now survive Russian speech.** New `TAPEBACK_HOTWORDS` setting, shipping a software-meeting glossary by default, biases decoding towards domain vocabulary. On a 31-minute recording it raised distinct English terms preserved in Latin script from 25 to 33 and cut the share of low-confidence words from 81.5 to 59.4 per 1000 — "tapeback", previously transcribed as "ты пупа ты бэк", now comes back as "tapeback". It reduces hallucinations rather than adding them. Replace the list with your own domain vocabulary, or set it empty to disable.
- The default glossary moved to its own module, [`src/tapeback/glossary.py`](src/tapeback/glossary.py), and was rebuilt from terms harvested out of real transcripts: `RabbitMQ` (seen as "RebitMQ"), `LayoutLMv3` ("Layout MV3"), `ONNX Runtime` ("Onyx, Runtime"), `Qwen` ("QVN"), `OpenVINO` ("OpenSWIFO"). `docTR` was tried and dropped — it still came back as "Dr. OCR", the hotword losing to the phonetic match with an ordinary English word, and a term that does not work still spends the token budget.
- **`TAPEBACK_COMPUTE_TYPE=auto` now resolves to `int8_float16` on CUDA instead of `float16`** — faster *and* smaller, which is not the usual trade-off. Measured on a GTX 1650 Ti with large-v3-turbo, same clip twice each: 14.16× vs 3.90× real time, 1115 MiB vs 2139 MiB. Quality does not pay for it: decoding the same audio both ways gave near-identical text, and across a benchmark grid int8_float16 had the lower share of low-confidence words. ctranslate2 falls back on its own where the type is unsupported.
- **Transcription is several times faster: `TAPEBACK_CHUNK_LENGTH` default raised 7 → 30.** Whisper's encoder is fixed at 30 seconds and faster-whisper zero-pads every window back to it, so a smaller value never made a pass cheaper — it only made the run need more of them. Measured on a 145 s recording: `2` → 390.6 s (slower than real time), `7` → 116.8 s, `30` → 41.4 s. Quality improved rather than regressed: on a 13-minute reference recording the new value produced no subtitle-corpus hallucinations at all, against 4 at `10` and 2 at `7`. If you set `TAPEBACK_CHUNK_LENGTH` yourself, remove the override.
- Enabling `TAPEBACK_BATCH_SIZE` now warns which settings batched inference silently ignores (`no_speech_threshold`, `condition_on_previous_text`, and every `temperature` value after the first). These are anti-hallucination settings, and the run otherwise looked identical.
- `nvidia-smi` is now queried through one helper shared by transcription and diarization, so a missing binary, a driver error and a hung query all degrade the same way instead of each module handling it differently.

### Fixed
- **Processing a long recording needs a third of the memory it did.** Channel analysis kept both stereo channels as float32 for the whole run and built them by converting the entire interleaved buffer at once, so a 37-minute 48 kHz recording peaked at 1272 MB and held 868 MB throughout. The channels are now read in chunks and kept as int16 — the format they are stored in — for a peak of ~434 MB and 434 MB held. On a machine with little free RAM this is the difference between running and thrashing swap, which in turn heats the CPU and drives the GPU into a thermal clamp.
- Four RMS calculations widened their samples before squaring; three did not. With int16 channels a near-full-scale sample squares past the type's range: two sites would have silently reported the loudest audio as the quietest, and `identify_user_speaker` — which accumulates across a whole speaker — went negative and produced a complex square root.
- **A CUDA out-of-memory no longer costs the whole GPU until you restart.** Transcription now runs in a child process (`TAPEBACK_ISOLATE_TRANSCRIPTION`, on by default). The leak is real and unfixable from Python — ctranslate2 builds the model on the C++ side, and a load that fails partway leaves its allocation behind with nothing in Python holding a reference; neither dropping the exception's traceback nor `CT2_CUDA_ALLOCATOR=cuda_malloc_async` recovers it. A process that exits does, though: measured on the configuration that used to strand the card, free VRAM went 3674 MiB → 95 MiB in-process and 3674 MiB → **3674 MiB** with isolation. Segments stream back line by line, so a worker killed mid-run still leaves its finished work with the parent. Credentials are not passed to the worker.
- Below `TAPEBACK_MIN_FREE_VRAM_MIB` free VRAM, tapeback uses the CPU instead of attempting a CUDA load that would leak on failure.
- **Ctrl+C no longer throws away the transcription.** The interrupt propagated out of the segment loop, so everything decoded so far was discarded — a run going for over two hours could produce nothing at all, which is how sixteen recordings ended up with no transcript. The segments already decoded are now kept and saved, the note is tagged `partial` with a visible warning so it cannot be mistaken for a complete one, and the second channel is skipped rather than making you interrupt twice. A further Ctrl+C still stops the process.
- VRAM is released even when transcription fails or is interrupted, so a failure there no longer starves the diarization that runs next.
- **Transcription no longer grinds for hours on a thermally clamped GPU.** On laptops sharing one heatsink between CPU and GPU, the controller cuts the GPU's power budget in response to a hot *system* — measured at 5 W against a 50 W default, clocks pinned to 300 MHz, while the GPU itself sat at 74 °C and the CPU package at 93 °C. The CPU is roughly 8× faster in that state (2.39× real time against 0.31×), so tapeback checks before each transcription stage and moves to the CPU, saying so. The check is retaken per stage — each runs in its own worker — so a clamp that clears between channels puts the next one back on the GPU rather than stranding the whole recording. Tunable via `TAPEBACK_THERMAL_CLAMP_CHECK`, `TAPEBACK_THERMAL_CLAMP_WAIT` and `TAPEBACK_THERMAL_CLAMP_CPU_FALLBACK`; `TAPEBACK_STAGE_PAUSE_SECONDS` sheds heat between stages to avoid the clamp in the first place.
- The clamp is not waited out by default. It clears on *system* idle, not on the GPU cooling: the shortest release measured was 451 s, and with a browser keeping the CPU busy it stayed latched for over an hour with the GPU at 72 °C. A short wait therefore almost never succeeds and only delays the CPU fallback, so `TAPEBACK_THERMAL_CLAMP_WAIT` now defaults to 0. Any wait that does happen is reported with elapsed time — an earlier version waited silently, which looked exactly like a hang.
- **Re-processing a recording that already lives in the vault no longer copies the audio again.** The vault never overwrites, so each re-run left another `_1.wav`, `_2.wav` beside the original; on the author's machine roughly 2.2 GB of a 5.5 GB audio directory was duplicates of recordings simply transcribed more than once.
- The microphone channel is no longer transcribed in a language of its own. Each channel ran its own auto-detection, and the mic — gated to near silence while you are only listening — had almost nothing to detect from, so it guessed wrong: notes came out labelled `language: en` with Russian text and stray Cyrillic inside English sentences. The monitor channel is now transcribed first and its detected language is reused for the mic. An explicitly configured `TAPEBACK_LANGUAGE` still wins over both.
- The "Diarized Transcript" section is no longer emitted when it only renames speakers. Every transcript that carried it duplicated the whole thing — same timecodes, same text, same recognition errors, with `Other` replaced by `Speaker 1` — doubling the file for no information. It now appears only when diarization actually splits the conversation between speakers.
- A long uninterrupted stretch of one speaker no longer collapses into a single block. Merging was bounded only by speaker changes and pause length, so a 31-minute recording rendered as two blocks whose last timecode was `[00:00:45]` — the text was intact but unnavigable. Blocks are now capped at 60 seconds.
- Whisper's subtitle-corpus hallucinations are stripped from transcripts: "Субтитры DimaTorzok", "Редактор субтитров …", "Корректор …", "Продолжение следует…" and similar. They appear mid-sentence, so the phrase is cut out and the surrounding speech kept; a segment that was nothing else is dropped.
- Tests no longer read the developer's `~/.config/tapeback/.env`. `Settings` declares that file as a source, so every test constructing `Settings()` inherited the ambient machine configuration — a machine with `TAPEBACK_CHUNK_LENGTH=2` set produced different results than CI, and assertions on default values were silently machine-dependent.

## [0.9.7] — 2026-06-18

### Added
- `tapeback-cuda` package (AUR + `.deb`) — installs the CUDA 12 cuBLAS/cuDNN runtime into the bundled venv, so GPU transcription works on CUDA 13 systems without manual setup. `yay -S tapeback-cuda` (or the matching `.deb`).

## [0.9.6] — 2026-06-18

### Added
- Per-stage timing in processing output — merge, split, load model, transcribe (mic and monitor separately), diarize, and summarize each report how long they took.
- Settings to tame wrong-language detection and hallucinations on quiet channels: `TAPEBACK_LANGUAGE_DETECTION_SEGMENTS`, `TAPEBACK_MULTILINGUAL` (per-segment detection for code-switching), and `TAPEBACK_HALLUCINATION_SILENCE_THRESHOLD`.
- Optional batched inference (`TAPEBACK_BATCH_SIZE`, off by default) — faster-whisper's `BatchedInferencePipeline`, several× faster transcription on GPU.

### Changed
- Faster post-recording processing: dropped a redundant ffmpeg pass that mixed both channels into a mono file the dual-channel pipeline never used.
- Faster transcription: default beam size lowered 5→4. The temperature fallback ladder is now exposed via `TAPEBACK_TEMPERATURE` (default keeps the full ladder, which breaks Whisper out of hallucination loops on noisy audio).
- Cleaner, faster mic channel: the mic is now silenced where you're only listening (mic quiet / monitor dominant) before transcription, so Whisper no longer hallucinates repeat loops on the pauses. Toggle with `TAPEBACK_GATE_MIC_SILENCE`.
- Whisper model loads from the local cache without contacting HuggingFace on every start — faster startup and no hang when offline (after the first download).

### Fixed
- CPU fallback during transcription and diarization now triggers only on real CUDA / out-of-memory / cuBLAS errors; unrelated failures surface instead of being masked by a slow CPU retry.
- The CUDA error that triggers a CPU fallback is now printed in full, so an out-of-memory failure can be told apart from a cuDNN/driver problem.
- GPU transcription now works on CUDA 13 systems: tapeback preloads the CUDA 12 cuBLAS/cuDNN libraries (nvidia-cublas-cu12, nvidia-cudnn-cu12) on startup so ctranslate2 finds them — no manual LD_LIBRARY_PATH needed.

## [0.9.5] — 2026-05-21

### Fixed
- Tray icon now actually appears on GNOME Wayland (with the AppIndicator extension installed).
v0.9.4 registered with the watcher but the icon stayed invisible because our `WindowId` property was exposed as `u` (unsigned int) — the SNI spec mandates `i` (signed int), and GNOME Shell silently rejects mismatched signatures (`type u does not match expected type i`).
Fixed to `i`; regression test locks the signature so it can't regress.
- Added the AppIndicator accessibility-description SNI extensions (`IconAccessibleDesc`, `AttentionAccessibleDesc`, `OverlayIconAccessibleDesc`). Newer hosts query these and dbus-next raised `DBusError(UNKNOWN_PROPERTY)` when we didn't implement them; the error is gone now.
- `tapeback tray` no longer prints the AppIndicator hint twice. v0.9.4 both logged the message and printed it to stderr; the duplicate is removed (only the logger.warning remains).

### Changed
- AppIndicator hint message is now distro-neutral: lists install commands for Ubuntu/Debian (`apt`), Fedora/RHEL (`dnf`), and Arch (`yay`) instead of only Ubuntu.

## [0.9.4] — 2026-05-21

### Fixed
- `tapeback tray` on modern desktops (GNOME, KDE Plasma): rewrote the tray icon on top of [dbus-next](https://github.com/altdesktop/python-dbus-next) and a custom StatusNotifierItem / DBusMenu implementation. The previous pystray-based path defaulted to the legacy XEmbed protocol, which GNOME/KDE no longer provide on Wayland — the icon either failed to dock at all (`AssertionError: _systray_manager is None`) or appeared as a dead grey circle whose menu didn't open. pystray's AppIndicator backend needs system PyGObject (`gi`), which the bundled `.deb` Python can't access. The new SNI-direct implementation works in the bundled venv with no system C extensions. KDE Plasma works out of the box; GNOME still needs the AppIndicator Support extension installed (one-time, GNOME-side limitation, not ours).
- `.deb` build: switched from `uv python install` + `cp -a` to direct download of [python-build-standalone](https://github.com/astral-sh/python-build-standalone) by pinned date/version. The previous approach depended on `uv python find`'s layout, which differed between local and CI environments — the CI-built `.deb` had a broken bundled Python tree and `tapeback --version` failed with `exec: not found`. Direct download is byte-deterministic and fails fast if the tarball layout changes.

### Changed
- `tapeback-tray` optional-dependency now pulls in `dbus-next` instead of `pystray` + `Pillow`. The tray .deb's postinstall hook reflects the same.
- README install snippet for `.deb`: replaced the dynamic-version `curl | grep` snippet with a plain `VERSION=X.Y.Z` + `wget` + `apt install` form. Simpler to read, simpler to copy-paste.

### Added
- `.github/workflows/deb-e2e.yml` — PR-time end-to-end .deb smoke on a 5-image matrix (Ubuntu 22.04 / 24.04 / 26.04, Debian 12 / 13). Catches packaging regressions before they reach a release tag.
- `scripts/check-workflow-pins.py` + new CI step — validates every SHA-pinned GitHub Action against the GitHub API. Prevents a hallucinated 40-char hex from quietly slipping through to a release runner.
- `docs/release-testing.md` — layered checklist (local docker smoke → CI PR gate → optional `v0.9.X-rcN` tag → manual acceptance) so the .deb path is gated before every release.

## [0.9.3] — 2026-05-20

### Fixed
- Tray on GNOME Wayland: when `tapeback tray` starts on a GNOME-family Wayland session, it now prints an actionable warning explaining that the AppIndicator Support extension is required (the legacy XDG StatusNotifier protocol was removed upstream in GNOME 45+, so without the extension the icon appears but the menu does not respond). KDE Plasma Wayland is unaffected. Closes [#3](https://github.com/yastcher/tapeback/issues/3).

### Added
- Debian/Ubuntu `.deb` packages: built in CI via [nfpm](https://nfpm.goreleaser.com/) and attached to every GitHub Release. Mirrors the Arch split: `tapeback`, `tapeback-tray`, `tapeback-llm`, `tapeback-diarize`. Install with `sudo apt install ./tapeback_*.deb`. The base `.deb` bundles a standalone Python 3.13 (~75 MB extracted) from [python-build-standalone](https://github.com/astral-sh/python-build-standalone), so it works on any modern Ubuntu/Debian regardless of the system Python — including Ubuntu 26.04 LTS, which ships only Python 3.14 and would otherwise break our 3.13-tagged compiled wheels (faster-whisper, ctranslate2, pyav).
- `tapeback --version` flag — reads the installed package version via `importlib.metadata`.

## [0.9.2] — 2026-05-04

### Fixed
- `TAPEBACK_COMPUTE_TYPE=auto` always picked `int8` on 4 GiB cards: the 4096 MiB threshold could never be reached because the card's total VRAM is right at that limit.
`large-v3-turbo` actually needs ~1.5 GiB in float16, so the auto-quantize was spurious.
Auto now resolves purely from device (`cuda` → `float16`, `cpu` → `int8`); pin `TAPEBACK_COMPUTE_TYPE=int8` explicitly if your GPU is genuinely memory-tight.

### Changed
- Live transcription is now opt-in (`TAPEBACK_LIVE` defaults to `false`). Mid-recording transcription competes with the post-recording pipeline for GPU memory on small cards (4 GiB), causing long stalls. Set `TAPEBACK_LIVE=true` to re-enable; `--no-live` still works as a one-shot override.

## [0.9.1] — 2026-05-03

### Fixed
- CPU fallback didn't trigger when faster-whisper raised `RuntimeError` (e.g. `Library libcublas.so.12 is not found`) synchronously from `transcribe()` — eager language detection raises before yielding the segment generator, so the previous fallback (which only wrapped iteration) missed it. Both call-time and iteration-time CUDA failures now fall back to CPU and the recording survives.

## [0.9.0] — 2026-04-20

### Security
- `hf_token` and `llm_api_key` now stored as `pydantic.SecretStr` — prevents leakage through `repr(settings)`, tracebacks, or `model_dump_json()` output
- Path-traversal guard on `tapeback process --name`: session names are now validated (only `[\w-]+` allowed) before being used as vault path components; vault I/O also verifies the resolved destination stays under `vault_path`
- Atomic markdown writes: `save_markdown_to_vault` now uses write-temp + rename so Obsidian can't read a half-written transcript if tapeback crashes mid-write
- Upper version bounds pinned on all dependencies (`<2`, `<3`, etc.) to prevent unreviewed major-version upgrades from breaking the build

### Added
- Live transcription: Whisper transcribes audio in real-time during recording, writing a live markdown file to the vault that can be opened mid-meeting
- `--no-live` CLI flag to disable live transcription and use the old post-recording-only mode
- `TAPEBACK_LIVE` setting (default `true`) — enable/disable live transcription
- `TAPEBACK_LIVE_INTERVAL` setting (default `60`) — seconds between transcription cycles
- `TAPEBACK_LIVE_OVERLAP` setting (default `2.0`) — seconds of overlap between chunks for seamless transitions
- `TAPEBACK_LIVE_MIN_CHUNK` setting (default `5.0`) — minimum new audio (seconds) before triggering a transcription cycle
- `TAPEBACK_NO_SPEECH_THRESHOLD` setting (default `0.4`) — Whisper silence-rejection threshold; lower values suppress training-data hallucinations like "Субтитры DimaTorzok" on long pauses

### Fixed
- CPU fallback lost auto language detection: passed `"auto"` string to Whisper instead of `None`, causing errors on non-English transcripts
- Duplicate "## Diarized Transcript" section when diarization was skipped (via `--no-diarize` or missing HF token) — both sections were identical; now only "## Transcript" is rendered
- Whisper hallucinations on long pauses (e.g. "Субтитры DimaTorzok", "Продолжение следует") — `no_speech_threshold` now set to `0.4` (stricter than Whisper's default `0.6`)

### Changed
- Settings now fail-fast on invalid values: thresholds must be in `[0, 1]`, `pause_threshold` / `live_overlap` must be non-negative, `live_interval` / `live_min_chunk` must be positive, and `live_min_chunk` must be ≤ `live_interval` when live transcription is enabled — surfaced via `pydantic.ValidationError` at `get_settings()` instead of silent mis-behaviour deep in the pipeline
- Internal refactor: extracted `tapeback._gpu` (CUDA memory helper), `tapeback._lazy` (single lazy-load site for `Transcriber`), and `tapeback.speaker_merge` (spectral clustering) — no user-visible change, but `diarizer.py` and `channel.py` are now under the 500-line limit and no longer need `PLR0912` / `PLR0915` ignores
- Default language changed from `en` to `auto` — Whisper now auto-detects the spoken language
- `tapeback start` now detects when recording stops (e.g. via `tapeback stop`) using a polling loop instead of `signal.pause()`
- `TAPEBACK_CHUNK_LENGTH` default raised from `2` to `7`: 2-second chunks fragment Whisper's context and cause hallucinations and broken sentences on non-English speech; `7` balances context against hallucination risk on long pauses
- Low-confidence word threshold (italic marker) lowered from `0.5` to `0.35`: fewer false positives on English loanwords inside Russian/mixed-language speech

## [0.8.10] — 2026-04-04

### Fixed
- Speaker misattribution: words from one monitor speaker assigned to another; switched from whole-segment majority vote to word-level diarization split
- Interleaved single-word segments during simultaneous speech; consecutive same-speaker segments now consolidated
- False extra speaker from echo/crosstalk: minor speakers (< 15s and < 20% of dominant) absorbed with lower merge threshold (0.92)
- Headphone bleed falsely attributed to "You": crosstalk filter drops mic segments where monitor channel is louder

### Added
- Two-section transcript output: raw Whisper transcript (## Transcript) then diarized (## Diarized Transcript) for comparison
- Low-confidence word marking: Whisper words with probability < 0.5 shown in *italics*
- VRAM pre-check before diarization: skips CUDA when free VRAM < 1500 MiB to avoid slow OOM fallback
- Decomposed `diarizer.py` into `diarizer.py` + `channel.py` for channel-related utilities

### Changed
- `TAPEBACK_CHUNK_LENGTH` default lowered from `15` to `2` for finer VAD granularity

## [0.8.9] — 2026-04-02

### Fixed
- Lost speech after long pauses: Whisper VAD merged all speech chunks into one stream, losing speakers separated by silence; added `chunk_length=15` to split VAD output before transcription

### Added
- `TAPEBACK_CHUNK_LENGTH` setting (default `15s`) — max VAD chunk size before splitting for Whisper

## [0.8.8] — 2026-04-02

### Added
- Auto VRAM detection: `TAPEBACK_COMPUTE_TYPE=auto` (new default) picks `int8` when free GPU memory < 4 GiB, avoiding CUDA OOM with fallback to slow CPU
- `TAPEBACK_SPECTRAL_MERGE_THRESHOLD` setting (default `0.96`) for speaker merging sensitivity

## [0.8.7] — 2026-04-01

### Fixed
- Speaker diarization: two different speakers incorrectly merged into one; raised spectral merging cosine similarity threshold from 0.92 to 0.95
- PyAV `UnicodeDecodeError` crash on non-English locales: `os.environ` alone doesn't change the C locale after Python startup; added `locale.setlocale(LC_MESSAGES, "C")` to actually switch glibc's `strerror_r()` output to ASCII

## [0.8.6] — 2026-04-01

### Fixed
- PyAV `UnicodeDecodeError` crash on non-English locales (e.g. Russian): set `LC_MESSAGES=C` to force ASCII error messages from `strerror_r()`

## [0.8.5] — 2026-03-31

### Added
- AUR package `tapeback-tray`: system tray icon as separate meta-package
- `scripts/aur-publish.sh` now publishes all 4 AUR packages (added tapeback-tray)

### Changed
- Consolidated unit tests into integration flow tests, moved shared fixtures to `tests/fixtures.py`

## [0.8.4] — 2026-03-30

### Added
- System tray icon (`tapeback tray`): start/stop recording from the tray, no terminal needed
- `[tray]` optional extra: `uv pip install tapeback[tray]` (pystray + Pillow)

### Changed
- Extracted `LLMProvider` type alias from inline Literal in settings.py
- Replaced monkey-patched `Exception.status_code` in tests with proper `_HttpError` class
- Removed all `[[tool.ty.overrides]]` sections from pyproject.toml — fixed root causes instead
- Moved fixtures to `tests/fixtures.py`, reduced local imports in `pipeline.py`
- Replaced `assert` in summarizer with explicit `RuntimeError` check
- Diarizer: replaced `**kwargs` dispatch with explicit `_run_pipeline()` method

## [0.8.3] — 2026-03-29

### Fix
- install from AUR now worked

## [0.8.2] — 2026-03-29

### Added
- `scripts/aur-publish.sh`: one-command AUR update for all 3 packages (tapeback, tapeback-llm, tapeback-diarize)

### Changed
- `scripts/release.sh` now shows AUR publish step in next-steps output

## [0.8.1] — 2026-03-28

### Changed
- AUR publishing workflow documented in release process

## [0.8.0] — 2026-03-27

### Added
- `scripts/release.sh`: version bump across pyproject.toml, PKGBUILD
- AUR packages: `tapeback-diarize`, `tapeback-llm` (optional extras as separate packages)
- Version validation step in publish.yml (tag must match pyproject.toml)
- Lint + tests run before PyPI publish

### Changed
- **PyPI package renamed to `tapeback`**, CLI command `tapeback`, Python module renamed to `tapeback`
- CLI command renamed from `meetrec` → `echo-vault` → `tapeback` (entry point, help texts, temp dirs, state dir)
- Speaker diarization (pyannote/torch) moved to optional dependency: `tapeback[diarize]`
- LLM SDKs (anthropic/openai) moved to optional dependency: `tapeback[llm]`
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
- LLM provider fallback chain: if primary provider fails, tries next available provider
- Spectral similarity speaker merging to reduce pyannote over-segmentation of single speakers
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
- Markdown fence stripping: LLM responses wrapped in ```json``` now parsed correctly

## [0.6.0] — 2026-03-23

### Added
- Free/cheap LLM providers: Groq, Gemini, OpenRouter, DeepSeek, Qwen
- Retry with exponential backoff on 429/529 rate limit errors (3 retries, 5s→10s→20s)
- `models.py`: domain objects extracted into dedicated module
- `vault.py`: Obsidian vault I/O separated from formatting
- Session name validation (alphanumerics, dashes, underscores only)
- Restrictive permissions (0700) on `/tmp/tapeback` temp directories
- mypy strict mode
- Coverage threshold (85%) enforced in CI

### Changed
- `formatter.py`: pure formatting only, no I/O (moved to `vault.py`)
- `diarizer.py` no longer depends on `transcriber.py`, both import from `models.py`
- `summarizer.py` no longer defines domain models, imports from `models.py`
- Expanded ruff rules: W, G, PLC, PLR, S (security)

## [0.5.0] — 2026-03-22

### Added
- LLM summarization: brief summary, action items, key decisions
- `summarize` CLI command for re-summarizing existing transcripts
- `--no-summarize` flag for `start` and `process` commands
- Anthropic and OpenAI provider support
- Non-fatal summarization, transcript always saved first

### Changed
- Python 3.14 migration (PEP 758 exception syntax)
- Testing Trophy refactoring: more integration tests, fewer unit tests

### Fixed
- Audio device hot-switching: `@DEFAULT_MONITOR@`/`@DEFAULT_SOURCE@` follow device changes

## [0.4.0] — 2026-03-21

### Added
- Pause detection: split segments on word gaps >= threshold (configurable)
- CLI integration tests via CliRunner with mocked ML models
- Coverage reporting enabled by default in pytest

### Changed
- Strict typing refactoring across all modules
- Trophy testing approach: integration tests as primary coverage

### Fixed
- Segment splitting at actual silence gaps in raw mic audio

## [0.3.0] — 2026-03-18

### Added
- Stereo channel support: per-channel transcription (mic + monitor)
- Audio normalization (loudnorm) before transcription
- RMS-based crosstalk filtering to reject Whisper hallucinations on silent channels
- Channel-based speaker attribution (mic=You, monitor=Others)

### Fixed
- GPU memory management: free CUDA memory between transcription and diarization

## [0.2.0] — 2026-03-17

### Added
- Speaker diarization via pyannote, identifies "You" vs remote participants
- Speaker identification using mic/monitor RMS energy ratio
- Dual-channel recording (monitor + mic as separate WAV files)

## [0.1.0] — 2026-03-17

### Added
- Initial release
- Dual-channel audio recording via parecord (PulseAudio/PipeWire)
- Local transcription with faster-whisper (CUDA with CPU fallback)
- Markdown output with YAML frontmatter and `[HH:MM:SS]` timecodes
- Obsidian vault integration: saves audio + markdown
- pydantic-settings configuration with `TAPEBACK_` prefix
- `start`, `stop`, `process`, `status` CLI commands
