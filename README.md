# Echo vault (meetrec)

Local meeting audio recorder with transcription for Obsidian.

Records monitor source (other participants) + microphone (your voice) on Linux via PulseAudio/PipeWire, transcribes locally with faster-whisper, identifies speakers with pyannote, and saves markdown notes to your Obsidian vault.

## Features

- Records system audio (monitor) + microphone simultaneously — survives audio device hot-switching
- Transcribes locally with faster-whisper (CUDA with CPU fallback)
- LLM summarization — brief summary, action items, key decisions (Anthropic/OpenAI)
- Speaker diarization via pyannote — identifies "You" vs other speakers
- Audio channel normalization — mic and monitor balanced before transcription
- Saves audio to vault immediately, before transcription starts
- Markdown output with YAML frontmatter and `[HH:MM:SS]` timecodes
- Processes pre-recorded audio files (mp3, m4a, ogg, wav)

## Requirements

- Linux with PulseAudio or PipeWire
- Python 3.14+
- ffmpeg (`sudo apt install ffmpeg`)
- parecord (`sudo apt install pulseaudio-utils`)
- NVIDIA GPU recommended (CPU fallback available)

## Quick Start

```bash
# Install
pip install uv
uv sync

# Configure
cp .env.example .env
# Edit .env — set MEETREC_VAULT_PATH

# Record a meeting
meetrec start

# Stop and transcribe (or press Ctrl+C)
meetrec stop

# Process an existing audio file
meetrec process recording.mp3
```

## Commands

| Command | Description |
|---------|-------------|
| `meetrec start [NAME] [--no-diarize] [--no-summarize]` | Start recording (Ctrl+C to stop and transcribe) |
| `meetrec stop` | Stop recording, transcribe, save to vault |
| `meetrec process FILE [--name NAME] [--no-diarize] [--no-summarize]` | Process an existing audio file |
| `meetrec summarize FILE [--provider anthropic\|openai] [--model MODEL]` | Summarize an existing transcript |
| `meetrec status` | Show recording status and settings |

## Configuration

All settings via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `MEETREC_VAULT_PATH` | *required* | Path to Obsidian vault |
| `MEETREC_WHISPER_MODEL` | `large-v3-turbo` | Whisper model name |
| `MEETREC_LANGUAGE` | `en` | Transcription language (`auto` for detection) |
| `MEETREC_DEVICE` | `cuda` | Compute device (cuda/cpu) |
| `MEETREC_COMPUTE_TYPE` | `float16` | Model precision |
| `MEETREC_MONITOR_SOURCE` | `auto` | PulseAudio monitor source (`auto` uses `@DEFAULT_MONITOR@` — follows device switches) |
| `MEETREC_MIC_SOURCE` | `auto` | PulseAudio microphone source (`auto` uses `@DEFAULT_SOURCE@`) |
| `MEETREC_SAMPLE_RATE` | `48000` | Audio sample rate |
| `MEETREC_HF_TOKEN` | *(empty)* | HuggingFace token for speaker diarization |
| `MEETREC_DIARIZE` | `true` | Enable speaker diarization |
| `MEETREC_MAX_SPEAKERS` | *(auto)* | Max speakers hint for pyannote |
| `MEETREC_CLUSTERING_THRESHOLD` | *(pyannote default)* | Speaker clustering threshold (higher = fewer speakers, try 0.85 if over-segmented) |
| `MEETREC_PAUSE_THRESHOLD` | `1.0` | Seconds — split segments at silence gaps longer than this (lower = more granular segments) |
| `MEETREC_MEETINGS_DIR` | `meetings` | Subdirectory in vault for transcripts |
| `MEETREC_ATTACHMENTS_DIR` | `attachments/audio` | Subdirectory in vault for audio files |
| `MEETREC_BEAM_SIZE` | `5` | Whisper beam size |
| `MEETREC_VAD_FILTER` | `true` | Voice activity detection filter |
| `MEETREC_CONDITION_ON_PREVIOUS_TEXT` | `false` | Whisper conditions on prior text (disable to preserve repeated speech) |
| `MEETREC_SUMMARIZE` | `true` | Enable LLM summarization after transcription |
| `MEETREC_LLM_PROVIDER` | `anthropic` | LLM provider (see [Summarization](#summarization) for full list) |
| `MEETREC_LLM_API_KEY` | *(empty)* | API key (fallback: provider-specific env var, see below) |
| `MEETREC_LLM_MODEL` | *(provider default)* | Model name override |

## Speaker Diarization

meetrec can identify who said what using [pyannote](https://github.com/pyannote/pyannote-audio).

**Setup:**

1. Accept model terms (all three required):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/segmentation-3.0
2. Create a HuggingFace token at https://huggingface.co/settings/tokens
3. Add to `.env`: `MEETREC_HF_TOKEN=hf_your_token_here`

When recording with two channels (mic + monitor), meetrec automatically identifies
your voice ("You") vs other participants ("Speaker 1", "Speaker 2", ...).

Without the token, transcription works normally — just without speaker labels.

## Summarization

After transcription, meetrec sends the transcript to an LLM to extract:
- Brief summary (2-3 sentences)
- Action items (who promised what, with deadlines if mentioned)
- Key decisions

**Supported providers:**

| Provider | `MEETREC_LLM_PROVIDER` | Env var fallback | Default model |
|----------|------------------------|------------------|---------------|
| Anthropic | `anthropic` (default) | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` |
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-4o` |
| Groq | `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| Google Gemini | `gemini` | `GEMINI_API_KEY` | `gemini-2.0-flash` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | `google/gemma-3-27b-it:free` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| Qwen | `qwen` | `DASHSCOPE_API_KEY` | `qwen-turbo` |

**Setup:**

Set an API key via `MEETREC_LLM_API_KEY` or the provider-specific env var:
```bash
# Anthropic (default)
export ANTHROPIC_API_KEY=sk-ant-...

# Free alternative — Groq
export MEETREC_LLM_PROVIDER=groq
export GROQ_API_KEY=gsk_...

# Free alternative — OpenRouter
export MEETREC_LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-...
```

Without an API key, summarization is silently skipped — the transcript is saved normally.

Use `meetrec summarize transcript.md` to (re-)summarize an existing transcript file.
Use `--no-summarize` with `start` or `process` to skip summarization for a single run.

## How It Works

### Live recording (dual-channel pipeline)

1. **Recording** — parecord captures monitor + mic as separate WAV files (uses `@DEFAULT_MONITOR@`/`@DEFAULT_SOURCE@` to follow device switches)
2. **Merging** — ffmpeg creates stereo WAV (left=mic, right=monitor)
3. **Audio saved** — stereo WAV copied to vault immediately
4. **Channel splitting** — stereo split into two mono 16kHz WAVs, each independently normalized (loudnorm)
5. **Dual transcription** — faster-whisper transcribes each channel separately: mic segments are labeled "You", monitor segments are labeled by pyannote
6. **Pause splitting** — mic segments are split at silence gaps detected via raw audio RMS (adaptive threshold + monitor-relative check), so mic and monitor segments interleave chronologically
7. **Hallucination filter** — segments where the raw channel RMS is below noise floor are dropped (catches Whisper hallucinations on silent portions)
8. **GPU memory freed** — Whisper model unloaded before diarization
9. **Diarization** — pyannote runs on monitor channel only to distinguish remote speakers ("Speaker 1", "Speaker 2", ...)
10. **Merge** — segments from both channels combined and sorted by timestamp; overlapping segments (simultaneous speech) are preserved
11. **Output** — markdown with timecodes and speaker labels saved to vault
12. **Summarization** — transcript sent to LLM (Anthropic/OpenAI) for summary, action items, and key decisions (if API key configured)

This dual-channel approach avoids mono mixing, which degrades both voices during simultaneous speech and can lose quiet-channel audio entirely.

### Processing existing files

`meetrec process` converts any audio file to mono 16kHz and runs the original single-channel pipeline (Whisper + pyannote on mono).

CUDA is used when available; each model (Whisper, pyannote) runs sequentially to fit in limited VRAM. Automatic CPU fallback on OOM or missing CUDA libraries.

## Output

Markdown files with YAML frontmatter in `{vault}/meetings/`:

```markdown
---
date: 2026-03-17
time: "14:30"
duration: "01:23:45"
language: en
audio: "[[attachments/audio/2026-03-17_14-30-00.wav]]"
tags:
  - meeting
  - transcript
---

## Summary

Discussed roadmap priorities and assigned tasks for the upcoming sprint.

### Action Items

- [ ] **Speaker 1:** Prepare backend migration plan by Monday
- [ ] **You:** Review the slides and send feedback

### Key Decisions

- Start with backend changes before frontend redesign

---

# Meeting 2026-03-17 14:30

**Duration:** 1h 23m 45s | **Language:** en

---

[00:01:23] **You:** Let's discuss the roadmap.

[00:01:30] **Speaker 1:** Sure, I have the slides ready.

[00:02:45] **Speaker 2:** Can we start with the backend changes?
```

Stereo WAV archive saved in `{vault}/attachments/audio/`.

## Development

```bash
uv sync --group dev
uv run ruff check
uv run ruff format --check
uv run pytest
```

Coverage report is enabled by default (`--cov=meetrec --cov-report=term-missing` in pyproject.toml).

### Test Structure

```
tests/
├── fixtures.py          # Shared fixtures and WAV helpers
├── conftest.py          # Plugin registration for fixtures
├── test_cli.py          # CLI integration tests (CliRunner + mocked ML models)
├── test_integration.py  # Multi-module pipeline integration tests
├── test_audio.py        # ffmpeg audio processing
├── test_diarizer.py     # Speaker diarization and channel analysis
├── test_formatter.py    # Markdown formatting and vault I/O
├── test_recorder.py     # PulseAudio recording control
├── test_settings.py     # Configuration / env vars
├── test_summarizer.py   # LLM summarization, markdown injection
├── test_transcriber.py  # Whisper transcription
└── regressions/         # Bug-fix regression tests
    ├── test_diarizer_regressions.py
    ├── test_recorder_regressions.py
    └── test_transcriber_regressions.py
```

Tests follow the Testing Trophy approach — integration tests cover CLI commands and multi-module pipelines, unit tests cover individual functions. ML models (Whisper, pyannote) are always mocked; ffmpeg and file I/O run for real. Tests requiring ffmpeg are skipped when it's not installed (`@pytest.mark.skipif`).

## License

Proprietary. All rights reserved.
