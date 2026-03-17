# meetrec

Local meeting audio recorder with transcription for Obsidian.

Records monitor source (other participants) + microphone (your voice) on Linux via PulseAudio/PipeWire, transcribes locally with faster-whisper, and saves markdown notes to your Obsidian vault.

## Requirements

- Linux with PulseAudio or PipeWire
- Python 3.11+
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
| `meetrec start [NAME] [--no-diarize]` | Start recording (Ctrl+C to stop and transcribe) |
| `meetrec stop` | Stop recording, transcribe, save to vault |
| `meetrec process FILE [--name NAME] [--no-diarize]` | Process an existing audio file |
| `meetrec status` | Show recording status and settings |

## Configuration

All settings via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `MEETREC_VAULT_PATH` | *required* | Path to Obsidian vault |
| `MEETREC_WHISPER_MODEL` | `large-v3-turbo` | Whisper model name |
| `MEETREC_LANGUAGE` | `en` | Transcription language |
| `MEETREC_DEVICE` | `cuda` | Compute device (cuda/cpu) |
| `MEETREC_COMPUTE_TYPE` | `float16` | Model precision |
| `MEETREC_MONITOR_SOURCE` | `auto` | PulseAudio monitor source |
| `MEETREC_MIC_SOURCE` | `auto` | PulseAudio microphone source |
| `MEETREC_SAMPLE_RATE` | `48000` | Audio sample rate |
| `MEETREC_HF_TOKEN` | *(empty)* | HuggingFace token for speaker diarization |
| `MEETREC_DIARIZE` | `true` | Enable speaker diarization |
| `MEETREC_MAX_SPEAKERS` | *(auto)* | Max speakers hint for pyannote |

## Speaker Diarization

meetrec can identify who said what using [pyannote](https://github.com/pyannote/pyannote-audio).

**Setup:**

1. Accept pyannote model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
2. Create a HuggingFace token at https://huggingface.co/settings/tokens
3. Add to `.env`: `MEETREC_HF_TOKEN=hf_your_token_here`

When recording with two channels (mic + monitor), meetrec automatically identifies
your voice ("You") vs other participants ("Speaker 1", "Speaker 2", ...).

Without the token, transcription works normally — just without speaker labels.

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

# Meeting 2026-03-17 14:30

**Duration:** 1h 23m 45s | **Language:** en

---

[00:01:23] Lorem ipsum dolor sit amet.

[00:02:45] Sed do eiusmod tempor incididunt.
```

Stereo WAV archive saved in `{vault}/attachments/audio/`.

## Development

```bash
uv sync
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run pytest
```

## License

Proprietary. All rights reserved.
