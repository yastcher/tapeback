# Echo vault (meetrec)

Local meeting audio recorder with transcription for Obsidian.

Records monitor source (other participants) + microphone (your voice) on Linux via PulseAudio/PipeWire, transcribes locally with faster-whisper, identifies speakers with pyannote, and saves markdown notes to your Obsidian vault.

## Features

- Records system audio (monitor) + microphone simultaneously
- Transcribes locally with faster-whisper (CUDA with CPU fallback)
- Speaker diarization via pyannote — identifies "You" vs other speakers
- Audio channel normalization — mic and monitor balanced before transcription
- Saves audio to vault immediately, before transcription starts
- Markdown output with YAML frontmatter and `[HH:MM:SS]` timecodes
- Processes pre-recorded audio files (mp3, m4a, ogg, wav)

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
| `MEETREC_LANGUAGE` | `en` | Transcription language (`auto` for detection) |
| `MEETREC_DEVICE` | `cuda` | Compute device (cuda/cpu) |
| `MEETREC_COMPUTE_TYPE` | `float16` | Model precision |
| `MEETREC_MONITOR_SOURCE` | `auto` | PulseAudio monitor source |
| `MEETREC_MIC_SOURCE` | `auto` | PulseAudio microphone source |
| `MEETREC_SAMPLE_RATE` | `48000` | Audio sample rate |
| `MEETREC_HF_TOKEN` | *(empty)* | HuggingFace token for speaker diarization |
| `MEETREC_DIARIZE` | `true` | Enable speaker diarization |
| `MEETREC_MAX_SPEAKERS` | *(auto)* | Max speakers hint for pyannote |
| `MEETREC_MEETINGS_DIR` | `meetings` | Subdirectory in vault for transcripts |
| `MEETREC_ATTACHMENTS_DIR` | `attachments/audio` | Subdirectory in vault for audio files |
| `MEETREC_BEAM_SIZE` | `5` | Whisper beam size |
| `MEETREC_VAD_FILTER` | `true` | Voice activity detection filter |

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

## How It Works

1. **Recording** — parecord captures monitor + mic as separate WAV files
2. **Merging** — ffmpeg creates stereo WAV (left=mic, right=monitor)
3. **Audio saved** — stereo WAV copied to vault immediately
4. **Normalization** — channels normalized independently (loudnorm) before mixing to mono 16kHz for Whisper, so quiet mic isn't drowned out by loud monitor
5. **Transcription** — faster-whisper transcribes the normalized mono file
6. **GPU memory freed** — Whisper model unloaded before diarization
7. **Diarization** — pyannote identifies speakers, RMS energy on stereo channels determines which speaker is "You"
8. **Output** — markdown with timecodes and speaker labels saved to vault

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

### Test Structure

```
tests/
├── fixtures.py          # Shared fixtures and WAV helpers
├── regressions/         # Bug-fix regression tests
│   ├── test_diarizer_regressions.py
│   ├── test_recorder_regressions.py
│   └── test_transcriber_regressions.py
├── test_audio.py
├── test_diarizer.py
├── test_formatter.py
├── test_recorder.py
├── test_settings.py
└── test_transcriber.py
```

Tests with real ffmpeg are skipped on CI (marked with `skipif`). All other tests use mocks at system boundaries.

## License

Proprietary. All rights reserved.
