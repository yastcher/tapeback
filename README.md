# echo-vault

Local meeting recorder for Linux. Records system audio + microphone via
PipeWire/PulseAudio, transcribes with Whisper, identifies speakers, saves
Markdown to your Obsidian vault. Everything runs on your machine — no cloud,
no bots, no API calls for transcription.

Works with any video call platform: Google Meet, Zoom, Teams, Telegram, Discord, Slack huddles.

<!-- TODO: Add demo GIF here -->
<!-- ![echo-vault demo](docs/demo.gif) -->

## Features

- **Platform-agnostic** — captures OS-level audio, works with any app
- **Local transcription** — faster-whisper, runs on CPU or CUDA GPU
- **Speaker diarization** — pyannote identifies who said what
- **Stereo channel separation** — your mic (left) vs. others (right) for accurate "You" attribution
- **Obsidian-native output** — Markdown with YAML frontmatter, wikilinks to audio files
- **LLM summarization** — optional summaries via Anthropic, OpenAI, Groq, Gemini, DeepSeek, OpenRouter, Qwen (with automatic provider fallback)
- **CLI-first** — `echo-vault start`, Ctrl+C to stop, done

## Requirements

- Linux (PipeWire or PulseAudio)
- Python 3.13+
- ffmpeg
- parecord (usually comes with `pulseaudio-utils` or `pipewire-pulse`)
- NVIDIA GPU (optional, for faster transcription and diarization)

## Installation

### 1. System dependencies

```bash
# Arch / Manjaro
sudo pacman -S python uv ffmpeg pipewire-pulse

# Ubuntu / Debian
sudo apt install python3 pipx ffmpeg pulseaudio-utils
pipx ensurepath  # adds ~/.local/bin to PATH

# Fedora
sudo dnf install python3 pipx ffmpeg pipewire-pulseaudio
pipx ensurepath
```

### 2. Install echo-vault

The base package records audio and transcribes locally. Optional extras add
speaker diarization and LLM summaries:

| Extra | What it adds | Size |
|---|---|---|
| *(none)* | Recording + transcription | ~150 MB |
| `[llm]` | LLM summarization (Anthropic, OpenAI, Gemini, etc.) | +50 MB |
| `[diarize]` | Speaker diarization (pyannote + PyTorch) | +2 GB |
| `[llm,diarize]` | Everything | +2 GB |

#### With uv (recommended)

```bash
uv tool install echo-vault                  # basic
uv tool install "echo-vault[llm]"           # + summaries
uv tool install "echo-vault[diarize]"       # + speaker diarization
uv tool install "echo-vault[llm,diarize]"   # everything
```

#### With pipx

```bash
pipx install echo-vault                     # basic
pipx install "echo-vault[llm]"              # + summaries
pipx install "echo-vault[diarize]"          # + speaker diarization
pipx install "echo-vault[llm,diarize]"      # everything
```

#### Arch Linux (AUR)

```bash
yay -S echo-vault                  # basic
yay -S echo-vault-llm              # + summaries
yay -S echo-vault-diarize          # + speaker diarization (~2 GB PyTorch)
```

#### Nix

```bash
nix run github:yastcher/echo-vault              # basic
nix run github:yastcher/echo-vault#llm          # + summaries
nix run github:yastcher/echo-vault#diarize      # + speaker diarization
nix run github:yastcher/echo-vault#full         # everything
```

#### From source (development)

```bash
git clone https://github.com/yastcher/echo-vault
cd echo-vault
uv sync --group dev    # all dependencies + dev tools
```

### 3. Uninstall

```bash
# Remove echo-vault
uv tool uninstall echo-vault    # if installed with uv
pipx uninstall echo-vault       # if installed with pipx

# Remove cached ML models (~2-5 GB)
# ⚠ Skip if you have other HuggingFace projects
rm -rf ~/.cache/huggingface/

# Arch Linux
yay -R echo-vault echo-vault-diarize echo-vault-llm
```

## Quick start

### 1. Configure

```bash
# Required: set your Obsidian vault path
export MEETREC_VAULT_PATH=~/Documents/obsidian/vault

# Or create a .env file in the project root:
echo 'MEETREC_VAULT_PATH=~/Documents/obsidian/vault' > .env
```

### 2. Record a meeting

```bash
# Start recording (blocks, Ctrl+C to stop and transcribe)
echo-vault start

# Optionally give the session a name
echo-vault start "weekly-standup"
```

### 3. Check your vault

The recording is saved as a Markdown note with audio attachment:

```
vault/
  meetings/2026-03-23_14-30-00.md
  attachments/audio/2026-03-23_14-30-00.wav
```

### Process an existing recording

```bash
# Transcribe any audio file (mp3, m4a, ogg, wav)
echo-vault process meeting.mp3

# With options
echo-vault process call.wav --name "client-call" --no-diarize
```

### Add summary to existing transcript

```bash
echo-vault summarize vault/meetings/2026-03-23.md
echo-vault summarize transcript.md --provider gemini
```

## Configuration

All settings via environment variables (prefix `MEETREC_`) or `.env` file.
Copy `.env.example` to `.env` and adjust:

```bash
cp .env.example .env
```

### Core

| Variable | Default | Description |
|---|---|---|
| `MEETREC_VAULT_PATH` | *(required)* | Path to Obsidian vault |
| `MEETREC_MEETINGS_DIR` | `meetings` | Subdirectory for meeting notes |
| `MEETREC_ATTACHMENTS_DIR` | `attachments/audio` | Subdirectory for audio files |

### Transcription

| Variable | Default | Description |
|---|---|---|
| `MEETREC_WHISPER_MODEL` | `large-v3-turbo` | Whisper model (`tiny`, `base`, `small`, `medium`, `large-v3-turbo`) |
| `MEETREC_LANGUAGE` | `en` | Transcription language code |
| `MEETREC_DEVICE` | `cuda` | `cuda` or `cpu` |
| `MEETREC_COMPUTE_TYPE` | `float16` | `float16`, `int8`, or `float32` |
| `MEETREC_BEAM_SIZE` | `5` | Whisper beam search width |
| `MEETREC_PAUSE_THRESHOLD` | `1.0` | Seconds — split segments on silence gaps >= this |

### Audio

| Variable | Default | Description |
|---|---|---|
| `MEETREC_MONITOR_SOURCE` | `auto` | PulseAudio monitor source name |
| `MEETREC_MIC_SOURCE` | `auto` | PulseAudio mic source name |
| `MEETREC_SAMPLE_RATE` | `48000` | Recording sample rate |

### Speaker diarization

| Variable | Default | Description |
|---|---|---|
| `MEETREC_DIARIZE` | `true` | Enable speaker diarization (requires `echo-vault[diarize]`) |
| `MEETREC_HF_TOKEN` | *(empty)* | HuggingFace token for pyannote models |
| `MEETREC_MAX_SPEAKERS` | *(auto)* | Maximum number of speakers |

Speaker diarization requires the `diarize` extra (`uv tool install "echo-vault[diarize]"`)
and a HuggingFace token with access to pyannote models:

1. Create account at [huggingface.co](https://huggingface.co)
2. Accept license at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Accept license at [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Accept license at [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
5. Create token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
6. Set `MEETREC_HF_TOKEN=hf_your_token_here`

Without a token, echo-vault still works — it just skips diarization.

### LLM summarization

Requires the `llm` extra: `uv tool install "echo-vault[llm]"`

| Variable | Default | Description |
|---|---|---|
| `MEETREC_SUMMARIZE` | `true` | Enable LLM summarization |
| `MEETREC_LLM_PROVIDER` | `anthropic` | Primary LLM provider |
| `MEETREC_LLM_API_KEY` | *(empty)* | API key (or use provider-specific env var) |
| `MEETREC_LLM_MODEL` | *(provider default)* | Override model name |

Supported providers and their env vars:

| Provider | Env var | Default model |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | claude-sonnet-4-20250514 |
| `openai` | `OPENAI_API_KEY` | gpt-4o |
| `groq` | `GROQ_API_KEY` | llama-3.3-70b-versatile |
| `gemini` | `GEMINI_API_KEY` | gemini-2.5-flash |
| `openrouter` | `OPENROUTER_API_KEY` | google/gemini-2.5-flash:free |
| `deepseek` | `DEEPSEEK_API_KEY` | deepseek-chat |
| `qwen` | `DASHSCOPE_API_KEY` | qwen-turbo |

If the primary provider fails, echo-vault automatically tries the next available provider (any provider with an API key set).

## CLI reference

```
echo-vault --help                    Show help and quick start guide
echo-vault start [NAME]              Start recording (Ctrl+C to stop)
echo-vault stop                      Stop recording from another terminal
echo-vault process <FILE> [--name N] Transcribe an existing audio file
echo-vault summarize <FILE>          Add LLM summary to transcript
echo-vault status                    Show recording status and settings
```

### Common options

```bash
echo-vault start --no-diarize        # Skip speaker identification
echo-vault start --no-summarize      # Skip LLM summary
echo-vault process file.mp3 --name "weekly-standup"
echo-vault summarize file.md --provider gemini --model gemini-2.5-pro
```

## Output format

```markdown
---
date: 2026-03-23
time: "14:30"
duration: "01:23:45"
language: en
tags:
  - meeting
  - transcript
---

## Summary

Brief overview of the meeting.

### Action Items

- [ ] **You:** Send the report by Friday
- [ ] **Speaker 1:** Review the PR

### Key Decisions

- Use PostgreSQL instead of MongoDB

---

# Meeting 2026-03-23 14:30

![[attachments/audio/2026-03-23_14-30-00.wav]]

[00:00:01] **You:** Hello, let's start with the backend changes.

[00:01:23] **Speaker 1:** Sure, I have the slides ready.

[00:02:45] **Speaker 2:** Can we start with the backend changes?
```

## Architecture

```
src/meetrec/
  cli.py          Click CLI — start, stop, process, summarize, status
  recorder.py     PulseAudio recording via parecord
  audio.py        ffmpeg audio processing (split channels, normalize, convert)
  transcriber.py  faster-whisper transcription
  diarizer.py     pyannote speaker diarization + spectral speaker merging
  formatter.py    Markdown generation (pure formatting, no I/O)
  vault.py        Obsidian vault file I/O
  summarizer.py   LLM summarization with multi-provider fallback
  models.py       Domain objects (Segment, Word, DiarizationSegment, Summary)
  settings.py     pydantic-settings configuration
```

## Development

```bash
git clone https://github.com/yastcher/echo-vault
cd echo-vault
uv sync --group dev

uv run ruff check       # lint
uv run ruff format      # format
uv run ty check         # type check
uv run pytest           # test (coverage >= 85%)
```

## Roadmap

Future development directions:

- **Custom diarization model** — train a speaker embedding model optimized for meeting audio, replacing generic pyannote for better accuracy
- **Windows client** — native Windows support via WASAPI loopback capture
- **Real-time transcription** — live streaming transcription with partial results
- **Web dashboard** — browser UI for reviewing and searching meeting history
- **Speaker profiles** — learn and remember recurring speakers across meetings
- **Multi-language meetings** — detect and handle language switches mid-meeting

<!-- QR codes for donations -->
<!-- TODO: Insert QR codes here -->

## License

Apache-2.0. See [LICENSE](LICENSE).
