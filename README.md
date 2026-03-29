# tapeback

Local meeting recorder for Linux. Records system audio + microphone via
PipeWire/PulseAudio, transcribes with Whisper, identifies speakers, saves
Markdown to your Obsidian vault. Everything runs on your machine, no cloud
services or API calls needed for transcription.

Works with any video call platform: Google Meet, Zoom, Teams, Telegram, Discord, Slack huddles.

![tapeback in Obsidian](docs/obsidian-screenshot.png)

## Features

- **Platform-agnostic**: captures OS-level audio, works with any app
- **Local transcription**: faster-whisper on CPU or CUDA GPU
- **Speaker diarization**: pyannote identifies who said what
- **Stereo channel separation**: your mic (left) vs. others (right) for accurate "You" attribution
- **Obsidian-native output**: Markdown with YAML frontmatter, wikilinks to audio files
- **LLM summarization**: optional, via Anthropic, OpenAI, Groq, Gemini, DeepSeek, OpenRouter, Qwen (with automatic provider fallback)
- **CLI-first**: `tapeback start`, Ctrl+C to stop, done

## Requirements

- Linux (PipeWire or PulseAudio)
- Python 3.13+
- ffmpeg
- parecord (usually comes with `pulseaudio-utils` or `pipewire-pulse`)
- NVIDIA GPU (optional, for faster transcription and diarization)

## Installation

The base package records audio and transcribes locally. Optional extras add
speaker diarization and LLM summaries:

| Extra | What it adds | Size      |
|---|---|-----------|
| *(none)* | Recording + transcription | ~320 MB |
| `[llm]` | LLM summarization (Anthropic, OpenAI, Gemini, etc.) | +50 MB    |
| `[diarize]` | Speaker diarization (pyannote + PyTorch) | +2 GB     |
| `[llm,diarize]` | Everything | +2 GB     |

### Arch Linux (AUR)

```bash
yay -S tapeback                  # basic
yay -S tapeback-llm              # + summaries
yay -S tapeback-diarize          # + speaker diarization (~2 GB PyTorch)
```

### pip / uv / pipx

1. Install system dependencies:

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

2. Install tapeback:

```bash
uv tool install tapeback                  # basic
uv tool install "tapeback[llm]"           # + summaries
uv tool install "tapeback[diarize]"       # + speaker diarization
uv tool install "tapeback[llm,diarize]"   # everything
```

or with pipx:

```bash
pipx install tapeback                     # basic
pipx install "tapeback[llm,diarize]"      # everything
```

### Nix

```bash
nix run github:yastcher/tapeback              # basic
nix run github:yastcher/tapeback#llm          # + summaries
nix run github:yastcher/tapeback#diarize      # + speaker diarization
nix run github:yastcher/tapeback#full         # everything
```

## Quick start

```bash
tapeback start                     # start recording, Ctrl+C to stop
```

That's it. The transcript is saved to `~/tapeback/meetings/`.

To save to your Obsidian vault instead:

```bash
mkdir -p ~/.config/tapeback
echo 'TAPEBACK_VAULT_PATH=~/Documents/obsidian/vault' > ~/.config/tapeback/.env
```

### More examples

```bash
tapeback start "weekly-standup"    # name the session
tapeback process meeting.mp3       # transcribe an existing file
tapeback summarize transcript.md   # add LLM summary to a transcript
tapeback status                    # show current settings
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

## Roadmap

- **System tray icon**: start/stop recording from the tray, no terminal needed
- **Speaker profiles**: learn and remember recurring speakers across meetings
- **Real-time transcription**: live streaming with partial results
- **Multi-language meetings**: detect and handle language switches mid-meeting
- **Windows support**: WASAPI loopback capture

---

<details>
<summary><h2>Advanced configuration</h2></summary>

All settings via environment variables (prefix `TAPEBACK_`) or
`~/.config/tapeback/.env` file.

### Core

| Variable | Default | Description |
|---|---|---|
| `TAPEBACK_VAULT_PATH` | `~/tapeback` | Path to output directory (Obsidian vault) |
| `TAPEBACK_MEETINGS_DIR` | `meetings` | Subdirectory for meeting notes |
| `TAPEBACK_ATTACHMENTS_DIR` | `attachments/audio` | Subdirectory for audio files |

### Transcription

| Variable | Default | Description |
|---|---|---|
| `TAPEBACK_WHISPER_MODEL` | `large-v3-turbo` | Whisper model (`tiny`, `base`, `small`, `medium`, `large-v3-turbo`) |
| `TAPEBACK_LANGUAGE` | `en` | Transcription language code |
| `TAPEBACK_DEVICE` | `cuda` | `cuda` or `cpu` |
| `TAPEBACK_COMPUTE_TYPE` | `float16` | `float16`, `int8`, or `float32` |
| `TAPEBACK_BEAM_SIZE` | `5` | Whisper beam search width |
| `TAPEBACK_PAUSE_THRESHOLD` | `1.0` | Seconds; split segments on silence gaps >= this |

### Audio

| Variable | Default | Description |
|---|---|---|
| `TAPEBACK_MONITOR_SOURCE` | `auto` | PulseAudio monitor source name |
| `TAPEBACK_MIC_SOURCE` | `auto` | PulseAudio mic source name |
| `TAPEBACK_SAMPLE_RATE` | `48000` | Recording sample rate |

### Speaker diarization

| Variable | Default | Description |
|---|---|---|
| `TAPEBACK_DIARIZE` | `true` | Enable speaker diarization (requires `tapeback[diarize]`) |
| `TAPEBACK_HF_TOKEN` | *(empty)* | HuggingFace token for pyannote models |
| `TAPEBACK_MAX_SPEAKERS` | *(auto)* | Maximum number of speakers |

Speaker diarization requires the `diarize` extra (`uv tool install "tapeback[diarize]"`)
and a HuggingFace token with access to pyannote models:

1. Create account at [huggingface.co](https://huggingface.co)
2. Accept license at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Accept license at [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Accept license at [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
5. Create token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
6. Set `TAPEBACK_HF_TOKEN=hf_your_token_here`

Without a token, tapeback still works but skips diarization.

### LLM summarization

Requires the `llm` extra: `uv tool install "tapeback[llm]"`

| Variable | Default | Description |
|---|---|---|
| `TAPEBACK_SUMMARIZE` | `true` | Enable LLM summarization |
| `TAPEBACK_LLM_PROVIDER` | `anthropic` | Primary LLM provider |
| `TAPEBACK_LLM_API_KEY` | *(empty)* | API key (or use provider-specific env var) |
| `TAPEBACK_LLM_MODEL` | *(provider default)* | Override model name |

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

If the primary provider fails, tapeback automatically tries the next available provider (any provider with an API key set).

### CLI reference

```
tapeback --help                    Show help and quick start guide
tapeback start [NAME]              Start recording (Ctrl+C to stop)
tapeback stop                      Stop recording from another terminal
tapeback process <FILE> [--name N] Transcribe an existing audio file
tapeback summarize <FILE>          Add LLM summary to transcript
tapeback status                    Show recording status and settings
```

```bash
tapeback start --no-diarize        # Skip speaker identification
tapeback start --no-summarize      # Skip LLM summary
tapeback process file.mp3 --name "weekly-standup"
tapeback summarize file.md --provider gemini --model gemini-2.5-pro
```

### Uninstall

```bash
# pip / uv
uv tool uninstall tapeback

# pipx
pipx uninstall tapeback

# Arch Linux
yay -R tapeback tapeback-diarize tapeback-llm

# Remove cached ML models (~2-5 GB)
# Skip if you have other HuggingFace projects
rm -rf ~/.cache/huggingface/
```

</details>

<details>
<summary><h2>Development</h2></summary>

### Architecture

```
src/tapeback/
  cli.py          Click CLI (start, stop, process, summarize, status)
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

### Setup

```bash
git clone https://github.com/yastcher/tapeback
cd tapeback
uv sync --group dev

uv run ruff check       # lint
uv run ruff format      # format
uv run ty check         # type check
uv run pytest           # test (coverage >= 85%)
```

### Releasing

```bash
scripts/release.sh 0.9.0          # bump version in pyproject.toml + PKGBUILDs
# commit, tag, push → CI publishes to PyPI
scripts/aur-publish.sh 0.9.0      # update all 3 AUR packages
```

</details>

## Support

If you find tapeback useful, consider a small donation:

| USDT (TRC-20) | ADA (Cardano) |
|:-:|:-:|
| <img src="docs/qr-usdt.png" width="180"> | <img src="docs/qr-ada.png" width="180"> |
| `TAECw9FebnoSN2n3H2Fk9Bv5aA8fwpCuBB` | `addr1q9tqg2g8wxpxawsrvea84lms3ampuda0ygzawuxq77sxwr48mxj2vq2rzd4nsmhpdhy6lftp30tz78tetzr29mtvkqmsskrmp7` |

## License

Apache-2.0. See [LICENSE](LICENSE).
