# Glossary

The words this project uses for its own things. Not the transcription glossary: the
terms Whisper is biased towards are the product's `hotwords` (`src/tapeback/glossary.py`,
`TAPEBACK_HOTWORDS`). When a word in a discussion means something other than what is
written here, this file is corrected, not the discussion. Decisions behind the terms live
in [`docs/adr/`](adr/README.md).

## Recording

- **Session**: one recording. Its directory, `~/.local/state/tapeback/sessions/<name>/`, holds
  `monitor.wav`, `mic.wav`, the merged `stereo.wav` and each recorder's log; `session.json`
  in the state directory names the session in progress. The name is the start time in
  `TAPEBACK_TIMEZONE`, or the one given to `tapeback start`/`process`. A session is removed
  once processed and kept when processing fails (ADR-008).
- **Channel**: one side of a session. **mic** is the user's microphone, left in the
  stereo file, labelled **You**. **monitor** is the system audio, the other participants,
  right in the stereo file.
- **Recorder**: the `parecord` process that records one channel, started and stopped by
  pid (ADR-018).
- **Note**: the Markdown file a session becomes, `<vault>/meetings/<name>.md`, with its
  frontmatter, transcript and summary. The audio goes to `attachments/audio/<name>.wav`.
- **Vault**: the directory notes are written to, `TAPEBACK_VAULT_PATH`, usually an Obsidian
  vault.
- **Live transcript**: the note-in-progress `TAPEBACK_LIVE` writes while a meeting runs.

## Processing

- **Stage**: one timed step of processing, printed as `Stage '<name>' took …`: `merge`,
  `load channels`, `split`, `gate mic`, `load model`, `transcribe monitor`,
  `transcribe mic`, `diarize`, `summarize`.
- **Mic gating**: silencing the mic channel where the user only listens, before
  transcription (`TAPEBACK_GATE_MIC_SILENCE`). Not the gate below.
- **Worker**: the child process that transcribes one channel (ADR-003).
- **Resume cache**: finished channels kept against the audio and every output-affecting
  setting, reused by the next run (`TAPEBACK_RESUME_CACHE`).
- **Run record**: one JSON file per run in `~/.local/share/tapeback/runs/`: the settings
  used, every status line, the outcome (ADR-007).
- **Clamp** (thermal clamp): the embedded controller holding the GPU at its lowest clock
  on a hot laptop; it clears only on system idle (ADR-004).
- **Telemetry line**: `GPU: sm … throttled NN%`, what the card did during a stage
  (ADR-005).
- **Diarization**: telling the remote speakers apart, with pyannote. Its labels are
  **Speaker 1**, **Speaker 2** and so on; without it the monitor side is **Other**.
- **Spectral merge**: joining two pyannote speakers whose spectra are alike
  (`speaker_merge.py`, `TAPEBACK_SPECTRAL_MERGE_THRESHOLD`). A **minor cluster** is a
  speaker with little speech, usually echo or cross-talk.
- **Hallucination markers**: subtitle-corpus phrases Whisper invents on silence
  ("Продолжение следует..."), counted and filtered by `_quality.py`.

## The summary

- **Provider**: who writes the summary, `TAPEBACK_LLM_PROVIDER`. A **cloud provider** is one
  of the vendors in `_llm.PROVIDERS`; the **fallback chain** is the next cloud provider
  whose key is set (ADR-010).
- **Endpoint**: the `openai_compatible` provider, an address the user gives
  (`TAPEBACK_LLM_BASE_URL`). The **local server** is the `llama-server` tapeback starts
  for provider `local` (ADR-011, ADR-012).
- **Masking**: replacing emails, phones and the user's terms with placeholders such as
  `[EMAIL_1]` before the transcript leaves the machine (ADR-009).
- **Remote STT**: transcription by an uploaded copy of the audio,
  `TAPEBACK_STT_BACKEND=openai`; opt-in (ADR-010).

## Development

- **The gate**: `scripts/gate.sh`, every check CI runs, plus the stands when the diff
  touches their paths. It has passed only when its last line reads `all gates passed`.
- **Stand**: a packaging check in containers that builds a package as users get it and
  runs it: the `.deb` stand (`scripts/deb-smoke.sh`) and the Arch stand
  (`scripts/arch-smoke.sh`).
- **Companion** (also written as extra): an optional package next to `tapeback`:
  `tapeback-tray`, `-llm`, `-llm-local`, `-diarize`, `-cuda`. The pip extras are
  `tray`, `stt`, `llm`, `diarize`.
- **Pre-release QA**: `scripts/pre_release_qa.sh`, the gate with every stand, then the e2e
  suite on real recordings. Its **stamp** is the tree it passed, which `scripts/release.sh`
  requires.
- **Bench**: `scripts/bench_transcribe.py`, the table configuration choices come from
  (ADR-001, ADR-002).
- **Probe**: an ad-hoc script that measures or checks one thing, kept under `tmp/`.
- **Ratchet**: the coverage floor, `fail_under` in `pyproject.toml`, which only moves up.
- **ADR, spec, BACKLOG**: a decision that holds (`docs/adr/`), a plan with its
  measurements (`docs/spec/`), an open item (`docs/spec/BACKLOG.md`).
