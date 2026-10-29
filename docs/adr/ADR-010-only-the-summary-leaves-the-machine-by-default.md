# ADR-010 — By default only the summary request leaves the machine

**Status:** accepted 2026-08-07, extended 2026-09-16 (`settings.py`: `stt_backend`,
`summarize`, `llm_provider`; `_llm.py`: `provider_chain`).

**Context.** Recording, Whisper and pyannote run locally. A cloud summary sends the
transcript out; remote STT (2026-09) uploads the meeting's audio.

**Decision.**
- Local transcription and diarization are the default; remote STT
  (`TAPEBACK_STT_BACKEND=openai`) is opt-in, and README says the audio leaves the
  machine and masking cannot apply to it.
- A cloud summary falls back to the next cloud provider whose key is set, which can
  hand the same transcript to a second company; README says so.
- `TAPEBACK_SUMMARIZE=false` and `TAPEBACK_LLM_PROVIDER=local` keep everything local.

**Why.** A meeting is personal data; sending it anywhere is the user's choice, made
once, knowingly. The fallback exists because a summary that silently fails is worse
for most users than a second provider, and masking (ADR-009) covers that path.

**Consequences.** Each new outbound path asks for a rerun of the security review
(BACKLOG §10). Providers that never join the chain: ADR-011.
