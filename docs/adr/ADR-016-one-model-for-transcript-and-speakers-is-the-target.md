# ADR-016 — One model for transcript and speakers is the target

**Status:** accepted 2026-10-16 (direction; nothing in code yet).

**Context.** Whisper and pyannote run one after the other on a 4 GB card that clamps,
and diarization took 0.7–1.2x the recording. pyannote also finds a small third cluster
and its speaker merge is weak. Some engines produce transcript, speakers and more in one
pass.

**Decision.** Aim for one model that transcribes and tells the speakers apart, in place
of Whisper and pyannote. pyannote's own speed-ups (a longer segmentation step,
speech-only input, diarizing on the CPU while Whisper holds the GPU) are measured only
if pyannote stays. The pipeline and `tapeback setup` must not assume one engine per
stage.

**Why.** One pass over the audio instead of two halves the time the card works, and
speakers aligned with words remove the re-segmentation step where words get lost.

**Consequences.** Candidates are chosen from a bench table on Russian meetings, as
Whisper was (ADR-001). Remote engines do not cover this: OpenAI's diarizing model shuts
down on 2027-02-26 and Anthropic takes no audio (BACKLOG §11). Plan:
`docs/spec/2026-10-06-diarization.md`, `docs/spec/2026-10-15-setup-and-local-llm.md`
("Later").
