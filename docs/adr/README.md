# Architecture decisions

A decision that is still in force and has a reason is written here, one file per
decision: `ADR-NNN-<the-decision-as-a-phrase>.md`. Specs, BACKLOG and reports describe
their own day; this folder describes what holds now.

Sections: Status (the date and the files that implement it), Context, Decision, Why,
Consequences. A decision that changes keeps its text and gains a dated line,
"Now (date): …", or a new ADR that names the old one. ADR-001 to ADR-017 were moved out
of the specs on 2026-10-28; each spec points here where the decision used to be.

- [ADR-001](ADR-001-whisper-large-v3-turbo-int8-with-a-glossary-is-the-local-stt.md) — Whisper large-v3-turbo in int8_float16 with a glossary is the local STT.
- [ADR-002](ADR-002-decoding-defaults-come-from-measurement.md) — Decoding defaults come from measurement, and the anti-hallucination set stays whole.
- [ADR-003](ADR-003-transcription-runs-in-a-child-process-and-resumes-per-channel.md) — Transcription runs in a child process, and resumes per whole channel.
- [ADR-004](ADR-004-a-clamped-gpu-sends-a-stage-to-the-cpu.md) — A thermally clamped GPU sends a stage to the CPU; only the summary waits.
- [ADR-005](ADR-005-gpu-telemetry-observes-and-never-changes-the-card.md) — GPU telemetry observes and never changes the card.
- [ADR-006](ADR-006-loudnorm-stays-and-runs-before-resampling.md) — loudnorm stays the normaliser and runs before resampling.
- [ADR-007](ADR-007-settings-leave-the-process-only-through-an-allow-list.md) — Settings leave the process only through an allow-list.
- [ADR-008](ADR-008-recordings-live-in-a-private-state-directory.md) — Recordings in progress live in a private state directory, verified, never repaired.
- [ADR-009](ADR-009-pii-masking-is-opt-in-literal-and-at-the-summary-seam.md) — PII masking is opt-in, literal, and sits at the summary seam.
- [ADR-010](ADR-010-only-the-summary-leaves-the-machine-by-default.md) — By default only the summary request leaves the machine.
- [ADR-011](ADR-011-an-endpoint-and-the-local-server-never-fall-back.md) — An OpenAI-compatible endpoint and the local server never fall back.
- [ADR-012](ADR-012-local-summaries-run-a-managed-llama-server.md) — Local summaries run a llama-server tapeback starts and stops.
- [ADR-013](ADR-013-a-note-holds-one-transcript-and-its-summary-speaks-its-language.md) — A note holds one transcript, and its summary is in the note's language.
- [ADR-014](ADR-014-settings-come-from-the-environment-and-env-files-only.md) — Settings come from the environment and .env files only; setup edits the user's file.
- [ADR-015](ADR-015-packages-pin-what-they-bundle.md) — Distro packages pin what they bundle.
- [ADR-016](ADR-016-one-model-for-transcript-and-speakers-is-the-target.md) — One model for transcript and speakers is the target.
- [ADR-017](ADR-017-only-the-release-script-writes-the-version.md) — Only the release script writes the version.
- [ADR-018](ADR-018-recorders-are-bare-pids-that-log-to-the-session.md) — Recorders are bare pids, and each logs to its session directory.
