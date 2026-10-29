# ADR-004 — A thermally clamped GPU sends a stage to the CPU; only the summary waits

**Status:** accepted 2026-08-06 for transcription, 2026-10-16 for diarization and the
local summary (`_gpu.py`: `thermal_clamp_active`, `wait_for_clamp_release`;
`_device.py`: `resolve_device`; `diarizer.py`: `DIARIZATION_VRAM_MIN_MIB`;
`settings.py`: `thermal_clamp_*`, `stage_pause_seconds`, `llm_local_clamp_wait`).

**Context.** On laptops where CPU and GPU share a heatsink the embedded controller
latches the GPU at 300 MHz and a fraction of its power limit. The latch clears only on
system idle: 451 s at best, not after 900 s once the chassis was saturated, never while
the user keeps the CPU busy. A clamped card transcribed at 0.31x real time against the
CPU's 2.39x, and diarized at 0.86–1.20x the recording against 0.67x on the CPU.

**Decision.**
- Before each transcription stage and before diarization, look for the clamp once; if
  it is on, run the stage on the CPU and say so.
- `TAPEBACK_THERMAL_CLAMP_WAIT` defaults to 0: no waiting.
- No pacing between stages by default (`TAPEBACK_STAGE_PAUSE_SECONDS=0`).
- Diarization uses the same policy with its own VRAM floor, 1500 MiB.
- The local summary is the exception: it waits up to 600 s for the clamp before the
  server starts, then goes ahead, with its own 1800 s timeout.

**Why.**
- The alternative wins below the floor: a clamped GPU is about 8x slower than the CPU
  for transcription and 1.5x for diarization, so persisting on it is strictly worse.
- A short wait almost never succeeds (the shortest release measured is 451 s), so a
  60 s wait was 60 s of nothing, then the fallback that could have started at once.
- The summary is the last stage, nothing runs after it, and a local LLM on this
  machine's CPU does not fit in RAM; a clamped card read an hour's prompt at
  13–17 tok/s and timed out at 600 s.
- Diarization's policy was chosen without the long-file measurement, by the
  maintainer.

**Consequences.** The check is retaken per stage, so a run returns to the GPU once the
card recovers. A clamp that latches during a stage is not seen until the next one
(BACKLOG §1, §6). Clock caps need root and stay a manual step (ADR-005).
