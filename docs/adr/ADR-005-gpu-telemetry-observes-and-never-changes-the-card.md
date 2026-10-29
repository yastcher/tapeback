# ADR-005 — GPU telemetry observes and never changes the card

**Status:** accepted 2026-08-05 (`_gpu.py`: `sample_gpu`, `THROTTLE_MASK`;
`_timing.py`: `ProgressReporter`; `settings.py`: `gpu_telemetry`).

**Context.** "Transcription is slow" looked the same for a clamped card, a silent CPU
fallback and a bad setting. Runs printed one line and were killed by hand.

**Decision.**
- tapeback samples clocks, temperature, VRAM and throttle reasons per stage and prints
  one summary line. It never sets clocks or power caps and never asks for root;
  `thermal_profile.sh` stays a manual tool.
- Throttling counts SW power cap, HW slowdown, SW/HW thermal and HW power brake, and
  excludes `GpuIdle` and `ApplicationsClocksSetting`.
- Progress comes from our own `ProgressReporter`, not faster-whisper's `log_progress`.
- `nvidia-smi` is reached through one function; a missing binary, a driver error and a
  hung query all degrade to "no answer".

**Why.**
- tapeback is an unprivileged CLI; a cap needs root and changes the whole machine.
- Idling is not throttling, and a cap the user applied is a setting, not a symptom.
- `log_progress` draws a tqdm bar straight to the terminal, bypassing `on_status`, so
  it never reaches the tray log or the run record.

**Consequences.** The device decision itself is ADR-004; this line only reports what
already happened.
