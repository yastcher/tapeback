# ADR-006 — loudnorm stays the normaliser and runs before resampling

**Status:** accepted 2026-08-06 (`audio.py`: `split_channels_16k`).

**Context.** The `split` stage is almost all `loudnorm` (31.6 s of 32 on a 16-minute
recording; 0.5 s without it), and it heats the CPU package by about 9 °C just before the
GPU is asked to work. `dynaudnorm` takes 1 s.

**Decision.** Keep EBU R128 `loudnorm` as the default, and keep `aresample=16000` after
it.

**Why.**
- An A/B on three recordings found the monitor channel identical and the quiet mic
  channel a third less punctuated with `dynaudnorm` (92 against 140 commas per 1000
  words); readability of the quiet channel was traded away before and regretted.
- Resampling first is not faster (32.4 s against 31.6 s) and `loudnorm` emits at its
  internal 192 kHz, so the output WAVs came out at 192 kHz.

**Consequences.** A cheaper path for hot machines would be an opt-in setting
(`TAPEBACK_NORMALISATION`), keeping `loudnorm` as the default: BACKLOG §3.
