# ADR-002 — Decoding defaults come from measurement, and the anti-hallucination set stays whole

**Status:** accepted 2026-08-05 (`settings.py`: `chunk_length`, `temperature`,
`batch_size`, `vad_filter`, `condition_on_previous_text`; `transcriber.py`:
`_batched_warning`).

**Context.** `chunk_length` had been lowered to 7, and locally to 2, to fight
hallucinations on pauses. A 15-minute recording then took 2h17m for one channel.

**Decision.**
- `chunk_length` is 30.
- The temperature ladder stays `0.0 … 1.0`.
- VAD stays on; `condition_on_previous_text` stays off.
- Batching is off by default; turning it on prints which configured settings it drops.
- A change to any of these is made from a `scripts/bench_transcribe.py` table.

**Why.**
- Whisper's encoder is fixed at 30 s and pads every window to it: `chunk_length=2`
  took 390.6 s on a 145 s file, 30 took 41.4 s, with fewer hallucination markers.
  VAD, `no_speech_threshold` and mic gating are what stop pause hallucinations.
- A shorter temperature ladder pushed the model into repeat loops: slower and worse.
- faster-whisper's batched pipeline silently ignores `no_speech_threshold`,
  `condition_on_previous_text`, `hallucination_silence_threshold` and all but the
  first temperature.

**Consequences.** README states each reason next to the setting, so it is not lowered
again on a hunch. Measurements:
`docs/spec/done/2026-08-05-transcription-speed-and-quality.md`.
