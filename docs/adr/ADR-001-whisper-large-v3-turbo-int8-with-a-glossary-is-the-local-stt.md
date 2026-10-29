# ADR-001 — Whisper large-v3-turbo in int8_float16 with a glossary is the local STT

**Status:** accepted 2026-08-05 (`settings.py`: `stt_model`, `compute_type`, `hotwords`;
`_device.py`; `glossary.py`).

**Context.** Meetings are Russian speech with English technical terms inside one
phrase, transcribed on a 4 GB laptop GPU (TU117, no tensor cores). Before 2026-08 the
model was turbo in float16 with no glossary: one term came out in six spellings.

**Decision.** Stay on Whisper. The default is `large-v3-turbo` with `int8_float16`
(`compute_type=auto`) and the project glossary as `hotwords`.

**Why.** Picked from an 18-point grid on real recordings (`scripts/bench_transcribe.py`),
not from the literature:
- turbo + int8_float16 + hotwords was best or tied-best in every column: RTF 16.05,
  13/19 terms, 4.9% low-confidence words;
- `large-v3` recognised fewer terms at under half the speed, though it punctuates
  better; in float16 it does not fit 4 GB at all;
- `float16` is 3.6x slower than `int8_float16` on this card and takes twice the VRAM;
- GigaAM v3 and Canary break on English terms inside Russian, Parakeet detects the
  language per utterance, Voxtral does not fit 4 GB.

**Consequences.** A glossary term stays only if a measurement shows it helps (docTR was
removed for still failing). A model or compute-type change needs a new bench table.
Measurements: `docs/spec/done/2026-08-05-transcription-speed-and-quality.md`.

Now (2026-10-16): the long-term target is one model for transcript and speakers,
ADR-016.
