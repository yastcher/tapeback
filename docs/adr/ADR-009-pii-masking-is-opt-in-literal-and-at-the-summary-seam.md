# ADR-009 — PII masking is opt-in, literal, and sits at the summary seam

**Status:** accepted 2026-08-06 (`_mask.py`: `Masker`; `summarizer.py`: `summarize`;
`settings.py`: `mask_pii`, `mask_terms`).

**Context.** The transcript sent for a summary is the most personal text the machine
produces, and the fallback chain can hand it to a second company. A scan of 56 real
transcripts found zero emails, zero phone numbers and many names.

**Decision.**
- Off by default (`TAPEBACK_MASK_PII`); off changes not a byte of the request.
- Rules: EMAIL, PHONE and the user's own TERM list; matching is literal,
  case-insensitive, word-bounded, longest first.
- The seam is `summarize()`: the first request, the JSON retry and every fallback
  provider get the masked text, and the parsed `Summary` is unmasked, not the raw JSON.
- The mask map lives in memory for one call. The transcript on disk is never masked.
- `tapeback summarize --show-masked` prints what would be sent and sends nothing.

**Why.**
- It is the user's data and their call; the source project's mandatory masking served
  a legal requirement this one does not have.
- Generating word forms over-matches common nouns; a morphology library is a
  dependency and must fail closed if it ever ships. Card, INN and passport rules had
  zero instances.
- Substituting a value with a quote back into a JSON string corrupts it.
- Without a way to see the payload, a privacy claim cannot be checked.

**Consequences.** Every new outbound text call goes through `summarize()`. Names the
user did not list are not masked (NER is a backlog idea). Remote STT uploads audio and
masking cannot apply there (ADR-010). Spec: `docs/spec/done/2026-08-06-optional-pii-masking.md`.
