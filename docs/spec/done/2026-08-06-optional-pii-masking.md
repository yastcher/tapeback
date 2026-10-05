# Spec: optional PII masking at the LLM boundary

Status: all stages landed (1, 2, 3).

## Context

Everything tapeback does is local — recording, Whisper, pyannote — with exactly one
exception: `summarizer.summarize()` sends the whole transcript to an external provider
(Anthropic, or one of six OpenAI-compatible ones, plus a fallback chain that can hand the
same text to a *second* provider when the first fails). A meeting transcript is about the
most personal text this machine produces.

`backlog_profiler` already solved the same problem — see
`backend/src/llm/_mask.py`, `clients/telegram-bot/bot/mask.py` and
`backend/tests/test_llm_mask.py`. This spec adapts that design, with one difference the
user asked for: **there, masking is mandatory (152-ФЗ, P0); here it is opt-in and off by
default**, and it only has any meaning when a summarization request is actually made.

### Pre-code understanding check

1. **Real problem.** Transcript text crosses to a third party with no way to hold
   anything back, and today there is not even a switch.
2. **Already clear.** Off by default; only relevant when `summarize` is on; the reference
   implementation exists and is small.
3. **Ambiguities.** Which PII classes to cover, and whether names are in scope — resolved
   by measurement below.
4. **Most likely to get wrong.** Porting the profiler's rule set verbatim and calling it
   done. The measurement says that would mask *nothing* in this project's real data.

## What the reference implementation does

- A `Masker` object per LLM call. Regex rules produce `[LABEL_N]` placeholders; the same
  value always gets the same placeholder within a call, so the model sees a consistent
  entity and the map stays small.
- The map is per-call and dies with the call — it is never persisted.
- The provider is wrapped in a decorator (`MaskingLLMProvider`), so no individual call
  site can forget to mask.
- The **response is unmasked before downstream code sees it**, so storage keeps real
  values while the model never saw them.
- Rules are deliberately conservative: email and phone only, with `_RULES` as the
  documented seam for more detectors. Names are an acknowledged gap needing NER.

All five properties carry over. The decorator shape does not — tapeback has one call
path, not a provider abstraction.

## Measurement: what PII is actually in the transcripts

Scanned the real vault: **56 transcripts, 63 051 words**.

| pattern | occurrences |
|---|---|
| email addresses | **0** |
| phone numbers (`+7`/`8` forms and international) | **0** |
| URLs | **0** |
| digit runs ≥ 4 | 170 — of which 163 are the literal `2026`, the rest round numbers and years |
| person names | present |

**This is the finding that shapes the spec.** The profiler's rule set, ported as-is,
would fire on zero characters of this corpus. That is not an argument against porting it
— it is insurance, and it also produces zero false positives — but shipping only that
and describing the feature as "PII masking" would be misleading. What people actually
say out loud in meetings is *names*.

Two second-order reasons the structured rules stay near-zero here: Whisper writes
speech, and a dictated address arrives as "ivan dot petrov at example dot com" as often
as `ivan.petrov@example.com`; and people read addresses out in meetings far less often
than they type them in tickets, which is where the profiler's corpus comes from.

## Design

### Seam

`summarizer.summarize()`, not `_call_llm()`.

`_call_llm` is called twice (normal prompt, then the JSON-retry prompt), and both must be
masked — so the seam has to be above them. Placing it at `summarize()` also puts the
unmasking **after** JSON parsing, on the `Summary` domain object, rather than on the raw
response string. That matters: a mask term may contain a quote or a backslash, and
substituting it back into a JSON string before `json.loads` would corrupt the document.
The profiler is not exposed to this because its provider returns an already-parsed dict.

```
summarize(transcript, settings)
  ├─ masker = Masker(settings)          # no-op object when mask_pii is off
  ├─ masked = masker.mask(transcript)
  ├─ raw = _call_llm(system, masked, settings)      # and the retry, same masked text
  ├─ summary = _parse_response(raw)
  └─ return masker.unmask_summary(summary)
```

`_call_llm`, `_call_provider_with_retry` and the fallback chain need no changes — every
provider in the chain receives the already-masked text, which is what we want, since the
chain is precisely the path that hands the same transcript to a *second* company.

### Module

`src/tapeback/_mask.py` — private helper, matching `_gpu.py` / `_quality.py` / `_resume.py`.
Public surface: `Masker`, and nothing else. No domain types belong here.

### Settings

| setting | default | meaning |
|---|---|---|
| `TAPEBACK_MASK_PII` | `false` | Master switch. Off means not a single byte changes. |
| `TAPEBACK_MASK_TERMS` | `""` | Comma-separated literal terms to redact — names, company and project names. Same comma-separated-string shape as `TAPEBACK_HOTWORDS`. |

No interaction validator with `summarize`. `mask_pii=true` with summarization off is a
silent no-op, which is correct: nothing is sent, so nothing needs masking. Documented,
not enforced — a user setting it once in `~/.config/tapeback/.env` should not get an
error every time they pass `--no-summarize`.

### Rules

Ported from the profiler, in order:

1. `EMAIL` — same regex.
2. `PHONE` — same regex (Russian `+7`/`8` and generic international, tolerant of spaces,
   dashes, parentheses; look-around prevents it grabbing bare digit runs). Verified
   against this corpus: no false positives on the 163 occurrences of `2026`.
3. `TERM` — user list from `TAPEBACK_MASK_TERMS`. Case-insensitive, word-bounded,
   longest-first so a two-word term wins over its first word alone.

Stopping there. Card numbers, ИНН and passport numbers were considered and rejected for
now: zero instances in the corpus, and a Luhn-checked card rule is only meaningful once
something in the corpus justifies it. `_RULES` stays the documented seam, exactly as in
the profiler.

### The names problem, stated honestly

`TERM` requires the user to know and list the names. In Russian it also runs into
inflection: listing `Роман` does not mask `Романа`, `Роману`, `Романом`. A privacy
feature that silently covers three of five occurrences is worse than none, because the
user believes they are covered.

Three options, in the order they were considered:

- **Require the user to list every form.** Deterministic and dependency-free, but the
  failure mode is a missed form the user never notices.
- **Stem plus bounded suffix** (`Ром` + `[а-яё]{0,3}`). Dependency-free, over-matches —
  the common noun `роман` collides with the name. Note the asymmetry: a false positive
  costs summary quality, a false negative leaks, so over-matching is the *safe*
  direction. Still rejected: it silently rewrites unrelated words, and a name appearing
  where the transcript said a common noun reads as a transcription defect.
- **Morphological expansion** — generate the paradigm from the lemma (`pymorphy3`,
  offline, dictionary-based). Precise in both directions. Costs a dependency.

**Decision: literal matching, and say so plainly in the README** — a term is masked in
the exact forms the user lists, nothing more. Morphology is not shipped; if it ever is,
it must fail closed, refusing to run rather than under-masking, because a privacy feature
that degrades quietly is the failure mode being designed against.

The generic version of this problem — masking names nobody listed — needs NER
(`natasha` for Russian, spaCy multilingual), degraded further by ASR output that has no
reliable capitalization. That is a backlog item, not this spec.

### Invariants

- The mask map is in memory, per call, and is **never written anywhere** — not the run
  log (`_runlog.py` is an allow-list of config fields, so it is already safe; add an
  assertion), not the resume cache, not the vault.
- The transcript **on disk is never masked.** Masking applies to the wire only. The vault
  is local and the user wants real values there.
- Masking never changes behaviour when off: `mask_pii=false` must produce a byte-identical
  request to today's. Pinned by a test — this is the same class of bug as the
  `strip_hallucinations` regression, where a "clean-up" path silently altered untouched
  input.
- Placeholders left unresolved after unmasking (the model invented `[EMAIL_7]`, or
  reformatted one) are reported once as a warning. Silent placeholder residue in a saved
  summary is a visible defect.

### Verifiability

Given that the structured rules fire on nothing in the measured corpus, a user has no way
to tell whether the feature works. Add `tapeback summarize --show-masked`: print exactly
what would be sent and exit without calling any provider. Small, and it is the difference
between a privacy claim and a checkable one.

## Stages

**Stage 1 — seam and structured rules.** `_mask.py` with `Masker`, `EMAIL` and `PHONE`;
`mask_pii` setting; wire into `summarize()`; unmask over `Summary`
(`brief`, `key_decisions`, and each `ActionItem`'s `assignee` / `action` / `deadline`).
Tests, README, CHANGELOG.

**Stage 2 — user terms.** `mask_terms` setting, `TERM` rule, longest-first ordering,
collision guard against speaker labels (`You`, `Speaker N`) so a mask term cannot corrupt
attribution. This is the stage that makes the feature real for this project's data.

**Stage 3 — `--show-masked`.** CLI flag on the `summarize` command.

## Testing

Mirrors `backend/tests/test_llm_mask.py`, plus what this project's rules demand:

- **Round trip** — masked text contains no raw value, and unmask restores the exact
  original.
- **Stable placeholders** — same value → same placeholder; distinct values → distinct
  placeholders; the counter continues across several texts in one call.
- **Phone formats**, parametrized over the same five forms the profiler pins.
- **No false positives**, parametrized: `2026`, `8 items left`, bare digit runs.
- **Off by default** — a fresh `Settings()` has `mask_pii is False`, and with it off the
  text handed to the provider is byte-identical to the input.
- **Both branches** of every conditional (`mask_pii` on/off, terms empty/non-empty,
  residue present/absent), per the project's testing rules.
- **Boundary**: a term that is a strict prefix of another (`Ann` vs `Anna`) must not be
  masked inside the longer word.
- **Regression** (`../../../tests/regressions`): with masking on, capture what the provider layer
  received and assert no raw email, phone or listed term appears in it — including on the
  **retry** prompt and on the **second provider in the fallback chain**, which are the two
  paths most likely to be missed.
- Assert exact strings, hardcoded, not constants shared with production code.

## Non-goals

- Masking anything but the LLM request. Local Whisper, local pyannote and the vault are
  out of scope by construction.
- Masking the summary written to the vault.
- Encrypting or redacting stored transcripts.
- NER-based name detection (backlog).
- Making masking mandatory or default-on. The user's call: opt-in.

## Backlog candidates arising

- NER for unlisted names, and its accuracy on uncased ASR output.
- Morphological expansion of listed terms (`pymorphy3`), fail-closed when absent.
- Harvesting name candidates from accumulated transcripts and proposing them for
  `TAPEBACK_MASK_TERMS` — the same shape as the glossary harvesting that already worked
  in this project, and the natural answer to "the user has to know the names".
