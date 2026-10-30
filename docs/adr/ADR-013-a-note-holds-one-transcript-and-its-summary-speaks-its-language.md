# ADR-013 — A note holds one transcript, and its summary is in the note's language

**Status:** accepted 2026-10-15 and 2026-10-16 (`formatter.py`; `summarizer.py`:
`_LANGUAGE_RULE`, `note_language`).

**Context.** A diarized note carried the raw You/Other transcript and then the same
text again with speakers told apart; readers deleted the first by hand, and the summary
prompt sent both (59.4k characters for an hour against 26.9k). A 4B local model followed
the English instructions and summarised a Russian meeting in English.

**Decision.**
- A diarized note keeps one section, the diarized transcript. For an older note
  `tapeback summarize` sends only its diarized half.
- The summary prompt names the language Whisper detected, read from the note's
  frontmatter.

**Why.**
- Word counts over real notes found a few words in a thousand only in the raw section,
  and that loss was fixed at its cause (short pieces join the speech around them), not
  by keeping a copy.
- Every provider gets half the input; an hour fits one local pass.
- "The same language as the transcript" is too weak an instruction for a small model.

**Consequences.** Diarization must not drop words, since nothing behind it keeps them;
regressions in `tests/regressions/test_diarized_short_reply.py`.
