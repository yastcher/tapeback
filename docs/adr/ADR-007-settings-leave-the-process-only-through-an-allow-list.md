# ADR-007 — Settings leave the process only through an allow-list

**Status:** accepted 2026-08-05 (`_runlog.py`: `RECORDED_SETTINGS`; `_isolated.py`:
`job_settings`).

**Context.** Every run writes a JSON record so a failed or interrupted run can be read
afterwards, and the transcription worker receives the settings it needs. `Settings`
carries `hf_token`, `llm_api_key` and `stt_api_key`.

**Decision.**
- The run record and the worker job name the fields they carry, never
  `settings.model_dump()`.
- The run record keeps status lines verbatim and the outcome (`completed`, `aborted`,
  `failed` with the error); exceptions are classified and re-raised unchanged.
- A failed record write degrades to no record; it never fails the run.

**Why.**
- A post-mortem file that leaks credentials is worse than no file. A test fails the
  build if a `SecretStr` field joins the allow-list.
- Re-parsing our own formatted lines would break on every reword.
- Losing a diagnostic file must not destroy the run it describes.

**Consequences.** A new setting that matters for a post-mortem is added to the list by
hand.
