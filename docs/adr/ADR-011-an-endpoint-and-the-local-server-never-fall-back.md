# ADR-011 — An OpenAI-compatible endpoint and the local server never fall back

**Status:** accepted 2026-10-15 (`_llm.py`: `ENDPOINT`, `LOCAL`, `provider_chain`,
`connected`; `settings.py`: `llm_base_url`, `llm_timeout`).

**Context.** Issue #11 asked for any OpenAI-compatible API (ollama.com, anyapi.ai, a
self-hosted server) and for summaries on the machine itself. The openai SDK reads
`OPENAI_API_KEY` from the environment when given no key.

**Decision.**
- Two providers: `openai_compatible` for an address the user gives, `local` for the
  llama-server tapeback starts (ADR-012).
- Neither joins the cloud fallback chain, and neither falls back to it. The endpoint's
  key comes only from `TAPEBACK_LLM_API_KEY`.
- Endpoint guards: masking applies whatever the host; an address with credentials is
  refused; plain http to another host warns; `tapeback status` shows host and port
  only.
- No SDK retries; one answer is bounded by `TAPEBACK_LLM_TIMEOUT` (600 s). A
  misconfigured endpoint skips the summary and says why; it never blocks a recording.

**Why.**
- Someone who points tapeback at their own server chose where the text goes; a cloud
  fallback found in the environment would break that choice silently.
- "Loopback" can be a forwarded port to another machine.
- Each SDK retry sends the whole transcript again.

**Consequences.** `TAPEBACK_LLM_MODEL` applies to the primary provider only; a cloud
fallback uses its own default model.
