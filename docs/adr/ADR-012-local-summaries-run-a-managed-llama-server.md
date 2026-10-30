# ADR-012 — Local summaries run a llama-server tapeback starts and stops

**Status:** accepted 2026-10-16 (`_llama_server.py`: `running`, `DEFAULT_MODEL`;
`settings.py`: `llm_local_*`).

**Context.** The test machine has 4 GB of VRAM shared with Whisper and pyannote, and
about 2.3 GB of free RAM. Through Ollama a 4B model took 182 s against 6.5 s on
llama.cpp, and Ollama's default context on such a card is 4k: a long transcript is cut
without an error.

**Decision.**
- `TAPEBACK_LLM_PROVIDER=local` starts llama.cpp's `llama-server` once transcription
  and diarization have given the GPU back, and stops it after the answer; the summary
  and its JSON retry share one start.
- The server binds a free loopback port, takes its key from the environment, not argv,
  and runs in a private state directory (ggml loads backend libraries from the working
  directory).
- One local summary at a time (`flock`); a PID file with the process's start time finds
  a server a crashed run left behind.
- The default model is Qwen3.5-4B Q4_K_M, pinned by revision and sha256 and downloaded
  once; `TAPEBACK_LLM_LOCAL_MODEL` names another GGUF.
- Out of VRAM, the context is halved once and free VRAM is reported.
- Built before the bench, with flags measured earlier; the bench tunes them.

**Why.**
- The model holds the VRAM only for the summary, so it never competes with the earlier
  stages.
- A pinned file is the model that was measured; a moving tag is not.
- Qwen3.5-4B with thinking off was the best measured fit on this machine.

**Consequences.** Ollama is compared on the bench; a managed server within 10% of tuned
Ollama is not worth its code. Clamp policy: ADR-004. Packages: ADR-015. Plan and
measurements: `docs/spec/2026-10-15-setup-and-local-llm.md`.
