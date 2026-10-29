# ADR-003 — Transcription runs in a child process, and resumes per whole channel

**Status:** accepted 2026-08-05 (`settings.py`: `isolate_transcription`,
`resume_cache`; `_worker.py`, `_isolated.py`, `_resume.py`).

**Context.** A CUDA out-of-memory inside ctranslate2 keeps the allocation for the rest
of the process: free VRAM went from 3674 to 95 MiB and stayed there, so every later
stage silently ran on the CPU. `torch.cuda.empty_cache()` cannot reach ctranslate2's
allocator. Separately, an interrupted run redid finished channels.

**Decision.**
- Each transcription channel runs in a worker process by default; segments stream back
  line by line, so a killed worker leaves its finished segments with the parent.
- A finished channel is cached against the audio and every output-affecting setting
  and reused on the next run.
- Resume is per whole channel, never from the middle of one.

**Why.**
- A child that exits returns its VRAM to the kernel: measured 3674 → 3674 MiB isolated
  against 3674 → 95 MiB in-process. One process start and model load per run is the
  price.
- Mid-channel resume needs `clip_timestamps`, which faster-whisper documents as
  ignoring `vad_filter`, and VAD is half of why silence hallucinations went away.

**Consequences.** Settings cross into the worker through an explicit allow-list
(`_isolated.job_settings`), as in ADR-007. A way to resume mid-channel without losing
VAD is an open item, BACKLOG §4.
