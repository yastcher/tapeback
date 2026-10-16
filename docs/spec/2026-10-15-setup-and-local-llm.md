# `tapeback setup` and local LLM summaries

Status: done on `feat/setup-local-llm` but for commit 8, which waits for a
menu-driven setup (Later). Synthesised from three proposals (simplest, fastest on a
4 GB laptop GPU, cleanest code).

## The request

- `tapeback setup`: for now, diarization on/off and summary on/off. Decided: a dialog
  with the current values as defaults, plus the same choices as flags for scripts.
- In the same PR, local summaries (issue #11, "more llm providers"; the plan in its
  comments: a generic OpenAI-compatible endpoint, base URL + model). Installed in the
  way most convenient for the user, possibly a separate package like
  `tapeback-diarize`. Qwen is the first candidate.
- Later, designed for now: presets for weak and strong machines, cloud or local per
  stage, own model paths or downloads. Presets are a template of values; the storage
  (maybe TOML) is undecided.
- Some engines cover several stages at once: transcript, speakers, emotion tags, even
  a short summary. The setup must not assume one engine per stage.

## Facts

The test machine: a 4 GB Turing laptop GPU, often thermally clamped to 300 MHz;
12 threads; 7.6 GB RAM with ~2.3 GB available and swap in use. Local LLMs were
measured on it earlier, outside this repo:

- Best model there: Qwen3.5-4B Q4_K_M with thinking off; earlier Qwen3-4B-Instruct-2507
  Q4_K_M. Runtime: llama.cpp `llama-server` with `-ngl 99 --flash-attn on
  --cache-type-k q8_0 --cache-type-v q8_0 --reasoning off --cache-ram 0 --parallel 1`,
  context 12288–24576, 3.1–3.5 GB VRAM.
- Without `--cache-ram 0` the default 8 GiB prompt cache pushed the host into swap.
- Cool GPU: prompt processing 127–260 tok/s, generation 20–38 tok/s. Clamped: 28 and
  ~5 tok/s.
- Through Ollama, qwen3 with hidden thinking took 182 s against 6.5 s on llama.cpp.
  `ollama serve --help` on 0.30.10 gives a context of 4k on a card this size, so a
  long transcript sent through `/v1` would be cut without an error.
- Only extraction tasks with prompts up to ~2.6k tokens were measured; summaries of
  long transcripts never were.

In tapeback:

- The summary prompt sends both `## Transcript` and `## Diarized Transcript`
  (`summarizer.py:22`, `formatter.py:282`). For a 60-minute Russian meeting that is
  59.4k characters; the diarized section alone without markup is 26.9k (~9k tokens),
  which fits one pass in a 12k context.
- After transcription the pipeline frees the GPU (`pipeline.py:226`); after
  diarization it does not (`pipeline.py:271`, `:388`). The pyannote weights and torch's
  CUDA context likely still hold VRAM when the summary starts.
- `summarizer.py` has 466 lines; the limit is 500. Provider data lives in five tables
  (`settings.py:14`, `:24`; `summarizer.py:58-84`).
- The openai SDK reads `OPENAI_API_KEY` when `api_key=None`
  (`openai/_client.py:155`), so a generic endpoint must always pass a string.
- `_stt_caps.py` already lets a remote model cover diarization
  (`allows_local_diarize`): the seed of a stage model.

## Design for this PR

### `tapeback setup`

- `--diarize/--no-diarize`, `--summarize/--no-summarize`, default `None`. Any flag:
  write it, no prompts. No flags in a terminal: one `click.confirm` per toggle,
  defaulting to the value in effect. No terminal: print the report only.
- The writer, `_user_env.py`, edits the first file of
  `Settings.model_config["env_file"]` with python-dotenv's `set_key(...,
  follow_symlinks=True)`: other lines byte-identical, the mode kept, a new file 0600
  in a 0700 directory, an unchanged value not written. An `OSError` (a read-only
  target) prints the lines to add by hand. python-dotenv becomes a direct dependency.
- The report: each written key's effective value and its source, in order
  environment > `./.env` (unless it is the user file) > user file > default. A
  shadowed key names what shadows it. Secrets only as set/not set.
- Engine-aware: when the STT model provides speakers (`allows_local_diarize` is
  false), no diarization question; setup says where speakers come from.
- Hints: diarization without an HF token; a summary without a usable key or endpoint;
  a running tray keeps its old settings until restarted.

### The endpoint

- Move-only refactoring first: keys, chain, retries and SDK calls go to `_llm.py`
  with one provider registry; `summarizer.py` keeps prompts, masking, parsing,
  markdown. The test isolation list is driven off the registry.
- A new provider, `openai_compatible`: base URL + model, both required; the key only from
  `TAPEBACK_LLM_API_KEY`. It covers llama-server, Ollama, LM Studio, ollama.com,
  anyapi.ai.
- Privacy: the endpoint and the cloud providers are mutually exclusive settings: the
  endpoint is never in the cloud fallback chain and never falls back to a cloud key
  found in the environment; masking applies whatever the host (a forwarded port
  makes "loopback" remote); a URL with credentials is refused; plain http to a
  non-loopback host warns. `status` shows host and port only.
- No SDK retries (each one re-reads the whole prompt); a `TAPEBACK_LLM_TIMEOUT`
  with the SDK's own default of 600 s, so cloud calls do not change.
- A misconfigured endpoint skips the summary and says why; it never blocks a
  recording.
- Chunking (map-reduce over speaker turns, mask before split, unmask after merge) is
  built only if the bench shows an hour does not fit one pass; with one transcript
  section (below) an hour is ~9k tokens.

### One transcript in the note

A diarized note keeps one section, the diarized transcript (`You` from the mic,
`Speaker N` from diarization), instead of the raw `You`/`Other` transcript followed by
the diarized copy of the same text, which readers deleted by hand.
`raw_segments` and `_adds_speaker_information` go; the summary prompt describes one
section; for older notes `summarize` sends only their diarized section. Every provider
gets the shorter input.

### The bench

Run later, in a session of its own.

`scripts/bench_summarize.py` drives the real `summarize()` with
`Settings(_env_file=None, ...)` over a manifest of meeting notes kept outside the
repo, at least one an hour-long Russian meeting. Candidates: Qwen3.5-4B Q4_K_M
(thinking off) and Qwen3-4B-Instruct-2507 Q4_K_M (no thinking mode, so nothing to
misconfigure in Ollama). Columns: model, runtime, device, input, tokens in, load s,
prompt and generation tok/s, wall s, peak VRAM, minimum free RAM, swap-in, clamp,
valid JSON on the first try, action/decision agreement with the note's existing cloud
summary, summary language. Clamped rows are never averaged with cool ones. The table
goes here, and the README follows it.

## After the bench: the convenient package

Decided by the bench, not by reasoning:

- Managed llama-server (provider `local`): started per summary with the measured flags
  on a free loopback port, the key in an env var rather than argv, stopped afterwards
  so the next recording's Whisper gets the VRAM, orphans reaped through a PID file in
  the state directory, half the context after a CUDA OOM. The model as a local path or
  a pinned GGUF downloaded through huggingface_hub, offered by `setup`. AUR
  `tapeback-llm-local` depends on `llama.cpp` (cuda or vulkan); the `.deb` bundles a
  pinned llama.cpp release checked by sha256; pip users need `llama-server` on PATH.
- Ollama: `ollama` from Arch extra; needs `OLLAMA_CONTEXT_LENGTH`, a non-thinking model
  and a short keep-alive, otherwise the model holds VRAM after the summary.
- Rule: if tuned Ollama comes within 10% of the managed server, the managed server is
  not worth its code.
- With a managed server, a clamp policy becomes possible: `thermal_clamp_active()`
  before the stage and between chunks; on clamp `gpu`, `cpu` (only if RAM allows) or
  `defer` (`summary: pending` in the note and `tapeback summarize --pending`).

## Later

- A menu-driven setup, beyond today's two switches: the summary's cloud provider or
  server, with a `/v1/models` reachability probe and a line saying whether the text
  stays on this machine, and a check of what fits the machine (the probe below).
- Presets: named sets of `TAPEBACK_*` values in code, applied by `setup --preset`
  through the same writer and report. Built-ins along the lines of `weak`,
  `laptop-4gb`, `desktop`, `cloud`. A hardware probe (VRAM from nvidia-smi with a
  `/proc/driver/nvidia` fallback, RAM) recommends a model; a preset steps one size
  down or up from it (fastest / balanced / accurate). Questions only with a TTY, auto
  mode without one; `--auto` and `--preset` skip the dialog. Thresholds from our own
  benches, never from vendor tables. User
  presets in TOML conflict with AGENTS.md ("no config files besides pyproject.toml");
  `.env` fragments would not.
- Per stage: a read-only view over existing fields first (`stt_backend`; diarization
  local, off or covered by the STT model; summary provider or endpoint or off). Model
  references (a name, a path, an HF file) grow out of this PR's resolver into a
  `_models.py`.
- Model downloads: progress with sizes, our own directory, a pinned revision, a
  sha256 + size manifest; download to a temporary file, verify, rename, then a
  completion marker; hashed once at download, not on every start; a cancelled download
  leaves nothing half-installed. A user-supplied path is referenced, never copied or
  deleted; only files under our model root are ever removed.
- Multi-stage engines: `_stt_caps.RemoteSttCapabilities` becomes engine capabilities
  (transcript, word timestamps, speakers, emotion, audio events, summary, egress); the
  pipeline skips covered stages, as `allows_local_diarize` does today, and setup shows
  them as "provided by X". An audio engine that also summarizes bypasses the text
  masking seam and is treated like remote STT. Requirements for the descriptor:
  - capabilities are keyed by engine, model and language: a vendor may diarize some
    languages and not others, or offer sentiment for English only;
  - limits per request (maximum duration, maximum bytes), since speaker labels hold
    only within one request: chunked audio needs a cross-chunk speaker merge;
  - one job per channel: speakers come from the monitor channel, `You` from the mic;
  - local diarization is skipped only once remote speaker labels have actually
    arrived, never on the promise of a flag;
  - emotion and audio-event tags are an optional stage, off by default;
  - setup shows each cloud provider's data-retention and training default.
  Built when a second multi-stage engine arrives; `models.py` gains `emotion` fields
  only when an engine produces them. Local candidates for a Russian bench on 4 GB:
  GigaAM-v3 (with GigaAM-Emo for emotion), Parakeet TDT 0.6B v3, Qwen3-ASR-0.6B, and
  Streaming Sortformer against pyannote. SenseVoice-Small has no Russian.
- A generic OpenAI-compatible STT endpoint, the twin of the LLM one. Audio leaves the
  machine, so it follows the remote-STT rules.

## Defects found on the way (a bug goes first)

1. VRAM is not freed after diarization (`pipeline.py:271`, `:388`). Fixed first on
   this branch: `_diarize()` frees the GPU in a `finally`; regression test
   `tests/regressions/test_diarization_vram_release.py`.
2. `transcriber.py` has 578 lines, over the 500-line limit.
3. `.env.example:17` suggested `TAPEBACK_CHUNK_LENGTH=2`, which the README measures as
   much slower. Fixed on this branch: the example shows the default, 30.
4. A diarized transcript drops short replies. Diarization cuts a Whisper segment where
   the speaker changes, and the one-second VAD-artifact rule measured the piece, not
   the segment. On recorded meetings the diarized section lacked about two words in a
   thousand of the raw one; in a stereo note whose labels then matched the raw ones,
   the raw section was hidden and the reply lost outright. Keeping each piece under
   its own pyannote speaker lost nothing, but added a speaker who was not in the
   meeting: lines of a word or three cut from the others' sentences. Fixed on this
   branch: a run of words under a second joins the speech around it
   (`diarizer.py:270`), and a segment Whisper returned longer than a second keeps that
   length when its words span less (`formatter.py:151`); regression test
   `tests/regressions/test_diarized_short_reply.py`. Replayed under each rule, a
   recording keeps every word and loses the phantom's fragments; what is left of a
   phantom is a whole Whisper segment that pyannote clusters apart, §6 of the BACKLOG.
5. Text-only remote STT repeats the mic text. `split_on_silence` cut a segment without
   word timestamps at mic pauses and gave every piece the whole text; gpt-transcribe
   returns one such segment per chunk. Fixed on this branch: such a segment is not
   split (`channel.py:226`), and the silence and bleed checks judge it stretch by
   stretch between the pauses, as they judged the pieces (`channel.py:272`). Judged
   whole it goes to whichever side talked more: a 600 s mic chunk of a recorded
   meeting came out "monitor", so whatever the user said in those ten minutes would
   have gone. Regression test `tests/regressions/test_text_only_mic_once.py`.
6. The default LLM models were retired. Vendors had retired most of them or closed
   them to new keys (anthropic's on 2026-06-15, groq's on 2026-08-16), so a summary
   without `TAPEBACK_LLM_MODEL` failed; openai's `gpt-4o` shuts down on 2027-01-14.
   Fixed on this branch: current defaults from each vendor's deprecation page
   (`const.py:116`). The new anthropic default thinks adaptively, and its reply can
   open with a thinking block, which the transport took for the text; it now reads
   the text block (`_llm.py:251`). Regression test
   `tests/regressions/test_anthropic_thinking_reply.py`. What is still open: §8 of
   the BACKLOG.

## Commits

Done on `feat/setup-local-llm`: 1–7 and 9. Commit 8 waits for the menu-driven setup.

1. `fix:` free the GPU after diarization.
2. `fix:` a diarized transcript keeps short replies.
3. `fix:` a segment without word timestamps is not split on pauses.
4. `feat:` one transcript in a diarized note.
5. `feat:` `tapeback setup` with the writer and the report, and python-dotenv.
6. `refactoring:` move the transport to `_llm.py`.
7. `feat:` the `openai_compatible` provider, tested against a loopback HTTP fixture that records
   raw request bodies: no `OPENAI_API_KEY` in any byte sent, no cloud call when the
   endpoint fails, the masked body is what crossed the wire.
8. `feat:` setup configures the summary.
9. `docs:` README, `.env.example`, CHANGELOG through the script.

The bench, then the managed server, clamp policy, chunking and packaging follow as
their own work once the table exists.

## Decided (2026-10-15)

- The provider is `openai_compatible`; `local` stays for a server tapeback manages.
- The endpoint and the cloud providers are mutually exclusive: no fallback either way.
- A diarized note keeps one transcript; every provider gets that shorter input.
- The bench runs in a separate session; neither Ollama nor llama-server is ruled out.
  Qwen has matched a cloud model on other extraction tasks.
- Preset storage is decided later.
- (2026-10-16) The local summaries stay in this PR. `setup` keeps its two switches;
  choosing the cloud or a server moves to a menu-driven setup that also checks what
  fits the machine.
