# Backlog

Items that need design work rather than a patch. Each says what is known, what is not,
and why it is not simply done.

---

## Why runs used to stall with no way out

Every stall we chased had the same shape: **a resource was quietly taken away, and the
tool kept trying anyway.** Not one of them raised an error. Recording them together
because the shape, not the specific cause, is what generalises.

| what was removed | how it showed | how it was detectable | escape |
|---|---|---|---|
| GPU power budget (50 W → 5 W, clocks 300 MHz) | 25x slower; a 15-minute job ran 2h17m | `clocks_event_reasons` thermal bits set **at idle**, where a healthy card reports only `GpuIdle` | idle ~451 s; after chassis saturation >900 s |
| GPU entirely (CUDA OOM leaked the allocation) | next stage found 95 MiB free and silently used CPU | `memory.free` collapsed and never recovered | process exit only |
| GPU silently (CPU fallback) | ~10x slower, output identical | nothing — the warning scrolled past | — |
| RAM (channel arrays + other apps) | swap thrash; CPU I/O-bound, everything crawls | `si`/`so` in vmstat, available MB | free memory |
| GPU throughput (encoder doing 15x redundant work) | RTF 0.37x, looked like "the model is slow" | nothing observable — it was a config semantics bug | — |

### What the shape implies

1. **Measure the resource, not just the outcome.** "Transcription is slow" is
   indistinguishable between a clamped card, a CPU fallback, and a bad setting. Clocks,
   power limit, free VRAM and free RAM tell them apart; elapsed time does not.
2. **Have a floor, and change strategy at it rather than persisting.** A clamped GPU is
   ~8x slower than the CPU, so waiting it out was strictly the wrong move. The general
   rule: when the resource drops below the point where the alternative wins, take the
   alternative.
3. **Make degradation loud.** Every one of these was survivable; what made them
   expensive was that a degraded run looked exactly like a healthy one.

### What is already implemented

Clamp detection with CPU fallback, pre-flight VRAM floor, process isolation, per-stage
GPU telemetry, positive device reporting, progress percentage, run records, bounded
channel memory. See CHANGELOG 0.9.8; the decisions are ADR-003 to ADR-005 in `docs/adr/`.

---

## 1. Progress-rate watchdog

**Why.** Every guard so far is a *pre-flight* check: it looks once, before the work.
Nothing notices a collapse that happens mid-run — the clamp can latch after transcription
has started, and the current design will ride it to the end of the channel.

**The idea.** We now emit progress (`transcribe monitor: 40% (2:31 / 6:18)`), so a live
real-time factor is available for free. Establish a baseline from the first minute, and
if the rate falls below some fraction of it for a sustained window, act: report loudly,
and optionally restart the channel on CPU (cheap now that the worker is a separate
process and partial results survive).

**Why it is not done.** Choosing the thresholds needs data we do not have. A rate drop is
not always pathological — a dense passage genuinely decodes slower than silence, and the
temperature ladder makes per-window cost vary by up to 6x legitimately. Acting on noise
would abandon a healthy GPU for a slower CPU. Wants: a distribution of per-minute RTF
across a set of real recordings, then a threshold with a false-positive rate one can
state.

**Attractive because** it is the one guard that generalises past NVIDIA — it needs no
vendor telemetry at all, only our own progress numbers.

---

## 2. Resource guards beyond this laptop

Everything implemented reads `nvidia-smi`. Where that is absent the code degrades to "no
answer, do not block", which is right but blind.

- **Other NVIDIA machines.** Desktops rarely see an EC clamp, but power limits, driver
  throttling and shared-GPU contention all produce the same shape. The existing
  bit-mask logic should carry over; unverified on any card but a GTX 1650 Ti Mobile.
- **AMD / Intel GPUs.** `rocm-smi` / `intel_gpu_top` expose comparable counters. The
  `_gpu.py` seam is one function (`query_nvidia_smi`) plus a parser, so a second backend
  is small — but untestable without the hardware.
- **Apple Silicon.** No CUDA path at all; faster-whisper runs CPU-only. The thermal
  story is different and probably needs nothing.
- **Containers and cloud.** `nvidia-smi` may report the *host* while a cgroup limits us
  to less. Free-RAM checks read the host too, so `min_free_vram_mib` and any future RAM
  guard would be measuring the wrong number. Reading `/sys/fs/cgroup/memory.max` when
  present is the fix.

**Why it is not done.** None of it is verifiable here. Shipping a code path for hardware
one cannot test is how the `int8_float16`-on-CPU crash got in — the fallback existed but
had never actually run.

---

## 3. `loudnorm` is our own worst thermal contributor

**Measured 2026-08-06**, same 16-minute recording, each variant after a cooldown:

| filter | time | CPU package start → peak |
|---|---|---|
| `loudnorm` (current) | **39 s** | 85 → **94 °C** |
| `dynaudnorm` | **1 s** | 86 → 86 °C |
| `speechnorm` | 1 s | 93 → 93 °C |
| none (resample only) | 3 s | 92 → 93 °C |

So the whole cost of the `split` stage is `loudnorm`, and it is also the entirety of
tapeback's own contribution to chassis heat — 9 °C on top of whatever the desktop is
already doing, immediately before the GPU is asked to work.

**Thread limiting was tried and does nothing.**

| variant | time | peak | ffmpeg CPU |
|---|---|---|---|
| baseline | 37.0 s | 94 °C | 96 % |
| `-filter_threads 1` | 36.0 s | 95 °C | 99 % |
| `-threads 1 -filter_threads 1` | 38.0 s | 94 °C | 99 % |

ffmpeg uses ~96-99 % — one core — even though the graph asks for two `loudnorm` chains;
it interleaves them on a single thread rather than running them in parallel. There is no
load to spread, so "spread it thinner" is not an available move. Only doing *less work*
is.

### The A/B was run — quality is equivalent except on one axis

Three recordings, both channels, 6 minutes sampled from across each (3x2-minute slices),
rendered through the production filter chain and transcribed with everything else fixed
(large-v3-turbo, int8, CPU, glossary on).

| set | hallu | rep-w | rep-p | low-conf % | punct/1k | terms | chars |
|---|---|---|---|---|---|---|---|
| mon / loudnorm | 1 | 1 | 1 | 1.3 | 248 | 4/19 | 4061 |
| mon / dynaudnorm | 1 | 1 | 1 | 1.3 | 254 | 4/19 | 4143 |
| mic / loudnorm | 0 | 1 | **3** | **11.5** | **140** | 10/19 | 4208 |
| mic / dynaudnorm | 0 | 1 | **0** | **10.1** | **92** | 9/19 | 4362 |

**Monitor channel: identical.** Every metric matches; the punctuation difference
(248 vs 254) is noise.

**Mic channel — the quiet one normalisation exists for:** `dynaudnorm` is *better* on
recognition confidence (10.1 % vs 11.5 % low-confidence words) and on phrase loops
(0 vs 3), level on hallucinations, and one term behind out of nineteen — noise at this
sample size.

**The one real difference is punctuation on the quiet channel: 92 against 140 per 1000
words.** That is the same axis on which `large-v3` beat turbo in stage 3a, and the axis
the metric suite was originally blind to — it was found by reading, not measuring. A
third less punctuation is probably not noise.

### The default stays: ADR-006

**Worth doing instead:** make it a setting (`TAPEBACK_NORMALISATION=loudnorm |
dynaudnorm | none`) keeping `loudnorm` as the default, so a hot or weak machine can take
the cheap path deliberately without changing anyone else's output.

### Caveats on the measurement

- **Timings from that run are void.** The machine was thrashing swap throughout —
  load 10–18, 33–38 % iowait, the worker sometimes at 0.4 % CPU. Transcript *text* is
  deterministic with respect to the audio, so the quality columns stand; RTF does not.
- Small sample: 3 recordings x 6 minutes per side.
- **Silence gating was not reproduced.** In production the mic channel is zeroed where
  the user is only listening, before transcription; these clips were ungated, so Whisper
  saw more silence than it really would. The gate measures the *pre*-normalisation
  signal (`SILENCE_RMS_THRESHOLD` on raw channels), so it is unaffected by the filter
  choice either way — but the real quiet-channel behaviour is gentler than this test.

---

## 4. Mid-channel resume

Resuming part-way through a channel needs `clip_timestamps`, which faster-whisper
documents as ignoring `vad_filter`. VAD is half of why the silence hallucinations went
away, so that trade is bad as stated.

**The way round it** is to run the VAD ourselves (`faster_whisper.vad.get_speech_timestamps`),
drop the chunks already transcribed, and pass the remainder as `clip_timestamps` — same
VAD output, just filtered. Plausible, but the timestamp restoration path differs when
clip timestamps are supplied and would need verifying against a known-good transcript.

Worth doing only if long single-channel runs stay common. With the current speed a
channel is minutes, so the payoff is smaller than it was.

---

## 5. Personal glossary

`TAPEBACK_HOTWORDS` replaces the shipped list wholesale. Two things are missing:

- **Adding** to the default rather than replacing it (`TAPEBACK_HOTWORDS_EXTRA`), so a
  user keeps upstream improvements while adding their own vocabulary.
- **Harvesting candidates** from accumulated transcripts — proper nouns and acronyms by
  frequency, terms appearing in several spellings. Done by hand once; the method worked
  (it found `RabbitMQ`/`RebitMQ`, `LayoutLMv3` as "Layout MV3", `Qwen` as "QVN") and
  belongs in a script beside `bench_transcribe.py`.

**Constraint that makes this fiddly:** faster-whisper silently truncates hotwords past
`max_length // 2 - 1` = 223 tokens. A merged default-plus-personal list can exceed that
without warning, so whatever is built has to measure and refuse rather than truncate.

---

## 6. Diarization: quality, and models to choose from

Found by the 0.9.9 pre-release QA; full numbers in `2026-10-06-diarization.md`. In
short: pyannote reports the real speakers plus a small third cluster; the spectral
merge of real speakers is too weak a signal to rely on; diarization ran at roughly the
recording's length because it met the thermal clamp with no check, no telemetry and
warnings that bypassed the run record. 0.9.9 fixed only the bridging defect; the clamp
check, the telemetry and the warnings came on 2026-10-16. The rest belongs with
pluggable diarization models, chosen from a bench table.

Measured on 2026-10-16 with a recording of about an hour, before that check: the clamp
latched three minutes into the first channel's transcription and held for the next
90 minutes, idle CPU-only stages included. Diarization ran clamped for 3522 s, 1.20x
the recording. A check made before a stage cannot see a clamp that latches during it.

---

## 7. Found 2026-10-15, not fixed yet

- **The tray dies on Python 3.15.** `dbus_next/service.py:9` imports
  `typing.no_type_check_decorator`, which CPython 3.15 removed (gh-133601). dbus-next's
  last release is 0.2.3 (2021); `dbus-fast` is the maintained fork. All 43 pytest
  warnings are this one deprecation, so `filterwarnings = ["error"]` would need exactly
  one filter until the move. Python 3.15 in CI: not yet (decided 2026-10-15).
- **Packaging resolves dependencies at build time and lets extras rewrite the base.**
  The AUR and `.deb` venvs are not built from `uv.lock`, the extras' scriptlets pip into
  the pacman-owned `/opt/tapeback`, and the AUR venv borrows undeclared system
  packages through `--system-site-packages`. Redesign undecided: build from the lock,
  or create the venv in `post_install`.
- **The user's own short replies are dropped.** The VAD-artifact rule
  (`formatter.py:151`, under a second) measures diarization's pieces by the segment
  they were cut from, but not the mic's: pause splitting (`channel.py:226`) and the
  trim to loud words (`channel.py:32`) leave a "Да." between the other side's turns
  shorter than a second, and it is dropped. Needs a decision, not a fix: shown as lines
  of their own, the user's backchannels cut the other side's speech into more blocks,
  and noise Whisper took for a word would show as well.
- **A note can open with Speaker 2.** Labels are numbered as diarization assigns them
  (`diarizer.py:346`), before the formatter drops artifacts under a second, so an
  artifact that carried a new voice first takes "Speaker 1" with it. Numbering by first
  appearance in the note would fix it.
- The setup and local-LLM work and the defects it turned up: `2026-10-15-setup-and-local-llm.md`.

## 8. Found 2026-10-16: the new default LLM models

- **They all reason by default, and nothing measures it.** anthropic thinks
  adaptively, gemini 3 cannot turn thinking off, groq's `openai/gpt-oss-120b` always
  reasons, `deepseek-flash` thinks by default, openai's `reasoning_effort` defaults to
  medium: slower and costlier than the old defaults. On anthropic the thinking shares
  `max_tokens=4096` (`_llm.py:275`), so a long meeting can cut the JSON off, and the
  parse retry (`summarizer.py:161`) resends the whole transcript under the same cap.
  No test or QA step summarizes through a live vendor (the e2e suite passes
  `--no-summarize`): one `tapeback summarize` per provider on a long note is the check.
- **qwen's address is moving.** Alibaba tells Singapore users to move from
  `dashscope-intl.aliyuncs.com` to
  `https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1`. The
  workspace id is the user's, so it cannot be a constant in `const.py`. No retirement
  date for the old host is given.
- **The data policy of OpenRouter's free models is unknown.** Their model data has
  no policy fields, and a free model may keep or train on prompts
  (`poolside/laguna-s-2.1` says it trains on inputs). The README warns; the default's
  own policy is not checked. The free tier allows 50 requests a day.

## 9. Found 2026-10-16: the local LLM

- **`-ngl 99` switches llama.cpp's `--fit` off.** With an explicit layer count, fit
  only warns, and a model that does not fit fails to load instead of leaving layers on
  the CPU; `auto` keeps a 1 GiB margin, which on a 4 GB card may move layers off the
  GPU. A row for the bench, not a guess.
- **JSON from the local model is not enforced.** llama-server enforces
  `response_format` only with a non-empty schema; `json_object` alone is ignored for
  this model's template. Measured earlier outside this repo, Qwen3.5-4B with thinking
  off answered valid JSON 10 times in 10 under a `json_schema` grammar, and without
  bounds on arrays and strings a 4B model looped to its token limit. The JSON retry it
  would spare reprocesses the whole transcript: its system prompt differs, so nothing
  is reused. The bench's "valid JSON on the first try" column decides.
- **The local request sets no `max_tokens` and no temperature, and ignores
  `finish_reason`.** The server's default temperature is 0.8, and a truncated answer
  reads as "not JSON" and costs a second full pass. For the bench.
- **Halving the context frees little for the default model.** Qwen3.5-4B is hybrid:
  measured earlier, it took 3090 MiB at a 12288-token context, and 24576 ran on the
  same card, while 8192 tokens do not hold an hour of Russian. An out-of-memory more
  likely means another program holds the card, so the message reports free VRAM; the
  halving stays for other models a user names.
- **The summary's clamp wait may be held up by tapeback's own CUDA context.** On a
  shorter meeting the clamp cleared by itself between transcription and
  diarization, when no process held a CUDA context. After diarization the main
  process kept torch's 106 MiB context, and the clamp held through the whole 600 s
  wait (64-72 °C, 300 MHz) and the summary after it. One run, so a correlation; if it
  holds, diarization in a worker process of its own, as transcription runs, would let
  the card idle before the summary.
- **A local summary on the CPU was ruled out on the test machine**: the model needs
  2.7 GB of RAM where about 2.3 GB are free. A summary deferred to a cooler moment
  (`summary: pending`, `tapeback summarize --pending`) stays open, if the wait for
  the clamp (spec, defect 7) proves too long in practice.
- **Vulkan against CUDA on a GTX 1650 Ti is not measured on the same input.** Each
  build was run once, on different prompts (spec, "The convenient package"). The
  `.deb` carries the Vulkan build for its reach (Ubuntu 22.04+, 30 MB).
- **The pinned GGUF's Hugging Face scan reads "safe" with the code PAIT-GGUF-100**,
  whose meaning is not public. Its chat template differs from Qwen's in two places: the
  tool-call arguments and the thinking default.
- **`tests/fixtures.py` is over the 500-line limit** (`wc -l tests/fixtures.py`). AGENTS.md keeps every
  fixture there, so a split needs a decision on the layout first.
- **The diarize, tray and cuda companions never run on a stand.** They are built, and
  released as `.deb`s, but only `tapeback-llm-local` (with `tapeback-llm`) is installed
  and run.

## 10. Found 2026-10-17

- **The security review was not rerun for three new outbound paths.** The 2026-08-07
  review (`done/2026-08-07-security-review.md`) asks for a rerun whenever a call that
  leaves the machine is added. Since then remote STT uploads audio, `openai_compatible`
  sends the transcript to an address the user gives, and `local` downloads its model
  from Hugging Face.
- **A second stop at the same moment dies with a raw error.** The tray and `tapeback
  stop` both read `session.json` (`recorder.py:384`); the first removes it after up to
  5 s of waiting for the recorders (`recorder.py:394`), and the second then fails with
  `FileNotFoundError` instead of "No recording in progress". Read from the code, not
  reproduced: no second note should come of it.

## 11. Found 2026-10-28: two of the three remote STT models shut down on 2027-02-26

- OpenAI deprecated `whisper-1` and `gpt-4o-transcribe-diarize` on 2026-08-26 and
  shuts them down on 2027-02-26 (developers.openai.com/api/docs/deprecations). The
  replacement it names, `gpt-transcribe`, returns text without timestamps and without
  speakers, and no diarizing successor is announced. After that date remote STT is
  `gpt-transcribe` only: no remote speaker labels, and no timestamps to align with
  local pyannote turns. Local diarization stays required.
- Needs a decision before the date: drop the two models from `_stt_caps.py`, warn at
  load until then, or align `gpt-transcribe` text to audio locally (forced alignment).
- Anthropic offers no audio input or speech-to-text API, so it can replace neither
  stage. Details and sources: `docs/local/2026-10-28-remote-stt-survey.md` (not in git).

## 12. Found 2026-10-28, while turning warnings into errors

- **A recorder that dies mid-meeting ends the recording silently.** When one parecord
  exits (a source vanished), `Recorder.is_recording()` removes `session.json`
  (`recorder.py`, `is_recording`), and `tapeback start` takes it for a stop from another
  terminal and returns with exit code 0 (`cli.py`, `start`): no error, no processing,
  no note, and the other parecord keeps recording. Read from the code, not reproduced. Needs a
  decision: process what both channels hold so far, or stop the survivor and fail
  with the tail of the dead recorder's log (`monitor.log`/`mic.log` in the session
  directory).
- **The python-dotenv leak is filtered, not fixed.** Up to 1.2.4, `set_key` leaves the
  `.env` open when it cannot create its temporary file; fixed upstream in 6a49d6d8,
  unreleased. When a release carries it, raise the bound in `pyproject.toml` and drop
  its two `filterwarnings` lines.

## 13. Found 2026-10-30, while setting up mutation testing

- **The transcription worker may load torch for nothing.** `ctranslate2/__init__.py`
  imports its converters, and `ctranslate2/specs/model_spec.py` tries `import torch`:
  where the diarize extra is installed, every process that imports faster-whisper loads
  torch, several hundred MB, though transcription never uses it. Not measured in the
  worker yet: compare the worker's peak RSS with torch installed and with its import
  blocked (a `sys.meta_path` finder in `_worker.py`) before changing anything. Every
  pytest process loads torch the same way: collecting
  `tests/regressions/test_bench_measures_its_own_grid.py` loads the bench, which imports
  the transcriber.
- **A FIFO as the user's settings file becomes a plain file holding all it served.**
  Needs a decision. pydantic-settings reads a FIFO env file, but `_user_env` asks
  `Path.is_file()`, which is false for one: `tapeback setup` reports the values it
  serves as defaults, and its first write (`set_key`, then `os.replace`) leaves a
  regular file with every line the FIFO gave, `TAPEBACK_HF_TOKEN` included. Refuse a
  file that is not regular, or read it and never write it; a mutant of `write` that only
  a FIFO tells apart survives until then.
- **A pause longer than half its segment goes unfound when the monitor is silent.**
  Needs a decision. `_find_split_points` sets the silence level from the segment's median
  energy, so 0.5 s of speech, 2 s of silence and 0.5 s of speech put the median in the
  silence and nothing is split; a loud monitor during the pause still finds it.
