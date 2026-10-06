# Spec: transcription speed and quality

Status: all stages landed (1a, 1b, 1c, 2, 3a, 3, 3b, 3c, 4) plus process isolation and resume.

## Stage 3a result — defaults chosen

Grid of 18 points on 6-minute sampled excerpts, every point run with the thermal clamp
released (see the thermal note below):

| model | compute | hotwords | RTF | hallu | loops | low-conf % | punct/1k | terms |
|---|---|---|---|---|---|---|---|---|
| large-v3 | int8_float16 | off | 7.55 | 0 | 4 | 6.6 | 174 | 5/19 |
| large-v3 | int8_float16 | on | 6.09 | 0 | 4 | 10.0 | 263 | 9/19 |
| large-v3-turbo | float16 | off | 3.65 | 1 | 3 | 5.4 | 187 | 10/19 |
| large-v3-turbo | float16 | on | 4.11 | 0 | 1 | 6.1 | 147 | 13/19 |
| large-v3-turbo | int8_float16 | off | 13.41 | 2 | 3 | 10.0 | 152 | 10/19 |
| **large-v3-turbo** | **int8_float16** | **on** | **16.05** | **0** | **1** | **4.9** | 165 | **13/19** |

**Chosen: `large-v3-turbo` + `int8_float16` + hotwords** — best or tied-best in every
column. Confirmed on the full 31-minute reference recording: 33 distinct English terms
against 15 originally, low-confidence words 59.4 per 1000 against 124.5, and mic+monitor
in 194 s against 814 s.

`large-v3` was rejected despite reading better by hand (highest punctuation density, 263)
because it recognised fewer terms at less than half the speed. The disagreement between
"reads better" and "recognises terms better" is real and unresolved — it is the reason
both metrics now exist.

**Thermal caveat that invalidated the first attempt.** The first grid ran with the card in
an EC clamp and timed the same configuration at **3519 s (RTF 0.31x) against 41 s
unclamped** — a 25x distortion, at only 68-70 °C, so not heat in the moment. A clamped
card reports `SW Thermal Slowdown` **at idle** (`0x24`) where a healthy one reports only
`GpuIdle` (`0x1`); on this laptop it took **451 s of idle** to release. The harness now
waits for that before every grid point, and the grid runs on short excerpts so no single
point saturates the chassis.

## Context

Two reported symptoms:

1. **Runs never finish.** A real run on 2026-07-30: a 15m55s recording produced
   `Stage 'transcribe mic' took 8203.0s` (2h17m) — for one of two channels. Aborted with Ctrl+C.
2. **Poor quality**, especially on Russian speech carrying English technical terms.

Scale of the problem: **16 of 53 recordings have no transcript at all**, and they are exactly the
longest ones (37, 36, 25 min). Since 2026-06-28, no recording over 20 minutes has been completed.

The goal is to find root causes rather than treat symptoms, decide whether Whisper should be
replaced, and produce a fix plan.

---

## Measurements taken while writing this spec

Hardware: GTX 1650 Ti Mobile 4 GB (Turing, CC 7.5), hard 50 W limit, driver 610.43.03, CUDA 13.3.

- CUDA works: `ctranslate2.get_cuda_device_count() == 1`, `float16` / `int8_float16` supported. The
  cu12 wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) are installed and `preload_cuda_libs()`
  succeeds. **A silent CPU fallback is not the cause of the 8203 s.**
- Under Whisper load: 2428 MiB VRAM, sm 1860–1890 MHz against a 2100 MHz maximum,
  `clocks_event_reasons.active = 0x4` (SW Power Cap), temperature 53 → 86 °C over ~5 minutes.

---

## Root causes: speed

### 1. `chunk_length=7` — the main offender (~4.3x redundant encoder work)

`settings.py:61` sets `chunk_length: int = 7`. In faster-whisper this overwrites the feature
extractor's window (`feature_extractor.py:203-205`: `nb_max_frames = 7*16000/160 = 700`), and the
decode loop advances `seek` by 700 frames per iteration (`transcribe.py:1173-1178`).

But Whisper's encoder is fixed at 30 seconds. Every window is zero-padded back to 3000 frames
before encoding:

```python
segment = pad_or_trim(segment)                      # transcribe.py:1180
def pad_or_trim(array, length: int = 3000, ...):    # audio.py:111
    """Pad or trim the Mel features array to 3000, as expected by the encoder."""
```

So **every encoder pass costs a full 30 s but advances only 7 s of audio** — 4.3x redundant passes,
a multiplier on top of everything else.

**The value in force on this machine is 2, not the repo default of 7.**
`~/.config/tapeback/.env` sets `TAPEBACK_CHUNK_LENGTH=2`, so production runs pad 200 frames to
3000 — **15x redundant encoder passes**, not 4.3x. This was found by the stage 1c work (a test
asserting the default 7 read 2 instead) and is the single largest factor behind the 8203 s.

**Measured** on `2026-05-03_17-56-45.wav` (145.5 s), large-v3-turbo / float16 / cuda, all other
parameters as in production:

| `chunk_length` | time | RTF | segments | characters |
|---|---|---|---|---|
| **2** (actual production value) | **390.6 s** | **0.37x** | 60 | — |
| 7 (repo default) | 116.8 s | 1.25x | 28 | 1659 |
| **30** | **41.4 s** | **3.51x** | 17 | 1514 |

At the value actually in use, transcription runs **2.7x slower than real time**. Moving to 30 is
**9.4x faster**, with the same content (the character delta comes from less ragged segmentation) and
larger, more navigable timecodes.

The `chunk_length=2` run also carried GPU telemetry: `sm 1773 MHz avg / 1590 min, max 87 °C,
2429 MiB peak, **throttled 90 % of 78 samples**`. So the thermal clamp is real and heavy on a long
run — but it is a consequence of the run being long, not the reason it is long.

`chunk_length` was originally lowered to 7 to fight hallucinations on long pauses
("Субтитры DimaTorzok"). The cure for that is VAD + `no_speech_threshold` + silence gating — all of
which the project already does — not shrinking the encoder window.

### 2. The temperature ladder fires on 7-second fragments

`beam_size=4` times the full `(0.0…1.0)` ladder means up to 6 re-decodes per window. The ladder
itself is necessary (shortening it was verified to push the model into repeat loops), but combined
with cause 1 it triggers on four times as many windows.

### 3. Everything else, by descending contribution

- `word_timestamps=True` is hardcoded (`transcriber.py:163`), adding a DTW alignment pass per
  segment. It is load-bearing (`channel.filter_silent_segments`, `diarizer._resegment_by_words`) and
  cannot simply be removed.
- Two full sequential passes (mic, then monitor) — `transcriber.py:191-194`.
- `batch_size=0` — `BatchedInferencePipeline` is off.
- The `split` stage costs 36 s: `loudnorm` (EBU R128) runs **twice over the full 48 kHz stream** and
  only then `aresample=16000` (`audio.py:113-116`). Moving `aresample` ahead of `loudnorm` cuts the
  work roughly threefold.

### 4. Thermals — real, but secondary

`backlog_profiler/reports/THERMAL-LLM-REPORT.md` documents this machine's behaviour:

- The EC clamp is **persistent**: after chassis saturation it did not release within 15 minutes, and
  only idling clears it.
- Share of samples carrying throttle flags: **93–96 % uncapped** → 37 % (10 min capped) → **0.0 %
  across 76 minutes** with caps plus pacing.
- Its key conclusion: *"speed does NOT depend on the cap"* — the 50 W limit already pins average
  clocks; a cap only shaves peaks, i.e. watts, at no throughput cost.
- Tooling already exists: `scripts/thermal_profile.sh on <MHz>` (nvidia-smi `-lgc` + CPU `no_turbo` +
  RAPL written into both `intel-rapl:0` and `intel-rapl-mmio:0`).

Conclusion for tapeback: **clamp monitoring is worth having as observability, not as a cure.** A
two-hour run will always end up clamped; a 90-second one never will. Remove the 4.3x of redundant
work first, then measure what remains.

### 5. No progress output, so runs get killed by hand

The only line for the entire run was `"Transcribing (this may take a few minutes)..."`
(`pipeline.py:196`). faster-whisper offers `log_progress` (`transcribe.py:259`), which tapeback never
passed. No timeouts, no cancellation, no partial results: on `KeyboardInterrupt` even
`free_gpu_memory()` is skipped (`pipeline.py:228-229` sits on the non-exception path).

Note: `Stage 'transcribe mic' took 8203.0s` is printed from a `finally` block **immediately** — it is
not a slow reaction to Ctrl+C but an honest report of how long the stage had already been running.

---

## Root causes: quality

### 1. The model is `large-v3-turbo`, not `large-v3`

`settings.py:45`. Turbo is a distillation with 4 decoder layers instead of 32; it is noticeably more
prone to repeat loops and weaker on non-English audio.

### 2. No glossary — neither `hotwords` nor `initial_prompt`

`grep` over `../../../src` returns zero hits. faster-whisper supports `hotwords` (`transcribe.py:1545`).
This is the most direct lever for the "technical terms must survive" requirement.

What its absence costs, from the actual transcripts:

- RAG within a single meeting: `rec` / `Rick` / `Ray` / `REC` / `Wreck` / `racist` / `rails` — six
  spellings of one term
- ONNX → `on an X Runtime`, OpenVINO → `OpenSWIFO`, Excalidraw → `Escalidraw`
- tapeback → `тейп *B*`, `тейпбэком`
- "ML and CV pipeline integration" → `ML and CVS. pipeline integration` / `pipeline. immigration`

### 3. `language="auto"` with `language_detection_segments=1`

The language is decided from the first segment of each channel, independently for mic and monitor.
Hence notes with `language: en` in the front matter containing raw Cyrillic:
`but I *stopped* the audio ... исчез`.

### 4. Subtitle-corpus hallucinations are not filtered

At least 11 occurrences across 6 files: `Субтитры DimaTorzok` (x2), `Редактор субтитров .Семкин`,
`Корректор .Кулакова`, `Продолжение следует...` (x6). A recognisable set — filterable by list.

### 5. Timecode collapse — a formatter bug

`formatter.py:41-75`, `_merge_consecutive_speakers`: blocks keep merging while the speaker is
unchanged and `gap < pause_threshold`. **There is no upper bound on block length.** Result, in
`2026-04-16_18-10-19.md`: 31m39s of recording rendered as **two segments, last timecode
`[00:00:45]`**. The text survives; navigation does not.

Three more files have timecode coverage below 20 % of their audio.

### 6. The "Diarized Transcript" section duplicates the transcript

In all 16 files that have it. The only difference is `**Other:**` → `**Speaker 1:**`, including
identical recognition errors. Diarization additionally splits single utterances across two speakers
(`Speaker 2: I got to` + `Speaker 1: see.`).

### 7. The batching trap

With `TAPEBACK_BATCH_SIZE>0`, faster-whisper 1.2.1 **silently ignores** `no_speech_threshold`,
`condition_on_previous_text` and `hallucination_silence_threshold`, and takes only the first value
from `temperature` (`transcribe.py:316-317, 351-369`). Enabling batching therefore quietly disables
the entire anti-hallucination configuration. The README does not warn about this.

---

## Research: should Whisper be replaced

The requirement is Russian speech with English technical terms **inside a single phrase**. That rules
out almost everything.

| Model | Verdict |
|---|---|
| **GigaAM v3** (Sber, MIT, 240M, Russian SOTA, punctuation + normalisation) | Best on pure Russian. But in a benchmark specifically on RU/EN code-switching it **breaks on English terms** (`Gemini` → `Jemni`). Not viable as the primary model. |
| **Canary-1B-v2** (NVIDIA, 25 languages, WER 8.1 % vs 9.9 % for large-v3) | Same code-switching weakness as GigaAM. |
| **Parakeet-TDT-0.6B-v3** (25 languages including Russian, RTFx ~3300, fits 4 GB easily) | Very fast, but detects language **per utterance**, not within a phrase. Mixed phrases are its weak spot. |
| **Voxtral Mini 4B Realtime** (Apache-2.0, WER 5.9 % on FLEURS) | ~9.5 GB in fp16. **Does not fit 4 GB.** |
| **Whisper large-v3** | Trained multilingually with translation, so mixed text is its normal mode. Supports `hotwords`. |

**Conclusion: stay on Whisper.** The choice between `large-v3-turbo` and `large-v3` is **settled by
measurement** (stage 3a), not by literature: turbo wins on speed and VRAM, large-v3 on Russian and
loop resistance. VRAM: turbo in float16 measured at 2428 MiB; large-v3 in `int8_float16` is expected
around 2.9 GB, which fits into 4 GB but only just.

GigaAM v3 as an optional second backend is **deferred, out of scope here**: it does not suit the
primary scenario, and an ASR-backend abstraction is substantial separate work.

---

## Plan

### Stage 1a — progress and timings — DONE

Landed as a 357-line diff. `ruff` and `ty` clean; **277 tests pass, coverage 93.52 %** (up from
92.96 % / 263 tests).

- `_timing.py`: `ProgressReporter` + `format_stage_progress`, throttled to one line per 10 s.
- `transcriber.py`: `describe()` reports the resolved model / device / compute type (plus
  `batch_size` when batching is on); progress is emitted from `_collect_segments`; `transcribe()`
  gained keyword-only `stage` / `on_status` so existing call sites keep working. faster-whisper's own
  `log_progress` was rejected: its tqdm bar writes straight to the terminal, bypassing `on_status`,
  so it would never reach the tray log.
- `pipeline.py`: `stage_timer` around `load_stereo_channels` and `gate_wav_inactive`; `describe()`
  reported in both the stereo and mono paths.
- Tests added: 4 on `ProgressReporter` / formatting, 5 on the transcriber, 2 pipeline regressions.

**What the first instrumented run revealed** (`2026-05-03_17-56-45.wav`, 2:25):

```
Whisper: large-v3-turbo on cuda/float16
  transcribe mic: 25% (0:36 / 2:25)
Stage 'transcribe mic' took 52.1s
  transcribe monitor: 76% (1:50 / 2:25)
Stage 'transcribe monitor' took 409.1s
```

- **The bottleneck is the monitor channel, not the mic.** The original report showed 8203 s on
  `transcribe mic`, and all previous work ("cleaner, faster mic channel", silence gating) targeted
  that path. The gating works — mic now finishes in 52 s. The problem moved.
- The same file took 116.8 s end-to-end in the isolated benchmark but 409 s for a single channel
  inside the pipeline. **Initial hypothesis — that `loudnorm` was inflating VAD speech and firing
  the temperature ladder — was wrong.** The pipeline reads `~/.config/tapeback/.env`
  (`TAPEBACK_CHUNK_LENGTH=2`) while the benchmark passed 7 explicitly. An isolated run at 2 takes
  390.6 s, against 409 s in the pipeline: `loudnorm` accounts for roughly 5 %, not the bulk. Stage 2
  should therefore treat the `aresample`/`loudnorm` reordering as an ffmpeg-time optimisation only,
  and not weaken `loudnorm` on the strength of a hypothesis that measurement has now refuted.

### Stage 1b — GPU telemetry — DONE

Landed as a ~400-line code diff. `ruff` and `ty` clean; **296 tests pass, coverage 93.73 %**.

- `_gpu.py`: `sample_gpu()` context manager polling
  `clocks.sm,temperature.gpu,memory.used,clocks_event_reasons.active` every 5 s on a daemon thread,
  reporting one `GpuStats.format()` line per stage. **Observation only — no cap management, no
  sudo**: tapeback stays an unprivileged CLI, and `thermal_profile.sh` is run separately by hand.
- Throttle accounting uses the NVML bitmask and deliberately excludes `GpuIdle` (0x1) and
  `ApplicationsClocksSetting` (0x2) — idling is not throttling, and an applied cap is a setting
  rather than a symptom. Only SW power cap, HW slowdown, SW/HW thermal and HW power brake count.
- `query_nvidia_smi()` is now the single point of contact with the binary; `diarizer`'s private
  `_get_free_vram_mib` was folded into it, so a missing binary, a driver error and a hung query all
  degrade to `None` the same way instead of each module inventing its own handling.
- `sample_gpu` waits for the first poll before entering the block, so a stage shorter than the
  sampling interval still reports, and the sampler's output is deterministic rather than dependent
  on thread scheduling.
- New setting `TAPEBACK_GPU_TELEMETRY` (default on); no-op without `nvidia-smi` or on `cpu`.
- Verified against real `nvidia-smi` output, not only mocks: `['300', '57', '51',
  '0x0000000000000001']` parsed to `GPU: sm 300 MHz avg / 300 min, max 57°C, 51 MiB peak,
  throttled 0% of 6 samples` — idle correctly not counted as throttled.

### Stage 1c — run metadata — DONE

Split out of 1b to stay under the ~500-line commit limit. **312 tests pass, coverage 93.88 %.**

- `_runlog.py`: `run_log()` context manager writing one JSON record per run to
  `~/.local/share/tapeback/runs/` (honours `XDG_DATA_HOME`; `TAPEBACK_RUN_LOG`,
  `TAPEBACK_RUN_LOG_DIR`). Holds the settings actually used, every status line verbatim, and the
  outcome — `completed` / `aborted` (Ctrl+C) / `failed` (with the error). Exceptions are classified
  and re-raised unchanged, never swallowed.
- Status lines are stored verbatim rather than parsed back into fields: re-parsing our own formatted
  output would break on every reword.
- **Recorded settings are an explicit allow-list, never `settings.model_dump()`** — `Settings`
  carries `hf_token` and `llm_api_key`, and a post-mortem file that leaks credentials is worse than
  no file. Two tests guard this, one of which fails the build if a new `SecretStr` field is ever
  added to the allow-list.
- A failed record write degrades to `None`: losing a diagnostic file must never destroy the run that
  produced it. Directory is pruned to the newest 200 records.

**What it caught immediately.** A test asserting the default `chunk_length == 7` read `2`: the test
suite was never isolated from `~/.config/tapeback/.env`, so every `Settings()` in every test
inherited the developer's machine configuration. Fixed with an autouse fixture that detaches both
env-file sources and clears ambient `TAPEBACK_*` variables, plus a regression test. The same finding
corrected the headline speed number above from 2.82x to **9.4x**.

### Stage 2 — speed — DONE

**315 tests pass, coverage 93.88 %.** The user's `TAPEBACK_CHUNK_LENGTH=2` override has been removed
from `~/.config/tapeback/.env`, so the new default is what runs.

- `chunk_length` default raised `7` → `30` (`settings.py`), with the encoder-window reasoning in the
  comment so it does not get lowered again to fight hallucinations.
- Batching now warns which settings it silently drops (`transcriber.py::_batched_warning`), listing
  only the ones actually configured. Verified line by line against faster-whisper 1.2.1's own
  "Unused Arguments" docstring rather than taken on trust.

**End-to-end on the reference recording** `2026-04-17_19-08-24.wav` (13 m 37 s), `--no-diarize
--no-summarize`:

| stage | time |
|---|---|
| split | 30.4 s |
| load model | 13.3 s |
| transcribe mic | 140.7 s |
| transcribe monitor | 199.7 s |

Transcription total 340.4 s for 817 s of audio — **RTF 2.40x**, against 0.37x at the value previously
in force.

**Quality did not regress — it improved.** Same file, counting subtitle-corpus hallucination markers
and 3x word repeat loops:

| `chunk_length` | hallucination markers | repeat loops | content chars |
|---|---|---|---|
| 2 | 0 | 4 | ~10 100 |
| 10 | **4** | 2 | 10 130 |
| 7 | 2 | 2 | 10 556 |
| **30** | **0** | **1** | 9 468 |

(The 2 run's file measures 20 246 chars only because it also carries the duplicate "Diarized
Transcript" section — its actual content is comparable.) The historical fear that raising
`chunk_length` brings back "Субтитры DimaTorzok" did not materialise: those came from windows too
short to hold speech context, and `no_speech_threshold=0.4` plus `gate_mic_silence` — neither of
which existed when the value was first lowered — now cover the case they were meant to.

**The ffmpeg reorder was dropped: the hypothesis was wrong twice over.**

- ffmpeg's `loudnorm` upsamples internally to 192 kHz for EBU R128 true-peak measurement and *emits*
  at that rate. Putting `aresample` before it produces **192 kHz output WAVs** — caught by the
  existing `test_split_channels_16k`.
- It is not even faster: measured on a 16-minute recording, resample-first 32.4 s vs current 31.6 s.
  `loudnorm`'s cost does not scale with its input rate.
- The real distribution: the split without `loudnorm` takes **0.5 s**, with it **31.6 s** — so
  `loudnorm` is ~98 % of the stage. `-filter_threads 6` does not help (34.4 s).

The order is now documented as load-bearing in `split_channels_16k`'s docstring. Whether ASR input
needs EBU R128 normalisation at all is a *quality* question, not a speed one, and belongs in the
stage 3a grid — it must not be dropped on a hunch.

**Thermals are now the visible next bottleneck.** The run reported `sm 1552 MHz avg / 480 min,
max 87 °C, throttled 99 % of 68 samples`: the card spent essentially the whole run clamped, with the
SM clock falling as low as 480 MHz against a 2100 MHz maximum. Shortening the run did not avoid the
clamp on this chassis, which weakens the stage 1b assumption that it would.

### Stage 3a — model benchmark, default chosen by numbers

The model default is fixed **only after measurement**. Grid: `large-v3-turbo` x `large-v3`, each in
`float16` and `int8_float16`, with `chunk_length` already corrected to 30.

**Model-fit probe done first — it removed a grid point and found a bigger lever than the
model choice.** Each configuration measured twice on a 90 s clip, in its own process (the
first attempt was contaminated by the OOM leak described in stage 3b):

| configuration | model VRAM | RTF run1 / run2 |
|---|---|---|
| large-v3-turbo / float16 (current default) | 2139 MiB | 3.91x / 3.90x |
| large-v3-turbo / **int8_float16** | **1115 MiB** | **13.33x / 14.16x** |
| large-v3 / **int8_float16** | 1883 MiB | **7.21x / 7.04x** |
| large-v3 / float16 | — | **impossible: OOM at load** |

- **`int8_float16` is 3.6x faster than `float16` on this GPU** and uses half the VRAM.
  `run1 ≈ run2` in every row, so this is not a warm-up artefact. The current `auto` → `float16`
  default (set in 0.9.2 on VRAM grounds, without measuring speed) costs a factor of 3.6.
  Likely cause: the GTX 1650 Ti is TU117, a Turing part **without tensor cores**, so fp16 gets
  no acceleration while int8 benefits from DP4A integer paths. That explanation is a
  hypothesis; the measurement is not.
- **`large-v3` fits, but only in `int8_float16`** — and at 7x it is still ~1.8x faster than the
  current turbo/float16 default. Confirmed end-to-end: the mic channel of a 31-minute
  recording took 183.8 s against turbo/float16's 309.3 s.
- **`large-v3` in `float16` is impossible**, retested with `beam_size=1` and 3674 MiB free —
  it OOMs on the weights themselves. This grid point is dropped; leaving it in would have
  spent hours silently transcribing on CPU.

Material — real files with known failures and a pre-written list of terms the model must get right:

- `2026-07-22_18-33-18.wav` — RAG (currently six different spellings), vector search, LLM as judge,
  Excalidraw, Notion, Jira
- `2026-06-15_02-52-59.wav` — ONNX Runtime (currently `on an X Runtime`), OpenVINO (`OpenSWIFO`),
  ML/CV pipeline, the number "75 fps"
- `2026-04-17_19-08-24.wav` — previous iterations ran on this file, so there is a baseline

Per-run metrics: time, RTF, peak VRAM, sm avg / max temp / % throttled, **how many listed terms were
recognised correctly**, presence of known hallucination strings, number of repeat loops.

**Two metrics the first hand comparison proved were missing.** Reading three versions of the same
recording showed large-v3 winning decisively on dimensions nothing automated was measuring:

- **Punctuation and capitalisation.** turbo emitted `я слышал что -то периодически попадаются в
  новостях то есть это такая вещь которую вроде бы сделали` — one unpunctuated run; large-v3 gave
  the same speech as `Я слышал что -то, знаешь, периодически попадаются в новостях, то есть это
  такая вещь, которую вроде бы сделали,` — punctuated, capitalised, and carrying a filler word
  turbo dropped entirely. Commas per 1000 words: OLD 117, turbo 165, large-v3 167.
- **Low-confidence word rate** — the italic spans the formatter already emits for
  `probability < 0.35`, i.e. Whisper's own uncertainty. Per 1000 words: OLD **124.5**, turbo
  **81.5**, large-v3 **75.1**. This is free to compute and is the single best available proxy for
  recognition quality.

Both belong in `_quality.py` before the grid's numbers are used to pick a default.

Each configuration is measured twice — without and with `hotwords` — to separate the model's
contribution from the glossary's. Output: a table in CHANGELOG/README and fixed defaults for
`whisper_model` and `compute_type`.

### Stage 3 — quality

- `whisper_model` and `compute_type` set from stage 3a results (`compute_type=auto` on cuda currently
  yields `float16`, `transcriber.py:38-39`; large-v3 would need `int8_float16`).
- New `TAPEBACK_HOTWORDS` setting (string) threaded into `_invoke_transcribe`. Default: the project
  glossary — RAG, ONNX, OpenVINO, LLM, Whisper, Obsidian, tapeback, Excalidraw, Jira, Notion,
  embeddings, vector search, and so on.
- `language`: stop detecting the language independently per channel. Either detect once on the more
  speech-heavy channel and reuse it, or raise `language_detection_segments` from its default of 1
  (`settings.py:70`) to 4.
- Hallucination filter: a list of known subtitle artefacts in `const.py`, stripped at segment level.
  Must start from a failing test built on real strings from the transcripts.
- `formatter.py`: bound merged block length with a new constant (`MAX_BLOCK_SECONDS`, ~30–60 s) in
  `_merge_consecutive_speakers`. Failing test: 31 minutes of single-speaker segments must produce
  more than one timecode.
- Drop the "Diarized Transcript" section when it matches the main one segment for segment.

### Stage 3b — VRAM is not released after a CUDA OOM fallback

Observed three times while probing models. When a model fails to load with
`CUDA failed with error out of memory` and `Transcriber` falls back to CPU, the GPU
allocation is not returned: free VRAM went 3674 MiB → **95 MiB** and stayed there for the
remainder of the process.

`free_gpu_memory()` calls `torch.cuda.empty_cache()`, but the memory is held by
**ctranslate2**, which has its own allocator and is unaffected by torch's cache. The failed
`WhisperModel` object is also still referenced by the exception's traceback frame at the
point the fallback runs.

Consequences in the real pipeline: after a transcription OOM the diarizer's
`get_free_vram_mib()` check sees almost nothing and silently drops to CPU as well, so one
OOM degrades the whole run rather than just the transcription stage. Anything else measured
in the same process afterwards is measured on a starved card — this invalidated one of the
model-probe grid points until it was re-run in a clean process.

Candidate fixes to evaluate: drop the reference before retrying (so the failed model is
collectable), and call ctranslate2's own release path if one exists, otherwise isolate model
loading so a failure cannot leave the allocation behind.

### Stage 3c — the thermal clamp is the remaining cause of runs that never finish

Promoted from "secondary factor". Stage 1b assumed a shorter run would outrun the clamp;
it does not. On this laptop the clamp is the dominant cost of any long recording.

**What it is.** Not ordinary temperature throttling: the embedded controller latches the
GPU to its 300 MHz floor and holds it there. A clamped card reports `SW Thermal Slowdown`
**at idle** (`clocks_event_reasons.active = 0x24`) where a healthy one reports only
`GpuIdle` (`0x1`), and the state is uncorrelated with the temperature at that moment —
observed latched at 63 °C and released at 76 °C.

**How persistent.** Measured: **451 s of idle** to release after moderate heating; after
13 consecutive transcriptions it had **not released after 900 s**. Only idle clears it;
finishing the work does not.

**What it costs.** 300 MHz against a 2100 MHz maximum. Measured end to end, the same
configuration on the same audio: **3519 s clamped against 140 s clear — 25x.**

**This is what the original 8203 s was.** Two multipliers stacked: `chunk_length=2` doing
15x redundant encoder work (fixed in stage 2) and a clamped card doing the rest. The
first is gone; the second is not.

**The decisive measurement — CPU beats a clamped GPU.** Same 180 s clip,
large-v3-turbo/int8:

| where | RTF |
|---|---|
| GPU, unclamped | ~14x |
| **CPU (i7-10750H, int8)** | **2.39x** |
| GPU, clamped | ~0.31x |

**CPU is roughly 7.7x faster than the clamped GPU**, and that CPU figure is pessimistic —
it was taken while a GPU transcription was competing for the same cores. So persisting on
a clamped card is strictly worse than moving to CPU: it turns a ~15-minute job into a
multi-hour one.

**Proposed behaviour** (needs a decision before implementing):

1. **Check for the clamp before transcribing.** If it is active, wait a short, bounded
   time for release; if it does not clear, transcribe on CPU and say so. Reuses
   `thermal_clamp_active()` / `wait_for_clamp_release()` from stage 1b, which currently
   only the benchmark uses.
2. **Pace between channels and stages** so a long recording stops saturating the chassis
   in the first place. backlog_profiler's Gate 4 measured 0 % throttled samples over
   76 minutes with pacing, against 93–96 % without.
3. External clock caps (`thermal_profile.sh on 1050`) remain the only thing that removes
   the clamp entirely, and remain a manual step: they need root, and tapeback stays an
   unprivileged CLI.

Both 1 and 2 need settings, and both change runtime behaviour materially rather than only
reporting — so they are worth agreeing on before they ship.

### Stage 4 — stop losing work — DONE

- `KeyboardInterrupt` inside the segment loop keeps what was decoded instead of discarding it;
  the note is tagged `partial` with a visible warning; the second channel is skipped rather than
  making the user interrupt twice; a further Ctrl+C still stops the process.
- `try/finally` around transcription so `free_gpu_memory()` runs on the exception path too.
- **Process isolation** (`TAPEBACK_ISOLATE_TRANSCRIPTION`): transcription runs in a child, so the
  unfixable ctranslate2 OOM leak dies with it. Verified on the configuration that used to strand
  the card: free VRAM 3674 → 95 MiB in-process, 3674 → **3674 MiB** isolated. Segments stream back
  as complete lines, so a killed worker still leaves its finished work with the parent.
- **Resume** (`TAPEBACK_RESUME_CACHE`): a completed channel is cached against the audio plus every
  output-affecting setting. Measured on a real clip, a repeat run went **85.8 s → 0.0 s**.

**Two limits worth stating.** Resuming part-way through a channel was rejected on evidence, not
effort: it needs `clip_timestamps`, which faster-whisper documents as ignoring `vad_filter`, and
VAD is half of why the silence hallucinations went away. And because the monitor channel runs
first and is the expensive one, an interrupt during *it* has nothing to reuse — the cache pays off
when the second channel is the one interrupted.

### Order

All stages ship as separate commits (~500-line diff limit): **1a → 1b → 1c → 2 → 3a → 3 → 4**.

Stage 1 comes first deliberately: without before/after numbers, stages 3a and 3 would again become
guesswork, exactly as `chunk_length` once was. Stage 1a has already justified this — it moved the
suspected bottleneck from the mic channel to the monitor channel.

---

## Verification

Reference file: `2026-04-17_19-08-24.wav` (13m37s) — previous iterations ran on it, so comparisons
exist. Long case: `2026-07-30_13-25-08.wav` (15m55s) — the run that never finished.

```bash
uv run tapeback process <file> --no-diarize --no-summarize
```

Record per run: stage timings, RTF, sm avg / max temp / % throttled, segment count, presence of
strings from the hallucination list, timecodes per minute.

Required before finishing (from CLAUDE.md): `uv run ruff check --fix`, `uv run ruff format`,
`uv run ty check`, `uv run pytest` (coverage >= 90 %), update README.md and CHANGELOG.md (latest tag
is `v0.9.7`, so a new `0.9.8` section dated today).

Tests: every bug fix (timecode collapse, hallucinations, duplicate Diarized section, segment loss on
Ctrl+C) starts with a failing test under `../../../tests/regressions`.
