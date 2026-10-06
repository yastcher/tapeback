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
channel memory. See CHANGELOG 0.9.8.

---

## 0. `thermal_clamp_wait=60` is the worst of both options

**Observed 2026-08-06**, transcribing a 31-minute recording while a video played in the
browser. The monitor channel finished unclamped in 322 s. The clamp then latched, the
mic channel's worker checked it, and sat waiting — 0.7 % CPU, 14 MB RSS, nothing on the
GPU — for the full configured timeout before giving up and using the CPU.

**The wait cannot succeed while the user is doing something else.** The clamp clears on
*system* idle, and the browser held the CPU package at 90–94 °C throughout. Waiting for
a condition that depends on the user's own workload is not a strategy.

**The numbers make the default indefensible.** The shortest measured release is 451 s of
genuine idle. A 60 s wait therefore essentially never succeeds — it is 60 s of doing
nothing, then the CPU fallback that could have started immediately. Meanwhile the CPU
runs at ~2.39x real time against a clamped GPU's ~0.31x.

So the default should be one of:

- **0** — never wait, fall back to CPU the moment a clamp is seen. Costs the case where
  the clamp really was about to clear.
- **~500+** — actually long enough to succeed, and only worth it when the machine will
  genuinely be idle.

60 buys neither. Deciding between them wants one more measurement: how often a clamp
seen at the *start* of a run clears within a few minutes on an otherwise idle machine.

**A better policy than either** is to stop treating this as a waiting problem. We know
both throughputs — CPU ~2.39x, clamped GPU ~0.31x — so the decision is arithmetic, not a
timeout: if clamped, use the CPU, and re-check the GPU only between channels where the
switch is free. That needs the worker to report throughput back, which it now can.

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

### Recommendation: do not switch the default

The saving is real and large — 39 s and 9 °C on every recording. But the only quality
difference found lands on readability of the quiet channel, which this project has
repeatedly under-valued and then regretted. Trading it for 39 seconds is the same deal
that was reverted twice before.

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
