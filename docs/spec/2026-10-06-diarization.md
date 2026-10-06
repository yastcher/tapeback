# Diarization: what the 0.9.9 pre-release QA found

Status: findings and a plan for the next release. 0.9.9 only fixes the plain defects
(see "Done in 0.9.9"); the rest waits for pluggable diarization models.

## How it surfaced

`scripts/pre_release_qa.sh` counts a skipped e2e test as a failure. Run with a token
for the first time, `test_stereo_pipeline_with_diarization` failed: one narrator on
the monitor channel came out as two speakers. The same failure reproduces on the
v0.9.8 code in the same environment, so it is not a 0.9.9 regression.
pyannote-audio moved to 4.0.4 in v0.9.6 (2026-06-17) while the test dates from
2026-04-20 — most likely it has failed since June, unseen, because e2e only ever ran
by hand.

## Quality

Measured on four recordings of 49–78 s, each with two remote speakers, monitor
channel only, on the GPU. `3:18/7/1` reads "three speakers with 18, 7 and 1 seconds
of speech"; the arrow is tapeback's spectral merge (`TAPEBACK_SPECTRAL_MERGE_THRESHOLD`,
0.96) applied after pyannote.

| config | 18-41 | 23-15 | 17-31 | 22-01 |
|---|---|---|---|---|
| 3.1, threshold 0.705 (model default) | 3:18/7/1 → 1 | 4:9/8/8/3 → 3 | 3:16/8/3 → 3 | 3:16/6/1 → 1 |
| 3.1, threshold 0.85 (tapeback default) | 3:18/7/1 → 1 | 3:14/9/5 → 1 | 3:16/8/3 → 3 | 3:16/6/1 → 1 |
| 3.1, threshold 0.95 | 3:18/7/1 → 1 | 3:14/9/5 → 1 | 3:16/8/3 → 3 | 2:16/7 → 2 |
| community-1, threshold 0.4–0.8 | 3:18/7/1 → 3 | 3:14/9/5 → 1 | 3:17/7/4 → 3 | 3:17/6/1 → 3 |

What it says:

1. **The spectral merge joins different people.** With the shipped settings it turns
   two real speakers into one in three recordings of four — the reason past notes
   show a single "Speaker 1" where two people talked. Its own docstring calls power
   spectrum a weak signal for voice identity; the channel response dominates it.
2. **pyannote finds the two real speakers plus a small third cluster** (1–5 s: echo,
   cross-talk) with either model. The minor-speaker rule meant to absorb such clusters
   only fires through the same spectral gate.
3. **community-1 is no better here**, and its clustering threshold changes nothing
   between 0.4 and 0.8.
4. A removed single-narrator clip showed the opposite error: 3 speakers at the model
   default, 2 at 0.85. No threshold fixes both directions on clips this short.

Caveat: four one-minute clips. Speech shares are noisy at this length, which is why
nothing here is tuned to them.

## Speed

- In 10 of 12 real runs (September–October) diarization took 0.86–1.18x the
  recording's length — 7387 s for a 120-minute meeting.
- On a cool GPU a 47 s clip diarizes in 7.5 s including model load; on the CPU, 31.6 s.
- Caught live during this investigation: pyannote at 100% GPU utilisation with the
  SM clock pinned at 300 MHz and the card at 71 °C, throttle reasons `0x24`
  (`SwPowerCap` + `SwThermalSlowdown`). The same sweep step took 17 s at first and
  220–242 s later in the run.

So diarization runs into the thermal clamp described in BACKLOG ("Why runs used to
stall with no way out"), and nothing in it reacts: transcription checks for the clamp
before each stage and falls back to the CPU; diarization has no check, no GPU
telemetry line, and its own fallback warnings go to `print()`, so they never reach
the run record.

## Plan for the next release

1. **Pluggable diarization models**, chosen from a table rather than by reasoning —
   the same rule `scripts/bench_transcribe.py` enforces for Whisper.
2. **`scripts/bench_diarize.py`**: drives the real `Diarizer` and post-processing over a
   manifest of recordings with known speaker counts, including several meetings of
   20–60 minutes; reports speakers found, error, wall time and GPU clocks.
3. **Minor clusters absorbed by speech share**, not by spectral similarity — decided
   from the bench, not from the four clips above.
4. **Clamp-aware diarization**: measure a clamped GPU against the CPU on a long file,
   then pick the policy transcription already has (check before the stage, fall back
   when the alternative wins).
5. **Observability**: diarization device and a GPU telemetry line for the stage in the
   run record; Diarizer warnings through the status callback.

## Done in 0.9.9

Only defects, nothing tuned:

- **A minor cluster no longer bridges two real speakers.** Absorption compared a short
  cluster with every speaker at the relaxed 0.92 and merged pair by pair into one
  group, so a 1 s cluster resembling both people joined them — although the two were
  never close enough (about 0.92 against the 0.96 standard threshold) to merge on
  their own. A minor cluster now joins the one speaker it resembles most. With
  default settings the four recordings went from 1, 3, 3, 1 remote speakers to
  2, 3, 3, 2 against a truth of 2, 2, 2, 2 — no recording worse, the threshold
  untouched. The remaining over-count is the small third cluster pyannote produces,
  left for the next release.
- The e2e test lost its recording and now uses `2026-04-02_22-01-53` (two remote
  speakers), asserting exactly two. It reads the token from `HF_TOKEN` only:
  `TAPEBACK_HF_TOKEN` never reached it under the settings isolation.
