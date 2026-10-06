## Friction (where the agent went wrong)

- 2026-10-06. Probe scripts built `Settings()` and silently read the developer's `.env` (a local clustering threshold of 0.85 skewed a whole sweep). Ad-hoc probes outside pytest get no isolation fixture: construct `Settings(_env_file=None)` and pass every knob explicitly.
- 2026-10-06. A CLI flag was assumed from current docs (`uv lock --check`) and broke on the pinned uv 0.4.30. Verify a tool's flag against the installed version (`--help`, one dry run) before wiring it into a gate script.

## What worked (patterns worth keeping)

- 2026-10-06. Counting a skipped e2e test as a failure in `pre_release_qa.sh` exposed a diarization regression that had been red, unseen, since pyannote 4 landed in v0.9.6.
- 2026-10-06. Measuring final speaker counts on recordings with a known answer (four clips, two remote speakers each) located the defect in our own post-processing (a minor cluster bridging two speakers), not in pyannote, before any threshold was touched.
- 2026-10-05. Switching a protection off and rerunning its flow test showed one test green for the wrong reason (ffmpeg's Popen reaped the zombies); running the suite under CPU load found 2 flaky tests in 25 runs.

## Doubts (not sure this should become a rule)

- An exact speaker-count assertion on a one-minute clip is a noisy oracle: speech shares at that length swing between runs and models. Maybe e2e should assert on longer meetings, or allow a documented tolerance, once `bench_diarize.py` exists.
