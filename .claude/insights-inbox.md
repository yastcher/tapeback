## Friction (where the agent went wrong)

- 2026-10-15. The weekly insights review proposed harness edits that did not survive a check against HEAD. Its deny patterns `*.env*` also blocked the tracked `.env.example` (deny beats allow, so only a narrower pattern such as `*.env` works); it called the local uv 0.4.30 "pinned"; its line numbers were off by one or two; it deleted two inbox entries as "already rules" that were not (Arch stand: base plus extras and `pacman -Qk`; stress under CPU load); it read the 43 pytest warnings as noise, though they are one dbus-next deprecation that kills the tray on Python 3.15; and it missed that its own new probes rule is already broken by `scripts/bench_transcribe.py`. A review's proposals are hypotheses: verify each against HEAD before applying it.
- 2026-10-09. The Arch stand was presented as covering AUR packaging, but it builds only the base package. The first real `yay` upgrade showed what it could not see: the extras' pip rewrote files pacman owns, and `tapeback-cuda` had never been published. A packaging stand should install what users install (base plus extras) and end with `pacman -Qk`.

## What worked (patterns worth keeping)

- 2026-10-05. Running the suite under CPU load found 2 flaky tests in 25 runs that a quiet machine never showed.

- 2026-10-06. A skipped e2e test counts as a failure: the mechanism lives in `scripts/pre_release_qa.sh`.
- 2026-10-06. Measuring final speaker counts on recordings with a known answer (four clips, two remote speakers each) located the defect in our own post-processing (a minor cluster bridging two speakers), not in pyannote, before any threshold was touched.

## Doubts (not sure this should become a rule)

- An exact speaker-count assertion on a one-minute clip is a noisy oracle: speech shares at that length swing between runs and models. Maybe e2e should assert on longer meetings, or allow a documented tolerance, once `bench_diarize.py` exists.
