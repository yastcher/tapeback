## Friction (where the agent went wrong)

- 2026-10-16. The default LLM model ids had gone stale unnoticed: vendors had retired most of them or closed them to new keys, and no check can see that, since no test reaches a live vendor and the e2e suite runs with `--no-summarize`. A new default also changed the reply's shape: the current Claude model opens with a thinking block, which the transport read as the text. Touching the LLM path means reading each vendor's deprecation page and its reply format.
- 2026-10-15. A probe that called the pipeline directly ran Whisper on the CPU: the repo venv has no `nvidia-cublas-cu12`, so `preload_cuda_libs` does nothing and ctranslate2 falls back with one warning line. Run GPU probes with `/opt/tapeback/bin/python` and `PYTHONPATH=src`, and read the "Whisper: ... on cuda" line before waiting on the result.
- 2026-10-15. The weekly insights review proposed harness edits that did not survive a check against HEAD. Its deny patterns `*.env*` also blocked the tracked `.env.example` (deny beats allow, so only a narrower pattern such as `*.env` works); it called the local uv 0.4.30 "pinned"; its line numbers were off by one or two; it deleted two inbox entries as "already rules" that were not (Arch stand: base plus extras and `pacman -Qk`; stress under CPU load); it read the 43 pytest warnings as noise, though they are one dbus-next deprecation that kills the tray on Python 3.15; and it missed that its own new probes rule is already broken by `scripts/bench_transcribe.py`. A review's proposals are hypotheses: verify each against HEAD before applying it.
- 2026-10-09. The Arch stand was presented as covering AUR packaging, but it builds only the base package. The first real `yay` upgrade showed what it could not see: the extras' pip rewrote files pacman owns, and `tapeback-cuda` had never been published. A packaging stand should install what users install (base plus extras) and end with `pacman -Qk`.

## What worked (patterns worth keeping)

- 2026-10-05. Running the suite under CPU load found 2 flaky tests in 25 runs that a quiet machine never showed.

- 2026-10-06. A skipped e2e test counts as a failure: the mechanism lives in `scripts/pre_release_qa.sh`.
- 2026-10-06. Measuring final speaker counts on recordings with a known answer (four clips, two remote speakers each) located the defect in our own post-processing (a minor cluster bridging two speakers), not in pyannote, before any threshold was touched.
- 2026-10-15. "The raw transcript duplicates the diarized one" was measured before it was deleted: word counts of the two sections over real notes found a few words in a thousand only in the raw one. Replaying one recording once and formatting the same segments under the old and the new rule proved the fix (the loss went to zero) without a second transcription.
- 2026-10-15. Filters written for one stage's output change meaning when a later change cuts or merges segments before them: the 1 s VAD-artifact rule and the bleed check both ended up judging pieces, and the fix for one nearly made the silence check judge a whole 10-minute chunk. When a change alters what a filter sees, list every filter after it and ask what each was written for.
- 2026-10-16. A fake SDK reply built from the SDK's own types (`anthropic.types.TextBlock`, `ThinkingBlock`) instead of `MagicMock(text=...)`: a MagicMock has every attribute, so it could never show that a reply's first block has no `.text`.

## Doubts (not sure this should become a rule)

- An exact speaker-count assertion on a one-minute clip is a noisy oracle: speech shares at that length swing between runs and models. Maybe e2e should assert on longer meetings, or allow a documented tolerance, once `bench_diarize.py` exists.
