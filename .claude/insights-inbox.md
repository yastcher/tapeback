## Friction (where the agent went wrong)

- 2026-10-16. The default LLM model ids had gone stale unnoticed: vendors had retired most of them or closed them to new keys, and no check can see that, since no test reaches a live vendor and the e2e suite runs with `--no-summarize`. A new default also changed the reply's shape: the current Claude model opens with a thinking block, which the transport read as the text. Touching the LLM path means reading each vendor's deprecation page and its reply format.
- 2026-10-09. The Arch stand was presented as covering AUR packaging, but it builds only the base package. The first real `yay` upgrade showed what it could not see: the extras' pip rewrote files pacman owns, and `tapeback-cuda` had never been published. A packaging stand should install what users install (base plus extras) and end with `pacman -Qk`.

## What worked (patterns worth keeping)

- 2026-10-05. Running the suite under CPU load found 2 flaky tests in 25 runs that a quiet machine never showed.
- 2026-10-30. Triage of 317 surviving mutants went to five read-only subagents, one per module group: they rebuilt each mutant from its diff and computed every expected value by running the original, with no pytest, no mutmut and no torch, so the five fit in memory together. The tests written from their reports left 24 survivors, all explained, and the reports found four defects the suite could not see. Their claims were still checked against the code before acting: two of three earlier guesses about `_mask` turned out wrong.

- 2026-10-06. A skipped e2e test counts as a failure: the mechanism lives in `scripts/pre_release_qa.sh`.
- 2026-10-06. Measuring final speaker counts on recordings with a known answer (four clips, two remote speakers each) located the defect in our own post-processing (a minor cluster bridging two speakers), not in pyannote, before any threshold was touched.
- 2026-10-15. "The raw transcript duplicates the diarized one" was measured before it was deleted: word counts of the two sections over real notes found a few words in a thousand only in the raw one. Replaying one recording once and formatting the same segments under the old and the new rule proved the fix (the loss went to zero) without a second transcription.
- 2026-10-15. Filters written for one stage's output change meaning when a later change cuts or merges segments before them: the 1 s VAD-artifact rule and the bleed check both ended up judging pieces, and the fix for one nearly made the silence check judge a whole 10-minute chunk. When a change alters what a filter sees, list every filter after it and ask what each was written for.
- 2026-10-16. A pytest plugin under `tmp/probes/` (`PYTHONPATH=tmp/probes uv run pytest -p off_x -k ...`) switches one protection off without touching the repo. The orphan reaping, the lock, the PID start-time check, the server's working directory and diarization's clamp policy each failed their own test once disabled, which is what "a protection is proven by switching it off" asks.

## Doubts (not sure this should become a rule)

- An exact speaker-count assertion on a one-minute clip is a noisy oracle: speech shares at that length swing between runs and models. Maybe e2e should assert on longer meetings, or allow a documented tolerance, once `bench_diarize.py` exists.
