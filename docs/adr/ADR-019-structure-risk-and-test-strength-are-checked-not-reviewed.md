# ADR-019 — Structure, risk and test strength are checked, not reviewed

**Status:** accepted 2026-10-30 (`tests/test_architecture.py`, `scripts/crap.py`,
`scripts/mutation_ratchet.py`, `[tool.mutmut]` in `pyproject.toml`,
`scripts/pre_release_qa.sh`).

**Context.** Line coverage says which lines ran, not whether a test would notice them
change. The boundaries AGENTS.md draws (the masking seam, lazy extras, where models
live, the line limit) were held only by review, and a complex, thinly tested function
showed up nowhere.

**Decision.**
- **Architecture rules are tests.** Each rule of AGENTS.md that code can break is a
  function over the repository in `tests/test_architecture.py`, run in the normal suite:
  over `ast`, plus one fresh interpreter that imports the CLI. Each rule also runs on a
  small tree built to break it and must find exactly that break.
- **CRAP in pre-release QA.** CRAP = cc² × (1 − coverage)³ + cc per function, from the
  gate's coverage and McCabe's complexity counted over `ast` (`scripts/crap.py`). The
  limit sat just above the worst function when it was set; it comes down as that
  function improves and is never raised to let a change in.
- **Mutation testing in pre-release QA.** mutmut over a slice of modules whose behaviour
  unit tests pin, from scratch every time. The counts of surviving and untested mutants
  may only go down (`scripts/mutation_ratchet.py`); the slice widens a module at a time.

**Why.**
- The rules are per-module facts (where a class is defined, which module imports an
  SDK, how long a file is). import-linter answers import-graph questions, adds a
  dependency, and cannot tell an import inside a function from one at the top.
- radon, the usual complexity tool, last shipped in 2023 with no Python 3.14 support
  stated; McCabe's count is a short walk over `ast`.
- Coverage the gate collects is the coverage QA has (the e2e suite runs without it), but
  a CRAP failure should not block an urgent fix, so the check waits for a release.
- A mutant that survives is a change no test notices: the switch-off proof AGENTS.md
  asks for, made by a tool over every function of the slice.

**Consequences.**
- mutmut runs pytest in its own copy of the tree, `mutants/`: tests that read files
  outside `tests/` need them in `also_copy`, coverage is off, and the architecture tests
  are left out (mutmut rewrites the sources they read).
- Workers start from a forkserver, two at a time, inside a memory cap
  (`systemd-run --user --scope`, no swap, `OOMPolicy=continue`). A worker per core, each
  forked from the process whose first run had loaded torch, ran an 8 GB laptop out of
  memory, and the kernel killed the IDE. The fixtures never import a heavy package to
  ask whether it is installed (`tests/regressions/test_fixtures_leave_torch_alone.py`).
- A mutant can make code allocate without end: one that dropped the step of
  `speaker_merge`'s frame loop passed 2 GB in seconds on a minute of real audio. The cap
  kills that one worker, mutmut records a segfault, and the ratchet counts the mutant as
  died: its tests did not pass. The ratchet also fails a run that left a mutant
  unchecked, which a stopped run does, with fewer survivors than a whole one.
- A survivor is a test gap, code that does nothing (simplified away), a question for the
  maintainer (BACKLOG), or an equivalent mutant. The first triage, from 317 survivors,
  found six defects: four bugs, and two behaviours the maintainer ruled on. The
  equivalents stay counted under the maximum: `astype(None)` is float64 in numpy,
  `wave.open` reads by default, `reshape(-2, 2)` equals `reshape(-1, 2)`, a bound whose
  other value changes nothing (an empty slice, a read past the data, a clamp that binds
  only below 20 Hz, a loop with no full phrase left, the shortcut for one speaker), the
  separator of a resume key, and a threshold no signal can be built to hit exactly.
