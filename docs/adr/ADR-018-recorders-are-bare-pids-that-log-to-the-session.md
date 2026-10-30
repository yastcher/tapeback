# ADR-018 — Recorders are bare pids, and each logs to its session directory

**Status:** accepted 2026-10-28 (`recorder.py`: `_spawn_recorder`, `MONITOR_LOG`,
`MIC_LOG`).

**Context.** `tapeback start` launches two `parecord` processes and may not be the one
that stops them: `tapeback stop` in another terminal and the tray stop them by the pids
in `session.json`. They were started with `subprocess.Popen(stderr=PIPE)` and the Popen
objects dropped. Nothing read the pipe, and each dropped Popen warned that its child
was still running.

**Decision.**
- parecord starts through `os.posix_spawnp`: a bare pid, stopped and reaped by pid
  (`_reap`), with the signal defaults Popen would restore.
- Its stderr goes to `monitor.log` or `mic.log` in the private session directory, mode
  0600, truncated at start; stdout goes to `/dev/null`.

**Why.**
- Python has no way to let go of a Popen whose child should outlive it; the process
  is managed by pid already.
- An unread pipe blocks the writer after 64 KiB, and once the starter exits, the next
  message kills the writer with SIGPIPE.
- A file is readable by whichever process stops the recording, needs no reader alive,
  and lives exactly as long as the session: removed after processing, kept with a
  session that failed. `llama-server` logs the same way (ADR-012).

**Consequences.** A recorder that dies mid-meeting should surface its log's tail; today
the recording ends silently (BACKLOG §12).
