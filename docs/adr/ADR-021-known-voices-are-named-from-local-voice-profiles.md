# ADR-021 — Known voices are named from local voice profiles

**Status:** accepted 2026-11-24 (`src/tapeback/voice_profiles.py`,
`src/tapeback/_speaker_name.py`, `src/tapeback/cli_voices.py`, `src/tapeback/diarizer.py`,
`src/tapeback/summarizer.py`).

**Context.** After ADR-020 a note tells the remote speakers apart, but calls them
"Speaker 1", "Speaker 2"; the reader puts the names back by ear, and the summary hands
its action items to labels. The voice embeddings that tell speakers apart also
recognise a person from one call to the next (`docs/spec/2026-10-06-diarization.md`).

**Decision.**
- A voice profile is a person's name and samples of their voice: pyannote's embeddings
  of that person in past calls, each kept with its session and label. Profiles live in
  `~/.local/share/tapeback/voices/profiles.json` (XDG data, as the resume cache and the
  run records): the directory 0700, the file 0600, written whole or not at all.
- After the voice merge, each speaker is compared with every person's samples, by the
  closest one. A speaker whose voice reaches cosine 0.8 to a person, ahead of every other
  person by 0.15, takes that person's name; any other speaker stays "Speaker N", counted
  among the speakers without a name. Two speakers may take one name.
- The file records the model whose embeddings it holds (`const.PYANNOTE_MODEL`); another
  model's file, or a malformed one, names no one and says so.
- `TAPEBACK_YOUR_NAME` puts the user's own name in place of "You" on the mic side, in the
  note, the live note and the summary's prompt.
- The run record keeps status lines verbatim and is meant to be shareable, so the match
  is reported by label and similarity, never by name.
- With `TAPEBACK_MASK_PII` on, every name tapeback wrote in as a speaker, the user's and
  the profiles', joins the masker's terms, and the prompt names the one who recorded by
  their placeholder.
- A name is refused that would read as tapeback's own label ("You", "Other",
  "Speaker N") or break the note's line or the masker's term list.
- `tapeback voices learn <note>` makes a profile from a note: each name the note gives a
  speaker, renamed there by hand or given as `Speaker 2=Anna` (which renames the note),
  takes the voice heard most in that name's lines in a new diarization of the note's
  recording. A name already known gains a sample from that call. `voices list` shows the
  people, `voices forget` deletes one.

**Why.**
- Measured on recorded calls: a profile built from a person's clusters in the other
  calls reached that person at cosine 0.88 or more by the samples' mean, 0.93 by the
  closest sample, and another person at 0.56 at most. A person with no profile got 0.56
  from the closest one and stays unnamed at 0.8. The closest sample widens the gap: a
  headset in one call and a phone in another are both the person.
- A wrong name is worse than "Speaker N": the summary would hand a commitment to the
  wrong person. Hence a threshold in the gap's middle and the margin over the next.
- No two speakers resembled one profile at 0.8 in the same call; when pyannote splits a
  person, both halves are still that person.
- XDG data, not `~/.config/tapeback`: ADR-014 keeps configuration to env and `.env`, the
  test isolation already redirects XDG_DATA_HOME, and a config directory often ends up in
  a dotfiles repository.
- Learning diarizes the recording again instead of keeping every call's voices: only the
  people the user names have their voice kept, and the vault already holds the recording.

**Consequences.**
- Learning a voice costs a second diarization of the call, about as long as the first.
- A mono recording, a remote STT backend's own speakers and the live note's monitor side
  get no names.
- The vectors compare only within one model: when ADR-016's single model replaces
  pyannote, profiles are rebuilt from the recordings their samples name.
- A voice profile is biometric data; whose voice is kept is the user's decision.
