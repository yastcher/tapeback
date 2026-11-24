# ADR-020 — Speakers are told apart by pyannote's voice embeddings

**Status:** accepted 2026-11-23 (`src/tapeback/speaker_merge.py`,
`src/tapeback/diarizer.py`, `src/tapeback/models.py`).

**Context.** pyannote splits the other side of a call into clusters, more than there are
people: besides each person's cluster come fragments, a few words in other conditions,
speech a lagging connection garbled, two people at once. tapeback joined clusters whose
long-term spectra were alike (the spectral merge, `TAPEBACK_SPECTRAL_MERGE_THRESHOLD`).
A call's codec makes every voice's spectrum alike: on recorded calls it joined people
whose voices were far apart, and a note showed one remote speaker where there were more
(`docs/spec/2026-10-06-diarization.md`).

**Decision.**
- pyannote's own speaker embeddings decide (`DiarizeOutput.speaker_embeddings` in 4.x;
  3.x is asked for them). `Diarizer.diarize` returns them with the turns.
- A cluster with at least a twentieth of the call's speech and a turn of ten seconds or
  more is a speaker. Two speakers are one person only when their voices reach cosine 0.8,
  and a person keeps the label of their largest part.
- A cluster with a twentieth of the speech whose voice is no one's, under 0.5 to every
  speaker before it (larger clusters first), is a speaker too: someone who only ever
  answered briefly, or anyone in a short recording.
- Any other cluster, a fragment, joins the speaker whose voice it is (cosine 0.5 or more,
  ahead of the runner-up by 0.15). Otherwise its turns are dropped, and `assign_speakers`
  gives its words to the nearest speaker left.
- The spectral merge and its setting are gone. The thresholds are constants of
  `speaker_merge.py`, set from recorded calls with a probe that scores the rule on
  pyannote's saved output against moments checked by ear.

**Why.**
- Measured on recorded calls: one voice in two calls had cosine 0.87–0.97, different
  voices 0.20–0.29, within a call or across calls. Spectra do not separate them.
- On recorded calls, short and long, every speaker had a turn of 11.7 s or more; the
  clusters whose share only a short call pushed past a twentieth were fragments, 7.4 s at
  most, and resembled their speaker at 0.56–0.72.
- In the 49-second recording of the e2e suite no one spoke ten seconds at a stretch: the
  two remote speakers' longest turns were 9.3 s and 6.2 s, their voices at cosine 0.33.
  Only a voice unlike everyone's tells such a speaker from a fragment.
- A fragment of garbled or overlapping speech resembles no one, and a turn's own
  embedding can be near zero when the connection lagged; the nearest speaker in time was
  right at every checked moment.
- The same embeddings are what recognising a speaker from one meeting to the next needs,
  so the hook for speaker profiles is the one the merge already uses.

**Consequences.**
- A real participant with under a twentieth of a call's speech and a voice unlike the
  others loses their words to the nearest speaker until profiles can name them.
- A cluster of garbled speech with a twentieth of a short call and a voice like no one's
  shows as a speaker of its own; none of the recorded calls had one.
- Without embeddings (a legacy pipeline that returns a bare Annotation) the clusters come
  out as pyannote gave them, fragments included.
- ADR-016 stands: one model for transcript and speakers remains the target; this
  decision is about what tapeback does with pyannote's clusters.
