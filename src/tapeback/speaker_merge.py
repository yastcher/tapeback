"""Speakers told apart by their voices, as pyannote embeds them.

pyannote splits a call into clusters and gives each one's voice an embedding. A cluster
with a fair share of the speech and a turn of some length is a speaker, and two speakers
merge only when their voices are one person's. So is a cluster with a fair share whose
voice is no one else's: someone who only ever answered briefly, or anyone in a short
recording. Any other is a fragment: a few words of a speaker in other conditions, speech
a lagging connection garbled, two people at once. It joins the speaker whose voice it
clearly is, or its turns go, and `assign_speakers` gives its words to the nearest speaker
left.

Measured on recorded calls (docs/spec/2026-10-06-diarization.md): one voice in two calls
had cosine 0.87-0.97, different voices 0.20-0.29. The spectral merge this replaces
compared long-term spectra, which a call's codec makes alike for everyone, and joined
voices that far apart.
"""

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from tapeback.models import Diarization, DiarizationSegment

# A cluster with less of the call's speech than this is a fragment, not a speaker...
MINOR_SPEECH_SHARE = 0.05
# ...and one that never spoke this long at a stretch is a speaker only by a voice like
# no one's: on recorded calls every speaker had a turn of 11.7 s or more, the fragments
# 7.4 s at most, however short the call made their share.
MIN_LONGEST_TURN_SECONDS = 10.0
# A fragment joins the speaker whose voice it resembles this much or more; under it for
# every speaker, its voice is no one's...
JOIN_SIMILARITY = 0.5
# ...and joins only when no other speaker comes within this of it: a fragment of two
# voices at once resembles both and belongs to neither.
JOIN_MARGIN = 0.15
# Two speakers whose voices are this alike are one person pyannote split in two.
SAME_VOICE_SIMILARITY = 0.8


def merge_speakers_by_voice(diarization: Diarization) -> list[DiarizationSegment]:
    """The segments relabeled: one person split in two made one, each other cluster joined
    to its speaker, kept as a speaker of its own, or dropped. Without the voice of any
    cluster with a fair share, the segments as they came."""
    voices = {speaker: _unit(vector) for speaker, vector in diarization.voices.items()}
    speech = _speech(diarization.segments)
    longest = _longest_turn(diarization.segments)
    total = sum(speech.values())
    by_size = sorted(speech, key=lambda s: (-speech[s], s))
    fair = {s for s in by_size if speech[s] >= MINOR_SPEECH_SHARE * total}
    if not any(s in voices for s in fair):
        return diarization.segments
    speakers = [s for s in by_size if s in fair and longest[s] >= MIN_LONGEST_TURN_SECONDS]
    label = _merge_split_speakers(speakers, voices)
    for cluster in by_size:
        if cluster not in label:
            label[cluster] = _owner(cluster, speakers, voices, label, fair)
            if label[cluster] == cluster:
                speakers.append(cluster)
    return [
        DiarizationSegment(speaker=owner, start=seg.start, end=seg.end)
        for seg in diarization.segments
        if (owner := label[seg.speaker]) is not None
    ]


def _merge_split_speakers(
    speakers: list[str], voices: dict[str, np.ndarray]
) -> dict[str, str | None]:
    """Each speaker's label: a larger speaker's whose voice it shares, else its own.

    speakers come largest first, so a person keeps the label of their larger part. A
    speaker without a voice keeps its own: there is nothing to compare.
    """
    label: dict[str, str | None] = {}
    for speaker in speakers:
        twin = next(
            (
                other
                for other in speakers
                if label.get(other) == other
                and speaker in voices
                and other in voices
                and float(voices[speaker] @ voices[other]) >= SAME_VOICE_SIMILARITY
            ),
            None,
        )
        label[speaker] = twin or speaker
    return label


def _owner(
    cluster: str,
    speakers: list[str],
    voices: dict[str, np.ndarray],
    label: dict[str, str | None],
    fair: set[str],
) -> str | None:
    """The speaker whose voice a cluster clearly is. A voice that is no one's, or none to
    compare, makes a cluster with a fair share a speaker of its own: the cluster itself.
    None for a smaller one, and for one like two speakers at once."""
    closest: dict[str, float] = {}
    for speaker in speakers:
        person = label[speaker]
        if cluster in voices and speaker in voices and person is not None:
            similarity = float(voices[cluster] @ voices[speaker])
            closest[person] = max(similarity, closest.get(person, similarity))
    owner = max(closest, key=closest.__getitem__, default=None)
    if owner is None or closest[owner] < JOIN_SIMILARITY:
        return cluster if cluster in fair else None
    others = [similarity for person, similarity in closest.items() if person != owner]
    if others and closest[owner] - max(others) < JOIN_MARGIN:
        return None
    return owner


def _speech(segments: Iterable[DiarizationSegment]) -> dict[str, float]:
    speech: dict[str, float] = defaultdict(float)
    for seg in segments:
        speech[seg.speaker] += seg.end - seg.start
    return dict(speech)


def _longest_turn(segments: Iterable[DiarizationSegment]) -> dict[str, float]:
    longest: dict[str, float] = defaultdict(float)
    for seg in segments:
        longest[seg.speaker] = max(longest[seg.speaker], seg.end - seg.start)
    return dict(longest)


def _unit(vector: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(vector))
    norm = float(np.linalg.norm(array))
    return array / norm if norm else array
