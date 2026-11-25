"""pyannote's diarization output, faked at the boundary without importing pyannote.

Importing pyannote.audio to build its own `DiarizeOutput` would load torch into every
test process that collects the module, so the fake mirrors the parts tapeback reads, and
no more: a MagicMock would answer any attribute and hide one tapeback must not touch.
"""

from types import SimpleNamespace

import numpy as np


class FakeAnnotation:
    """pyannote's Annotation as tapeback reads it: its tracks, and its labels sorted."""

    def __init__(self, tracks: list[tuple[float, float, str]]) -> None:
        self._tracks = [(SimpleNamespace(start=s, end=e), None, label) for s, e, label in tracks]

    def itertracks(self, yield_label: bool = False) -> list[tuple]:
        if yield_label:
            return list(self._tracks)
        return [(turn, track) for turn, track, _ in self._tracks]

    def labels(self) -> list[str]:
        return sorted({label for _, _, label in self._tracks})


def pyannote_output(
    tracks: list[tuple[float, float, str]], voices: dict[str, list[float]]
) -> SimpleNamespace:
    """What pyannote 4 returns: the turns, and one embedding per speaker in labels() order.

    A speaker missing from voices gets the zero row pyannote pads in for a speaker it
    has no centroid for.
    """
    annotation = FakeAnnotation(tracks)
    width = len(next(iter(voices.values())))
    rows = [voices.get(label, [0.0] * width) for label in annotation.labels()]
    return SimpleNamespace(
        speaker_diarization=annotation,
        exclusive_speaker_diarization=annotation,
        speaker_embeddings=np.array(rows, dtype=float),
    )
