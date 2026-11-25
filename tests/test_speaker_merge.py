"""The speakers a call's clusters make, decided by pyannote's voice embeddings.

Voices are a few dimensions wide here; the merge compares directions only. The cosines
are chosen like those measured on recorded calls: different people near 0.2, a fragment
near 0.6 to its speaker, one person split in two near 0.95.
"""

from collections.abc import Mapping

import numpy as np
import pytest

from tapeback import speaker_merge
from tapeback.models import Diarization, DiarizationSegment
from tapeback.speaker_merge import merge_speakers_by_voice

A = (1.0, 0.0, 0.0)
B = (0.25, 0.968, 0.0)  # 0.25 to A: a different person


def _merge(turns: list[tuple[str, float, float]], voices: Mapping[str, tuple[float, ...]]):
    segments = [DiarizationSegment(speaker=s, start=a, end=b) for s, a, b in turns]
    merged = merge_speakers_by_voice(Diarization(segments=segments, voices=voices))
    return [(seg.speaker, seg.start) for seg in merged]


def _cosine(u: tuple[float, ...], v: tuple[float, ...]) -> float:
    """Computed as the merge computes it, each side made unit length first, so a
    threshold set to it sits exactly on the merge's own number."""
    a, b = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    return float((a / np.linalg.norm(a)) @ (b / np.linalg.norm(b)))


def test_two_voices_stay_two_speakers():
    turns = [("SPEAKER_00", 0.0, 10.0), ("SPEAKER_01", 10.0, 20.0)]

    assert _merge(turns, {"SPEAKER_00": A, "SPEAKER_01": B}) == [
        ("SPEAKER_00", 0.0),
        ("SPEAKER_01", 10.0),
    ]


def test_one_person_split_in_two_keeps_the_larger_parts_label():
    turns = [("SPEAKER_00", 0.0, 12.0), ("SPEAKER_01", 12.0, 40.0)]
    voices = {"SPEAKER_00": (0.95, 0.312, 0.0), "SPEAKER_01": A}

    assert _merge(turns, voices) == [("SPEAKER_01", 0.0), ("SPEAKER_01", 12.0)]


def test_a_relabeled_turn_keeps_its_times():
    segments = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=12.0),
        DiarizationSegment(speaker="SPEAKER_01", start=12.0, end=40.0),
    ]
    voices = {"SPEAKER_00": (0.95, 0.312, 0.0), "SPEAKER_01": A}

    assert merge_speakers_by_voice(Diarization(segments=segments, voices=voices)) == [
        DiarizationSegment(speaker="SPEAKER_01", start=0.0, end=12.0),
        DiarizationSegment(speaker="SPEAKER_01", start=12.0, end=40.0),
    ]


def test_a_fragment_joins_the_speaker_whose_voice_it_is():
    turns = [("SPEAKER_00", 0.0, 50.0), ("SPEAKER_01", 50.0, 100.0), ("SPEAKER_02", 100.0, 101.0)]
    fragment = (0.6, 0.0, 0.8)  # 0.6 to A, 0.15 to B

    assert _merge(turns, {"SPEAKER_00": A, "SPEAKER_01": B, "SPEAKER_02": fragment}) == [
        ("SPEAKER_00", 0.0),
        ("SPEAKER_01", 50.0),
        ("SPEAKER_00", 100.0),
    ]


@pytest.mark.parametrize(
    "fragment",
    [
        pytest.param((0.6, 0.55, 0.581), id="two voices at once: close to both"),
        pytest.param((0.2, 0.05, 0.978), id="no one's voice"),
    ],
)
def test_a_fragment_that_is_not_clearly_one_speakers_goes(fragment):
    """Its words fall to the nearest speaker left, in assign_speakers."""
    turns = [("SPEAKER_00", 0.0, 50.0), ("SPEAKER_01", 50.0, 100.0), ("SPEAKER_02", 100.0, 101.0)]

    assert _merge(turns, {"SPEAKER_00": A, "SPEAKER_01": B, "SPEAKER_02": fragment}) == [
        ("SPEAKER_00", 0.0),
        ("SPEAKER_01", 50.0),
    ]


def test_a_fragment_joins_a_person_by_the_closer_of_their_parts():
    """One person split in two: a fragment like the larger part is theirs, however unlike
    the smaller."""
    turns = [("SPEAKER_00", 0.0, 50.0), ("SPEAKER_01", 50.0, 80.0), ("SPEAKER_02", 80.0, 81.0)]
    part = (0.9, 0.436, 0.0)  # 0.9 to A: the same person
    fragment = (0.6, -0.4, 0.693)  # 0.6 to A, 0.37 to the part

    assert _merge(turns, {"SPEAKER_00": A, "SPEAKER_01": part, "SPEAKER_02": fragment}) == [
        ("SPEAKER_00", 0.0),
        ("SPEAKER_00", 50.0),
        ("SPEAKER_00", 80.0),
    ]


def test_voices_are_compared_by_direction_not_length():
    """pyannote's embeddings are not unit length: two people stay two however long their
    vectors, and a fragment joins by the angle alone."""
    turns = [("SPEAKER_00", 0.0, 50.0), ("SPEAKER_01", 50.0, 100.0), ("SPEAKER_02", 100.0, 101.0)]
    voices = {
        "SPEAKER_00": (30.0, 0.0, 0.0),  # A
        "SPEAKER_01": (7.5, 29.04, 0.0),  # B
        "SPEAKER_02": (0.06, 0.0, 0.08),  # 0.6 to A, 0.15 to B
    }

    assert _merge(turns, voices) == [
        ("SPEAKER_00", 0.0),
        ("SPEAKER_01", 50.0),
        ("SPEAKER_00", 100.0),
    ]


def test_a_voice_of_zeros_is_like_no_one():
    turns = [("SPEAKER_00", 0.0, 50.0), ("SPEAKER_01", 50.0, 100.0), ("SPEAKER_02", 100.0, 101.0)]

    assert _merge(turns, {"SPEAKER_00": A, "SPEAKER_01": B, "SPEAKER_02": (0.0, 0.0, 0.0)}) == [
        ("SPEAKER_00", 0.0),
        ("SPEAKER_01", 50.0),
    ]


def test_a_fragment_without_a_voice_goes():
    turns = [("SPEAKER_00", 0.0, 50.0), ("SPEAKER_01", 50.0, 100.0), ("SPEAKER_02", 100.0, 101.0)]

    assert _merge(turns, {"SPEAKER_00": A, "SPEAKER_01": B}) == [
        ("SPEAKER_00", 0.0),
        ("SPEAKER_01", 50.0),
    ]


def test_a_speaker_without_a_voice_keeps_its_label():
    turns = [("SPEAKER_00", 0.0, 50.0), ("SPEAKER_01", 50.0, 100.0), ("SPEAKER_02", 100.0, 101.0)]

    assert _merge(turns, {"SPEAKER_00": A, "SPEAKER_02": A}) == [
        ("SPEAKER_00", 0.0),
        ("SPEAKER_01", 50.0),
        ("SPEAKER_00", 100.0),
    ]


@pytest.mark.parametrize(
    "voices",
    [pytest.param({}, id="no voices"), pytest.param({"SPEAKER_02": A}, id="only a fragment's")],
)
def test_without_a_speakers_voice_the_turns_come_back_as_they_were(voices):
    turns = [("SPEAKER_00", 0.0, 50.0), ("SPEAKER_01", 50.0, 100.0), ("SPEAKER_02", 100.0, 101.0)]

    assert _merge(turns, voices) == [
        ("SPEAKER_00", 0.0),
        ("SPEAKER_01", 50.0),
        ("SPEAKER_02", 100.0),
    ]


def test_no_turns_no_speakers():
    assert _merge([], {"SPEAKER_00": A}) == []


@pytest.mark.parametrize("pieces", [1, 2], ids=["at a stretch", "in short answers"])
@pytest.mark.parametrize(
    ("other", "kept"),
    [
        pytest.param(190.0, True, id="exactly a twentieth: a speaker"),
        pytest.param(191.0, False, id="under it: it goes"),
    ],
)
def test_a_voice_like_no_ones_is_a_speaker_from_a_twentieth_of_the_speech(pieces, other, kept):
    """Ten seconds of a voice unlike anyone's, said at once or by someone who only ever
    answered briefly."""
    piece = 10.0 / pieces
    answers = [("SPEAKER_01", other + i * piece, other + (i + 1) * piece) for i in range(pieces)]
    voices = {"SPEAKER_00": A, "SPEAKER_01": (0.0, 0.0, 1.0)}

    assert (("SPEAKER_01", other) in _merge([("SPEAKER_00", 0.0, other), *answers], voices)) is kept


@pytest.mark.parametrize(
    ("turn", "kept"),
    [
        pytest.param(10.0, True, id="ten seconds at a stretch: a speaker"),
        pytest.param(9.99, False, id="never that long: a fragment"),
    ],
)
def test_a_speaker_has_spoken_ten_seconds_at_a_stretch(turn, kept):
    """A fragment joins a voice it resembles where two speakers stay apart, however large
    its share: garbled speech can hold a tenth of a short call."""
    turns = [
        ("SPEAKER_00", 0.0, 60.0),
        ("SPEAKER_01", 60.0, 60.0 + turn),
        ("SPEAKER_01", 80.0, 80.0 + turn),
    ]
    voices = {"SPEAKER_00": A, "SPEAKER_01": (0.6, 0.8, 0.0)}  # 0.6 to A

    assert (("SPEAKER_01", 60.0) in _merge(turns, voices)) is kept


def test_in_a_short_recording_the_voices_still_decide():
    """No one spoke ten seconds at a stretch: two people and a fragment of the second, as
    the e2e suite's 49-second recording comes out of pyannote."""
    turns = [
        ("SPEAKER_00", 0.0, 9.0),
        ("SPEAKER_01", 9.0, 15.0),
        ("SPEAKER_00", 15.0, 22.0),
        ("SPEAKER_02", 22.0, 23.0),
    ]
    fragment = (0.3, 0.542, 0.785)  # 0.6 to B, 0.3 to A

    assert _merge(turns, {"SPEAKER_00": A, "SPEAKER_01": B, "SPEAKER_02": fragment}) == [
        ("SPEAKER_00", 0.0),
        ("SPEAKER_01", 9.0),
        ("SPEAKER_00", 15.0),
        ("SPEAKER_01", 22.0),
    ]


def test_a_cluster_like_two_speakers_at_once_goes_however_much_it_holds():
    """A tenth of the call, never ten seconds at a stretch: two people over each other."""
    turns = [
        ("SPEAKER_00", 0.0, 50.0),
        ("SPEAKER_01", 50.0, 100.0),
        ("SPEAKER_02", 100.0, 106.0),
        ("SPEAKER_02", 106.0, 112.0),
    ]
    two_at_once = (0.6, 0.55, 0.581)  # 0.6 to A, 0.68 to B

    assert _merge(turns, {"SPEAKER_00": A, "SPEAKER_01": B, "SPEAKER_02": two_at_once}) == [
        ("SPEAKER_00", 0.0),
        ("SPEAKER_01", 50.0),
    ]


@pytest.mark.parametrize(("nudge", "merged"), [(0, True), (1, False)], ids=["at", "above"])
def test_two_speakers_merge_from_the_same_voice_threshold(monkeypatch, nudge, merged):
    twin = (0.8, 0.6, 0.0)
    edge = _cosine(A, twin)
    monkeypatch.setattr(
        speaker_merge, "SAME_VOICE_SIMILARITY", np.nextafter(edge, 2) if nudge else edge
    )
    turns = [("SPEAKER_00", 0.0, 10.0), ("SPEAKER_01", 10.0, 20.0)]

    labels = {s for s, _ in _merge(turns, {"SPEAKER_00": A, "SPEAKER_01": twin})}

    assert labels == ({"SPEAKER_00"} if merged else {"SPEAKER_00", "SPEAKER_01"})


@pytest.mark.parametrize(("nudge", "joined"), [(0, True), (1, False)], ids=["at", "above"])
def test_a_fragment_joins_from_the_join_threshold(monkeypatch, nudge, joined):
    fragment = (0.5, 0.0, 0.866)
    edge = _cosine(fragment, A)
    monkeypatch.setattr(speaker_merge, "JOIN_SIMILARITY", np.nextafter(edge, 2) if nudge else edge)
    turns = [("SPEAKER_00", 0.0, 50.0), ("SPEAKER_01", 50.0, 100.0), ("SPEAKER_02", 100.0, 101.0)]

    merged = _merge(turns, {"SPEAKER_00": A, "SPEAKER_01": B, "SPEAKER_02": fragment})

    assert (("SPEAKER_00", 100.0) in merged) is joined


@pytest.mark.parametrize(("nudge", "joined"), [(0, True), (1, False)], ids=["at", "above"])
def test_a_fragment_joins_from_the_margin_over_the_runner_up(monkeypatch, nudge, joined):
    fragment = (0.7, 0.4, 0.592)
    edge = _cosine(fragment, A) - _cosine(fragment, B)
    monkeypatch.setattr(speaker_merge, "JOIN_SIMILARITY", 0.0)
    monkeypatch.setattr(speaker_merge, "JOIN_MARGIN", np.nextafter(edge, 2) if nudge else edge)
    turns = [("SPEAKER_00", 0.0, 50.0), ("SPEAKER_01", 50.0, 100.0), ("SPEAKER_02", 100.0, 101.0)]

    merged = _merge(turns, {"SPEAKER_00": A, "SPEAKER_01": B, "SPEAKER_02": fragment})

    assert (("SPEAKER_00", 100.0) in merged) is joined
