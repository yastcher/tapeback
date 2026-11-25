"""Learning a voice from a note: reading the note back, finding whose voice each name is,
and keeping it in the profiles. The names and the notes are made up."""

import numpy as np
import pytest

from tapeback import voice_profiles
from tapeback.models import DiarizationSegment, VoiceProfile, VoiceSample
from tapeback.voice_profiles import (
    add_samples,
    forget,
    note_recording,
    note_speakers,
    rename_speakers,
    voices_for_names,
)

A = (1.0, 0.0, 0.0)
B = (0.0, 1.0, 0.0)
NEAR_A = (0.95, 0.312, 0.0)  # 0.95 to A: the same person in another call


def _profile(name: str, *vectors: tuple[float, ...]) -> VoiceProfile:
    samples = tuple(
        VoiceSample(vector=vector, session="2026-01-05_10-00-00", speaker=f"SPEAKER_0{i}")
        for i, vector in enumerate(vectors)
    )
    return VoiceProfile(id=f"id-{name}", name=name, samples=samples)


NOTE = (
    "---\n"
    "date: 2026-01-05\n"
    'audio: "[[attachments/audio/2026-01-05_10-00-00.wav]]"\n'
    "---\n"
    "\n"
    "## Summary\n"
    "\n"
    "Speaker 2 tests the build; Speaker 20 reviews it.\n"
    "\n"
    "- [ ] **Speaker 2:** Test the build\n"
    "\n"
    "---\n"
    "\n"
    "[00:00:00] **You:** Shall we start?\n"
    "\n"
    "[00:00:04] **Speaker 1:** Yes.\n"
    "\n"
    "[00:00:10] **Speaker 2:** I have the build.\n"
    "\n"
    "[00:00:10] **Speaker 1:** Good.\n"
    "\n"
    "[00:01:00] **Speaker 2:** It passes.\n"
)


def test_a_notes_recording_is_its_audio_field():
    assert note_recording(NOTE) == "attachments/audio/2026-01-05_10-00-00.wav"
    assert note_recording("---\ndate: 2026-01-05\n---\n") is None


def test_a_notes_speakers_hold_the_spans_of_their_lines():
    """From a line's timecode to the next one's, the last to the end given; a line at
    the very time of the next spans nothing."""
    assert note_speakers(NOTE, 75.0) == {
        "You": [(0.0, 4.0)],
        "Speaker 1": [(4.0, 10.0), (10.0, 60.0)],
        "Speaker 2": [(60.0, 75.0)],
    }


def test_a_speaker_is_renamed_wherever_their_label_stands_whole():
    renamed = rename_speakers(NOTE, {"Speaker 2": "Ann"})

    assert "Ann tests the build; Speaker 20 reviews it." in renamed
    assert "- [ ] **Ann:** Test the build" in renamed
    assert "[00:00:10] **Ann:** I have the build." in renamed
    assert "Speaker 2:" not in renamed


def test_a_name_goes_in_as_it_is():
    assert rename_speakers("**Speaker 1:** hi", {"Speaker 1": r"Ann\1"}) == r"**Ann\1:** hi"


def _speech(*turns: tuple[str, float, float]) -> list[DiarizationSegment]:
    return [DiarizationSegment(speaker=s, start=a, end=b) for s, a, b in turns]


def test_a_name_takes_the_voice_heard_most_in_their_lines():
    found, missed = voices_for_names(
        {"Ann": [(0.0, 10.0)], "Boris": [(10.0, 20.0)]},
        _speech(("SPEAKER_00", 0.0, 8.0), ("SPEAKER_01", 8.0, 20.0)),
        {"SPEAKER_00": A, "SPEAKER_01": B},
    )

    assert (found, missed) == ({"Ann": "SPEAKER_00", "Boris": "SPEAKER_01"}, {})


@pytest.mark.parametrize(("nudge", "learned"), [(0, True), (1, False)], ids=["at", "above"])
def test_a_name_is_learned_from_the_majority_share(monkeypatch, nudge, learned):
    share = 6.0 / 10.0  # SPEAKER_00's share of the speech heard in Ann's lines
    monkeypatch.setattr(
        voice_profiles, "LEARN_MAJORITY", np.nextafter(share, 2) if nudge else share
    )

    found, _ = voices_for_names(
        {"Ann": [(0.0, 10.0)]},
        _speech(("SPEAKER_00", 0.0, 6.0), ("SPEAKER_01", 6.0, 10.0)),
        {"SPEAKER_00": A, "SPEAKER_01": B},
    )

    assert found == ({"Ann": "SPEAKER_00"} if learned else {})


def test_a_name_whose_lines_hold_no_speech_is_not_learned():
    found, missed = voices_for_names(
        {"Ann": [(30.0, 40.0)]}, _speech(("SPEAKER_00", 0.0, 8.0)), {"SPEAKER_00": A}
    )

    assert (found, missed) == ({}, {"Ann": "no one voice holds most of their lines"})


def test_a_voice_pyannote_gave_no_embedding_is_not_learned():
    found, missed = voices_for_names({"Ann": [(0.0, 10.0)]}, _speech(("SPEAKER_00", 0.0, 10.0)), {})

    assert (found, missed) == ({}, {"Ann": "pyannote gave their voice no embedding"})


def test_two_names_on_one_voice_learn_neither():
    """The note says two people where the diarization heard one: no telling which."""
    found, missed = voices_for_names(
        {"Ann": [(0.0, 10.0)], "Boris": [(10.0, 20.0)], "Cleo": [(20.0, 30.0)]},
        _speech(("SPEAKER_00", 0.0, 20.0), ("SPEAKER_01", 20.0, 30.0)),
        {"SPEAKER_00": A, "SPEAKER_01": B},
    )

    assert found == {"Cleo": "SPEAKER_01"}
    assert missed == {
        "Ann": "their lines are the same voice as another name's",
        "Boris": "their lines are the same voice as another name's",
    }


def test_a_new_name_makes_a_new_profile():
    sample = VoiceSample(vector=A, session="2026-01-06_10-00-00", speaker="SPEAKER_01")

    [ann] = add_samples([], {"Ann": sample})

    assert (ann.name, ann.samples) == ("Ann", (sample,))
    assert len(ann.id) == 36  # a fresh uuid, which renaming will not change


def test_a_known_name_keeps_its_samples_and_gains_one():
    known = _profile("Ann", A)
    sample = VoiceSample(vector=B, session="2026-01-06_10-00-00", speaker="SPEAKER_01")

    [ann] = add_samples([known], {"Ann": sample})

    assert (ann.id, ann.samples) == ("id-Ann", (*known.samples, sample))


def test_learning_a_note_again_replaces_its_sample():
    first = VoiceSample(vector=A, session="2026-01-06_10-00-00", speaker="SPEAKER_01")
    again = VoiceSample(vector=NEAR_A, session="2026-01-06_10-00-00", speaker="SPEAKER_01")

    [ann] = add_samples(add_samples([], {"Ann": first}), {"Ann": again})

    assert ann.samples == (again,)


def test_forgetting_a_person_takes_every_profile_of_their_name():
    ann, boris = _profile("Ann", A), _profile("Boris", B)

    assert forget([ann, boris, ann], "Ann") == ([boris], [ann, ann])


def test_a_line_past_the_first_hour_starts_where_its_timecode_says():
    note = "[01:02:03] **Ann:** Late.\n\n[02:00:00] **Boris:** Later.\n"

    assert note_speakers(note, 7300.0) == {"Ann": [(3723.0, 7200.0)], "Boris": [(7200.0, 7300.0)]}


def test_a_line_that_only_touches_a_turn_holds_none_of_its_speech():
    found, missed = voices_for_names(
        {"Ann": [(8.0, 10.0)]}, _speech(("SPEAKER_00", 0.0, 8.0)), {"SPEAKER_00": A}
    )

    assert (found, missed) == ({}, {"Ann": "no one voice holds most of their lines"})


def test_half_a_second_of_speech_is_enough_to_tell_whose_it_is():
    found, _ = voices_for_names(
        {"Ann": [(0.0, 0.5)]}, _speech(("SPEAKER_00", 0.0, 0.5)), {"SPEAKER_00": A}
    )

    assert found == {"Ann": "SPEAKER_00"}


def test_every_turn_of_a_voice_in_a_names_lines_counts():
    """SPEAKER_00 speaks 4 s and 1 s of Ann's lines, SPEAKER_01 2 s in between."""
    found, _ = voices_for_names(
        {"Ann": [(0.0, 7.0)]},
        _speech(("SPEAKER_00", 0.0, 4.0), ("SPEAKER_01", 4.0, 6.0), ("SPEAKER_00", 6.0, 7.0)),
        {"SPEAKER_00": A, "SPEAKER_01": B},
    )

    assert found == {"Ann": "SPEAKER_00"}


def test_two_new_names_make_two_profiles():
    ann = VoiceSample(vector=A, session="2026-01-06_10-00-00", speaker="SPEAKER_00")
    boris = VoiceSample(vector=B, session="2026-01-06_10-00-00", speaker="SPEAKER_01")

    assert [(p.name, p.samples) for p in add_samples([], {"Ann": ann, "Boris": boris})] == [
        ("Ann", (ann,)),
        ("Boris", (boris,)),
    ]
