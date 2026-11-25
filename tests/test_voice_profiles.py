"""Voice profiles: the people tapeback can name, and the speakers of a call they name.

Voices are a few dimensions wide here; the match compares directions only. The cosines
are chosen like those measured on recorded calls: one person in another call near 0.9,
another person near 0.2.
"""

import json
import stat
from pathlib import Path

import numpy as np
import pytest

from tapeback import voice_profiles
from tapeback._speaker_name import speaker_name_problem
from tapeback.models import DiarizationSegment, VoiceProfile, VoiceSample
from tapeback.voice_profiles import (
    ProfileError,
    load_profiles,
    name_speakers,
    profiles_path,
    save_profiles,
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


def _turns(*labels: str) -> list[DiarizationSegment]:
    return [
        DiarizationSegment(speaker=label, start=float(i), end=float(i) + 1.0)
        for i, label in enumerate(labels)
    ]


def _cosine(u: tuple[float, ...], v: tuple[float, ...]) -> float:
    """As the match computes it, each side made unit length first."""
    a, b = np.asarray(u), np.asarray(v)
    return float((a / np.linalg.norm(a)) @ (b / np.linalg.norm(b)))


def test_profiles_live_in_tapebacks_data_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    assert profiles_path() == tmp_path / "data" / "tapeback" / "voices" / "profiles.json"


@pytest.mark.parametrize("data_home", ["relative/data", None], ids=["relative", "unset"])
def test_without_an_absolute_data_home_profiles_live_under_home(tmp_path, monkeypatch, data_home):
    if data_home is None:
        monkeypatch.delenv("XDG_DATA_HOME")
    else:
        monkeypatch.setenv("XDG_DATA_HOME", data_home)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert profiles_path() == tmp_path / ".local/share/tapeback/voices/profiles.json"


def test_saved_profiles_come_back_as_they_were():
    profiles = [_profile("Ann", A, NEAR_A), _profile("Boris", B)]

    save_profiles(profiles)

    assert load_profiles() == profiles


def test_the_file_and_its_directory_are_the_users_alone():
    path = save_profiles([_profile("Ann", A)])

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert [p.name for p in path.parent.iterdir()] == ["profiles.json"]


def test_the_file_records_the_model_whose_voices_it_holds():
    path = save_profiles([_profile("Ann", A)])

    assert json.loads(path.read_text())["model"] == "pyannote/speaker-diarization-3.1"


def test_no_file_no_profiles():
    assert load_profiles() == []


def _write(data: object) -> Path:
    path = profiles_path()
    path.parent.mkdir(parents=True, mode=0o700)
    path.write_text(json.dumps(data) if not isinstance(data, str) else data)
    return path


def test_another_models_voices_are_refused():
    _write({"model": "other/embedder", "profiles": []})

    with pytest.raises(ProfileError, match="other/embedder"):
        load_profiles()


@pytest.mark.parametrize(
    ("content", "why"),
    [
        pytest.param(
            '{"model": "pyannote/speaker-diarization-3.1", "profiles": [{"name": "Ann"',
            "JSONDecodeError",
            id="cut",
        ),
        pytest.param({"model": "pyannote/speaker-diarization-3.1"}, "KeyError", id="no profiles"),
        pytest.param(
            {
                "model": "pyannote/speaker-diarization-3.1",
                "profiles": [{"id": "1", "name": "Ann", "samples": [{"vector": ["x"]}]}],
            },
            "ValueError",
            id="no number",
        ),
    ],
)
def test_a_broken_file_is_refused_without_quoting_it(content, why):
    path = _write(content)

    with pytest.raises(ProfileError) as refused:
        load_profiles()

    assert str(refused.value) == f"{path} cannot be read ({why})"


def test_a_cyrillic_name_is_kept_in_ascii_json():
    """Whatever the locale's encoding, an ASCII file reads back as it was written."""
    save_profiles([_profile("Аня", A)])

    assert profiles_path().read_bytes().isascii()
    assert [p.name for p in load_profiles()] == ["Аня"]


def test_a_write_cut_short_before_does_not_stay_beside_the_file():
    stale = _write({"model": "pyannote/speaker-diarization-3.1", "profiles": []})
    stale.rename(stale.with_name("profiles.json.partial"))

    save_profiles([_profile("Ann", A)])

    assert [p.name for p in profiles_path().parent.iterdir()] == ["profiles.json"]


def test_voices_of_different_lengths_are_refused():
    entry = {"session": "s", "speaker": "SPEAKER_00"}
    _write(
        {
            "model": "pyannote/speaker-diarization-3.1",
            "profiles": [
                {"id": "1", "name": "Ann", "samples": [{**entry, "vector": [1.0, 0.0]}]},
                {"id": "2", "name": "Boris", "samples": [{**entry, "vector": [1.0, 0.0, 0.0]}]},
            ],
        }
    )

    with pytest.raises(ProfileError, match="lengths"):
        load_profiles()


@pytest.mark.parametrize(
    "name",
    ["You", "Other", "speaker 3", "Ann:", "A*nn", "[Ann]", "Ann, Boris", "", " Ann", "Ann\n"],
)
def test_a_name_that_would_read_as_tapebacks_own_label_is_not_saved(name):
    with pytest.raises(ValueError, match="cannot name a speaker"):
        save_profiles([_profile(name, A)])

    assert not profiles_path().exists()


def test_a_second_save_replaces_the_first_whole():
    save_profiles([_profile("Ann", A)])
    save_profiles([_profile("Boris", B)])

    assert [p.name for p in load_profiles()] == ["Boris"]
    assert [p.name for p in profiles_path().parent.iterdir()] == ["profiles.json"]


def test_a_known_voice_is_named():
    names, closest = name_speakers(
        _turns("SPEAKER_00", "SPEAKER_01"),
        {"SPEAKER_00": NEAR_A, "SPEAKER_01": B},
        [_profile("Ann", A)],
    )

    assert names == {"SPEAKER_00": "Ann"}
    assert closest == {"SPEAKER_00": pytest.approx(0.95, abs=0.001), "SPEAKER_01": 0.0}


def test_the_closest_of_a_persons_samples_decides():
    """A headset in one call, a phone in another: either sample is the person."""
    names, _ = name_speakers(_turns("SPEAKER_00"), {"SPEAKER_00": B}, [_profile("Ann", A, B)])

    assert names == {"SPEAKER_00": "Ann"}


def test_two_speakers_may_take_one_name():
    """pyannote can split one person heard in two conditions."""
    names, _ = name_speakers(
        _turns("SPEAKER_00", "SPEAKER_01"),
        {"SPEAKER_00": A, "SPEAKER_01": NEAR_A},
        [_profile("Ann", A)],
    )

    assert names == {"SPEAKER_00": "Ann", "SPEAKER_01": "Ann"}


def test_two_profiles_of_one_name_are_one_person():
    """Not a runner-up to itself: the margin is over other people."""
    names, _ = name_speakers(
        _turns("SPEAKER_00"), {"SPEAKER_00": B}, [_profile("Ann", A), _profile("Ann", B)]
    )

    assert names == {"SPEAKER_00": "Ann"}


def test_a_speaker_without_a_voice_is_neither_named_nor_measured():
    """It comes first, and the speaker after it is still named."""
    names, closest = name_speakers(
        _turns("SPEAKER_00", "SPEAKER_01"), {"SPEAKER_01": A}, [_profile("Ann", A)]
    )

    assert (names, closest) == ({"SPEAKER_01": "Ann"}, {"SPEAKER_01": 1.0})


def test_a_stranger_first_does_not_stop_the_naming_of_the_rest():
    names, _ = name_speakers(
        _turns("SPEAKER_00", "SPEAKER_01"),
        {"SPEAKER_00": B, "SPEAKER_01": NEAR_A},
        [_profile("Ann", A)],
    )

    assert names == {"SPEAKER_01": "Ann"}


def test_a_voice_between_two_people_first_does_not_stop_the_naming_of_the_rest():
    """Close to Ann and to Boris alike, the first is neither; the second is Cleo."""
    cleo = (0.0, 0.0, 1.0)
    names, _ = name_speakers(
        _turns("SPEAKER_00", "SPEAKER_01"),
        {"SPEAKER_00": A, "SPEAKER_01": cleo},
        [_profile("Ann", A), _profile("Boris", NEAR_A), _profile("Cleo", cleo)],
    )

    assert names == {"SPEAKER_01": "Cleo"}


def test_a_voice_of_zeros_is_like_no_one():
    names, closest = name_speakers(
        _turns("SPEAKER_00"), {"SPEAKER_00": (0.0, 0.0, 0.0)}, [_profile("Ann", A)]
    )

    assert (names, closest) == ({}, {"SPEAKER_00": 0.0})


@pytest.mark.parametrize(
    "profiles",
    [
        pytest.param([], id="no profiles"),
        pytest.param([_profile("Ann")], id="a profile without samples"),
    ],
)
def test_without_samples_no_one_is_named(profiles):
    assert name_speakers(_turns("SPEAKER_00"), {"SPEAKER_00": A}, profiles) == ({}, {})


def test_voices_are_compared_by_direction_not_length():
    names, _ = name_speakers(
        _turns("SPEAKER_00", "SPEAKER_01"),
        {"SPEAKER_00": (28.5, 9.36, 0.0), "SPEAKER_01": (0.0, 30.0, 0.0)},  # NEAR_A, B
        [_profile("Ann", (0.5, 0.0, 0.0))],
    )

    assert names == {"SPEAKER_00": "Ann"}


def test_a_voice_of_another_length_than_the_profiles_is_refused():
    with pytest.raises(ProfileError) as refused:
        name_speakers(_turns("SPEAKER_00"), {"SPEAKER_00": (1.0, 0.0)}, [_profile("Ann", A)])

    assert str(refused.value) == "the profiles hold voices of another length than this call's"


@pytest.mark.parametrize(
    ("name", "why"),
    [
        ("", "it is empty, or starts or ends with a space"),
        ("Ann ", "it is empty, or starts or ends with a space"),
        ("Ann:", "it holds one of * : [ ] , or a line break"),
        ("Other", "it is one of tapeback's own labels"),
        ("Speaker 12", "it is one of tapeback's own labels"),
        ("Ann", None),
    ],
)
def test_why_a_name_cannot_stand_as_a_speaker(name, why):
    assert speaker_name_problem(name) == why


@pytest.mark.parametrize(("nudge", "named"), [(0, True), (1, False)], ids=["at", "above"])
def test_a_person_is_named_from_the_similarity_threshold(monkeypatch, nudge, named):
    edge = _cosine(NEAR_A, A)
    monkeypatch.setattr(
        voice_profiles, "PERSON_SIMILARITY", np.nextafter(edge, 2) if nudge else edge
    )

    names, _ = name_speakers(_turns("SPEAKER_00"), {"SPEAKER_00": NEAR_A}, [_profile("Ann", A)])

    assert names == ({"SPEAKER_00": "Ann"} if named else {})


@pytest.mark.parametrize(("nudge", "named"), [(0, True), (1, False)], ids=["at", "above"])
def test_a_person_is_named_from_the_margin_over_the_next(monkeypatch, nudge, named):
    voice = (0.9, 0.436, 0.0)
    edge = _cosine(voice, A) - _cosine(voice, B)
    monkeypatch.setattr(voice_profiles, "PERSON_SIMILARITY", 0.0)
    monkeypatch.setattr(voice_profiles, "PERSON_MARGIN", np.nextafter(edge, 2) if nudge else edge)

    names, _ = name_speakers(
        _turns("SPEAKER_00"), {"SPEAKER_00": voice}, [_profile("Ann", A), _profile("Boris", B)]
    )

    assert names == ({"SPEAKER_00": "Ann"} if named else {})
