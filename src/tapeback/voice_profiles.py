"""The voices tapeback knows by name, and the speakers of a call they name.

A profile is a person's name and samples of their voice: pyannote's embeddings of that
person in past calls. A speaker of a new call whose voice is close to one of a person's
samples, and clearly closer than to anyone else's, takes that person's name in the note;
any other stays "Speaker N". Two speakers may take one name: pyannote can split a person
it heard in two conditions. Measured on recorded calls (docs/spec/2026-10-06-diarization.md):
a profile built from other calls reached the same person at cosine 0.88 or more, another
person at 0.56 at most.

The file is biometric data. It lives in tapeback's own data directory, private to the
user, and nothing but this module reads it. Its vectors compare only with embeddings of
the model that made them, which the file records.
"""

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from tapeback import const
from tapeback._speaker_name import speaker_name_problem
from tapeback.models import DiarizationSegment, VoiceProfile, VoiceSample
from tapeback.recorder import ensure_private_dir

# A speaker is a person when their voice reaches this cosine to one of the person's
# samples: the same voice in another call measured 0.88 or more, another voice 0.56 at
# most...
PERSON_SIMILARITY = 0.8
# ...and no other person's samples come within this of it.
PERSON_MARGIN = 0.15
PROFILES_FILE = "profiles.json"


class ProfileError(Exception):
    """The profiles cannot be used: unreadable, malformed, or another model's voices."""


def profiles_path() -> Path:
    """Where the profiles live: tapeback's own data directory, as the resume cache does.

    A relative XDG_DATA_HOME is ignored, as the XDG spec requires.
    """
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg and Path(xdg).is_absolute() else Path.home() / ".local" / "share"
    return base / "tapeback" / "voices" / PROFILES_FILE


def load_profiles(path: Path | None = None) -> list[VoiceProfile]:
    """The profiles in the file; none when there is no file."""
    path = path or profiles_path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_bytes())
        model = data["model"]
        profiles = [
            VoiceProfile(
                id=str(entry["id"]),
                name=str(entry["name"]),
                samples=tuple(
                    VoiceSample(
                        vector=tuple(float(x) for x in sample["vector"]),
                        session=str(sample["session"]),
                        speaker=str(sample["speaker"]),
                    )
                    for sample in entry["samples"]
                ),
            )
            for entry in data["profiles"]
        ]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # The class, not the message: a message can quote the file, names included.
        raise ProfileError(f"{path} cannot be read ({type(exc).__name__})") from exc
    if model != const.PYANNOTE_MODEL:
        raise ProfileError(f"{path} holds voices of {model}, not of {const.PYANNOTE_MODEL}")
    if len({len(sample.vector) for profile in profiles for sample in profile.samples}) > 1:
        raise ProfileError(f"{path} holds voices of different lengths")
    return profiles


def save_profiles(profiles: Iterable[VoiceProfile], path: Path | None = None) -> Path:
    """Write the profiles, readable by the user alone; the old file stays until the new
    one is whole."""
    path = path or profiles_path()
    profiles = list(profiles)
    for profile in profiles:
        problem = speaker_name_problem(profile.name, reserved=[const.SPEAKER_YOU])
        if problem:
            raise ValueError(f"{profile.name!r} cannot name a speaker: {problem}")
    data = {
        "model": const.PYANNOTE_MODEL,
        "profiles": [
            {
                "id": profile.id,
                "name": profile.name,
                "samples": [
                    {"session": s.session, "speaker": s.speaker, "vector": list(s.vector)}
                    for s in profile.samples
                ],
            }
            for profile in profiles
        ],
    }
    ensure_private_dir(path.parent)
    partial = path.with_name(path.name + ".partial")
    partial.unlink(missing_ok=True)  # left by a write that did not finish
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    # ASCII JSON, non-ASCII names escaped: the file reads the same under any locale.
    with os.fdopen(fd, "wb") as out:
        out.write(json.dumps(data).encode())
    partial.replace(path)
    return path


def name_speakers(
    turns: Iterable[DiarizationSegment],
    voices: Mapping[str, tuple[float, ...]],
    profiles: list[VoiceProfile],
) -> tuple[dict[str, str], dict[str, float]]:
    """The speakers of the turns that are clearly a known person: (label -> name, and
    label -> the closest cosine to anyone, for every speaker with a voice)."""
    known: dict[str, list[np.ndarray]] = {}
    for profile in profiles:
        known.setdefault(profile.name, []).extend(_unit(s.vector) for s in profile.samples)
    known = {name: samples for name, samples in known.items() if samples}
    names: dict[str, str] = {}
    closest: dict[str, float] = {}
    for label in sorted({turn.speaker for turn in turns}):
        if label not in voices or not known:
            continue
        voice = _unit(voices[label])
        if any(sample.shape != voice.shape for samples in known.values() for sample in samples):
            raise ProfileError("the profiles hold voices of another length than this call's")
        by_person = {
            name: max(float(voice @ sample) for sample in samples)
            for name, samples in known.items()
        }
        person = max(by_person, key=by_person.__getitem__)
        closest[label] = by_person[person]
        others = [similarity for name, similarity in by_person.items() if name != person]
        if by_person[person] < PERSON_SIMILARITY:
            continue
        if others and by_person[person] - max(others) < PERSON_MARGIN:
            continue
        names[label] = person
    return names, closest


def _unit(vector: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(vector))
    norm = float(np.linalg.norm(array))
    return array / norm if norm else array
