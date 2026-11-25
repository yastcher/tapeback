"""`tapeback voices`: the people tapeback names in notes by their voices (ADR-021).

The profile code imports numpy, so each command imports it when it runs: importing the
CLI stays light (`tests/test_architecture.py`).
"""

from pathlib import Path

import click

from tapeback import const
from tapeback._speaker_name import speaker_name_problem
from tapeback.settings import get_settings


def _status(msg: str) -> None:
    click.echo(msg, err=True)


@click.group()
def voices() -> None:
    """The people tapeback names in notes by their voices.

    \b
      tapeback voices learn NOTE 'Speaker 2=Anna'   # learn a voice, name it in the note
      tapeback voices learn NOTE                    # learn the names the note already has
      tapeback voices list
      tapeback voices forget Anna
    """


@voices.command("list")
def list_people() -> None:
    """Each person tapeback knows by voice, and the calls their voice was heard in."""
    from tapeback.voice_profiles import ProfileError, load_profiles

    try:
        profiles = load_profiles()
    except ProfileError as exc:
        raise click.ClickException(str(exc)) from exc
    if not profiles:
        click.echo("No voice profiles yet: see `tapeback voices learn --help`.")
    for profile in profiles:
        sessions = ", ".join(sorted({sample.session for sample in profile.samples}))
        click.echo(f"{profile.name}: heard in {sessions}")


@voices.command()
@click.argument("name")
def forget(name: str) -> None:
    """Forget a person's voice: their profile is deleted from this machine."""
    from tapeback.voice_profiles import (
        ProfileError,
        load_profiles,
        save_profiles,
    )
    from tapeback.voice_profiles import forget as split_off

    try:
        kept, gone = split_off(load_profiles(), name)
    except ProfileError as exc:
        raise click.ClickException(str(exc)) from exc
    if not gone:
        raise click.ClickException(f"No voice profile is named {name!r}.")
    save_profiles(kept)
    click.echo(f"Forgot {name}.")


@voices.command()
@click.argument("note", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("renames", nargs=-1, metavar="['LABEL=NAME' ...]")
def learn(note: Path, renames: tuple[str, ...]) -> None:
    """Learn the voices of the people a note names.

    Name a speaker here, as 'Speaker 2=Anna', and the note says Anna too; or rename the
    speakers in the note first and give the note alone. Several at once learn in one go.
    The note's recording is diarized again, about as long as its diarization took.
    """
    from tapeback.diarizer import diarization_available
    from tapeback.models import VoiceSample
    from tapeback.pipeline import voices_of_recording
    from tapeback.vault import replace_note
    from tapeback.voice_profiles import (
        ProfileError,
        add_samples,
        load_profiles,
        note_recording,
        note_speakers,
        rename_speakers,
        save_profiles,
        voices_for_names,
    )

    settings = get_settings()
    own = [const.SPEAKER_YOU, settings.your_name]
    markdown = note.read_text(encoding="utf-8")
    renamed = rename_speakers(markdown, _renames(renames, markdown, own))
    recording = note_recording(markdown)
    if recording is None or not (settings.vault_path / recording).is_file():
        raise click.ClickException(f"{note} names no recording in {settings.vault_path}.")
    if not settings.hf_token.get_secret_value():
        raise click.ClickException(
            "Learning a voice diarizes the recording: set TAPEBACK_HF_TOKEN."
        )
    if not diarization_available():
        raise click.ClickException(
            "Learning a voice diarizes the recording: install tapeback-diarize or "
            "tapeback[diarize]."
        )
    try:
        profiles = load_profiles()
    except ProfileError as exc:
        raise click.ClickException(str(exc)) from exc

    turns, diarization = voices_of_recording(settings.vault_path / recording, settings, _status)
    end = max((turn.end for turn in turns), default=0.0)
    named = {
        label: spans
        for label, spans in note_speakers(renamed, end).items()
        if speaker_name_problem(label, reserved=own) is None
    }
    found, missed = voices_for_names(named, turns, diarization.voices)
    samples = {
        name: VoiceSample(vector=diarization.voices[label], session=note.stem, speaker=label)
        for name, label in found.items()
    }
    if samples:
        save_profiles(add_samples(profiles, samples))
    if renamed != markdown:
        replace_note(note, renamed, settings)
    for name in sorted(found):
        click.echo(f"Learned {name}'s voice.")
    for name, why in sorted(missed.items()):
        click.echo(f"Not learned, {name}: {why}.", err=True)
    if not found:
        raise click.ClickException("No voice learned.")


def _renames(pairs: tuple[str, ...], markdown: str, own: list[str]) -> dict[str, str]:
    """'LABEL=NAME' pairs, checked: the label speaks in the note, the name can stand as a
    speaker, and neither is the user's own side, which is not diarized."""
    renames: dict[str, str] = {}
    for pair in pairs:
        label, sep, name = (part.strip() for part in pair.partition("="))
        if not sep or not label or not name:
            raise click.ClickException(f"{pair!r} is not LABEL=NAME, as 'Speaker 2=Anna'.")
        if label in own:
            raise click.ClickException(
                f"{label!r} is your side of the call, which is not learned: set "
                "TAPEBACK_YOUR_NAME instead."
            )
        if f"**{label}:**" not in markdown:
            raise click.ClickException(f"{label!r} speaks nowhere in the note.")
        problem = speaker_name_problem(name, reserved=own)
        if problem:
            raise click.ClickException(f"{name!r} cannot name a speaker: {problem}.")
        renames[label] = name
    return renames
