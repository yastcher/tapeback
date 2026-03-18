import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

import click

from meetrec.settings import get_settings


@click.group()
def cli():
    """meetrec — local meeting recorder for Obsidian."""


@cli.command()
@click.argument("name", required=False)
@click.option("--no-diarize", is_flag=True, help="Skip speaker diarization")
def start(name, no_diarize):
    """Start recording monitor source + microphone.

    Runs in foreground — press Ctrl+C to stop and transcribe.
    """
    settings = get_settings()

    from meetrec.recorder import Recorder, detect_devices

    recorder = Recorder()

    monitor, mic = detect_devices(settings)
    session_name = recorder.start(settings, session_name=name)

    click.echo(f"Recording started: {session_name}", err=True)
    click.echo(f"Monitor: {monitor}", err=True)
    click.echo(f"Mic: {mic}", err=True)
    click.echo("Run 'meetrec stop' to finish and transcribe.", err=True)
    click.echo("Or press Ctrl+C to stop and transcribe now.", err=True)

    # Block and wait for Ctrl+C
    try:
        signal.pause()
    except KeyboardInterrupt:
        click.echo("\nStopping...", err=True)
        try:
            _stop_and_process(recorder, settings, diarize=not no_diarize)
        except KeyboardInterrupt:
            click.echo("\nAborted during processing. Audio files kept in /tmp/meetrec/", err=True)


@cli.command()
def stop():
    """Stop recording, transcribe, and save to vault."""
    settings = get_settings()

    from meetrec.recorder import Recorder

    recorder = Recorder()
    _stop_and_process(recorder, settings, diarize=True)


def _stop_and_process(recorder, settings, *, diarize=True):
    """Stop recording and run the full processing pipeline."""
    click.echo("Stopping recording...", err=True)
    monitor_path, mic_path = recorder.stop()

    from meetrec.audio import merge_channels
    from meetrec.formatter import format_markdown, save_audio_to_vault, save_markdown_to_vault
    from meetrec.transcriber import Transcriber

    click.echo("Merging audio channels...", err=True)
    output_dir = monitor_path.parent
    stereo_path, mono_16k_path = merge_channels(monitor_path, mic_path, output_dir)

    session_name = monitor_path.parent.name

    # Save audio to vault immediately (before transcription)
    audio_dest = save_audio_to_vault(stereo_path, settings, session_name)
    click.echo(f"Audio saved: {audio_dest}", err=True)

    click.echo("Transcribing (this may take a few minutes)...", err=True)
    transcriber = Transcriber(settings)
    segments, info = transcriber.transcribe(mono_16k_path)

    # Free GPU memory before diarization
    del transcriber
    _free_gpu_memory()

    segments = _maybe_diarize(segments, settings, mono_16k_path, stereo_path, diarize=diarize)

    audio_rel_path = f"{settings.attachments_dir}/{session_name}.wav"

    markdown = format_markdown(
        segments=segments,
        session_name=session_name,
        audio_rel_path=audio_rel_path,
        duration_seconds=info.get("duration", 0.0),
        language=info.get("language", settings.language),
    )

    md_path = save_markdown_to_vault(markdown, settings, session_name)

    # Clean up temp files
    shutil.rmtree(monitor_path.parent, ignore_errors=True)

    click.echo(f"Saved: {md_path}", err=True)


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--name", default=None, help="Session name for output file")
@click.option("--no-diarize", is_flag=True, help="Skip speaker diarization")
def process(audio_file, name, no_diarize):
    """Process an existing audio file (mp3, m4a, ogg, wav)."""
    settings = get_settings()

    from meetrec.audio import convert_to_mono16k
    from meetrec.formatter import format_markdown, save_audio_to_vault, save_markdown_to_vault
    from meetrec.transcriber import Transcriber

    audio_path = Path(audio_file)

    if name is None:
        name = audio_path.stem

    click.echo("Converting audio...", err=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="meetrec_"))
    mono_16k_path = convert_to_mono16k(audio_path, tmp_dir)

    # Save audio to vault immediately (before transcription)
    audio_dest = save_audio_to_vault(audio_path, settings, name)
    click.echo(f"Audio saved: {audio_dest}", err=True)

    click.echo("Transcribing (this may take a few minutes)...", err=True)
    transcriber = Transcriber(settings)
    segments, info = transcriber.transcribe(mono_16k_path)

    # Free GPU memory before diarization
    del transcriber
    _free_gpu_memory()

    # For process: use original file for channel attribution if it's stereo WAV
    stereo_for_attribution = _get_stereo_source(audio_path)
    segments = _maybe_diarize(
        segments, settings, mono_16k_path, stereo_for_attribution, diarize=not no_diarize
    )

    audio_rel_path = f"{settings.attachments_dir}/{name}.wav"

    markdown = format_markdown(
        segments=segments,
        session_name=name,
        audio_rel_path=audio_rel_path,
        duration_seconds=info.get("duration", 0.0),
        language=info.get("language", settings.language),
    )

    md_path = save_markdown_to_vault(markdown, settings, name)

    # Clean up temp
    shutil.rmtree(tmp_dir, ignore_errors=True)

    click.echo(f"Saved: {md_path}", err=True)


def _free_gpu_memory():
    """Free GPU memory so diarizer can use CUDA."""
    try:
        import gc

        gc.collect()

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _maybe_diarize(segments, settings, mono_16k_path, stereo_path, *, diarize):
    """Run diarization if enabled, configured, and token available."""
    if not diarize or not settings.diarize:
        return segments

    if not settings.hf_token:
        click.echo(
            "Warning: MEETREC_HF_TOKEN not set, skipping diarization. "
            "See README for setup instructions.",
            err=True,
        )
        return segments

    click.echo("Diarizing speakers...", err=True)
    from meetrec.diarizer import Diarizer, assign_speakers, identify_user_speaker

    diarizer = Diarizer(settings)
    diarization_segments = diarizer.diarize(mono_16k_path)

    user_speaker = None
    if stereo_path is not None:
        user_speaker = identify_user_speaker(diarization_segments, stereo_path)

    return assign_speakers(segments, diarization_segments, user_speaker, stereo_path)


def _get_stereo_source(audio_path):
    """Return audio_path if it's a stereo WAV, else None."""
    try:
        from meetrec.audio import get_channel_count

        if get_channel_count(audio_path) == 2:
            return audio_path
    except Exception:
        pass
    return None


@cli.command()
def status():
    """Show current recording status and settings."""
    settings = get_settings()

    from meetrec.recorder import Recorder

    recorder = Recorder()
    session = recorder.get_session_info()

    if session:
        click.echo(f"Recording in progress: {session['session_name']}")
        click.echo(f"Started at: {session['started_at']}")
    else:
        click.echo("Not recording.")

    click.echo(f"\nVault: {settings.vault_path}")
    click.echo(f"Whisper model: {settings.whisper_model}")
    click.echo(f"Device: {settings.device}")
    click.echo(f"Language: {settings.language}")

    # Show available audio devices
    if shutil.which("pactl"):
        click.echo("\nAudio sources:")
        result = subprocess.run(
            ["pactl", "list", "sources", "short"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            click.echo(result.stdout)
