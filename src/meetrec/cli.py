import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import click

from meetrec.settings import Settings, get_settings

if TYPE_CHECKING:
    from meetrec.recorder import Recorder
    from meetrec.transcriber import Segment


@click.group()
def cli() -> None:
    """meetrec — local meeting recorder for Obsidian."""


@cli.command()
@click.argument("name", required=False)
@click.option("--no-diarize", is_flag=True, help="Skip speaker diarization")
@click.option("--no-summarize", is_flag=True, help="Skip LLM summarization")
def start(name: str | None, no_diarize: bool, no_summarize: bool) -> None:
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
            _stop_and_process(
                recorder, settings, diarize=not no_diarize, do_summarize=not no_summarize
            )
        except KeyboardInterrupt:
            click.echo("\nAborted during processing. Audio files kept in /tmp/meetrec/", err=True)


@cli.command()
def stop() -> None:
    """Stop recording, transcribe, and save to vault."""
    settings = get_settings()

    from meetrec.recorder import Recorder

    recorder = Recorder()
    _stop_and_process(recorder, settings, diarize=True, do_summarize=True)


def _stop_and_process(
    recorder: Recorder, settings: Settings, *, diarize: bool = True, do_summarize: bool = True
) -> None:
    """Stop recording and run the full dual-channel processing pipeline."""
    click.echo("Stopping recording...", err=True)
    monitor_path, mic_path = recorder.stop()

    from meetrec.audio import merge_channels, split_channels_16k
    from meetrec.diarizer import (
        filter_silent_segments,
        load_stereo_channels,
        merge_channel_segments,
        split_on_silence,
    )
    from meetrec.formatter import format_markdown, save_audio_to_vault, save_markdown_to_vault
    from meetrec.transcriber import Transcriber

    click.echo("Merging audio channels...", err=True)
    output_dir = monitor_path.parent
    stereo_path, _mono_16k_path = merge_channels(monitor_path, mic_path, output_dir)

    session_name = monitor_path.parent.name

    # Save audio to vault immediately (before transcription)
    audio_dest = save_audio_to_vault(stereo_path, settings, session_name)
    click.echo(f"Audio saved: {audio_dest}", err=True)

    # Load raw stereo channels for RMS filtering (before loudnorm)
    mic_raw, monitor_raw, raw_sr = load_stereo_channels(stereo_path)

    # Split stereo into per-channel mono 16kHz (with loudnorm for Whisper)
    click.echo("Splitting channels...", err=True)
    mic_16k, monitor_16k = split_channels_16k(stereo_path, output_dir)

    # Transcribe each channel separately
    click.echo("Transcribing (this may take a few minutes)...", err=True)
    transcriber = Transcriber(settings)
    mic_segments, monitor_segments, info = transcriber.transcribe_stereo(mic_16k, monitor_16k)

    # Split mic segments at actual silence gaps in raw audio
    # (only mic — monitor segmentation is handled by Whisper + pyannote)
    mic_segments = split_on_silence(
        mic_segments,
        mic_raw,
        raw_sr,
        settings.pause_threshold,
        monitor_samples=monitor_raw,
    )

    # Filter out Whisper hallucinations on silent portions (using raw RMS)
    mic_segments = filter_silent_segments(mic_segments, mic_raw, raw_sr)
    monitor_segments = filter_silent_segments(monitor_segments, monitor_raw, raw_sr)

    # Free GPU memory before diarization
    del transcriber
    _free_gpu_memory()

    # Diarize monitor channel to separate remote speakers
    if diarize and settings.diarize and settings.hf_token:
        click.echo("Diarizing speakers...", err=True)
        from meetrec.diarizer import Diarizer, assign_speakers

        diarizer = Diarizer(settings)
        diarization_segments = diarizer.diarize(monitor_16k)
        monitor_segments = assign_speakers(monitor_segments, diarization_segments)

    # Merge both channels sorted by time
    segments = merge_channel_segments(mic_segments, monitor_segments)

    audio_rel_path = f"{settings.attachments_dir}/{session_name}.wav"

    markdown = format_markdown(
        segments=segments,
        session_name=session_name,
        audio_rel_path=audio_rel_path,
        duration_seconds=info.get("duration", 0.0),
        language=info.get("language", settings.language),
    )

    md_path = save_markdown_to_vault(markdown, settings, session_name)
    click.echo(f"Saved: {md_path}", err=True)

    if do_summarize:
        from meetrec.summarizer import maybe_summarize

        maybe_summarize(md_path, settings)

    # Clean up temp files
    shutil.rmtree(monitor_path.parent, ignore_errors=True)


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--name", default=None, help="Session name for output file")
@click.option("--no-diarize", is_flag=True, help="Skip speaker diarization")
@click.option("--no-summarize", is_flag=True, help="Skip LLM summarization")
def process(audio_file: str, name: str | None, no_diarize: bool, no_summarize: bool) -> None:
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
    click.echo(f"Saved: {md_path}", err=True)

    if not no_summarize:
        from meetrec.summarizer import maybe_summarize

        maybe_summarize(md_path, settings)

    # Clean up temp
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _free_gpu_memory() -> None:
    """Free GPU memory so diarizer can use CUDA."""
    try:
        import gc

        gc.collect()

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _maybe_diarize(
    segments: list[Segment],
    settings: Settings,
    mono_16k_path: Path,
    stereo_path: Path | None,
    *,
    diarize: bool,
) -> list[Segment]:
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


def _get_stereo_source(audio_path: Path) -> Path | None:
    """Return audio_path if it's a stereo WAV, else None."""
    try:
        from meetrec.audio import get_channel_count

        if get_channel_count(audio_path) == 2:
            return audio_path
    except Exception:
        pass
    return None


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "groq", "gemini", "openrouter", "deepseek", "qwen"]),
    default=None,
    help="Override LLM provider",
)
@click.option("--model", default=None, help="Override LLM model")
def summarize(file: str, provider: str | None, model: str | None) -> None:
    """Summarize an existing transcript markdown file."""
    settings = get_settings()

    if provider:
        settings.llm_provider = provider
    if model:
        settings.llm_model = model

    from meetrec.summarizer import (
        extract_transcript_from_markdown,
        format_summary_markdown,
        inject_summary_into_markdown,
    )
    from meetrec.summarizer import summarize as do_summarize

    path = Path(file)
    md_content = path.read_text()

    transcript = extract_transcript_from_markdown(md_content)
    if not transcript.strip():
        click.echo("Error: No transcript content found in file.", err=True)
        raise SystemExit(1)

    click.echo("Summarizing...", err=True)
    summary = do_summarize(transcript, settings)
    summary_md = format_summary_markdown(summary)
    new_content = inject_summary_into_markdown(md_content, summary_md)
    path.write_text(new_content)
    click.echo(f"Summary added to {path}", err=True)


@cli.command()
def status() -> None:
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
