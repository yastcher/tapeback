import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

import click

from tapeback.models import Segment
from tapeback.recorder import Recorder, detect_devices
from tapeback.settings import Settings, get_settings


@click.group()
def cli() -> None:
    """tapeback — local meeting recorder for Obsidian.

    Records system audio + microphone via PipeWire/PulseAudio,
    transcribes locally with Whisper, identifies speakers with pyannote,
    saves Markdown notes to your Obsidian vault.

    Works with any video call platform (Meet, Zoom, Teams, Telegram, Discord).

    \b
    Quick start:
      export TAPEBACK_VAULT_PATH=~/Documents/obsidian/vault
      tapeback start          # record (Ctrl+C to stop)
      tapeback process a.mp3  # transcribe existing file

    \b
    Configuration:
      All settings via TAPEBACK_* env vars or .env file.
      See: https://github.com/yastcher/tapeback#configuration
    """


@cli.command()
@click.argument("name", required=False)
@click.option("--no-diarize", is_flag=True, help="Skip speaker diarization")
@click.option("--no-summarize", is_flag=True, help="Skip LLM summarization")
def start(name: str | None, no_diarize: bool, no_summarize: bool) -> None:
    """Start recording monitor source + microphone.

    Records system audio (what you hear) and microphone (what you say)
    into a stereo WAV file. Runs in foreground — press Ctrl+C to stop,
    transcribe, and save to vault.

    \b
    Optionally provide a NAME for the output file:
      tapeback start "weekly-standup"
    """
    settings = get_settings()

    recorder = Recorder()

    monitor, mic = detect_devices(settings)
    session_name = recorder.start(settings, session_name=name)

    click.echo(f"Recording started: {session_name}", err=True)
    click.echo(f"Monitor: {monitor}", err=True)
    click.echo(f"Mic: {mic}", err=True)
    click.echo("Run 'tapeback stop' to finish and transcribe.", err=True)
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
            click.echo(
                "\nAborted during processing. Audio files kept in /tmp/tapeback/",
                err=True,
            )


@cli.command()
def stop() -> None:
    """Stop recording from another terminal.

    Sends stop signal to a running 'tapeback start' process,
    then transcribes and saves the recording to vault.
    """
    settings = get_settings()
    recorder = Recorder()
    _stop_and_process(recorder, settings, diarize=True, do_summarize=True)


def _stop_and_process(
    recorder: Recorder, settings: Settings, *, diarize: bool = True, do_summarize: bool = True
) -> None:
    """Stop recording and run the full dual-channel processing pipeline."""
    click.echo("Stopping recording...", err=True)
    monitor_path, mic_path = recorder.stop()

    from tapeback.audio import merge_channels
    from tapeback.formatter import format_markdown
    from tapeback.vault import save_audio_to_vault, save_markdown_to_vault

    click.echo("Merging audio channels...", err=True)
    output_dir = monitor_path.parent
    stereo_path, _mono_16k_path = merge_channels(monitor_path, mic_path, output_dir)

    session_name = monitor_path.parent.name

    # Save audio to vault immediately (before transcription)
    audio_dest = save_audio_to_vault(stereo_path, settings, session_name)
    click.echo(f"Audio saved: {audio_dest}", err=True)

    segments, info = _process_stereo_file(stereo_path, output_dir, settings, diarize=diarize)

    audio_rel_path = f"{settings.attachments_dir}/{session_name}.wav"

    markdown = format_markdown(
        segments=segments,
        session_name=session_name,
        audio_rel_path=audio_rel_path,
        duration_seconds=float(info.get("duration", 0.0)),
        language=str(info.get("language", settings.language)),
    )

    md_path = save_markdown_to_vault(markdown, settings, session_name)
    click.echo(f"Saved: {md_path}", err=True)

    if do_summarize:
        from tapeback.summarizer import maybe_summarize

        maybe_summarize(md_path, settings)

    # Clean up temp files
    shutil.rmtree(monitor_path.parent, ignore_errors=True)


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--name", default=None, help="Session name for output file")
@click.option("--no-diarize", is_flag=True, help="Skip speaker diarization")
@click.option("--no-summarize", is_flag=True, help="Skip LLM summarization")
def process(audio_file: str, name: str | None, no_diarize: bool, no_summarize: bool) -> None:
    """Process an existing audio file (mp3, m4a, ogg, wav).

    Transcribes the file, identifies speakers, generates an LLM summary,
    and saves everything as a Markdown note in your vault.

    \b
    Stereo WAV files (from tapeback start) use the dual-channel pipeline
    with per-channel transcription and speaker attribution.
    All other files use mono processing.

    \b
    Examples:
      tapeback process meeting.mp3
      tapeback process call.wav --name "client-call" --no-diarize
    """
    settings = get_settings()

    from tapeback.formatter import format_markdown
    from tapeback.vault import save_audio_to_vault, save_markdown_to_vault

    audio_path = Path(audio_file)

    if name is None:
        name = audio_path.stem

    # Save audio to vault immediately (before transcription)
    audio_dest = save_audio_to_vault(audio_path, settings, name)
    click.echo(f"Audio saved: {audio_dest}", err=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="tapeback_"))

    # Stereo WAV → dual-channel pipeline (split channels, transcribe each)
    # Mono/other → single-channel pipeline
    if _is_stereo(audio_path):
        click.echo("Stereo file detected, using dual-channel pipeline...", err=True)
        segments, info = _process_stereo_file(audio_path, tmp_dir, settings, diarize=not no_diarize)
    else:
        segments, info = _process_mono_file(audio_path, tmp_dir, settings, diarize=not no_diarize)

    audio_rel_path = f"{settings.attachments_dir}/{name}.wav"

    markdown = format_markdown(
        segments=segments,
        session_name=name,
        audio_rel_path=audio_rel_path,
        duration_seconds=float(info.get("duration", 0.0)),
        language=str(info.get("language", settings.language)),
    )

    md_path = save_markdown_to_vault(markdown, settings, name)
    click.echo(f"Saved: {md_path}", err=True)

    if not no_summarize:
        from tapeback.summarizer import maybe_summarize

        maybe_summarize(md_path, settings)

    # Clean up temp
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _is_stereo(audio_path: Path) -> bool:
    """Check if an audio file is a stereo WAV."""
    try:
        from tapeback.audio import get_channel_count

        return get_channel_count(audio_path) == 2
    except Exception:  # noqa: S110 — non-WAV or unreadable files are expected
        pass
    return False


def _process_stereo_file(
    stereo_path: Path,
    output_dir: Path,
    settings: Settings,
    *,
    diarize: bool,
) -> tuple[list[Segment], dict[str, str | float]]:
    """Process a stereo WAV through the dual-channel pipeline.

    Splits left (mic) / right (monitor), transcribes each channel separately,
    applies silence filtering, diarizes monitor channel, merges by time.
    """
    from tapeback.audio import split_channels_16k
    from tapeback.diarizer import (
        filter_silent_segments,
        load_stereo_channels,
        merge_channel_segments,
        split_on_silence,
    )
    from tapeback.transcriber import Transcriber

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
    diarized = False
    if diarize and settings.diarize and settings.hf_token:
        from tapeback.diarizer import diarization_available

        if not diarization_available():
            click.echo(
                "Warning: pyannote-audio not installed, skipping diarization. "
                "Install with: uv pip install tapeback[diarize]",
                err=True,
            )
        else:
            click.echo("Diarizing speakers...", err=True)
            from tapeback.diarizer import Diarizer, assign_speakers, merge_similar_speakers

            diarizer = Diarizer(settings)
            diarization_segments = diarizer.diarize(monitor_16k)
            diarization_segments = merge_similar_speakers(diarization_segments, monitor_raw, raw_sr)
            monitor_segments = assign_speakers(monitor_segments, diarization_segments)
            diarized = True

    # Default speaker for monitor segments when diarization is skipped
    if not diarized and monitor_segments and monitor_segments[0].speaker is None:
        monitor_segments = [
            Segment(
                start=s.start,
                end=s.end,
                text=s.text,
                words=s.words,
                speaker="Other",
            )
            for s in monitor_segments
        ]

    # Merge both channels sorted by time
    segments = merge_channel_segments(mic_segments, monitor_segments)
    return segments, info


def _process_mono_file(
    audio_path: Path,
    output_dir: Path,
    settings: Settings,
    *,
    diarize: bool,
) -> tuple[list[Segment], dict[str, str | float]]:
    """Process a mono/non-stereo audio file through the single-channel pipeline."""
    from tapeback.audio import convert_to_mono16k
    from tapeback.transcriber import Transcriber

    click.echo("Converting audio...", err=True)
    mono_16k_path = convert_to_mono16k(audio_path, output_dir)

    click.echo("Transcribing (this may take a few minutes)...", err=True)
    transcriber = Transcriber(settings)
    segments, info = transcriber.transcribe(mono_16k_path)

    # Free GPU memory before diarization
    del transcriber
    _free_gpu_memory()

    # Diarize mono audio if enabled
    stereo_for_attribution = _get_stereo_source(audio_path)
    segments = _maybe_diarize(
        segments, settings, mono_16k_path, stereo_for_attribution, diarize=diarize
    )

    return segments, info


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
            "Warning: TAPEBACK_HF_TOKEN not set, skipping diarization. "
            "See README for setup instructions.",
            err=True,
        )
        return segments

    from tapeback.diarizer import diarization_available

    if not diarization_available():
        click.echo(
            "Warning: pyannote-audio not installed, skipping diarization. "
            "Install with: uv pip install tapeback[diarize]",
            err=True,
        )
        return segments

    click.echo("Diarizing speakers...", err=True)
    from tapeback.diarizer import Diarizer, assign_speakers, identify_user_speaker

    diarizer = Diarizer(settings)
    diarization_segments = diarizer.diarize(mono_16k_path)

    user_speaker = None
    if stereo_path is not None:
        user_speaker = identify_user_speaker(diarization_segments, stereo_path)

    return assign_speakers(segments, diarization_segments, user_speaker, stereo_path)


def _get_stereo_source(audio_path: Path) -> Path | None:
    """Return audio_path if it's a stereo WAV, else None."""
    try:
        from tapeback.audio import get_channel_count

        if get_channel_count(audio_path) == 2:
            return audio_path
    except Exception:  # noqa: S110 — non-stereo or unreadable files are expected
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
    """Summarize an existing transcript markdown file.

    Sends the transcript to an LLM and adds a summary section with
    brief overview, action items, and key decisions.

    \b
    Requires the llm extra: uv pip install tapeback[llm]
    and an API key — set TAPEBACK_LLM_API_KEY or a provider-specific
    env var (ANTHROPIC_API_KEY, GEMINI_API_KEY, etc.).

    \b
    Examples:
      tapeback summarize vault/meetings/2026-03-26.md
      tapeback summarize transcript.md --provider gemini
    """
    settings = get_settings()

    if provider:
        settings.llm_provider = provider  # type: ignore[assignment]  # validated by click.Choice
    if model:
        settings.llm_model = model

    from tapeback.summarizer import (
        extract_transcript_from_markdown,
        format_summary_markdown,
        inject_summary_into_markdown,
    )
    from tapeback.summarizer import summarize as do_summarize

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
    """Show current recording status and settings.

    Displays whether a recording is in progress, vault path,
    Whisper model, device, and available audio sources.
    """
    settings = get_settings()

    from tapeback.recorder import Recorder

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
