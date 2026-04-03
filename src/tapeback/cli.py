import shutil
import signal
import subprocess
from pathlib import Path

import click

from tapeback import const
from tapeback.recorder import Recorder, detect_devices
from tapeback.settings import get_settings


def _echo_status(msg: str) -> None:
    """Status callback that prints to stderr via click."""
    click.echo(msg, err=True)


@click.group()
def cli() -> None:
    """tapeback — local meeting recorder for Obsidian.

    Records system audio + microphone via PipeWire/PulseAudio,
    transcribes locally with Whisper, identifies speakers with pyannote,
    saves Markdown notes to your Obsidian vault.

    Works with any video call platform (Meet, Zoom, Teams, Telegram, Discord).

    \b
    Quick start:
      tapeback start          # record (Ctrl+C to stop)
      tapeback process a.mp3  # transcribe existing file

    \b
    Configuration:
      All settings via TAPEBACK_* env vars or ~/.config/tapeback/.env or .env file.
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

    try:
        signal.pause()
    except KeyboardInterrupt:
        click.echo("\nStopping...", err=True)
        try:
            from tapeback.pipeline import stop_and_process

            stop_and_process(
                recorder,
                settings,
                diarize=not no_diarize,
                do_summarize=not no_summarize,
                on_status=_echo_status,
            )
        except KeyboardInterrupt:
            click.echo(
                f"\nAborted during processing. Audio files kept in {const.TEMP_DIR}/",
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

    from tapeback.pipeline import stop_and_process

    stop_and_process(recorder, settings, on_status=_echo_status)


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

    from tapeback.pipeline import process_file

    process_file(
        Path(audio_file),
        settings,
        name=name,
        diarize=not no_diarize,
        do_summarize=not no_summarize,
        on_status=_echo_status,
    )


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

    overrides: dict[str, str] = {}
    if provider:
        overrides["llm_provider"] = provider
    if model:
        overrides["llm_model"] = model
    if overrides:
        settings = settings.model_copy(update=overrides)

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


@cli.command()
def tray() -> None:
    """Run tapeback as a system tray icon.

    Provides a graphical interface for start/stop recording
    without needing a terminal. Right-click the icon for the menu.

    \b
    Requires the tray extra:
      uv pip install tapeback[tray]
    """
    try:
        from tapeback.tray import run_tray
    except ImportError:
        raise click.ClickException(
            "System tray requires pystray and Pillow. Install with: uv pip install tapeback[tray]"
        ) from None

    run_tray()
