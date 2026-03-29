"""Processing pipeline shared between CLI and tray."""

import gc
import shutil
from collections.abc import Callable
from pathlib import Path

from tapeback.audio import convert_to_mono16k, get_channel_count, merge_channels, split_channels_16k
from tapeback.diarizer import (
    filter_silent_segments,
    load_stereo_channels,
    merge_channel_segments,
    split_on_silence,
)
from tapeback.formatter import format_markdown
from tapeback.models import Segment
from tapeback.recorder import Recorder
from tapeback.settings import Settings
from tapeback.vault import save_audio_to_vault, save_markdown_to_vault

StatusCallback = Callable[[str], None]


def _noop_status(msg: str) -> None:
    pass


def stop_and_process(
    recorder: Recorder,
    settings: Settings,
    *,
    diarize: bool = True,
    do_summarize: bool = True,
    on_status: StatusCallback = _noop_status,
) -> Path:
    """Stop recording and run the full dual-channel processing pipeline.

    Returns path to the saved markdown file.
    """
    on_status("Stopping recording...")
    monitor_path, mic_path = recorder.stop()

    on_status("Merging audio channels...")
    output_dir = monitor_path.parent
    stereo_path, _mono_16k_path = merge_channels(monitor_path, mic_path, output_dir)

    session_name = monitor_path.parent.name

    audio_dest = save_audio_to_vault(stereo_path, settings, session_name)
    on_status(f"Audio saved: {audio_dest}")

    segments, info = process_stereo_file(
        stereo_path, output_dir, settings, diarize=diarize, on_status=on_status
    )

    audio_rel_path = f"{settings.attachments_dir}/{session_name}.wav"

    markdown = format_markdown(
        segments=segments,
        session_name=session_name,
        audio_rel_path=audio_rel_path,
        duration_seconds=float(info.get("duration", 0.0)),
        language=str(info.get("language", settings.language)),
    )

    md_path = save_markdown_to_vault(markdown, settings, session_name)
    on_status(f"Saved: {md_path}")

    if do_summarize:
        _maybe_summarize(md_path, settings, on_status)

    shutil.rmtree(monitor_path.parent, ignore_errors=True)
    return md_path


def process_file(
    audio_path: Path,
    settings: Settings,
    *,
    name: str | None = None,
    diarize: bool = True,
    do_summarize: bool = True,
    on_status: StatusCallback = _noop_status,
) -> Path:
    """Process an existing audio file. Returns path to saved markdown."""
    import tempfile

    if name is None:
        name = audio_path.stem

    audio_dest = save_audio_to_vault(audio_path, settings, name)
    on_status(f"Audio saved: {audio_dest}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="tapeback_"))

    if is_stereo(audio_path):
        on_status("Stereo file detected, using dual-channel pipeline...")
        segments, info = process_stereo_file(
            audio_path, tmp_dir, settings, diarize=diarize, on_status=on_status
        )
    else:
        segments, info = process_mono_file(
            audio_path, tmp_dir, settings, diarize=diarize, on_status=on_status
        )

    audio_rel_path = f"{settings.attachments_dir}/{name}.wav"

    markdown = format_markdown(
        segments=segments,
        session_name=name,
        audio_rel_path=audio_rel_path,
        duration_seconds=float(info.get("duration", 0.0)),
        language=str(info.get("language", settings.language)),
    )

    md_path = save_markdown_to_vault(markdown, settings, name)
    on_status(f"Saved: {md_path}")

    if do_summarize:
        _maybe_summarize(md_path, settings, on_status)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return md_path


def is_stereo(audio_path: Path) -> bool:
    """Check if an audio file is a stereo WAV."""
    try:
        return get_channel_count(audio_path) == 2
    except Exception:  # noqa: S110 — non-WAV or unreadable files are expected
        pass
    return False


def process_stereo_file(
    stereo_path: Path,
    output_dir: Path,
    settings: Settings,
    *,
    diarize: bool,
    on_status: StatusCallback = _noop_status,
) -> tuple[list[Segment], dict[str, str | float]]:
    """Process a stereo WAV through the dual-channel pipeline."""
    from tapeback.transcriber import Transcriber

    mic_raw, monitor_raw, raw_sr = load_stereo_channels(stereo_path)

    on_status("Splitting channels...")
    mic_16k, monitor_16k = split_channels_16k(stereo_path, output_dir)

    on_status("Transcribing (this may take a few minutes)...")
    transcriber = Transcriber(settings)
    mic_segments, monitor_segments, info = transcriber.transcribe_stereo(mic_16k, monitor_16k)

    mic_segments = split_on_silence(
        mic_segments,
        mic_raw,
        raw_sr,
        settings.pause_threshold,
        monitor_samples=monitor_raw,
    )

    mic_segments = filter_silent_segments(mic_segments, mic_raw, raw_sr)
    monitor_segments = filter_silent_segments(monitor_segments, monitor_raw, raw_sr)

    del transcriber
    free_gpu_memory()

    diarized = False
    if diarize and settings.diarize and settings.hf_token:
        from tapeback.diarizer import diarization_available

        if not diarization_available():
            on_status(
                "Warning: pyannote-audio not installed, skipping diarization. "
                "Install with: uv pip install tapeback[diarize]"
            )
        else:
            on_status("Diarizing speakers...")
            from tapeback.diarizer import Diarizer, assign_speakers, merge_similar_speakers

            diarizer = Diarizer(settings)
            diarization_segments = diarizer.diarize(monitor_16k)
            diarization_segments = merge_similar_speakers(diarization_segments, monitor_raw, raw_sr)
            monitor_segments = assign_speakers(monitor_segments, diarization_segments)
            diarized = True

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

    segments = merge_channel_segments(mic_segments, monitor_segments)
    return segments, info


def process_mono_file(
    audio_path: Path,
    output_dir: Path,
    settings: Settings,
    *,
    diarize: bool,
    on_status: StatusCallback = _noop_status,
) -> tuple[list[Segment], dict[str, str | float]]:
    """Process a mono/non-stereo audio file through the single-channel pipeline."""
    from tapeback.transcriber import Transcriber

    on_status("Converting audio...")
    mono_16k_path = convert_to_mono16k(audio_path, output_dir)

    on_status("Transcribing (this may take a few minutes)...")
    transcriber = Transcriber(settings)
    segments, info = transcriber.transcribe(mono_16k_path)

    del transcriber
    free_gpu_memory()

    stereo_for_attribution = _get_stereo_source(audio_path)
    segments = _maybe_diarize_segments(
        segments,
        settings,
        mono_16k_path,
        stereo_for_attribution,
        diarize=diarize,
        on_status=on_status,
    )

    return segments, info


def free_gpu_memory() -> None:
    """Free GPU memory so diarizer can use CUDA."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _maybe_diarize_segments(
    segments: list[Segment],
    settings: Settings,
    mono_16k_path: Path,
    stereo_path: Path | None,
    *,
    diarize: bool,
    on_status: StatusCallback = _noop_status,
) -> list[Segment]:
    """Run diarization if enabled, configured, and token available."""
    if not diarize or not settings.diarize:
        return segments

    if not settings.hf_token:
        on_status(
            "Warning: TAPEBACK_HF_TOKEN not set, skipping diarization. "
            "See README for setup instructions."
        )
        return segments

    from tapeback.diarizer import diarization_available

    if not diarization_available():
        on_status(
            "Warning: pyannote-audio not installed, skipping diarization. "
            "Install with: uv pip install tapeback[diarize]"
        )
        return segments

    on_status("Diarizing speakers...")
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
        if get_channel_count(audio_path) == 2:
            return audio_path
    except Exception:  # noqa: S110 — non-stereo or unreadable files are expected
        pass
    return None


def _maybe_summarize(md_path: Path, settings: Settings, on_status: StatusCallback) -> None:
    """Run summarization if available."""
    from tapeback.summarizer import maybe_summarize

    on_status("Summarizing...")
    maybe_summarize(md_path, settings)
