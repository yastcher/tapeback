"""Processing pipeline shared between CLI and tray."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tapeback.live import LiveTranscriber

from tapeback import const
from tapeback._gpu import free_gpu_memory, sample_gpu
from tapeback._lazy import load_transcriber
from tapeback._runlog import run_log
from tapeback._stt_caps import allows_local_diarize
from tapeback._timing import stage_timer
from tapeback.audio import (
    convert_to_mono16k,
    gate_wav_inactive,
    get_channel_count,
    merge_channels,
    split_channels_16k,
)
from tapeback.channel import (
    classify_segment_by_channel,
    filter_silent_segments,
    identify_user_speaker,
    load_stereo_channels,
    split_on_silence,
)
from tapeback.diarizer import (
    Diarizer,
    assign_speakers,
    diarization_available,
    merge_channel_segments,
    merge_similar_speakers,
)
from tapeback.formatter import TranscriptMeta, format_markdown
from tapeback.models import Segment
from tapeback.recorder import Recorder, validate_session_name
from tapeback.settings import Settings
from tapeback.summarizer import maybe_summarize
from tapeback.vault import remove_live_markdown, save_audio_to_vault, save_markdown_to_vault

StatusCallback = Callable[[str], None]


def _noop_status(msg: str) -> None:
    pass


def _gpu_telemetry_enabled(settings: Settings) -> bool:
    """GPU sampling is only meaningful for a run that actually asked for the GPU."""
    return settings.gpu_telemetry and settings.device == "cuda"


def stop_and_process(
    recorder: Recorder,
    settings: Settings,
    *,
    live_transcriber: LiveTranscriber | None = None,
    diarize: bool = True,
    do_summarize: bool = True,
    on_status: StatusCallback = _noop_status,
) -> Path:
    """Stop recording and run the full dual-channel processing pipeline.

    If a live_transcriber is active, stops it first to free GPU memory
    before the full pipeline creates its own Whisper model.

    Returns path to the saved markdown file.
    """
    if live_transcriber is not None:
        on_status("Stopping live transcription...")
        live_transcriber.stop()

    on_status("Stopping recording...")
    monitor_path, mic_path = recorder.stop()

    session_name = monitor_path.parent.name

    with run_log(session_name, settings, on_status) as report:
        report("Merging audio channels...")
        output_dir = monitor_path.parent
        with stage_timer("merge", report):
            stereo_path = merge_channels(monitor_path, mic_path, output_dir)

        audio_dest = save_audio_to_vault(stereo_path, settings, session_name)
        report(f"Audio saved: {audio_dest}")

        segments, info, raw_segments = process_stereo_file(
            stereo_path, output_dir, settings, diarize=diarize, on_status=report
        )

        audio_rel_path = f"{settings.attachments_dir}/{session_name}.wav"

        markdown = format_markdown(
            segments=segments,
            meta=TranscriptMeta(
                session_name=session_name,
                audio_rel_path=audio_rel_path,
                duration_seconds=float(info.get("duration", 0.0)),
                language=str(info.get("language", settings.language)),
                partial=bool(info.get("partial")),
            ),
            raw_segments=raw_segments,
        )

        md_path = save_markdown_to_vault(markdown, settings, session_name)
        report(f"Saved: {md_path}")

        if live_transcriber is not None:
            remove_live_markdown(settings, session_name)

        if do_summarize:
            _maybe_summarize(md_path, settings, report)

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
    if name is None:
        name = audio_path.stem
    validate_session_name(name)

    tmp_dir = Path(tempfile.mkdtemp(prefix="tapeback_"))

    with run_log(name, settings, on_status) as report:
        audio_dest = save_audio_to_vault(audio_path, settings, name)
        report(f"Audio saved: {audio_dest}")

        if is_stereo(audio_path):
            report("Stereo file detected, using dual-channel pipeline...")
            segments, info, raw_segments = process_stereo_file(
                audio_path, tmp_dir, settings, diarize=diarize, on_status=report
            )
        else:
            segments, info, raw_segments = process_mono_file(
                audio_path, tmp_dir, settings, diarize=diarize, on_status=report
            )

        audio_rel_path = f"{settings.attachments_dir}/{name}.wav"

        markdown = format_markdown(
            segments=segments,
            meta=TranscriptMeta(
                session_name=name,
                audio_rel_path=audio_rel_path,
                duration_seconds=float(info.get("duration", 0.0)),
                language=str(info.get("language", settings.language)),
                partial=bool(info.get("partial")),
            ),
            raw_segments=raw_segments,
        )

        md_path = save_markdown_to_vault(markdown, settings, name)
        report(f"Saved: {md_path}")

        if do_summarize:
            _maybe_summarize(md_path, settings, report)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return md_path


def is_stereo(audio_path: Path) -> bool:
    """Check if an audio file is a stereo WAV."""
    try:
        return get_channel_count(audio_path) == const.STEREO_CHANNELS
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
) -> tuple[list[Segment], dict[str, str | float], list[Segment] | None]:
    """Process a stereo WAV through the dual-channel pipeline.

    Returns (diarized_segments, info, raw_segments).
    raw_segments have basic You/Other attribution (channel-based, no diarization).
    """
    with stage_timer("load channels", on_status):
        mic_raw, monitor_raw, raw_sr = load_stereo_channels(stereo_path)

    on_status("Splitting channels...")
    with stage_timer("split", on_status):
        mic_16k, monitor_16k = split_channels_16k(stereo_path, output_dir)

    if settings.gate_mic_silence:
        # Silence the mic where the user only listens, so Whisper doesn't loop on it.
        with stage_timer("gate mic", on_status):
            gate_wav_inactive(mic_16k, mic_raw, monitor_raw, raw_sr)

    on_status("Transcribing (this may take a few minutes)...")
    with stage_timer("load model", on_status):
        transcriber = load_transcriber(settings)
    on_status(transcriber.describe())
    try:
        with sample_gpu(on_status, enabled=_gpu_telemetry_enabled(settings)):
            mic_segments, monitor_segments, info = transcriber.transcribe_stereo(
                mic_16k, monitor_16k, on_status=on_status
            )
    finally:
        # Release VRAM even when the stage raised, so a failure here does not starve
        # the diarizer that runs next.
        del transcriber
        free_gpu_memory()

    mic_segments = split_on_silence(
        mic_segments,
        mic_raw,
        raw_sr,
        settings.pause_threshold,
        monitor_samples=monitor_raw,
    )

    mic_segments = filter_silent_segments(mic_segments, mic_raw, raw_sr)
    monitor_segments = filter_silent_segments(monitor_segments, monitor_raw, raw_sr)

    # Drop mic segments where monitor is louder (headphone bleed, not real speech)
    mic_segments = [
        s
        for s in mic_segments
        if classify_segment_by_channel(s.start, s.end, mic_raw, monitor_raw, raw_sr) != "monitor"
    ]

    # Raw transcript: basic channel attribution only (You vs Other)
    raw_monitor = [
        Segment(start=s.start, end=s.end, text=s.text, words=s.words, speaker=const.SPEAKER_OTHER)
        for s in monitor_segments
    ]
    raw_segments = merge_channel_segments(mic_segments, raw_monitor)

    want_diarize = diarize and settings.diarize
    if want_diarize and not allows_local_diarize(settings.stt_backend, settings.stt_model):
        on_status(
            "Skipping local diarization — remote STT model "
            f"'{settings.stt_model}' provides text only or remote speakers."
        )
        want_diarize = False

    diarized = False
    if want_diarize and settings.hf_token.get_secret_value():
        if not diarization_available():
            on_status(
                "Warning: pyannote-audio not installed, skipping diarization. "
                "Install with: uv pip install tapeback[diarize]"
            )
        else:
            on_status("Diarizing speakers...")
            with stage_timer("diarize", on_status):
                diarizer = Diarizer(settings)
                diarization_segments = diarizer.diarize(monitor_16k)
                if settings.spectral_merge_threshold > 0:
                    diarization_segments = merge_similar_speakers(
                        diarization_segments,
                        monitor_raw,
                        raw_sr,
                        similarity_threshold=settings.spectral_merge_threshold,
                    )
                monitor_segments = assign_speakers(monitor_segments, diarization_segments)
            diarized = True

    # Remote diarize already set speaker labels; channel fallback only when unset.
    if not diarized and monitor_segments and monitor_segments[0].speaker is None:
        monitor_segments = [
            Segment(
                start=s.start,
                end=s.end,
                text=s.text,
                words=s.words,
                speaker=const.SPEAKER_OTHER,
            )
            for s in monitor_segments
        ]

    segments = merge_channel_segments(mic_segments, monitor_segments)
    # Without diarization, raw_segments == segments — skip the duplicate section.
    return segments, info, raw_segments if diarized else None


def process_mono_file(
    audio_path: Path,
    output_dir: Path,
    settings: Settings,
    *,
    diarize: bool,
    on_status: StatusCallback = _noop_status,
) -> tuple[list[Segment], dict[str, str | float], list[Segment] | None]:
    """Process a mono/non-stereo audio file through the single-channel pipeline.

    Returns (diarized_segments, info, raw_segments).
    """
    on_status("Converting audio...")
    with stage_timer("convert", on_status):
        mono_16k_path = convert_to_mono16k(audio_path, output_dir)

    on_status("Transcribing (this may take a few minutes)...")
    with stage_timer("load model", on_status):
        transcriber = load_transcriber(settings)
    on_status(transcriber.describe())
    try:
        with (
            sample_gpu(on_status, enabled=_gpu_telemetry_enabled(settings)),
            stage_timer("transcribe", on_status),
        ):
            segments, info = transcriber.transcribe(mono_16k_path, on_status=on_status)
    finally:
        # Release VRAM even when the stage raised — see the stereo path.
        del transcriber
        free_gpu_memory()

    # Raw transcript before diarization
    raw_segments = list(segments)

    stereo_for_attribution = _get_stereo_source(audio_path)
    segments_before = segments
    segments = _maybe_diarize_segments(
        segments,
        settings,
        mono_16k_path,
        stereo_for_attribution,
        diarize=diarize,
        on_status=on_status,
    )

    # _maybe_diarize_segments returns the same list reference when it skips
    # diarization — use identity to tell whether the raw section is a duplicate.
    diarized = segments is not segments_before
    return segments, info, raw_segments if diarized else None


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

    if not allows_local_diarize(settings.stt_backend, settings.stt_model):
        on_status(
            "Skipping local diarization — remote STT model "
            f"'{settings.stt_model}' provides text only or remote speakers."
        )
        return segments

    if not settings.hf_token.get_secret_value():
        on_status(
            "Warning: TAPEBACK_HF_TOKEN not set, skipping diarization. "
            "See README for setup instructions."
        )
        return segments

    if not diarization_available():
        on_status(
            "Warning: pyannote-audio not installed, skipping diarization. "
            "Install with: uv pip install tapeback[diarize]"
        )
        return segments

    on_status("Diarizing speakers...")
    with stage_timer("diarize", on_status):
        diarizer = Diarizer(settings)
        diarization_segments = diarizer.diarize(mono_16k_path)

        user_speaker = None
        if stereo_path is not None:
            user_speaker = identify_user_speaker(diarization_segments, stereo_path)

        result = assign_speakers(segments, diarization_segments, user_speaker, stereo_path)
    return result


def _get_stereo_source(audio_path: Path) -> Path | None:
    """Return audio_path if it's a stereo WAV, else None."""
    try:
        if get_channel_count(audio_path) == const.STEREO_CHANNELS:
            return audio_path
    except Exception:  # noqa: S110 — non-stereo or unreadable files are expected
        pass
    return None


def _maybe_summarize(md_path: Path, settings: Settings, on_status: StatusCallback) -> None:
    """Run summarization if available."""
    on_status("Summarizing...")
    with stage_timer("summarize", on_status):
        maybe_summarize(md_path, settings)
