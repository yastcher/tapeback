"""Live transcription — background thread that transcribes audio during recording."""

from __future__ import annotations

import struct
import sys
import threading
import wave
from pathlib import Path

import numpy as np

from tapeback import const
from tapeback._gpu import free_gpu_memory
from tapeback._lazy import TranscriberLike, load_transcriber
from tapeback.formatter import format_live_markdown
from tapeback.models import Segment, Word
from tapeback.settings import Settings
from tapeback.vault import save_live_markdown

# Tolerance for deduplication: segments within this many seconds are considered duplicates
DEDUP_TOLERANCE_SEC = 0.5

# Bytes per sample for s16le mono
BYTES_PER_SAMPLE = 2


def find_data_offset(path: Path) -> int:
    """Find the byte offset where PCM data starts in a WAV file.

    Scans RIFF chunks to locate the 'data' chunk. Returns the byte position
    immediately after the data chunk header (i.e. where raw PCM bytes begin).

    Falls back to the standard 44-byte offset if parsing fails.
    """
    try:
        with open(path, "rb") as f:
            riff = f.read(4)
            if riff != b"RIFF":
                return const.WAV_HEADER_FALLBACK
            f.read(4)  # file size (unreliable for growing files)
            wave_id = f.read(4)
            if wave_id != b"WAVE":
                return const.WAV_HEADER_FALLBACK
            # Scan sub-chunks until we find "data"
            while True:
                chunk_id = f.read(const.WAV_CHUNK_HEADER_BYTES)
                if len(chunk_id) < const.WAV_CHUNK_HEADER_BYTES:
                    return const.WAV_HEADER_FALLBACK
                chunk_size_bytes = f.read(const.WAV_CHUNK_HEADER_BYTES)
                if len(chunk_size_bytes) < const.WAV_CHUNK_HEADER_BYTES:
                    return const.WAV_HEADER_FALLBACK
                if chunk_id == b"data":
                    return f.tell()
                (chunk_size,) = struct.unpack("<I", chunk_size_bytes)
                f.seek(chunk_size, 1)
    except OSError:
        return const.WAV_HEADER_FALLBACK


def resample_48k_to_16k(pcm_bytes: bytes) -> np.ndarray:
    """Downsample raw s16le PCM from 48 kHz to 16 kHz.

    Simple decimation by factor 3 (no anti-aliasing filter).
    Adequate quality for a live preview — the final pipeline uses ffmpeg with loudnorm.
    """
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return samples[:: const.RESAMPLE_FACTOR]


def adjust_timestamps(segments: list[Segment], offset_seconds: float) -> list[Segment]:
    """Shift all segment and word timestamps by offset_seconds."""
    result: list[Segment] = []
    for seg in segments:
        words: list[Word] | None = None
        if seg.words:
            words = [
                Word(
                    start=w.start + offset_seconds,
                    end=w.end + offset_seconds,
                    word=w.word,
                    probability=w.probability,
                )
                for w in seg.words
            ]
        result.append(
            Segment(
                start=seg.start + offset_seconds,
                end=seg.end + offset_seconds,
                text=seg.text,
                words=words,
                speaker=seg.speaker,
            )
        )
    return result


def deduplicate_overlap(
    existing: list[Segment],
    new_segments: list[Segment],
    overlap_start: float,
) -> list[Segment]:
    """Remove segments from new_segments that duplicate existing ones in the overlap zone.

    A new segment is considered a duplicate if its start time is within
    DEDUP_TOLERANCE_SEC of any existing segment's start time AND it falls
    within the overlap region (before overlap_start + tolerance).
    """
    if not existing or overlap_start <= 0:
        return new_segments

    existing_starts = {s.start for s in existing}

    kept: list[Segment] = []
    for seg in new_segments:
        # Segments clearly past the overlap zone — always keep
        if seg.start >= overlap_start + DEDUP_TOLERANCE_SEC:
            kept.append(seg)
            continue
        # Check if this segment duplicates an existing one
        is_dup = any(abs(seg.start - es) < DEDUP_TOLERANCE_SEC for es in existing_starts)
        if not is_dup:
            kept.append(seg)
    return kept


class LiveTranscriber:
    """Background transcription thread that runs during recording.

    Periodically reads new audio from growing WAV files written by parecord,
    transcribes both channels (mic -> "You", monitor -> "Other"),
    and writes a live markdown transcript to the Obsidian vault.
    """

    def __init__(
        self,
        settings: Settings,
        session_name: str,
        mic_path: Path,
        monitor_path: Path,
    ) -> None:
        self._settings = settings
        self._session_name = session_name
        self._mic_path = mic_path
        self._monitor_path = monitor_path

        self._mic_data_offset: int | None = None  # parsed lazily on first read
        self._monitor_data_offset: int | None = None
        self._mic_byte_offset = 0  # bytes of PCM data already processed
        self._monitor_byte_offset = 0
        self._segments: list[Segment] = []

        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._transcription_loop,
            name="live-transcriber",
            daemon=True,
        )

        self._live_md_path = (
            settings.vault_path
            / settings.meetings_dir
            / f"{session_name}{const.FILE_LIVE_SUFFIX}.md"
        )

        # Backend is created lazily on the first chunk to avoid blocking
        # the main thread with a model / SDK load.
        self._transcriber: TranscriberLike | None = None

    @property
    def live_md_path(self) -> Path:
        return self._live_md_path

    def start(self) -> None:
        """Start the background transcription thread."""
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread, process final chunk, free GPU memory."""
        self._stop_event.set()
        self._thread.join(timeout=120)

        # Free GPU memory so the full pipeline can use it (no-op for remote STT).
        if self._transcriber is not None:
            del self._transcriber
            self._transcriber = None
            free_gpu_memory()

    def _ensure_transcriber(self) -> TranscriberLike:
        """Lazily create the configured transcription backend."""
        if self._transcriber is None:
            self._transcriber = load_transcriber(self._settings)
        return self._transcriber

    def _transcription_loop(self) -> None:
        """Main loop: wait for interval, then process a chunk."""
        # Write initial "waiting" markdown
        self._write_live_markdown()

        while not self._stop_event.wait(timeout=self._settings.live_interval):
            try:
                self._process_chunk()
            except Exception:
                import traceback  # noqa: PLC0415 — only on error

                print(
                    f"Warning: Live transcription error:\n{traceback.format_exc()}",
                    file=sys.stderr,
                )

        # Process final chunk on stop
        try:
            self._process_chunk()
        except Exception:
            import traceback  # noqa: PLC0415 — only on error

            print(
                f"Warning: Live transcription final chunk error:\n{traceback.format_exc()}",
                file=sys.stderr,
            )

    def _process_chunk(self) -> None:
        """Read new audio from both channels, transcribe, update markdown."""
        min_bytes = int(
            self._settings.live_min_chunk * self._settings.sample_rate * BYTES_PER_SAMPLE
        )
        overlap_bytes = int(
            self._settings.live_overlap * self._settings.sample_rate * BYTES_PER_SAMPLE
        )

        mic_pcm, mic_new_offset = self._read_new_pcm(
            self._mic_path,
            self._mic_byte_offset,
            min_bytes,
            overlap_bytes,
            is_mic=True,
        )
        monitor_pcm, monitor_new_offset = self._read_new_pcm(
            self._monitor_path,
            self._monitor_byte_offset,
            min_bytes,
            overlap_bytes,
            is_mic=False,
        )

        if mic_pcm is None and monitor_pcm is None:
            return

        transcriber = self._ensure_transcriber()
        new_segments: list[Segment] = []

        if mic_pcm is not None:
            mic_segs = self._transcribe_chunk(
                transcriber, mic_pcm, self._mic_byte_offset, overlap_bytes, is_mic=True
            )
            new_segments.extend(mic_segs)
            self._mic_byte_offset = mic_new_offset

        if monitor_pcm is not None:
            monitor_segs = self._transcribe_chunk(
                transcriber,
                monitor_pcm,
                self._monitor_byte_offset,
                overlap_bytes,
                is_mic=False,
            )
            new_segments.extend(monitor_segs)
            self._monitor_byte_offset = monitor_new_offset

        if new_segments:
            self._segments.extend(new_segments)
            self._segments.sort(key=lambda s: s.start)

        self._write_live_markdown()

    def _read_new_pcm(
        self,
        wav_path: Path,
        byte_offset: int,
        min_bytes: int,
        overlap_bytes: int,
        *,
        is_mic: bool,
    ) -> tuple[bytes | None, int]:
        """Read new raw PCM bytes from a growing WAV file.

        Returns (pcm_bytes_including_overlap, new_byte_offset) or (None, byte_offset)
        if not enough new data.
        """
        if not wav_path.exists():
            return None, byte_offset

        # Parse data offset lazily (once per file)
        if is_mic:
            if self._mic_data_offset is None:
                self._mic_data_offset = find_data_offset(wav_path)
            data_offset = self._mic_data_offset
        else:
            if self._monitor_data_offset is None:
                self._monitor_data_offset = find_data_offset(wav_path)
            data_offset = self._monitor_data_offset

        file_size = wav_path.stat().st_size
        available_pcm = file_size - data_offset
        new_bytes = available_pcm - byte_offset

        if new_bytes < min_bytes:
            return None, byte_offset

        # Include overlap from previous chunk
        read_start = max(0, byte_offset - overlap_bytes)
        read_length = available_pcm - read_start

        with open(wav_path, "rb") as f:
            f.seek(data_offset + read_start)
            pcm_bytes = f.read(read_length)

        # Ensure even number of bytes (s16le = 2 bytes per sample)
        if len(pcm_bytes) % BYTES_PER_SAMPLE != 0:
            pcm_bytes = pcm_bytes[: len(pcm_bytes) - (len(pcm_bytes) % BYTES_PER_SAMPLE)]

        new_offset = available_pcm
        return pcm_bytes, new_offset

    def _transcribe_chunk(
        self,
        transcriber: TranscriberLike,
        pcm_bytes: bytes,
        byte_offset: int,
        overlap_bytes: int,
        *,
        is_mic: bool,
    ) -> list[Segment]:
        """Resample, write temp WAV, transcribe, adjust timestamps, deduplicate."""
        samples_16k = resample_48k_to_16k(pcm_bytes)
        if len(samples_16k) == 0:
            return []

        # Write temp WAV for faster-whisper
        suffix = "mic" if is_mic else "monitor"
        chunk_path = self._mic_path.parent / f"chunk_{suffix}.wav"
        self._write_chunk_wav(samples_16k, chunk_path)

        # Transcribe
        segments, _info = transcriber.transcribe(chunk_path)

        # Clean up temp file
        chunk_path.unlink(missing_ok=True)

        # Calculate absolute time offset
        read_start = max(0, byte_offset - overlap_bytes)
        chunk_start_seconds = read_start / (self._settings.sample_rate * BYTES_PER_SAMPLE)

        # Adjust timestamps to absolute
        segments = adjust_timestamps(segments, chunk_start_seconds)

        # Assign speaker
        speaker = const.SPEAKER_YOU if is_mic else const.SPEAKER_OTHER
        segments = [
            Segment(
                start=s.start,
                end=s.end,
                text=s.text,
                words=s.words,
                speaker=speaker,
            )
            for s in segments
        ]

        # Deduplicate overlap with existing segments
        overlap_boundary = byte_offset / (self._settings.sample_rate * BYTES_PER_SAMPLE)
        segments = deduplicate_overlap(self._segments, segments, overlap_boundary)

        return segments

    @staticmethod
    def _write_chunk_wav(samples_16k: np.ndarray, path: Path) -> None:
        """Write a valid 16 kHz mono WAV file from int16 samples."""
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(const.SAMPLE_RATE_16K)
            wf.writeframes(samples_16k.tobytes())

    def _write_live_markdown(self) -> None:
        """Write (or overwrite) the live markdown file in the vault."""
        markdown = format_live_markdown(
            self._segments,
            self._session_name,
            self._settings.language,
        )
        save_live_markdown(markdown, self._settings, self._session_name)
