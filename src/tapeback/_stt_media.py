"""ffmpeg helpers for remote STT uploads (encode, slice, duration)."""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path
from typing import NamedTuple

from tapeback import const

_MAX_UPLOAD = const.STT_REMOTE_MAX_UPLOAD_BYTES
_BITRATE_K = const.STT_REMOTE_MP3_BITRATE_K
_UPLOAD_MARGIN = const.STT_REMOTE_UPLOAD_MARGIN
_FFMPEG_ERROR_TAIL = const.STT_REMOTE_FFMPEG_ERROR_TAIL


class _UploadJob(NamedTuple):
    """One MP3 upload: source WAV, dest MP3, timeline offset, and status labels."""

    wav_path: Path
    mp3_path: Path
    time_offset: float
    duration: float
    language: str | None
    stage: str
    chunk_label: str | None


def _wav_duration(path: Path) -> float:
    """Return WAV duration in seconds."""
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _check_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install: sudo apt install ffmpeg")


def _encode_mp3(wav_path: Path, mp3_path: Path) -> None:
    """Encode a mono WAV to MP3 for upload size."""
    _check_ffmpeg()
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            f"{_BITRATE_K}k",
            str(mp3_path),
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")[-_FFMPEG_ERROR_TAIL:]
        raise RuntimeError(f"ffmpeg failed to encode MP3 for remote STT upload:\n{err}")


def _max_chunk_seconds() -> float:
    """Longest audio that fits under the upload cap at the chosen bitrate."""
    bytes_per_sec = (_BITRATE_K * 1000) / 8
    return (_MAX_UPLOAD / bytes_per_sec) * _UPLOAD_MARGIN


def _slice_wav(src: Path, dest: Path, start: float, duration: float) -> None:
    """Write a time slice of src WAV to dest via ffmpeg."""
    _check_ffmpeg()
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(src),
            "-c",
            "copy",
            str(dest),
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")[-_FFMPEG_ERROR_TAIL:]
        raise RuntimeError(f"ffmpeg failed to slice WAV for remote STT upload:\n{err}")
