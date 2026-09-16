"""Remote STT ffmpeg upload helpers."""

import shutil
from unittest.mock import MagicMock, patch

import pytest

from tapeback import const
from tapeback._stt_media import (
    _check_ffmpeg,
    _encode_mp3,
    _max_chunk_seconds,
    _slice_wav,
    _wav_duration,
)
from tests.fixtures import create_silent_wav


def test_wav_duration(tmp_path):
    path = tmp_path / "clip.wav"
    create_silent_wav(path, duration=2.5, sample_rate=16000)
    assert _wav_duration(path) == 2.5


def test_max_chunk_seconds_matches_const_formula():
    bytes_per_sec = (const.STT_REMOTE_MP3_BITRATE_K * 1000) / 8
    expected = (const.STT_REMOTE_MAX_UPLOAD_BYTES / bytes_per_sec) * const.STT_REMOTE_UPLOAD_MARGIN
    assert _max_chunk_seconds() == expected


def test_check_ffmpeg_missing_raises():
    with (
        patch("tapeback._stt_media.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="ffmpeg not found"),
    ):
        _check_ffmpeg()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_encode_mp3_and_slice(tmp_path):
    src = tmp_path / "src.wav"
    create_silent_wav(src, duration=1.0, sample_rate=16000)
    mp3 = tmp_path / "out.mp3"
    _encode_mp3(src, mp3)
    assert mp3.exists()
    assert mp3.stat().st_size > 0

    sliced = tmp_path / "slice.wav"
    _slice_wav(src, sliced, start=0.0, duration=0.5)
    # stream-copy can snap to container frames; existence + non-empty proves the path.
    assert sliced.exists()
    assert sliced.stat().st_size > 0


def test_encode_mp3_failure_raises(tmp_path):
    src = tmp_path / "src.wav"
    create_silent_wav(src, duration=0.5, sample_rate=16000)
    mp3 = tmp_path / "out.mp3"
    failed = MagicMock(returncode=1, stderr=b"boom")
    with (
        patch("tapeback._stt_media._check_ffmpeg"),
        patch("tapeback._stt_media.subprocess.run", return_value=failed),
        pytest.raises(RuntimeError, match="ffmpeg failed to encode MP3"),
    ):
        _encode_mp3(src, mp3)
