import shutil
import wave
from unittest.mock import patch

import pytest

from tests.fixtures import create_silent_wav


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_merge_channels(tmp_path):
    """Merge two mono WAVs into stereo + 16kHz mono."""
    from meetrec.audio import merge_channels

    monitor = tmp_path / "monitor.wav"
    mic = tmp_path / "mic.wav"
    output_dir = tmp_path / "output"

    create_silent_wav(monitor)
    create_silent_wav(mic)

    stereo_path, mono_16k_path = merge_channels(monitor, mic, output_dir)

    assert stereo_path.exists()
    assert mono_16k_path.exists()

    # Verify stereo is 2 channels
    with wave.open(str(stereo_path), "rb") as wf:
        assert wf.getnchannels() == 2

    # Verify mono_16k is 1 channel at 16kHz
    with wave.open(str(mono_16k_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_convert_to_mono16k(tmp_path):
    """Convert a WAV file to 16kHz mono."""
    from meetrec.audio import convert_to_mono16k

    input_file = tmp_path / "input.wav"
    output_dir = tmp_path / "output"

    create_silent_wav(input_file, sample_rate=44100)

    result = convert_to_mono16k(input_file, output_dir)

    assert result.exists()
    with wave.open(str(result), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000


def test_empty_file_raises(tmp_path):
    """Empty WAV file should raise RuntimeError."""
    from meetrec.audio import merge_channels

    monitor = tmp_path / "monitor.wav"
    mic = tmp_path / "mic.wav"
    monitor.write_bytes(b"")
    mic.write_bytes(b"")

    with (
        patch("meetrec.audio.shutil.which", return_value="/usr/bin/ffmpeg"),
        pytest.raises(RuntimeError, match="No audio recorded"),
    ):
        merge_channels(monitor, mic, tmp_path / "output")


def test_ffmpeg_not_found(tmp_path):
    """Should give clear error when ffmpeg is not installed."""
    from meetrec.audio import merge_channels

    monitor = tmp_path / "monitor.wav"
    mic = tmp_path / "mic.wav"
    create_silent_wav(monitor)
    create_silent_wav(mic)

    with (
        patch("meetrec.audio.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="ffmpeg not found"),
    ):
        merge_channels(monitor, mic, tmp_path / "output")
