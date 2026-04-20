"""Live transcription tests — LiveTranscriber, WAV parsing, resampling, dedup."""

import struct
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tapeback.live import (
    LiveTranscriber,
    adjust_timestamps,
    deduplicate_overlap,
    find_data_offset,
    resample_48k_to_16k,
)
from tapeback.models import Segment, Word
from tapeback.settings import Settings
from tests.fixtures import create_mono_wav, mock_whisper_transcribe

# --- find_data_offset ---


def test_find_data_offset_standard_wav(tmp_path):
    """Standard 44-byte WAV header: data starts at byte 44."""
    wav_path = tmp_path / "test.wav"
    create_mono_wav(wav_path, duration=0.1, sample_rate=16000)

    offset = find_data_offset(wav_path)
    assert offset == 44


def test_find_data_offset_extended_wav(tmp_path):
    """WAV with extra chunks before 'data' should still find correct offset."""
    wav_path = tmp_path / "extended.wav"

    # Build a WAV with an extra "INFO" chunk before "data"
    with open(wav_path, "wb") as f:
        # RIFF header (placeholder size)
        f.write(b"RIFF")
        f.write(struct.pack("<I", 0xFFFFFFFF))
        f.write(b"WAVE")

        # fmt chunk (standard PCM, mono, 16kHz, 16-bit)
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16))

        # Extra "INFO" chunk (4 bytes of padding)
        f.write(b"INFO")
        f.write(struct.pack("<I", 4))
        f.write(b"\x00\x00\x00\x00")

        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", 100))
        data_start = f.tell()
        f.write(b"\x00" * 100)

    offset = find_data_offset(wav_path)
    assert offset == data_start


def test_find_data_offset_nonexistent_file(tmp_path):
    """Non-existent file falls back to 44."""
    offset = find_data_offset(tmp_path / "nope.wav")
    assert offset == 44


def test_find_data_offset_not_riff(tmp_path):
    """Non-RIFF file falls back to 44."""
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"NOT_RIFF_DATA" * 10)
    assert find_data_offset(bad) == 44


# --- resample_48k_to_16k ---


def test_resample_48k_to_16k_length():
    """Output length should be input_length / 3."""
    # 4800 samples at 48kHz = 0.1s → 1600 samples at 16kHz
    samples_48k = np.zeros(4800, dtype=np.int16)
    result = resample_48k_to_16k(samples_48k.tobytes())
    assert len(result) == 1600


def test_resample_48k_to_16k_values():
    """Decimation picks every 3rd sample."""
    samples = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int16)
    result = resample_48k_to_16k(samples.tobytes())
    np.testing.assert_array_equal(result, [1, 4, 7])


# --- adjust_timestamps ---


def test_adjust_timestamps_offsets_segments_and_words():
    """All timestamps should be shifted by offset_seconds."""
    segments = [
        Segment(
            start=0.0,
            end=2.0,
            text="Hello",
            words=[
                Word(start=0.0, end=0.5, word="Hello", probability=0.9),
            ],
            speaker="You",
        ),
        Segment(start=2.0, end=4.0, text="World", words=None, speaker="Other"),
    ]

    result = adjust_timestamps(segments, 60.0)

    assert result[0].start == pytest.approx(60.0)
    assert result[0].end == pytest.approx(62.0)
    assert result[0].words is not None
    assert result[0].words[0].start == pytest.approx(60.0)
    assert result[0].words[0].end == pytest.approx(60.5)
    assert result[0].speaker == "You"

    assert result[1].start == pytest.approx(62.0)
    assert result[1].end == pytest.approx(64.0)
    assert result[1].words is None


def test_adjust_timestamps_preserves_text():
    """Text and speaker should not change."""
    segments = [Segment(start=0.0, end=1.0, text="Test", speaker="You")]
    result = adjust_timestamps(segments, 10.0)
    assert result[0].text == "Test"
    assert result[0].speaker == "You"


# --- deduplicate_overlap ---


def test_deduplicate_overlap_removes_duplicates():
    """Segments matching existing ones in the overlap zone should be removed."""
    existing = [
        Segment(start=55.0, end=58.0, text="Old segment", speaker="You"),
        Segment(start=58.0, end=60.0, text="Boundary", speaker="Other"),
    ]

    new_segments = [
        # Duplicate of existing (within tolerance)
        Segment(start=55.1, end=58.0, text="Old segment", speaker="You"),
        Segment(start=58.2, end=60.0, text="Boundary", speaker="Other"),
        # New segment past overlap
        Segment(start=62.0, end=65.0, text="New content", speaker="You"),
    ]

    result = deduplicate_overlap(existing, new_segments, overlap_start=60.0)

    # Only the new segment past the overlap should remain
    assert len(result) == 1
    assert result[0].text == "New content"


def test_deduplicate_overlap_no_existing():
    """With no existing segments, all new ones are kept."""
    new_segments = [
        Segment(start=0.0, end=2.0, text="First", speaker="You"),
    ]
    result = deduplicate_overlap([], new_segments, overlap_start=0.0)
    assert len(result) == 1


def test_deduplicate_overlap_zero_overlap():
    """With overlap_start=0, all segments are kept."""
    existing = [Segment(start=0.0, end=2.0, text="Old", speaker="You")]
    new_segments = [
        Segment(start=0.0, end=2.0, text="Old", speaker="You"),
        Segment(start=2.0, end=4.0, text="New", speaker="You"),
    ]
    result = deduplicate_overlap(existing, new_segments, overlap_start=0.0)
    assert len(result) == 2


# --- LiveTranscriber ---


def test_live_transcriber_start_stop_lifecycle(tmp_path, tmp_vault):
    """LiveTranscriber should start a thread, process final chunk on stop, and clean up."""
    settings = Settings(vault_path=tmp_vault, live_interval=1, live_min_chunk=0.01)

    mic_path = tmp_path / "mic.wav"
    monitor_path = tmp_path / "monitor.wav"
    create_mono_wav(mic_path, duration=0.5, sample_rate=48000)
    create_mono_wav(monitor_path, duration=0.5, sample_rate=48000)

    mock_model = mock_whisper_transcribe([(0.0, 0.5, "Test speech.")])

    with patch("tapeback.transcriber.WhisperModel", return_value=mock_model):
        lt = LiveTranscriber(settings, "2026-04-18_10-00-00", mic_path, monitor_path)
        lt.start()
        # Let it run briefly — the thread will pick up the audio
        lt._stop_event.wait(timeout=0.1)
        lt.stop()

    # Live markdown should have been written
    live_md = tmp_vault / "meetings" / "2026-04-18_10-00-00_live.md"
    assert live_md.exists()


def test_live_transcriber_no_crash_on_empty_audio(tmp_path, tmp_vault):
    """LiveTranscriber should not crash when WAV files don't exist yet."""
    settings = Settings(vault_path=tmp_vault, live_interval=1, live_min_chunk=0.01)

    mic_path = tmp_path / "mic.wav"
    monitor_path = tmp_path / "monitor.wav"
    # Files don't exist!

    lt = LiveTranscriber(settings, "test-session", mic_path, monitor_path)
    lt.start()
    lt._stop_event.wait(timeout=0.1)
    lt.stop()

    # Should still write the "waiting" markdown
    live_md = tmp_vault / "meetings" / "test-session_live.md"
    assert live_md.exists()
    assert "Waiting for first transcription cycle" in live_md.read_text()


def test_live_transcriber_no_crash_on_transcription_error(tmp_path, tmp_vault):
    """LiveTranscriber should catch transcription errors and continue."""
    settings = Settings(vault_path=tmp_vault, live_interval=1, live_min_chunk=0.01)

    mic_path = tmp_path / "mic.wav"
    monitor_path = tmp_path / "monitor.wav"
    create_mono_wav(mic_path, duration=1.0, sample_rate=48000)
    create_mono_wav(monitor_path, duration=1.0, sample_rate=48000)

    mock_model = MagicMock()
    mock_model.transcribe.side_effect = RuntimeError("CUDA out of memory")

    with patch("tapeback.transcriber.WhisperModel", return_value=mock_model):
        lt = LiveTranscriber(settings, "error-session", mic_path, monitor_path)
        lt.start()
        lt._stop_event.wait(timeout=0.1)
        lt.stop()

    # Should not crash — live markdown still written (empty segments)
    live_md = tmp_vault / "meetings" / "error-session_live.md"
    assert live_md.exists()


def test_live_transcriber_process_chunk_accumulates_segments(tmp_path, tmp_vault):
    """_process_chunk should accumulate segments from transcription."""
    settings = Settings(
        vault_path=tmp_vault,
        live_interval=60,
        live_min_chunk=0.01,
        live_overlap=0.0,
        sample_rate=48000,
    )

    mic_path = tmp_path / "mic.wav"
    monitor_path = tmp_path / "monitor.wav"
    create_mono_wav(mic_path, duration=1.0, sample_rate=48000)
    create_mono_wav(monitor_path, duration=1.0, sample_rate=48000)

    mock_model = mock_whisper_transcribe([(0.0, 1.0, "Hello world.")])

    with patch("tapeback.transcriber.WhisperModel", return_value=mock_model):
        lt = LiveTranscriber(settings, "chunk-test", mic_path, monitor_path)
        lt._process_chunk()

    # Should have accumulated segments from both channels
    assert len(lt._segments) > 0
    speakers = {s.speaker for s in lt._segments}
    assert "You" in speakers
    assert "Other" in speakers


def test_live_markdown_written_atomically(tmp_path, tmp_vault):
    """Live markdown should be written via atomic temp+rename pattern."""
    settings = Settings(vault_path=tmp_vault, live_interval=60)

    mic_path = tmp_path / "mic.wav"
    monitor_path = tmp_path / "monitor.wav"

    lt = LiveTranscriber(settings, "atomic-test", mic_path, monitor_path)
    lt._write_live_markdown()

    live_md = tmp_vault / "meetings" / "atomic-test_live.md"
    assert live_md.exists()
    # No leftover .tmp file
    assert not live_md.with_suffix(".md.tmp").exists()


def test_write_chunk_wav_creates_valid_wav(tmp_path):
    """_write_chunk_wav should create a readable 16kHz mono WAV."""
    samples = np.array([0, 100, -100, 200, -200], dtype=np.int16)
    chunk_path = tmp_path / "chunk.wav"

    LiveTranscriber._write_chunk_wav(samples, chunk_path)

    with wave.open(str(chunk_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2
        assert wf.getnframes() == 5
