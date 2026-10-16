"""Regression tests for keeping work when transcription is interrupted."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tapeback.formatter import TranscriptMeta, format_markdown
from tapeback.models import Segment
from tapeback.transcriber import Transcriber


def _info(duration: float = 600.0):
    info = MagicMock()
    info.language, info.language_probability, info.duration = "ru", 0.9, duration
    return info


def _segment(start: float, end: float, text: str):
    seg = MagicMock()
    seg.start, seg.end, seg.text, seg.words = start, end, text, []
    return seg


@pytest.fixture
def gpu_ready(monkeypatch):
    monkeypatch.setattr("tapeback._whisper_device.wait_for_clamp_release", lambda *_a, **_k: True)
    monkeypatch.setattr("tapeback._whisper_device.get_free_vram_mib", lambda: 4096)


def _interrupting_segments(count_before_interrupt: int):
    """Yield a few segments, then raise KeyboardInterrupt mid-iteration."""

    def _gen():
        for i in range(count_before_interrupt):
            yield _segment(i * 10.0, i * 10.0 + 9.0, f"реплика {i}")
        raise KeyboardInterrupt

    return _gen()


def test_interrupt_keeps_the_segments_already_decoded(settings, gpu_ready):
    """Ctrl+C must not throw away hours of finished work.

    Bug: the interrupt propagated out of the segment loop, so everything decoded so
    far was discarded. Sixteen recordings ended up with no transcript this way — a
    run that had been going for over two hours produced nothing at all.
    """
    s = settings.model_copy(update={"device": "cuda"})

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.return_value = (_interrupting_segments(3), _info())

        segments, info = Transcriber(s).transcribe(Path("/fake/audio.wav"))

    assert len(segments) == 3
    assert segments[0].text == "реплика 0"
    assert info["partial"] is True


def test_uninterrupted_run_is_not_marked_partial(settings, gpu_ready):
    s = settings.model_copy(update={"device": "cuda"})

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.return_value = (iter([_segment(0.0, 5.0, "готово")]), _info())

        segments, info = Transcriber(s).transcribe(Path("/fake/audio.wav"))

    assert len(segments) == 1
    assert info["partial"] is False


def test_interrupt_in_the_second_channel_keeps_the_first(settings, gpu_ready):
    """Monitor runs first; losing it because the mic was interrupted wastes the lot."""
    s = settings.model_copy(update={"device": "cuda"})

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.side_effect = [
            (iter([_segment(0.0, 5.0, "монитор")]), _info()),
            (_interrupting_segments(2), _info()),
        ]

        mic, monitor, info = Transcriber(s).transcribe_stereo(
            Path("/fake/mic.wav"), Path("/fake/monitor.wav")
        )

    assert len(monitor) == 1
    assert len(mic) == 2
    assert info["partial"] is True


def test_interrupt_in_the_first_channel_skips_the_second(settings, gpu_ready):
    """Ctrl+C means stop, so the remaining channel must not be started."""
    s = settings.model_copy(update={"device": "cuda"})

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.side_effect = [
            (_interrupting_segments(1), _info()),
            (iter([_segment(0.0, 5.0, "never reached")]), _info()),
        ]

        mic, monitor, info = Transcriber(s).transcribe_stereo(
            Path("/fake/mic.wav"), Path("/fake/monitor.wav")
        )

    assert len(monitor) == 1
    assert mic == []
    assert info["partial"] is True
    assert instance.transcribe.call_count == 1


def test_partial_transcript_is_marked_in_the_note():
    """A partial transcript that reads as complete is worse than none — nothing
    would prompt a re-run."""
    markdown = format_markdown(
        segments=[Segment(start=0.0, end=5.0, text="реплика.", words=None, speaker="You")],
        meta=TranscriptMeta(
            session_name="2026-08-06_12-00-00",
            audio_rel_path="attachments/audio/x.wav",
            duration_seconds=1800.0,
            language="ru",
            partial=True,
        ),
    )

    assert "partial: true" in markdown
    assert "  - partial" in markdown
    assert "Interrupted" in markdown


def test_complete_transcript_carries_no_partial_marker():
    markdown = format_markdown(
        segments=[Segment(start=0.0, end=5.0, text="реплика.", words=None, speaker="You")],
        meta=TranscriptMeta(
            session_name="2026-08-06_12-00-00",
            audio_rel_path="attachments/audio/x.wav",
            duration_seconds=5.0,
            language="ru",
        ),
    )

    assert "partial" not in markdown
    assert "Interrupted" not in markdown


def test_second_interrupt_still_propagates(settings, gpu_ready):
    """Swallowing the first must not trap the user — a second Ctrl+C has to work."""
    s = settings.model_copy(update={"device": "cuda"})

    def _double_interrupt():
        yield _segment(0.0, 5.0, "one")
        raise KeyboardInterrupt

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.return_value = (_double_interrupt(), _info())
        transcriber = Transcriber(s)
        transcriber.transcribe(Path("/fake/audio.wav"))

        # A later interrupt (e.g. during saving) is a fresh one and must not be eaten.
        instance.transcribe.return_value = (_double_interrupt(), _info())
        with (
            patch.object(transcriber, "_collect_segments", side_effect=KeyboardInterrupt),
            pytest.raises(KeyboardInterrupt),
        ):
            transcriber.transcribe(Path("/fake/audio.wav"))
