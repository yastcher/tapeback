"""Tests for reusing an already-transcribed channel."""

import os
from unittest.mock import MagicMock, patch

import pytest

from tapeback import _resume
from tapeback.models import Segment, Word
from tapeback.transcriber import Transcriber


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "monitor_16k.wav"
    path.write_bytes(b"not really audio, but it has a size and an mtime")
    return path


@pytest.fixture
def cached_settings(settings, tmp_path):
    return settings.model_copy(
        update={"device": "cpu", "resume_cache": True, "resume_cache_dir": tmp_path / "resume"}
    )


def _info(partial: bool = False):
    info = MagicMock()
    info.language, info.language_probability, info.duration = "ru", 0.9, 60.0
    return info


def _whisper_segment(start: float, end: float, text: str):
    seg = MagicMock()
    seg.start, seg.end, seg.text = start, end, text
    word = MagicMock()
    word.start, word.end, word.word, word.probability = start, end, text, 0.8
    seg.words = [word]
    return seg


def test_key_changes_when_the_audio_changes(cached_settings, audio, tmp_path):
    first = _resume.resume_key(audio, cached_settings, "transcribe monitor")
    audio.write_bytes(b"different content, different size")
    second = _resume.resume_key(audio, cached_settings, "transcribe monitor")
    assert first is not None
    assert first != second


def test_key_changes_per_channel(cached_settings, audio):
    assert _resume.resume_key(audio, cached_settings, "transcribe mic") != _resume.resume_key(
        audio, cached_settings, "transcribe monitor"
    )


@pytest.mark.parametrize(
    "field",
    ["stt_model", "compute_type", "beam_size", "chunk_length", "hotwords", "language"],
)
def test_key_changes_when_an_output_affecting_setting_changes(cached_settings, audio, field):
    """A cached channel is only reusable if it would be produced the same way."""
    base = _resume.resume_key(audio, cached_settings, "transcribe")
    changed = {
        "stt_model": "tiny",
        "compute_type": "float32",
        "beam_size": 1,
        "chunk_length": 7,
        "hotwords": "different, glossary",
        "language": "de",
    }[field]
    other = cached_settings.model_copy(update={field: changed})
    assert base != _resume.resume_key(audio, other, "transcribe")


def test_key_is_none_for_missing_audio(cached_settings, tmp_path):
    assert _resume.resume_key(tmp_path / "gone.wav", cached_settings, "transcribe") is None


def test_round_trip_preserves_segments_and_words(cached_settings, audio, tmp_path):
    key = _resume.resume_key(audio, cached_settings, "transcribe")
    assert key is not None
    directory = tmp_path / "resume"
    segments = [
        Segment(
            start=0.0,
            end=5.0,
            text="реплика",
            speaker="You",
            words=[Word(start=0.0, end=5.0, word="реплика", probability=0.77)],
        )
    ]

    _resume.store(key, directory, segments, {"language": "ru", "duration": 5.0})
    restored = _resume.load(key, directory)

    assert restored is not None
    loaded, info = restored
    assert loaded == segments
    assert loaded[0].words is not None
    assert loaded[0].words[0].probability == 0.77
    assert info["language"] == "ru"


def test_corrupt_entry_is_ignored(cached_settings, audio, tmp_path):
    """A half-written cache file must cost a redo, not a failed run."""
    key = _resume.resume_key(audio, cached_settings, "transcribe")
    assert key is not None
    directory = tmp_path / "resume"
    directory.mkdir()
    (directory / key.filename).write_text("{ this is not json")

    assert _resume.load(key, directory) is None


def test_second_run_reuses_the_first(cached_settings, audio):
    """The point: an interrupted run must not redo a channel it already finished."""
    messages: list[str] = []
    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.return_value = (
            iter([_whisper_segment(0.0, 5.0, "готово")]),
            _info(),
        )
        first, _ = Transcriber(cached_settings).transcribe(audio, stage="transcribe monitor")

        # A second transcriber, as a re-run would build: Whisper must not be called.
        instance.transcribe.reset_mock()
        second, _ = Transcriber(cached_settings).transcribe(
            audio, stage="transcribe monitor", on_status=messages.append
        )

    assert [s.text for s in second] == [s.text for s in first]
    instance.transcribe.assert_not_called()
    assert any("Reusing" in m for m in messages)


def test_partial_results_are_not_cached(cached_settings, audio, tmp_path):
    """Caching a truncated channel would make the next run call it finished."""

    def _interrupting():
        yield _whisper_segment(0.0, 5.0, "начало")
        raise KeyboardInterrupt

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.return_value = (_interrupting(), _info())
        segments, info = Transcriber(cached_settings).transcribe(audio, stage="transcribe")

    assert info["partial"] is True
    assert len(segments) == 1
    assert list((tmp_path / "resume").glob("*.json")) == []


def test_cache_can_be_disabled(settings, audio, tmp_path):
    s = settings.model_copy(
        update={"device": "cpu", "resume_cache": False, "resume_cache_dir": tmp_path / "resume"}
    )
    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.return_value = (iter([_whisper_segment(0.0, 5.0, "x")]), _info())
        Transcriber(s).transcribe(audio)

    assert not (tmp_path / "resume").exists()


def test_prune_keeps_the_newest(tmp_path):
    directory = tmp_path / "resume"
    directory.mkdir()
    for index in range(5):
        entry = directory / f"{index}.json"
        entry.write_text("{}")
        stamp = (index + 1) * 1000
        os.utime(entry, (stamp, stamp))

    _resume._prune(directory, keep=3)

    remaining = sorted(p.name for p in directory.glob("*.json"))
    assert remaining == ["2.json", "3.json", "4.json"]


def test_default_dir_honours_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert _resume.default_resume_dir() == tmp_path / "xdg" / "tapeback" / "resume"
