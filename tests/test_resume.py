"""Tests for reusing an already-transcribed channel."""

import os
from pathlib import Path
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


def test_key_follows_the_audios_path_size_and_mtime_each(cached_settings, audio, tmp_path):
    stamp = (1_700_000_000, 1_700_000_000)
    os.utime(audio, stamp)
    base = _resume.resume_key(audio, cached_settings, "transcribe")
    assert base is not None
    assert len(base.digest) == 32

    twin = tmp_path / "twin.wav"
    twin.write_bytes(audio.read_bytes())
    os.utime(twin, stamp)
    assert _resume.resume_key(twin, cached_settings, "transcribe") != base

    os.utime(audio, (stamp[0] + 1, stamp[1] + 1))
    assert _resume.resume_key(audio, cached_settings, "transcribe") != base

    audio.write_bytes(audio.read_bytes() + b"!")
    os.utime(audio, stamp)
    assert _resume.resume_key(audio, cached_settings, "transcribe") != base


def test_a_chunk_key_is_as_long_as_a_channel_key(cached_settings, audio):
    key = _resume.chunk_resume_key(
        audio, cached_settings, "transcribe", chunk_index=2, time_offset=600.0, piece_duration=599.5
    )
    assert key is not None
    assert len(key.digest) == 32


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


def _entries(directory, stamps):
    directory.mkdir()
    for index, stamp in enumerate(stamps):
        entry = directory / f"{index}.json"
        entry.write_text("{}")
        os.utime(entry, (stamp, stamp))


def test_prune_keeps_the_newest(tmp_path):
    """By modification time, which here runs against the names."""
    directory = tmp_path / "resume"
    _entries(directory, [5000, 4000, 3000, 2000, 1000])

    _resume._prune(directory, keep=3)

    remaining = sorted(p.name for p in directory.glob("*.json"))
    assert remaining == ["0.json", "1.json", "2.json"]


@pytest.mark.parametrize(
    ("count", "remaining"),
    [
        (2, ["0.json", "1.json"]),
        (3, ["0.json", "1.json", "2.json"]),
        (4, ["1.json", "2.json", "3.json"]),
    ],
)
def test_prune_starts_past_the_limit(tmp_path, count, remaining):
    directory = tmp_path / "resume"
    _entries(directory, [(index + 1) * 1000 for index in range(count)])

    _resume._prune(directory, keep=3)

    assert sorted(p.name for p in directory.glob("*.json")) == remaining


def test_an_entry_another_run_pruned_first_is_no_failure(tmp_path, monkeypatch):
    """Two runs at once prune the same directory: the later unlink finds nothing."""
    directory = tmp_path / "resume"
    _entries(directory, [1000, 2000, 3000, 4000])
    unlink = Path.unlink

    def raced(path, missing_ok=False):
        os.remove(path)
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", raced)
    _resume._prune(directory, keep=3)

    assert sorted(p.name for p in directory.glob("*.json")) == ["1.json", "2.json", "3.json"]


@pytest.mark.parametrize("exists", [False, True], ids=["missing", "existing"])
def test_store_returns_the_file_it_wrote(tmp_path, exists):
    directory = tmp_path / "data" / "tapeback" / "resume"
    if exists:
        directory.mkdir(parents=True)
    key = _resume.ResumeKey("0123456789abcdef0123456789abcdef")

    path = _resume.store(key, directory, [Segment(start=0.0, end=5.0, text="x")], {})

    assert path == tmp_path / "data/tapeback/resume/0123456789abcdef0123456789abcdef.json"


def test_default_dir_honours_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert _resume.default_resume_dir() == tmp_path / "xdg" / "tapeback" / "resume"


@pytest.mark.parametrize("xdg", [None, ""], ids=["unset", "empty"])
def test_default_dir_falls_back_to_home(monkeypatch, tmp_path, xdg):
    if xdg is None:
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    else:
        monkeypatch.setenv("XDG_DATA_HOME", xdg)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert _resume.default_resume_dir() == tmp_path / ".local/share/tapeback/resume"
