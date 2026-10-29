"""Tests for out-of-process transcription."""

import io
import json
from pathlib import Path

import pytest

from tapeback._isolated import WorkerFailed, job_settings, transcribe_isolated
from tapeback._worker import EVENT_INFO, EVENT_SEGMENT, EVENT_STATUS, emit
from tapeback._worker import main as worker_main


class _FakeProcess:
    """Stands in for the worker: replays canned lines, records how it was stopped."""

    def __init__(self, lines: list[str], returncode: int = 0):
        self.stdin = io.StringIO()
        # A file, as the pipe is: the parent must close it.
        self.stdout = io.StringIO("".join(lines))
        self.returncode = returncode
        self.terminated = False
        self.killed = False
        self._alive = True

    def poll(self):
        return None if self._alive else self.returncode

    def terminate(self):
        self.terminated = True
        self._alive = False

    def kill(self):
        self.killed = True
        self._alive = False

    def wait(self, timeout=None):
        self._alive = False
        return self.returncode


def _line(event: str, **payload) -> str:
    return json.dumps({"type": event, **payload}) + "\n"


def _segment_line(start: float, end: float, text: str) -> str:
    return _line(
        EVENT_SEGMENT,
        data={
            "start": start,
            "end": end,
            "text": text,
            "speaker": None,
            "words": [{"start": start, "end": end, "word": text, "probability": 0.9}],
        },
    )


@pytest.fixture
def spawn(monkeypatch):
    """Replace Popen; returns a setter for the lines the fake worker will emit."""
    holder: dict[str, _FakeProcess] = {}

    def _install(lines: list[str], returncode: int = 0) -> _FakeProcess:
        process = _FakeProcess(lines, returncode)
        holder["process"] = process
        monkeypatch.setattr("tapeback._isolated.subprocess.Popen", lambda *_a, **_k: process)
        return process

    return _install


def test_credentials_never_reach_the_worker(settings):
    """A process that only transcribes audio has no business holding tokens."""
    payload = job_settings(settings)
    assert "hf_token" not in payload
    assert "llm_api_key" not in payload
    assert json.dumps(payload)  # must be JSON-serialisable


def test_worker_settings_disable_further_isolation(settings):
    """Without this the worker spawns a worker, forever."""
    assert job_settings(settings)["isolate_transcription"] is False


def test_segments_and_info_come_back(settings, spawn):
    spawn(
        [
            _line(EVENT_STATUS, message="Whisper: large-v3-turbo on cuda/int8_float16"),
            _segment_line(0.0, 5.0, "первая"),
            _segment_line(5.0, 9.0, "вторая"),
            _line(EVENT_INFO, data={"language": "ru", "duration": 9.0, "partial": False}),
        ]
    )
    messages: list[str] = []

    segments, info = transcribe_isolated(
        Path("/fake/audio.wav"), settings, on_status=messages.append
    )

    assert [s.text for s in segments] == ["первая", "вторая"]
    assert segments[0].words is not None
    assert segments[0].words[0].probability == 0.9
    assert info["language"] == "ru"
    assert "cuda/int8_float16" in messages[0]


def test_worker_dying_midway_keeps_what_it_sent(settings, spawn):
    """The whole point: an out-of-memory kills the child, not the run's work.

    Each segment is a complete line, so everything decoded before the crash is
    already in the parent. Losing it would put us back where an in-process OOM left
    us — except the VRAM is at least reclaimed now.
    """
    process = spawn([_segment_line(0.0, 5.0, "успел")], returncode=137)
    messages: list[str] = []

    segments, info = transcribe_isolated(
        Path("/fake/audio.wav"), settings, on_status=messages.append
    )

    assert [s.text for s in segments] == ["успел"]
    assert info["partial"] is True
    assert any("keeping the 1 segments" in m for m in messages)
    assert process.terminated or process.returncode == 137


def test_worker_dying_with_nothing_raises(settings, spawn):
    spawn([], returncode=1)

    with pytest.raises(WorkerFailed):
        transcribe_isolated(Path("/fake/audio.wav"), settings, on_status=lambda _m: None)


def test_worker_error_event_is_reported(settings, spawn):
    spawn([_line("error", message="RuntimeError: CUDA out of memory")], returncode=1)

    with pytest.raises(WorkerFailed, match="CUDA out of memory"):
        transcribe_isolated(Path("/fake/audio.wav"), settings, on_status=lambda _m: None)


def test_noise_on_stdout_is_ignored(settings, spawn):
    """Dependencies print to stdout; that must not derail the protocol."""
    spawn(
        [
            "loading model...\n",
            _segment_line(0.0, 5.0, "речь"),
            _line(EVENT_INFO, data={"language": "ru", "duration": 5.0, "partial": False}),
        ]
    )

    segments, info = transcribe_isolated(
        Path("/fake/audio.wav"), settings, on_status=lambda _m: None
    )

    assert [s.text for s in segments] == ["речь"]
    assert info["duration"] == 5.0


def test_worker_is_always_stopped(settings, spawn):
    """Leaving a worker running would defeat the reason it exists."""
    process = spawn([_line(EVENT_INFO, data={"language": "ru", "duration": 1.0, "partial": False})])

    transcribe_isolated(Path("/fake/audio.wav"), settings, on_status=lambda _m: None)

    assert process.terminated
    # An open pipe outlived every transcription, one descriptor each.
    assert process.stdout.closed


def test_emit_writes_one_flushed_line():
    stream = io.StringIO()
    emit(stream, EVENT_STATUS, message="hello")
    assert stream.getvalue() == '{"type": "status", "message": "hello"}\n'


def test_malformed_job_is_reported_as_an_error_event(monkeypatch, capsys):
    """A bad job must come back as a parsable event, not a traceback on stderr."""
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))

    assert worker_main() == 2

    event = json.loads(capsys.readouterr().out.strip())
    assert event["type"] == "error"
    assert "invalid job" in event["message"]
