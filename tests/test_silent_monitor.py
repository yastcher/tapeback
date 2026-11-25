"""A monitor that records nothing: warned about while recording, and said in the note."""

import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tapeback._monitor_watch import MonitorWatch
from tapeback.channel import is_silent_throughout
from tapeback.cli import cli
from tapeback.formatter import TranscriptMeta, format_markdown
from tapeback.models import Segment
from tests.fixtures import create_stereo_wav_segments, mock_whisper_transcribe, requires_pyannote

RATE = 16000
ANALOG = "alsa_output.pci-0000_00_1f.3.analog-stereo"
HDMI = "alsa_output.pci-0000_01_00.1.hdmi-stereo"
HEADER = b"RIFF\x24\x00\x00\x00WAVEfmt "  # what parecord writes first: not silence


def _stream(index: int, sink: int, app: str, *, corked: bool = False) -> dict:
    return {
        "index": index,
        "sink": sink,
        "corked": corked,
        "properties": {"application.name": app},
    }


ZOOM_ON_HDMI = [_stream(57, 1, "ZOOM VoiceEngine"), _stream(58, 0, "Firefox")]
ALL_ON_ANALOG = [_stream(58, 0, "Firefox")]


SOURCES = [
    {"index": 0, "name": f"{ANALOG}.monitor"},
    {"index": 1, "name": "alsa_input.pci-0000_00_1f.3.analog-stereo"},
    {"index": 2, "name": f"{HDMI}.monitor"},
]


class _Pactl:
    """`pactl -f json ...` as the sound server would answer it; `streams` can change.

    It knows no `info`: the recorded output must come from the recording's own stream,
    not from whichever output is the default.
    """

    def __init__(self, recording: Path, streams: list[dict], source: int) -> None:
        self.recording = recording
        self.streams = streams
        self.source = source
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        mic_path = str(self.recording.with_name("mic.wav"))
        replies = {
            ("list", "source-outputs"): [
                {"index": 48, "source": 1, "properties": {"media.name": mic_path}},
                {
                    "index": 49,
                    "source": self.source,
                    "properties": {"media.name": str(self.recording)},
                },
            ],
            ("list", "short", "sources"): SOURCES,
            ("list", "short", "sinks"): [{"index": 0, "name": ANALOG}, {"index": 1, "name": HDMI}],
            ("list", "sink-inputs"): self.streams,
        }
        return subprocess.CompletedProcess(argv, 0, json.dumps(replies[tuple(argv[3:])]), "")


@contextmanager
def _sound_server(tmp_path: Path, streams: list[dict], source: int = 0) -> Iterator[_Pactl]:
    """Never a real sound server: subprocess and which are the boundary.

    source is what the recording takes, an index into SOURCES: by default the monitor of
    the analog output.
    """
    fake = _Pactl(tmp_path / "monitor.wav", streams, source)
    with (
        patch("tapeback._monitor_watch.shutil.which", return_value="/usr/bin/pactl"),
        patch("tapeback._monitor_watch.subprocess.run", side_effect=fake),
    ):
        yield fake


def _watch(tmp_path, settings, check=1.0, warn=5.0):
    path = tmp_path / "monitor.wav"
    path.write_bytes(HEADER)
    s = settings.model_copy(
        update={
            "sample_rate": RATE,
            "monitor_silence_check_seconds": check,
            "monitor_silence_warn_seconds": warn,
        }
    )
    return MonitorWatch(path, s), path


def _record(path, seconds: float, level: int = 0) -> None:
    samples = np.full(round(seconds * RATE), level, dtype=np.int16)
    with path.open("ab") as f:
        f.write(samples.tobytes())


ELSEWHERE = (
    f"Warning: the monitor has recorded silence for 1 s, and the other side is not being "
    f"recorded: tapeback records {ANALOG}, but\n"
    f"  ZOOM VoiceEngine plays to {HDMI}. To record it, move it: "
    f"pactl move-sink-input 57 {ANALOG}\n"
    "  (or pick that output in the app's audio settings)"
)
SILENT = (
    "Warning: the monitor has recorded nothing for 0 min. If the other side is talking, "
    f"the meeting app plays to an output tapeback does not record (it records {ANALOG})."
)
SILENT_UNKNOWN_OUTPUT = (
    "Warning: the monitor has recorded nothing for 0 min. If the other side is talking, "
    "the meeting app plays to an output tapeback does not record."
)


def test_an_app_playing_elsewhere_is_named_once_the_check_time_is_reached(tmp_path, settings):
    with _sound_server(tmp_path, ZOOM_ON_HDMI):
        watch, path = _watch(tmp_path, settings)
        assert watch.poll() is None  # the header alone

        _record(path, 0.99)
        assert watch.poll() is None
        _record(path, 0.01)
        assert watch.poll() == ELSEWHERE


def test_the_warning_comes_once_a_silence_and_again_after_sound(tmp_path, settings):
    with _sound_server(tmp_path, ZOOM_ON_HDMI):
        watch, path = _watch(tmp_path, settings)
        _record(path, 1.0)
        assert watch.poll() == ELSEWHERE
        _record(path, 10.0)
        assert watch.poll() is None

        _record(path, 0.5, level=3000)
        _record(path, 1.0)
        assert watch.poll() == ELSEWHERE


@pytest.mark.parametrize(
    "streams",
    [ALL_ON_ANALOG, [_stream(57, 1, "ZOOM VoiceEngine", corked=True)]],
    ids=["every app plays to the recorded output", "the app elsewhere is paused"],
)
def test_silence_alone_warns_only_after_the_longer_wait(tmp_path, settings, streams):
    with _sound_server(tmp_path, streams):
        watch, path = _watch(tmp_path, settings, warn=5.0)
        _record(path, 4.99)
        assert watch.poll() is None
        _record(path, 0.01)
        assert watch.poll() == SILENT


def test_an_app_that_goes_elsewhere_later_is_still_named(tmp_path, settings):
    """A call often starts after the recording: the cautious warning comes first."""
    with _sound_server(tmp_path, ALL_ON_ANALOG) as server:
        watch, path = _watch(tmp_path, settings, warn=2.0)
        _record(path, 2.0)
        assert watch.poll() == SILENT

        server.streams = ZOOM_ON_HDMI
        _record(path, 1.0)
        assert watch.poll() == ELSEWHERE.replace("for 1 s", "for 3 s")


def test_the_recorded_output_is_the_recordings_own_not_the_default(tmp_path, settings):
    """The recording takes the HDMI monitor: a named source, or the default changed since."""
    with _sound_server(tmp_path, [_stream(57, 0, "ZOOM VoiceEngine")], source=2):
        watch, path = _watch(tmp_path, settings)
        _record(path, 1.0)
        assert watch.poll() == (
            f"Warning: the monitor has recorded silence for 1 s, and the other side is not "
            f"being recorded: tapeback records {HDMI}, but\n"
            f"  ZOOM VoiceEngine plays to {ANALOG}. To record it, move it: "
            f"pactl move-sink-input 57 {HDMI}\n"
            "  (or pick that output in the app's audio settings)"
        )


def test_a_recording_of_no_output_gets_only_the_cautious_warning(tmp_path, settings):
    """TAPEBACK_MONITOR_SOURCE may name an input: there is no output to move an app to."""
    with _sound_server(tmp_path, ZOOM_ON_HDMI, source=1):
        watch, path = _watch(tmp_path, settings, warn=2.0)
        _record(path, 1.0)
        assert watch.poll() is None
        _record(path, 1.0)
        assert watch.poll() == SILENT_UNKNOWN_OUTPUT


def test_zero_turns_both_warnings_off(tmp_path, settings):
    with _sound_server(tmp_path, ZOOM_ON_HDMI) as server:
        watch, path = _watch(tmp_path, settings, check=0.0, warn=0.0)
        _record(path, 600.0)
        assert watch.poll() is None

    assert server.calls == []


def test_without_pactl_only_silence_is_seen(tmp_path, settings):
    with patch("tapeback._monitor_watch.shutil.which", return_value=None):
        watch, path = _watch(tmp_path, settings, warn=2.0)
        _record(path, 2.0)
        assert watch.poll() == SILENT_UNKNOWN_OUTPUT


@pytest.mark.parametrize(("level", "silent"), [(199, True), (200, False), (201, False)])
def test_a_channel_is_silent_below_the_silence_floor(level, silent):
    # A second of nothing, then half a second at the level: the last second counts too.
    samples = np.concatenate([np.zeros(RATE, dtype=np.int16), np.full(RATE // 2, level, np.int16)])

    assert is_silent_throughout(samples, RATE) is silent


@pytest.mark.parametrize(("level", "silent"), [(199, True), (200, False), (201, False)])
def test_a_loud_second_between_silences_is_judged_on_its_own(level, silent):
    # Averaged with the silence around it, a second at the level would fall under the floor.
    zeros = np.zeros(RATE, dtype=np.int16)
    samples = np.concatenate([zeros, np.full(RATE, level, np.int16), zeros])

    assert is_silent_throughout(samples, RATE) is silent


def test_a_note_whose_system_audio_was_silent_says_so_above_the_transcript():
    """The flow test reads the same line, but runs only where pyannote is installed."""
    markdown = format_markdown(
        segments=[Segment(start=0.0, end=5.0, text="Hello?", words=None, speaker="You")],
        meta=TranscriptMeta(
            session_name="2026-08-06_12-00-00",
            audio_rel_path="attachments/audio/x.wav",
            duration_seconds=1800.0,
            language="en",
            monitor_silent=True,
        ),
    )

    assert markdown == (
        "---\n"
        "date: 2026-08-06\n"
        'time: "12:00"\n'
        'duration: "00:30:00"\n'
        "language: en\n"
        'audio: "[[attachments/audio/x.wav]]"\n'
        "tags:\n"
        "  - meeting\n"
        "  - transcript\n"
        "---\n"
        "\n"
        "# Meeting 2026-08-06 12:00\n"
        "\n"
        "**Duration:** 30m 0s | **Language:** en\n"
        "\n"
        "> [!warning] The system audio was silent for the whole recording, so only your side "
        "is here: if others spoke, the meeting app played to an output tapeback did not record.\n"
        "\n"
        "---\n"
        "\n"
        "[00:00:00] **You:** Hello?\n"
    )


@requires_pyannote
def test_a_recording_whose_system_audio_is_silent_says_so_and_skips_diarization(
    runner, tmp_path, monkeypatch, vault_env
):
    monkeypatch.setenv("TAPEBACK_HF_TOKEN", "hf_fake")
    audio = tmp_path / "2026-03-23_14-30-00.wav"
    create_stereo_wav_segments(audio, 48000, [(1.0, 0.8, 0.0), (1.0, 0.8, 0.0)])
    model = mock_whisper_transcribe([(0.0, 2.0, "Hello, can you hear me?")])

    with (
        patch("tapeback.transcriber.WhisperModel", return_value=model),
        patch("pyannote.audio.Pipeline") as pipeline,
    ):
        result = runner.invoke(cli, ["process", str(audio)])

    assert result.exit_code == 0, result.output
    assert "Skipping diarization: the system audio is silent." in result.output
    pipeline.from_pretrained.assert_not_called()
    note = (vault_env / "meetings" / "2026-03-23_14-30-00.md").read_text()
    assert (
        "> [!warning] The system audio was silent for the whole recording, so only your side "
        "is here: if others spoke, the meeting app played to an output tapeback did not record."
    ) in note.splitlines()


@pytest.mark.parametrize(
    "streams",
    [
        [{"index": 57, "corked": False, "properties": {"application.name": "ZOOM VoiceEngine"}}],
        [{"sink": 1, "corked": False}],
        ["not an object"],
    ],
    ids=["a stream without its output", "a stream without its index", "not an object"],
)
def test_an_answer_pactl_did_not_give_before_is_no_reason_to_stop_recording(
    tmp_path, settings, streams
):
    """The watch runs inside the recording loop: an exception there ends `tapeback start`."""
    with _sound_server(tmp_path, streams):
        watch, path = _watch(tmp_path, settings, warn=2.0)
        _record(path, 1.0)
        assert watch.poll() is None
        _record(path, 1.0)
        assert watch.poll() == SILENT
