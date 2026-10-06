import importlib.util
import json
import os
import shutil
import subprocess
import wave
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest
from click.testing import CliRunner
from pydantic import SecretStr

from tapeback.recorder import Recorder
from tapeback.settings import Settings
from tapeback.summarizer import _PROVIDER_ENV_VARS

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyannote_available() -> bool:
    try:
        import pyannote.audio  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


requires_pyannote = pytest.mark.skipif(
    not _pyannote_available(),
    reason="pyannote-audio not installed (install tapeback[diarize])",
)


def _dbus_next_available() -> bool:
    try:
        import dbus_next  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


requires_dbus_next = pytest.mark.skipif(
    not _dbus_next_available(),
    reason="dbus-next not installed (install tapeback[tray])",
)

# --- pytest fixtures ---


@pytest.fixture(autouse=True)
def isolate_session_root(monkeypatch, tmp_path_factory):
    """Keep recordings and session state made by tests out of the developer's home.

    recorder.session_root() and Recorder() resolve XDG_STATE_HOME at call time, so
    without this any test that starts a recording would create directories under
    ~/.local/state/tapeback and leave them there. Pointing it at a per-session tmp
    directory scopes every test to what it created itself. XDG_STATE_HOME rather than
    TAPEBACK_SESSIONS_DIR: isolate_settings_sources removes every TAPEBACK_ variable.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path_factory.mktemp("state")))


@pytest.fixture(autouse=True)
def isolate_settings_sources(monkeypatch):
    """Cut Settings off from the developer's real configuration.

    Settings reads ~/.config/tapeback/.env and ./.env, so without this every test
    that constructs Settings() inherits whatever the machine happens to have
    configured. That makes results machine-dependent: a developer with
    TAPEBACK_CHUNK_LENGTH=2 in their user config sees different values than CI.
    Isolation is by construction — no test should depend on the ambient config.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", ())
    for name in [key for key in os.environ if key.startswith("TAPEBACK_")]:
        monkeypatch.delenv(name, raising=False)
    # Provider keys are read under their own names, not the TAPEBACK_ prefix, so the
    # sweep above never touched them: a developer with ANTHROPIC_API_KEY exported gave
    # every test a live provider chain, and one forgotten mock would bill the vendor.
    # Driven off the production mapping on purpose — a new provider must be isolated
    # the moment it is added, not the next time someone remembers this list.
    for env_var in _PROVIDER_ENV_VARS.values():
        monkeypatch.delenv(env_var, raising=False)
    # Same rule applied to the machine's thermal state: without this every test that
    # builds a Transcriber on "cuda" polls the real GPU and can sit in the clamp wait,
    # so the suite's runtime would depend on how hot the developer's laptop is.
    monkeypatch.setenv("TAPEBACK_THERMAL_CLAMP_CHECK", "false")
    # Likewise for process isolation: a test that mocks WhisperModel would otherwise
    # spawn a real worker, which loads a real model and ignores the mock entirely.
    # Tests for the isolated path opt back in and mock the subprocess instead.
    monkeypatch.setenv("TAPEBACK_ISOLATE_TRANSCRIPTION", "false")


@pytest.fixture
def tmp_vault(tmp_path):
    """Temporary Obsidian vault for tests."""
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


@pytest.fixture
def vault_env(tmp_vault, monkeypatch):
    """tmp_vault + TAPEBACK_VAULT_PATH env var — for CLI tests that call get_settings()."""
    monkeypatch.setenv("TAPEBACK_VAULT_PATH", str(tmp_vault))
    return tmp_vault


@pytest.fixture
def settings(tmp_vault):
    """Settings with temporary vault."""
    return Settings(vault_path=tmp_vault)


@pytest.fixture
def session_wavs(tmp_path):
    """Factory: create session_dir with mic.wav + monitor.wav sine WAVs.

    Returns callable(session_name, duration=2.0) -> (session_dir, monitor_wav, mic_wav).
    """

    def _create(session_name: str, duration: float = 2.0) -> tuple[Path, Path, Path]:
        session_dir = tmp_path / session_name
        session_dir.mkdir()
        monitor_wav = session_dir / "monitor.wav"
        mic_wav = session_dir / "mic.wav"
        create_mono_wav(monitor_wav, duration=duration, sample_rate=48000, amplitude=0.5)
        create_mono_wav(mic_wav, duration=duration, sample_rate=48000, amplitude=0.5)
        return session_dir, monitor_wav, mic_wav

    return _create


@pytest.fixture
def recorder(tmp_path):
    """Recorder with temporary state directory."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    return Recorder(state_dir=state_dir)


@pytest.fixture
def session_file(recorder):
    """Path to session file."""
    return recorder.session_file


@pytest.fixture
def runner():
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def stereo_wav(tmp_path):
    """Factory fixture: returns callable(segments_spec) -> Path to stereo WAV."""

    def _create(segments_spec, sample_rate=16000):
        path = tmp_path / "stereo.wav"
        create_stereo_wav_segments(path, sample_rate, segments_spec)
        return path

    return _create


@pytest.fixture
def e2e_settings(tmp_path):
    """Settings for e2e tests with HF token from environment."""
    vault = tmp_path / "vault"
    vault.mkdir()
    return Settings(
        vault_path=vault,
        hf_token=SecretStr(os.environ.get("TAPEBACK_HF_TOKEN", os.environ.get("HF_TOKEN", ""))),
    )


@pytest.fixture
def summarize_settings(tmp_vault):
    """Settings pre-configured for summarizer tests."""
    return Settings(
        vault_path=tmp_vault,
        llm_provider="anthropic",
        llm_api_key=SecretStr("sk-ant-test-key"),
    )


@pytest.fixture
def tray_app(settings):
    """TrayApp with mocked SNI / DBusMenu / recorder — exercises state machine
    without ever touching the session bus."""
    from tapeback.tray import TrayApp  # noqa: PLC0415 — optional dep

    app = TrayApp(settings)
    app._sni = MagicMock()
    app._menu = MagicMock()
    app._recorder = MagicMock()
    app._recorder.is_recording.return_value = False
    app._recorder.get_session_info.return_value = None
    return app


@pytest.fixture
def e2e_output_dir(tmp_path):
    """Temporary directory for intermediate pipeline files."""
    d = tmp_path / "e2e_output"
    d.mkdir()
    return d


@pytest.fixture
def release_tree(tmp_path):
    """A tree where every copy of the version agrees on 1.2.3 — what
    scripts/release_from_tag.py accepts. Tests break one copy at a time."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "tapeback"\nversion = "1.2.3"\n')
    (tmp_path / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "click"\nversion = "8.1.7"\n\n'
        '[[package]]\nname = "tapeback"\nversion = "1.2.3"\nsource = { editable = "." }\n'
    )
    for directory in ("packaging", "packaging/tapeback-llm"):
        (tmp_path / directory).mkdir(exist_ok=True)
        (tmp_path / directory / "PKGBUILD").write_text("pkgname=tapeback\npkgver=1.2.3\npkgrel=1\n")
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [1.2.3] — 2026-10-05\n\n### Fixed\n"
        "- The fix.\n\n## [1.2.2] — 2026-10-01\n\n### Added\n- The feature.\n"
    )
    return tmp_path


@pytest.fixture
def run_gate(tmp_path):
    """Run the real scripts/gate.sh with every command it calls stubbed on PATH.

    Returns a callable: `run_gate(*args, changed=[...], fail="uv run ty check")` runs
    the gate on a branch whose diff against origin/main is `changed` (None: no merge
    base at all), makes the `fail` command exit 1, and gives back the finished process
    and every command the gate invoked, in order. The repository root reads `<root>`.
    """
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    gate = scripts / "gate.sh"
    shutil.copy2(REPO_ROOT / "scripts" / "gate.sh", gate)
    logging_stub = (
        "#!/usr/bin/env bash\n"
        'printf "%s %s\\n" "${0##*/}" "$*" >> "$GATE_LOG"\n'
        'if [ "${0##*/} $*" = "$GATE_FAIL" ]; then exit 1; fi\n'
    )
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for path in (stubs / "uv", stubs / "python3", stubs / "docker", scripts / "deb-smoke.sh"):
        path.write_text(logging_stub)
        path.chmod(0o755)
    # git answers only the two questions the gate asks it.
    git = stubs / "git"
    git.write_text(
        "#!/usr/bin/env bash\n"
        'case "$*" in\n'
        '  *merge-base*) [ -n "$GATE_NO_BASE" ] && exit 1; echo base ;;\n'
        '  *"diff --name-only"*) printf "%s" "$GATE_CHANGED" ;;\n'
        "esac\n"
    )
    git.chmod(0o755)
    log = tmp_path / "commands.log"

    def run(
        *args: str, changed: Sequence[str] | None = (), fail: str = ""
    ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        env = {
            "PATH": f"{stubs}:{os.environ['PATH']}",
            "GATE_LOG": str(log),
            "GATE_FAIL": fail,
            "GATE_NO_BASE": "1" if changed is None else "",
            "GATE_CHANGED": "".join(f"{name}\n" for name in changed or ()),
        }
        result = subprocess.run(
            [gate, *args],
            env=os.environ | env,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        lines = log.read_text().splitlines() if log.exists() else []
        return result, [line.replace(str(tmp_path), "<root>").rstrip() for line in lines]

    return run


# --- WAV file helpers ---


def create_silent_wav(path: Path, duration: float = 1.0, sample_rate: int = 48000) -> None:
    """Create a silent (zero-amplitude) WAV file for testing."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        n_frames = int(duration * sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)


def create_mono_wav(
    path: Path, duration: float = 1.0, sample_rate: int = 48000, amplitude: float = 0.5
) -> None:
    """Create a mono WAV with a 440Hz sine wave at given amplitude."""
    n_frames = int(duration * sample_rate)
    t = np.linspace(0, duration, n_frames, dtype=np.float32)
    samples = (amplitude * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def create_stereo_wav(path, duration, sample_rate, left_amplitude, right_amplitude):
    """Create a stereo WAV with sine wave on each channel at given amplitudes."""
    n_frames = int(duration * sample_rate)
    t = np.linspace(0, duration, n_frames, dtype=np.float32)

    left = (left_amplitude * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    right = (right_amplitude * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    stereo = np.empty(n_frames * 2, dtype=np.int16)
    stereo[0::2] = left
    stereo[1::2] = right

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())


def create_stereo_wav_segments(path, sample_rate, segments_spec):
    """Create a stereo WAV with different amplitudes per time segment.

    segments_spec: list of (duration, left_amp, right_amp)
    """
    all_left = []
    all_right = []

    for duration, left_amp, right_amp in segments_spec:
        n_frames = int(duration * sample_rate)
        t = np.linspace(0, duration, n_frames, dtype=np.float32)
        all_left.append((left_amp * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16))
        all_right.append((right_amp * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16))

    left = np.concatenate(all_left)
    right = np.concatenate(all_right)

    stereo = np.empty(len(left) * 2, dtype=np.int16)
    stereo[0::2] = left
    stereo[1::2] = right

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())


# --- Test data constants ---


SAMPLE_MD = """\
---
date: 2026-03-20
time: "14:30"
duration: "00:10:00"
language: en
---

# Meeting 2026-03-20 14:30

[00:00:01] **You:** Hello there.
[00:00:05] **Speaker 1:** Hi, let's discuss the plan.
"""

SAMPLE_MD_WITH_SUMMARY = """\
---
date: 2026-03-20
time: "14:30"
duration: "00:10:00"
language: en
---

## Summary

Old summary here.

---

# Meeting 2026-03-20 14:30

[00:00:01] **You:** Hello there.
"""

SAMPLE_TRANSCRIPT_MD = """\
---
date: 2026-03-20
time: "14:30"
duration: "00:10:00"
language: en
tags:
  - meeting
  - transcript
audio: "[[attachments/audio/2026-03-20_14-30-00.wav]]"
---

# Meeting 2026-03-20 14:30

[00:00:01] **You:** Hello there.
[00:00:05] **Speaker 1:** Hi, let's discuss the plan.
"""

VALID_LLM_RESPONSE = json.dumps(
    {
        "brief": "Discussed the project plan and assigned tasks.",
        "action_items": [
            {"assignee": "You", "action": "Send the report", "deadline": "Friday"},
            {"assignee": "Speaker 1", "action": "Review the code", "deadline": None},
        ],
        "key_decisions": ["Use PostgreSQL instead of MongoDB"],
        "is_trivial": False,
    }
)

VALID_LLM_RESPONSE_MINIMAL = json.dumps(
    {
        "brief": "Test summary.",
        "action_items": [],
        "key_decisions": [],
        "is_trivial": False,
    }
)


class HttpError(Exception):
    """Exception with status_code attribute, mimicking SDK exceptions (anthropic, openai)."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


# --- Test helpers ---


def load_script(name: str) -> ModuleType:
    """Load `scripts/<name>.py` by path — `scripts/` is not an importable package."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clear_all_provider_env_vars(monkeypatch) -> None:
    """Remove all provider-specific API key env vars.

    Redundant since isolate_settings_sources does the same for every test; kept because
    call sites read as an explicit statement of what the test depends on.
    """
    for env_var in _PROVIDER_ENV_VARS.values():
        monkeypatch.delenv(env_var, raising=False)


def mock_anthropic_response(text: str) -> MagicMock:
    """Create a mock anthropic Messages.create response."""
    response = MagicMock()
    response.content = [MagicMock(text=text)]
    return response


def voice_signal(duration: float, sr: int, fundamental: float) -> np.ndarray:
    """Generate a synthetic voice signal with harmonics."""
    t = np.linspace(0, duration, int(duration * sr), dtype=np.float32)
    return (
        np.sin(2 * np.pi * fundamental * t)
        + 0.5 * np.sin(2 * np.pi * fundamental * 2 * t)
        + 0.25 * np.sin(2 * np.pi * fundamental * 3 * t)
    ) * 10000


# --- Session file helpers ---


def create_session_file(session_file, **overrides):
    """Write a session.json file with sensible defaults, overridden by kwargs."""
    data = {
        "pid_monitor": 99998,
        "pid_mic": 99999,
        "session_name": "test_session",
        "monitor_path": "/run/user/1000/tapeback/test_session/monitor.wav",
        "mic_path": "/run/user/1000/tapeback/test_session/mic.wav",
        "started_at": "2026-03-17T14:30:00",
        **overrides,
    }
    session_file.write_text(json.dumps(data))
    return data


# --- Mock helpers ---


def mock_whisper_transcribe(segments_data):
    """Create a mock WhisperModel that returns given segments.

    segments_data: list of (start, end, text) tuples
    """
    mock_model = MagicMock()

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_info.duration = 10.0

    mock_segments = []
    for start, end, text in segments_data:
        seg = MagicMock()
        seg.start = start
        seg.end = end
        seg.text = f" {text} "
        seg.words = []
        mock_segments.append(seg)

    # Each call to transcribe() returns a fresh iterator
    mock_model.transcribe.side_effect = lambda *a, **kw: (iter(list(mock_segments)), mock_info)

    return mock_model


def mock_pyannote_annotation(tracks):
    """Create a mock pyannote annotation from (start, end, speaker) tuples."""
    mock_annotation = MagicMock()
    itertracks_result = []
    for start, end, speaker in tracks:
        turn = MagicMock()
        turn.start = start
        turn.end = end
        itertracks_result.append((turn, None, speaker))
    mock_annotation.itertracks.return_value = itertracks_result
    return mock_annotation


def process_state(pid: int) -> str | None:
    """The kernel's one-letter state for a process (R, S, Z, ...); None once it is gone,
    reaped included. A zombie is "Z": exited, but its parent has not collected it."""
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        # ProcessLookupError: reaped between opening the file and reading it.
        return None
    # The state follows the command name, which is parenthesised and may hold spaces.
    return stat_line.rsplit(")", 1)[1].split()[0]
