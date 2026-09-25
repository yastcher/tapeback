"""Unit tests for per-run metadata records."""

import json

import pytest
from pydantic import SecretStr

from tapeback._runlog import (
    MAX_RUN_RECORDS,
    RECORDED_SETTINGS,
    RunLog,
    _prune_old_records,
    default_run_log_dir,
    run_log,
    write_run_log,
)
from tapeback.settings import Settings


def _read_only_record(directory):
    """Return the single JSON record written into directory."""
    records = list(directory.glob("*.json"))
    assert len(records) == 1
    return json.loads(records[0].read_text())


def test_run_log_records_config_events_and_outcome(tmp_path):
    settings = Settings(vault_path=tmp_path / "vault", run_log_dir=tmp_path / "runs")

    printed: list[str] = []
    with run_log("2026-08-05_12-00-00", settings, printed.append) as report:
        report("Stage 'merge' took 1.0s")
        report("Whisper: large-v3-turbo on cuda/float16")

    # The reporter both prints and captures — neither replaces the other.
    assert printed == ["Stage 'merge' took 1.0s", "Whisper: large-v3-turbo on cuda/float16"]

    record = _read_only_record(tmp_path / "runs")
    assert record["session"] == "2026-08-05_12-00-00"
    assert record["outcome"] == "completed"
    assert record["error"] is None
    assert record["events"] == [
        "Stage 'merge' took 1.0s",
        "Whisper: large-v3-turbo on cuda/float16",
    ]
    assert record["config"]["stt_model"] == "large-v3-turbo"
    assert record["config"]["chunk_length"] == 30
    assert record["finished_at"] is not None


def test_run_log_never_records_credentials(tmp_path):
    """P0: a post-mortem file that leaks credentials is worse than no file at all."""
    settings = Settings(
        vault_path=tmp_path / "vault",
        run_log_dir=tmp_path / "runs",
        hf_token=SecretStr("hf_SUPERSECRET"),
        llm_api_key=SecretStr("sk-SUPERSECRET"),
    )

    with run_log("session", settings, lambda _m: None) as report:
        report("working")

    raw = next((tmp_path / "runs").glob("*.json")).read_text()
    assert "SUPERSECRET" not in raw
    assert "hf_token" not in raw
    assert "llm_api_key" not in raw


def test_recorded_settings_contains_no_secret_fields():
    """Guard the allow-list itself: adding a SecretStr field here must fail the build."""
    secret_fields = {
        name
        for name, info in Settings.model_fields.items()
        if info.annotation is not None and "Secret" in str(info.annotation)
    }
    assert secret_fields, "expected Settings to carry at least one SecretStr field"
    assert secret_fields.isdisjoint(RECORDED_SETTINGS)


def test_run_log_marks_keyboard_interrupt_as_aborted_and_reraises(tmp_path):
    """Ctrl+C is exactly the case worth a record — the interrupt must still propagate."""
    settings = Settings(vault_path=tmp_path / "vault", run_log_dir=tmp_path / "runs")

    with pytest.raises(KeyboardInterrupt), run_log("session", settings, lambda _m: None) as report:
        report("transcribing")
        raise KeyboardInterrupt

    record = _read_only_record(tmp_path / "runs")
    assert record["outcome"] == "aborted"
    assert record["error"] is None
    assert record["events"] == ["transcribing"]


def test_run_log_marks_exception_as_failed_and_reraises(tmp_path):
    settings = Settings(vault_path=tmp_path / "vault", run_log_dir=tmp_path / "runs")

    with (
        pytest.raises(RuntimeError, match="cuBLAS"),
        run_log("session", settings, lambda _m: None) as report,
    ):
        report("transcribing")
        raise RuntimeError("cuBLAS not found")

    record = _read_only_record(tmp_path / "runs")
    assert record["outcome"] == "failed"
    assert record["error"] == "RuntimeError: cuBLAS not found"


def test_run_log_disabled_writes_nothing_and_passes_reporter_through(tmp_path):
    settings = Settings(vault_path=tmp_path / "vault", run_log_dir=tmp_path / "runs", run_log=False)

    printed: list[str] = []
    with run_log("session", settings, printed.append) as report:
        # Passed straight through — not wrapped in a capturing reporter.
        assert report == printed.append
        report("working")

    assert printed == ["working"]
    assert not (tmp_path / "runs").exists()


def test_default_run_log_dir_honours_xdg_data_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert default_run_log_dir() == tmp_path / "xdg" / "tapeback" / "runs"


def test_default_run_log_dir_falls_back_to_local_share(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr("tapeback._runlog.Path.home", lambda: tmp_path / "home")
    assert default_run_log_dir() == tmp_path / "home" / ".local" / "share" / "tapeback" / "runs"


@pytest.mark.parametrize(
    ("existing", "keep", "expected_remaining"),
    [
        # Boundary: exactly at the limit keeps everything.
        (3, 3, 3),
        (2, 3, 2),
        (4, 3, 3),
    ],
)
def test_prune_old_records_boundaries(tmp_path, existing, keep, expected_remaining):
    directory = tmp_path / "runs"
    directory.mkdir()
    for index in range(existing):
        (directory / f"2026-08-0{index}_run.json").write_text("{}")

    _prune_old_records(directory, keep=keep)

    remaining = sorted(p.name for p in directory.glob("*.json"))
    assert len(remaining) == expected_remaining
    # Pruning drops the oldest first, so the newest name always survives.
    assert remaining[-1] == f"2026-08-0{existing - 1}_run.json"


def test_write_run_log_keeps_directory_bounded(tmp_path):
    """write_run_log prunes after writing, so the directory cannot grow forever."""
    directory = tmp_path / "runs"
    directory.mkdir()
    for index in range(MAX_RUN_RECORDS + 5):
        (directory / f"2026-01-{index:04d}_old.json").write_text("{}")

    record = RunLog(session="new", started_at="2099-01-01T00:00:00+00:00", config={})
    written = write_run_log(record, directory)

    assert written is not None
    assert len(list(directory.glob("*.json"))) == MAX_RUN_RECORDS
    # The record just written is the newest and must not be the one pruned.
    assert written.exists()


def test_run_log_defaults_to_unknown_outcome(tmp_path):
    """An exit neither branch classified (SystemExit, killed process) must not read as completed."""
    settings = Settings(vault_path=tmp_path / "vault", run_log_dir=tmp_path / "runs")

    with pytest.raises(SystemExit), run_log("session", settings, lambda _m: None) as report:
        report("working")
        raise SystemExit(1)

    record = _read_only_record(tmp_path / "runs")
    assert record["outcome"] == "unknown"


def test_write_run_log_returns_none_when_directory_is_unwritable(tmp_path):
    """A failed diagnostic write must never take down the run that produced it."""
    blocker = tmp_path / "runs"
    blocker.write_text("not a directory")

    record = RunLog(session="s", started_at="2026-08-05T00:00:00+00:00", config={})
    assert write_run_log(record, blocker) is None
