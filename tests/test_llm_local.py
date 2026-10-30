"""The local provider: a llama-server tapeback starts for the summary and stops after it.

The server is tests/fake_llama_server.py, a real process on a real loopback port. What
is asserted is how tapeback started it, what reached it and whether it is gone after.
"""

import fcntl
import hashlib
import io
import json
import subprocess
import sys
import threading
import time
import urllib.error
from email.message import Message
from pathlib import Path
from unittest.mock import patch

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from tapeback import _gpu, _llama_server
from tapeback._llama_server import PinnedModel, model_path
from tapeback.cli import cli
from tapeback.recorder import default_state_dir
from tapeback.settings import Settings
from tapeback.summarizer import maybe_summarize, summarize
from tests.fixtures import SAMPLE_MD, VALID_LLM_RESPONSE, process_state

TRANSCRIPT = "We ship on Friday."

# A tapeback that dies while its server is up, as a killed tray would.
CRASH_WHILE_SERVING = """
import os, signal, sys
from tapeback._llama_server import running
from tapeback.settings import Settings
settings = Settings(
    _env_file=None,
    vault_path=sys.argv[1],
    llm_provider="local",
    llm_local_server=sys.argv[2],
    llm_local_model=sys.argv[3],
)
with running(settings):
    os.kill(os.getpid(), signal.SIGKILL)
"""


def _settings(tmp_vault, fake, **overrides) -> Settings:
    settings = Settings(
        vault_path=tmp_vault,
        llm_provider="local",
        llm_local_server=str(fake.binary),
        llm_local_model=str(fake.model),
    )
    return settings.model_copy(update=overrides)


def _gone(pid: int) -> bool:
    """The process left /proc: exited and reaped, by us or by whoever adopted it."""
    deadline = time.monotonic() + 5
    while process_state(pid) is not None and time.monotonic() < deadline:
        time.sleep(0.01)
    return process_state(pid) is None


def test_a_summary_starts_the_server_with_the_measured_flags_and_stops_it(
    tmp_vault, fake_llama_server
):
    summary = summarize(TRANSCRIPT, _settings(tmp_vault, fake_llama_server))

    assert summary.brief == "Discussed the project plan and assigned tasks."
    [start] = fake_llama_server.starts()
    argv = start["argv"]
    assert argv[:6] == [
        "--model",
        str(fake_llama_server.model),
        "--ctx-size",
        "16384",
        "--host",
        "127.0.0.1",
    ]
    assert argv[6] == "--port"
    assert argv[8:] == [
        "--n-gpu-layers",
        "99",
        "--flash-attn",
        "on",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--reasoning",
        "off",
        "--cache-ram",
        "0",
        "--parallel",
        "1",
        "--no-webui",
    ]
    [request] = fake_llama_server.requests()
    assert request["path"] == "/v1/chat/completions"
    assert json.loads(request["body"])["messages"][1]["content"] == TRANSCRIPT
    assert _gone(start["pid"])
    assert not (default_state_dir() / "llama-server" / "pid").exists()


def test_the_server_runs_in_a_private_directory_not_where_tapeback_was_started(
    tmp_vault, fake_llama_server, tmp_path, monkeypatch
):
    """ggml loads backend libraries from the working directory: one planted where the
    user runs tapeback must not be loaded."""
    monkeypatch.chdir(tmp_path)

    summarize(TRANSCRIPT, _settings(tmp_vault, fake_llama_server))

    [start] = fake_llama_server.starts()
    assert start["cwd"] == str(default_state_dir() / "llama-server")


def test_the_key_reaches_the_server_through_its_environment_not_argv(tmp_vault, fake_llama_server):
    summarize(TRANSCRIPT, _settings(tmp_vault, fake_llama_server))

    [start] = fake_llama_server.starts()
    [request] = fake_llama_server.requests()
    assert len(start["key"]) == 43  # 32 random bytes, base64url
    assert request["auth"] == f"Bearer {start['key']}"
    assert start["key"] not in " ".join(start["argv"])


def test_the_json_retry_asks_the_same_server(tmp_vault, fake_llama_server, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_REPLIES", "Not JSON yet.\x1e" + VALID_LLM_RESPONSE)

    summary = summarize(TRANSCRIPT, _settings(tmp_vault, fake_llama_server))

    assert summary.brief == "Discussed the project plan and assigned tasks."
    assert len(fake_llama_server.starts()) == 1
    assert len(fake_llama_server.requests()) == 2


@pytest.mark.parametrize("what", ["weights", "context"])
def test_out_of_memory_halves_the_context_once(
    tmp_vault, fake_llama_server, monkeypatch, capsys, what
):
    monkeypatch.setenv("FAKE_LLAMA_FITS", "8192")
    monkeypatch.setenv("FAKE_LLAMA_OOM", what)

    summarize(TRANSCRIPT, _settings(tmp_vault, fake_llama_server))

    assert [start["argv"][3] for start in fake_llama_server.starts()] == ["16384", "8192"]
    assert "retrying with a 8192-token context" in capsys.readouterr().err


def test_out_of_memory_says_how_much_vram_is_free(
    tmp_vault, tmp_path, fake_llama_server, monkeypatch, capsys
):
    """Another program holding the card is the usual cause: the default model barely
    grows with its context, so what is free tells the user more."""
    nvidia_smi = tmp_path / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\necho 2100\n")
    nvidia_smi.chmod(0o755)
    monkeypatch.setattr(_gpu, "NVIDIA_SMI", str(nvidia_smi))
    monkeypatch.setenv("FAKE_LLAMA_FITS", "8192")

    summarize(TRANSCRIPT, _settings(tmp_vault, fake_llama_server))

    assert (
        "Warning: the summary model does not fit in VRAM with a 16384-token context "
        "(2100 MiB of VRAM free); retrying with a 8192-token context."
    ) in capsys.readouterr().err


def test_out_of_memory_at_half_the_context_gives_up(tmp_vault, fake_llama_server, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_FITS", "4096")

    with pytest.raises(_llama_server.OutOfMemory, match="with a 8192-token context"):
        summarize(TRANSCRIPT, _settings(tmp_vault, fake_llama_server))

    assert [start["argv"][3] for start in fake_llama_server.starts()] == ["16384", "8192"]
    assert not (default_state_dir() / "llama-server" / "pid").exists()


CLAMPED = "0x0000000000000024"  # SwPowerCap + SwThermalSlowdown, as caught live
IDLE = "0x0000000000000001"


def test_the_server_starts_once_a_thermal_clamp_releases(
    tmp_vault, tmp_path, fake_llama_server, fake_nvidia_smi, capsys
):
    """The summary is the last stage: a cool card is worth a wait."""
    fake_nvidia_smi(4096, CLAMPED, IDLE)  # clamped at the first look, released at the next
    settings = _settings(
        tmp_vault, fake_llama_server, thermal_clamp_check=True, llm_local_clamp_wait=30.0
    )

    summarize(TRANSCRIPT, settings)

    assert (tmp_path / "fake-nvidia-smi-looks").read_text() == "2"
    assert "still thermally clamped" not in capsys.readouterr().err
    assert len(fake_llama_server.starts()) == 1


def test_a_clamp_that_does_not_release_in_time_still_gets_a_summary(
    tmp_vault, fake_llama_server, fake_nvidia_smi, capsys
):
    fake_nvidia_smi(4096, CLAMPED)
    settings = _settings(
        tmp_vault, fake_llama_server, thermal_clamp_check=True, llm_local_clamp_wait=0.0
    )

    summary = summarize(TRANSCRIPT, settings)

    assert summary.brief == "Discussed the project plan and assigned tasks."
    assert "GPU still thermally clamped after 0s — proceeding anyway." in capsys.readouterr().err


def test_without_the_clamp_check_nvidia_smi_is_not_asked(
    tmp_vault, tmp_path, fake_llama_server, fake_nvidia_smi
):
    fake_nvidia_smi(4096, CLAMPED)

    summarize(TRANSCRIPT, _settings(tmp_vault, fake_llama_server, thermal_clamp_check=False))

    assert not (tmp_path / "fake-nvidia-smi-looks").exists()
    assert len(fake_llama_server.starts()) == 1


def test_a_local_answer_waits_for_its_own_timeout_not_the_clouds(
    tmp_vault, fake_llama_server, monkeypatch
):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "slow")
    settings = _settings(tmp_vault, fake_llama_server, llm_timeout=600.0, llm_local_timeout=0.5)

    with pytest.raises(Exception, match="timed out"):
        summarize(TRANSCRIPT, settings)


def test_a_server_that_fails_to_load_says_why_and_is_not_retried(
    tmp_vault, fake_llama_server, monkeypatch
):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "die")

    with pytest.raises(RuntimeError, match="invalid magic characters"):
        summarize(TRANSCRIPT, _settings(tmp_vault, fake_llama_server))

    assert len(fake_llama_server.starts()) == 1


def test_a_failing_local_server_never_falls_back_to_the_cloud(
    tmp_vault, fake_llama_server, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-cloud")
    monkeypatch.setenv("FAKE_LLAMA_MODE", "die")

    with patch("anthropic.Anthropic") as anthropic_client, pytest.raises(RuntimeError):
        summarize(TRANSCRIPT, _settings(tmp_vault, fake_llama_server))

    anthropic_client.assert_not_called()


def test_the_masked_text_is_what_reaches_the_server(tmp_vault, fake_llama_server):
    settings = _settings(tmp_vault, fake_llama_server, mask_pii=True, mask_terms="Alice")

    summarize("Alice ships it on Friday.", settings)

    [request] = fake_llama_server.requests()
    assert "Alice" not in request["body"]
    assert "[TERM_1] ships it on Friday." in request["body"]


def test_ctrl_c_during_the_answer_stops_the_server(tmp_vault, fake_llama_server):
    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt

    with (
        patch("openai.resources.chat.completions.Completions.create", interrupted),
        pytest.raises(KeyboardInterrupt),
    ):
        summarize(TRANSCRIPT, _settings(tmp_vault, fake_llama_server))

    [start] = fake_llama_server.starts()
    assert _gone(start["pid"])


def test_a_server_deaf_to_sigterm_is_killed(tmp_vault, fake_llama_server, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "deaf")
    monkeypatch.setattr(_llama_server, "STOP_TIMEOUT", 0.2)

    summarize(TRANSCRIPT, _settings(tmp_vault, fake_llama_server))

    [start] = fake_llama_server.starts()
    assert _gone(start["pid"])


def test_a_server_left_by_a_crashed_run_is_stopped_before_the_next_one(
    tmp_vault, fake_llama_server
):
    crash = [sys.executable, "-c", CRASH_WHILE_SERVING, str(tmp_vault)]
    subprocess.run([*crash, str(fake_llama_server.binary), str(fake_llama_server.model)])
    [orphan] = fake_llama_server.starts()
    assert process_state(orphan["pid"]) in ("R", "S")

    summarize(TRANSCRIPT, _settings(tmp_vault, fake_llama_server))

    assert _gone(orphan["pid"])
    assert len(fake_llama_server.starts()) == 2


def test_a_pid_file_naming_another_process_is_left_alone(tmp_vault, fake_llama_server):
    bystander = subprocess.Popen(["sleep", "30"])
    home = default_state_dir() / "llama-server"
    home.mkdir(parents=True, mode=0o700)
    # The same PID, another start time: the server it named is long gone.
    (home / "pid").write_text(f"{bystander.pid} 1\n")
    try:
        summarize(TRANSCRIPT, _settings(tmp_vault, fake_llama_server))

        assert process_state(bystander.pid) == "S"
    finally:
        bystander.kill()
        bystander.wait()


def test_a_second_summary_waits_for_the_first(tmp_vault, fake_llama_server, capsys):
    home = default_state_dir() / "llama-server"
    home.mkdir(parents=True, mode=0o700)
    second = threading.Thread(
        target=summarize, args=(TRANSCRIPT, _settings(tmp_vault, fake_llama_server))
    )
    with (home / "lock").open("w") as first:
        fcntl.flock(first, fcntl.LOCK_EX)
        second.start()
        said = ""
        deadline = time.monotonic() + 5
        while "Waiting for another summary" not in said and time.monotonic() < deadline:
            time.sleep(0.01)
            said += capsys.readouterr().err
        assert "Waiting for another summary to free the local model" in said
        assert fake_llama_server.starts() == []
    second.join(timeout=30)

    assert len(fake_llama_server.starts()) == 1


def test_a_missing_server_skips_the_summary_and_says_why(tmp_vault, fake_llama_server, capsys):
    note = tmp_vault / "meetings" / "2026-10-16_10-00-00.md"
    note.parent.mkdir(parents=True)
    note.write_text(SAMPLE_MD)
    settings = _settings(tmp_vault, fake_llama_server, llm_local_server="no-such-llama-server")

    maybe_summarize(note, settings)

    assert note.read_text() == SAMPLE_MD
    assert "no-such-llama-server not found" in capsys.readouterr().err
    assert fake_llama_server.starts() == []


def test_a_missing_model_file_is_refused(tmp_vault, tmp_path, fake_llama_server):
    absent = tmp_path / "absent.gguf"
    settings = _settings(tmp_vault, fake_llama_server, llm_local_model=str(absent))

    with pytest.raises(RuntimeError, match=f"TAPEBACK_LLM_LOCAL_MODEL: no file at {absent}"):
        summarize(TRANSCRIPT, settings)

    assert fake_llama_server.starts() == []


# --- The pinned default model ---


def test_the_default_model_is_pinned_by_commit_and_checksum():
    assert (
        PinnedModel(
            repo="unsloth/Qwen3.5-4B-GGUF",
            filename="Qwen3.5-4B-Q4_K_M.gguf",
            revision="e87f176479d0855a907a41277aca2f8ee7a09523",
            sha256="00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
            size=2740937888,
        )
        == _llama_server.DEFAULT_MODEL
    )


def _pin(tmp_path, monkeypatch) -> tuple[Path, PinnedModel]:
    """A seven-byte stand-in for the pinned GGUF, pinned in place of the real one."""
    blob = tmp_path / "hub" / "model.gguf"
    blob.parent.mkdir()
    blob.write_bytes(b"weights")
    pin = PinnedModel(
        repo="org/model-GGUF",
        filename="model.gguf",
        revision="0123456789abcdef0123456789abcdef01234567",
        sha256=hashlib.sha256(b"weights").hexdigest(),
        size=7,
    )
    monkeypatch.setattr(_llama_server, "DEFAULT_MODEL", pin)
    return blob, pin


def _hub(monkeypatch, blob: Path, *, cached: bool) -> list[tuple[str, str, str, bool]]:
    """hf_hub_download, the network boundary: the calls it got."""
    calls: list[tuple[str, str, str, bool]] = []

    def download(repo: str, filename: str, *, revision: str, local_files_only: bool = False):
        calls.append((repo, filename, revision, local_files_only))
        if local_files_only and not cached:
            raise LocalEntryNotFoundError("not in the cache")
        return str(blob)

    monkeypatch.setattr(_llama_server, "hf_hub_download", download)
    return calls


def test_the_pinned_model_is_downloaded_once_at_its_revision(
    tmp_path, tmp_vault, monkeypatch, capsys
):
    blob, _ = _pin(tmp_path, monkeypatch)
    calls = _hub(monkeypatch, blob, cached=False)

    assert model_path(Settings(vault_path=tmp_vault)) == blob

    revision = "0123456789abcdef0123456789abcdef01234567"
    assert calls == [
        ("org/model-GGUF", "model.gguf", revision, True),
        ("org/model-GGUF", "model.gguf", revision, False),
    ]
    assert "Downloading the summary model model.gguf" in capsys.readouterr().err


def test_a_cached_pinned_model_is_not_downloaded_again(tmp_path, tmp_vault, monkeypatch, capsys):
    blob, _ = _pin(tmp_path, monkeypatch)
    calls = _hub(monkeypatch, blob, cached=True)

    assert model_path(Settings(vault_path=tmp_vault)) == blob

    assert [call[3] for call in calls] == [True]
    assert "Downloading" not in capsys.readouterr().err


def test_a_download_that_fails_its_checksum_is_refused(tmp_path, tmp_vault, monkeypatch):
    blob, _ = _pin(tmp_path, monkeypatch)
    blob.write_bytes(b"weighed")  # the size it should have, other bytes
    _hub(monkeypatch, blob, cached=False)

    with pytest.raises(RuntimeError, match="does not match its pinned checksum"):
        model_path(Settings(vault_path=tmp_vault))


def test_a_cached_model_of_the_wrong_size_is_refused(tmp_path, tmp_vault, monkeypatch):
    blob, _ = _pin(tmp_path, monkeypatch)
    blob.write_bytes(b"weig")  # an interrupted copy
    _hub(monkeypatch, blob, cached=True)

    with pytest.raises(RuntimeError, match="has 4 bytes, not the 7 pinned"):
        model_path(Settings(vault_path=tmp_vault))


# --- The CLI ---


def test_status_names_the_local_model_and_server(runner, real_env_files, fake_llama_server):
    user_config, _ = real_env_files
    user_config.write_text(
        f"TAPEBACK_LLM_PROVIDER=local\nTAPEBACK_LLM_LOCAL_SERVER={fake_llama_server.binary}\n"
    )

    result = runner.invoke(cli, ["status"])

    assert result.exit_code == 0, result.output
    expected = f"LLM: local, model Qwen3.5-4B-Q4_K_M.gguf, server {fake_llama_server.binary}"
    assert expected in result.output.splitlines()


def test_status_says_when_the_local_server_is_missing(runner, real_env_files):
    user_config, _ = real_env_files
    user_config.write_text(
        "TAPEBACK_LLM_PROVIDER=local\nTAPEBACK_LLM_LOCAL_SERVER=no-such-llama-server\n"
    )

    result = runner.invoke(cli, ["status"])

    assert result.exit_code == 0, result.output
    expected = "LLM: local, model Qwen3.5-4B-Q4_K_M.gguf, server no-such-llama-server not found"
    assert expected in result.output.splitlines()


def test_setup_points_to_the_readme_when_the_local_server_is_missing(runner, real_env_files):
    user_config, _ = real_env_files
    user_config.write_text(
        "TAPEBACK_LLM_PROVIDER=local\n"
        "TAPEBACK_LLM_LOCAL_SERVER=no-such-llama-server\n"
        "TAPEBACK_HF_TOKEN=hf_fake\n"
    )

    result = runner.invoke(cli, ["setup"])

    assert result.exit_code == 0, result.output
    hints = [line for line in result.output.splitlines() if line.startswith("Hint:")]
    assert hints == [
        "Hint: local summaries need no-such-llama-server; see Local summaries in the README."
    ]


def test_summarize_takes_the_local_provider_on_the_command_line(
    runner, real_env_files, tmp_vault, fake_llama_server
):
    user_config, _ = real_env_files
    user_config.write_text(
        f"TAPEBACK_VAULT_PATH={tmp_vault}\n"
        f"TAPEBACK_LLM_LOCAL_SERVER={fake_llama_server.binary}\n"
        f"TAPEBACK_LLM_LOCAL_MODEL={fake_llama_server.model}\n"
    )
    note = tmp_vault / "meetings" / "2026-10-16_10-00-00.md"
    note.parent.mkdir(parents=True)
    note.write_text(SAMPLE_MD)

    result = runner.invoke(cli, ["summarize", str(note), "--provider", "local"])

    assert result.exit_code == 0, result.output
    assert "Discussed the project plan and assigned tasks." in note.read_text()
    assert len(fake_llama_server.starts()) == 1


def test_a_503_while_the_model_loads_is_closed(monkeypatch):
    """llama-server answers 503 until the model is loaded, and the error carries an
    open response: left unclosed, every readiness poll leaked a descriptor."""
    body = io.BytesIO(b'{"error": "Loading model"}')

    def loading(*_args, **_kwargs):
        raise urllib.error.HTTPError("http://127.0.0.1:1/health", 503, "Loading", Message(), body)

    monkeypatch.setattr(_llama_server.urllib.request, "urlopen", loading)

    assert _llama_server._healthy(1) is False
    assert body.closed
