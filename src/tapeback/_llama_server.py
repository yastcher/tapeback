"""A llama-server tapeback starts for one summary and stops right after it.

Transcription and diarization have given the GPU back by the time the summary starts,
and the model gives it back before the next recording needs Whisper. The flags and the
model are the ones measured on a 4 GB Turing GPU, where they take 3.1-3.5 GB of VRAM
(docs/spec/2026-10-15-setup-and-local-llm.md).
"""

import fcntl
import hashlib
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

import click
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

from tapeback._gpu import get_free_vram_mib, wait_for_clamp_release
from tapeback.recorder import default_state_dir, ensure_private_dir
from tapeback.settings import Settings

HOST = "127.0.0.1"

# As measured: every layer on the GPU, flash attention, an 8-bit KV cache, no thinking
# (it made answers many times slower), no prompt cache in RAM (its 8 GiB default pushed
# the host into swap), one slot, and no web UI.
SERVER_FLAGS = (
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
)

# What llama.cpp writes when the model or its context does not fit in VRAM: CUDA's
# allocator, Vulkan's, and the KV cache's, after which older builds crash outright.
OUT_OF_MEMORY_MARKERS = (
    "out of memory",
    "erroroutofdevicememory",
    "failed to initialize the context",
)

# Seconds. Loading reads the whole model from disk first: 2.7 GB from a cold cache.
READY_TIMEOUT = 300.0
POLL_INTERVAL = 0.25
HEALTH_TIMEOUT = 1.0
STOP_TIMEOUT = 10.0
# Lines of the server's log quoted when it fails to start.
LOG_TAIL_LINES = 20
HASH_CHUNK = 1 << 20

# Under tapeback's state directory: the lock, the PID file and the server's log.
STATE_SUBDIR = "llama-server"


@dataclass(frozen=True)
class PinnedModel:
    """A GGUF fixed by revision and checksum, so every install summarises with the same
    weights."""

    repo: str
    filename: str
    revision: str
    sha256: str
    size: int


DEFAULT_MODEL = PinnedModel(
    repo="unsloth/Qwen3.5-4B-GGUF",
    filename="Qwen3.5-4B-Q4_K_M.gguf",
    revision="e87f176479d0855a907a41277aca2f8ee7a09523",
    sha256="00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
    size=2740937888,
)


@dataclass(frozen=True)
class Server:
    url: str  # the OpenAI-compatible base, http://127.0.0.1:<port>/v1
    key: str  # random for each start; reaches the server through LLAMA_API_KEY, not argv
    model: str  # the name requests carry


class OutOfMemory(RuntimeError):
    """The model or its context did not fit in VRAM."""


def model_name(settings: Settings) -> str:
    """The model as a status line names it."""
    if settings.llm_local_model:
        return Path(settings.llm_local_model).name
    return DEFAULT_MODEL.filename


def find_binary(name: str) -> str | None:
    """llama-server on PATH or next to tapeback's interpreter, where the .deb companion
    puts it. A name with a slash is taken as a path."""
    here = str(Path(sys.executable).parent)
    path = os.pathsep.join(filter(None, (os.environ.get("PATH"), here)))
    return shutil.which(name, path=path)


@contextmanager
def running(settings: Settings) -> Iterator[Server]:
    """A server for the block: started on a free loopback port, stopped on the way out,
    Ctrl+C included. Running out of VRAM while loading halves the context once."""
    binary = find_binary(settings.llm_local_server)
    if binary is None:
        raise RuntimeError(
            f"{settings.llm_local_server} not found: install llama.cpp "
            "(Local summaries in the README) or set TAPEBACK_LLM_LOCAL_SERVER."
        )
    model = model_path(settings)
    home = default_state_dir() / STATE_SUBDIR
    ensure_private_dir(home)
    with _exclusive(home / "lock"):
        _reap_orphan(home / "pid")
        if settings.thermal_clamp_check:
            # The last stage, so a cool card is worth waiting for: clamped, an hour's
            # prompt took longer than the wait would.
            wait_for_clamp_release(settings.llm_local_clamp_wait, report=_warn)
        context = settings.llm_local_context
        try:
            process, server = _start(binary, model, context, home)
        except OutOfMemory as exc:
            context //= 2
            click.echo(f"Warning: {exc}; retrying with a {context}-token context.", err=True)
            process, server = _start(binary, model, context, home)
        try:
            yield server
        finally:
            _stop(process)
            (home / "pid").unlink(missing_ok=True)


def _warn(message: str) -> None:
    click.echo(message, err=True)


def model_path(settings: Settings) -> Path:
    """TAPEBACK_LLM_LOCAL_MODEL, or the pinned default, downloaded on first use."""
    if settings.llm_local_model:
        path = Path(settings.llm_local_model).expanduser()
        if not path.is_file():
            raise RuntimeError(f"TAPEBACK_LLM_LOCAL_MODEL: no file at {path}")
        return path
    return _pinned(DEFAULT_MODEL)


def _pinned(model: PinnedModel) -> Path:
    """The pinned file from the Hugging Face cache, downloading it the first time.

    The checksum is computed once, after the download; a later start checks the size,
    which an interrupted copy gets wrong, without reading 2.7 GB again.
    """
    try:
        cached = hf_hub_download(
            model.repo, model.filename, revision=model.revision, local_files_only=True
        )
    except LocalEntryNotFoundError:
        cached = None
    path = Path(cached) if cached else _download(model)
    size = path.stat().st_size
    if size != model.size:
        raise RuntimeError(
            f"{path} has {size} bytes, not the {model.size} pinned; delete it and retry."
        )
    return path


def _download(model: PinnedModel) -> Path:
    click.echo(
        f"Downloading the summary model {model.filename} ({model.size / 2**30:.1f} GiB), once...",
        err=True,
    )
    path = Path(hf_hub_download(model.repo, model.filename, revision=model.revision))
    digest = _sha256(path)
    if digest != model.sha256:
        raise RuntimeError(f"{path} does not match its pinned checksum; delete it and retry.")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _exclusive(lock: Path) -> Iterator[None]:
    """One local server at a time: a small card holds one model, and the PID file
    belongs to whoever holds this lock. The kernel drops the lock when its holder dies,
    so a PID file found under it was left by a crash."""
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            click.echo("Waiting for another summary to free the local model...", err=True)
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _start(
    binary: str, model: Path, context: int, home: Path
) -> tuple[subprocess.Popen[bytes], Server]:
    port = _free_port()
    key = secrets.token_urlsafe(32)
    log = home / "log"
    log_fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        process = subprocess.Popen(
            [
                binary,
                *("--model", str(model), "--ctx-size", str(context)),
                *("--host", HOST, "--port", str(port)),
                *SERVER_FLAGS,
            ],
            stdin=subprocess.DEVNULL,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            env={**os.environ, "LLAMA_API_KEY": key},
            # ggml loads its backend libraries from the working directory too: a
            # directory of our own, never wherever tapeback was started.
            cwd=home,
        )
    finally:
        os.close(log_fd)
    # The start time tells this process from a later one that reuses its PID.
    (home / "pid").write_text(f"{process.pid} {_start_time(process.pid)}\n")
    try:
        _wait_ready(process, port, log, context)
    except BaseException:
        _stop(process)
        (home / "pid").unlink(missing_ok=True)
        raise
    return process, Server(f"http://{HOST}:{port}/v1", key, model.stem)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]


def _wait_ready(process: subprocess.Popen[bytes], port: int, log: Path, context: int) -> None:
    deadline = time.monotonic() + READY_TIMEOUT
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            tail = _tail(log)
            if any(marker in tail.lower() for marker in OUT_OF_MEMORY_MARKERS):
                # The default model barely grows with its context, so what is free
                # tells more than the context does: often another program holds it.
                free = get_free_vram_mib()
                held = f" ({free} MiB of VRAM free)" if free is not None else ""
                raise OutOfMemory(
                    f"the summary model does not fit in VRAM with a {context}-token context{held}"
                )
            raise RuntimeError(f"llama-server exited with code {code} while loading:\n{tail}")
        if _healthy(port):
            return
        time.sleep(POLL_INTERVAL)
    raise RuntimeError(f"llama-server did not load the model in {READY_TIMEOUT:.0f} s; see {log}")


def _healthy(port: int) -> bool:
    """True once the model is loaded: /health answers 503 while it loads."""
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/health", timeout=HEALTH_TIMEOUT):
            return True
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return False


def _tail(log: Path) -> str:
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-LOG_TAIL_LINES:])


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=STOP_TIMEOUT)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _reap_orphan(pid_file: Path) -> None:
    """Stop a server a crashed run left behind: it would hold the VRAM until reboot."""
    try:
        pid_text, started = pid_file.read_text().split()
        pid = int(pid_text)
    except (FileNotFoundError, ValueError):
        return
    if _start_time(pid) == started:
        click.echo(f"Stopping a llama-server an earlier run left behind (pid {pid}).", err=True)
        _kill(pid)
    pid_file.unlink(missing_ok=True)


def _kill(pid: int) -> None:
    """SIGTERM, then SIGKILL, each followed by a wait: the VRAM is free only once the
    process is gone. Not our child, so its new parent reaps it."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with suppress(ProcessLookupError):
            os.kill(pid, sig)
        deadline = time.monotonic() + STOP_TIMEOUT
        while time.monotonic() < deadline:
            if _start_time(pid) is None:
                return
            time.sleep(POLL_INTERVAL)


def _start_time(pid: int) -> str | None:
    """Field 22 of /proc/<pid>/stat, or None once the process is gone or a zombie."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = stat[stat.rindex(")") + 2 :].split()
    if fields[0] == "Z":
        return None
    return fields[19]
