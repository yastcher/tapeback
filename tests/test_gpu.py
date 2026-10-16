"""Unit tests for GPU error classification, CUDA preloading, and GPU telemetry."""

import subprocess
from types import SimpleNamespace

import pytest

from tapeback import _gpu
from tapeback._gpu import (
    GpuStats,
    get_free_vram_mib,
    is_cuda_error,
    preload_cuda_libs,
    query_nvidia_smi,
    sample_gpu,
    thermal_clamp_active,
    wait_for_clamp_release,
)


@pytest.mark.parametrize(
    "message,expected",
    [
        ("CUDA failed with error out of memory", True),
        ("CUDA out of memory", True),
        ("Library libcublas.so.12 is not found", True),
        ("cuDNN failed to initialize", True),
        ("no CUDA-capable device is detected", True),
        ("corrupt audio frame at offset 1234", False),
        ("invalid sample rate", False),
    ],
)
def test_is_cuda_error(message, expected):
    assert is_cuda_error(RuntimeError(message)) is expected


def test_preload_cuda_libs_loads_present_libraries(monkeypatch, tmp_path):
    """When the cu12 wheels are installed, their cuBLAS/cuDNN get preloaded."""
    cublas = tmp_path / "cublas"
    cublas.mkdir()
    (cublas / "libcublas.so.12").touch()
    cudnn = tmp_path / "cudnn"
    cudnn.mkdir()
    (cudnn / "libcudnn.so.9").touch()

    fake = {
        "nvidia.cublas.lib": SimpleNamespace(__path__=[str(cublas)]),
        "nvidia.cudnn.lib": SimpleNamespace(__path__=[str(cudnn)]),
    }
    monkeypatch.setattr(_gpu.importlib, "import_module", lambda name: fake[name])
    loaded: list[str] = []
    monkeypatch.setattr(_gpu.ctypes, "CDLL", lambda path, mode=0: loaded.append(path))

    preload_cuda_libs()

    assert any(p.endswith("libcublas.so.12") for p in loaded)
    assert any(p.endswith("libcudnn.so.9") for p in loaded)


def test_preload_cuda_libs_noop_when_wheels_absent(monkeypatch):
    """Without the cu12 wheels, preload is a silent no-op (CPU fallback still works)."""

    def _missing(name):
        raise ImportError(name)

    monkeypatch.setattr(_gpu.importlib, "import_module", _missing)
    loaded: list[str] = []
    monkeypatch.setattr(_gpu.ctypes, "CDLL", lambda path, mode=0: loaded.append(path))

    preload_cuda_libs()  # must not raise

    assert loaded == []


def _fake_run(stdout: str, returncode: int = 0):
    """Stand-in for subprocess.run returning a canned nvidia-smi response."""

    def _run(*_args, **_kwargs):
        return SimpleNamespace(stdout=stdout, returncode=returncode)

    return _run


def test_query_nvidia_smi_splits_first_row(monkeypatch):
    monkeypatch.setattr(_gpu.subprocess, "run", _fake_run("1830, 83, 2428, 0x0000000000000004\n"))
    assert query_nvidia_smi(
        "clocks.sm,temperature.gpu,memory.used,clocks_event_reasons.active"
    ) == [
        "1830",
        "83",
        "2428",
        "0x0000000000000004",
    ]


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (FileNotFoundError("nvidia-smi"), "binary missing"),
        (subprocess.TimeoutExpired("nvidia-smi", 5), "driver hung"),
        (OSError("permission denied"), "not permitted"),
    ],
)
def test_query_nvidia_smi_returns_none_on_failure(monkeypatch, failure, reason):
    """A missing or hung nvidia-smi must never take down a transcription run."""

    def _raise(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(_gpu.subprocess, "run", _raise)
    assert query_nvidia_smi("memory.free") is None, reason


def test_query_nvidia_smi_returns_none_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(_gpu.subprocess, "run", _fake_run("", returncode=9))
    assert query_nvidia_smi("memory.free") is None


def test_the_suite_never_reads_the_real_gpu():
    """isolate_gpu keeps nvidia-smi out of reach, so no result depends on the card."""
    assert query_nvidia_smi("memory.free") is None


def test_get_free_vram_mib(monkeypatch):
    monkeypatch.setattr(_gpu.subprocess, "run", _fake_run("1620\n"))
    assert get_free_vram_mib() == 1620


def test_get_free_vram_mib_none_on_unparsable_output(monkeypatch):
    monkeypatch.setattr(_gpu.subprocess, "run", _fake_run("N/A\n"))
    assert get_free_vram_mib() is None


@pytest.mark.parametrize(
    ("reasons_hex", "expected"),
    [
        # A healthy idle card reports GpuIdle only.
        ("0x0000000000000001", False),
        # SW power cap alone is normal for a 50 W part under load, not a clamp.
        ("0x0000000000000004", False),
        # The real observed clamped-at-idle value on this laptop.
        ("0x0000000000000024", True),
        ("0x0000000000000020", True),
        ("0x0000000000000040", True),
    ],
)
def test_thermal_clamp_active(monkeypatch, reasons_hex, expected):
    monkeypatch.setattr(_gpu.subprocess, "run", _fake_run(f"{reasons_hex}\n"))
    assert thermal_clamp_active() is expected


def test_thermal_clamp_active_unknown_without_nvidia_smi(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(_gpu.subprocess, "run", _raise)
    assert thermal_clamp_active() is None


def test_wait_for_clamp_release_returns_when_clear(monkeypatch):
    monkeypatch.setattr(_gpu.subprocess, "run", _fake_run("0x0000000000000001\n"))
    slept: list[float] = []
    assert wait_for_clamp_release(60.0, sleep=slept.append) is True
    assert slept == []


def test_wait_for_clamp_release_polls_until_clear(monkeypatch):
    responses = iter(["0x24\n", "0x24\n", "0x1\n"])

    def _run(*_args, **_kwargs):
        return SimpleNamespace(stdout=next(responses), returncode=0)

    monkeypatch.setattr(_gpu.subprocess, "run", _run)
    slept: list[float] = []
    reported: list[str] = []
    assert (
        wait_for_clamp_release(60.0, poll_interval=5.0, report=reported.append, sleep=slept.append)
        is True
    )
    assert slept == [5.0, 5.0]
    # One opening line; the rest are rationed to CLAMP_REPORT_INTERVAL_SEC and this
    # wait is far shorter than that.
    assert len(reported) == 1
    assert "waiting for it to release" in reported[0]


def test_wait_for_clamp_release_rations_its_reporting(monkeypatch):
    """A long wait must stay visible without printing on every poll.

    At a 5 s poll and a 15-minute timeout, reporting each poll would be 180 lines.
    """
    monkeypatch.setattr(_gpu.subprocess, "run", _fake_run("0x24\n"))
    # Clock advances a minute per poll, so every poll crosses the reporting interval.
    ticks = iter([0.0, 0.0, 60.0, 120.0, 180.0, 900.0])
    reported: list[str] = []
    assert (
        wait_for_clamp_release(
            600.0,
            poll_interval=5.0,
            report=reported.append,
            clock=lambda: next(ticks),
            sleep=lambda _s: None,
        )
        is False
    )
    assert any("60s of 600s" in m for m in reported)
    assert "never release it" in reported[-1]


def test_wait_for_clamp_release_times_out(monkeypatch):
    monkeypatch.setattr(_gpu.subprocess, "run", _fake_run("0x24\n"))
    ticks = iter([0.0, 0.0, 10.0, 100.0])
    slept: list[float] = []
    reported: list[str] = []
    assert (
        wait_for_clamp_release(
            60.0,
            poll_interval=5.0,
            report=reported.append,
            clock=lambda: next(ticks),
            sleep=slept.append,
        )
        is False
    )
    assert "still thermally clamped" in reported[-1]


def test_wait_for_clamp_release_not_blocked_without_nvidia_smi(monkeypatch):
    """A machine that cannot answer the question must not be made to wait for it."""

    def _raise(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(_gpu.subprocess, "run", _raise)
    assert wait_for_clamp_release(60.0, sleep=lambda _s: None) is True


def test_gpu_stats_format():
    stats = GpuStats(
        samples=40,
        sm_avg_mhz=1832.5,
        sm_min_mhz=1005,
        max_temp_c=87,
        max_vram_mib=2428,
        throttled_samples=14,
    )
    assert stats.format() == (
        "GPU: sm 1832 MHz avg / 1005 min, max 87°C, 2428 MiB peak, throttled 35% of 40 samples"
    )


def test_gpu_stats_throttled_percent_with_no_samples():
    stats = GpuStats(
        samples=0, sm_avg_mhz=0.0, sm_min_mhz=0, max_temp_c=0, max_vram_mib=0, throttled_samples=0
    )
    assert stats.throttled_percent == 0.0


@pytest.mark.parametrize(
    ("reasons_hex", "counts_as_throttled"),
    [
        # GpuIdle and an applied clock cap are settings/states, not throttling.
        ("0x0000000000000001", False),
        ("0x0000000000000002", False),
        ("0x0000000000000004", True),  # SW power cap
        ("0x0000000000000008", True),  # HW slowdown
        ("0x0000000000000020", True),  # SW thermal
        ("0x0000000000000040", True),  # HW thermal
        ("0x0000000000000080", True),  # HW power brake
    ],
)
def test_sample_gpu_classifies_throttle_reasons(monkeypatch, reasons_hex, counts_as_throttled):
    monkeypatch.setattr(_gpu.subprocess, "run", _fake_run(f"1830, 83, 2428, {reasons_hex}\n"))

    reported: list[str] = []
    # sample_gpu waits for the first poll before entering the block, so even an
    # empty body yields exactly one sample.
    with sample_gpu(reported.append, interval=0.0):
        pass

    assert len(reported) == 1
    expected = "throttled 100%" if counts_as_throttled else "throttled 0%"
    assert expected in reported[0]


def test_sample_gpu_silent_when_nvidia_smi_unavailable(monkeypatch):
    """A CPU-only machine must not grow a warning it can do nothing about."""

    def _raise(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(_gpu.subprocess, "run", _raise)

    reported: list[str] = []
    with sample_gpu(reported.append, interval=0.0):
        pass

    assert reported == []


def test_sample_gpu_disabled_does_not_call_nvidia_smi(monkeypatch):
    calls: list[str] = []

    def _record(*args, **_kwargs):
        calls.append(str(args))
        return SimpleNamespace(stdout="", returncode=1)

    monkeypatch.setattr(_gpu.subprocess, "run", _record)

    reported: list[str] = []
    with sample_gpu(reported.append, enabled=False, interval=0.0):
        pass

    assert calls == []
    assert reported == []


def test_sample_gpu_reports_even_when_block_raises(monkeypatch):
    """Telemetry is the diagnostic for a failed run — it must survive the exception."""
    monkeypatch.setattr(_gpu.subprocess, "run", _fake_run("1830, 83, 2428, 0x1\n"))

    reported: list[str] = []
    with pytest.raises(ValueError, match="boom"), sample_gpu(reported.append, interval=0.0):
        raise ValueError("boom")

    assert len(reported) == 1
    assert reported[0].startswith("GPU: sm 1830 MHz avg")
