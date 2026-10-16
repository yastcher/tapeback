"""GPU helpers — shared between the pipeline, transcriber, and diarizer."""

import contextlib
import ctypes
import gc
import importlib
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

# Seconds between nvidia-smi samples during a long GPU stage. nvidia-smi costs a
# process spawn, so this is deliberately coarse — the question it answers ("did the
# card throttle during this run") does not need fine resolution.
GPU_SAMPLE_INTERVAL_SEC = 5.0

# The one binary every GPU query goes through (query_nvidia_smi).
NVIDIA_SMI = "nvidia-smi"
# nvidia-smi must never hold up the pipeline; a hung driver query is not worth waiting on.
NVIDIA_SMI_TIMEOUT_SEC = 5

# Bits of clocks_event_reasons.active that mean the card is actually being held back.
# GpuIdle (0x1) and ApplicationsClocksSetting (0x2) are excluded: idling is not
# throttling, and an applied clock cap is a deliberate setting, not a symptom.
THROTTLE_SW_POWER_CAP = 0x0000000000000004
THROTTLE_HW_SLOWDOWN = 0x0000000000000008
THROTTLE_SW_THERMAL = 0x0000000000000020
THROTTLE_HW_THERMAL = 0x0000000000000040
THROTTLE_HW_POWER_BRAKE = 0x0000000000000080
THROTTLE_MASK = (
    THROTTLE_SW_POWER_CAP
    | THROTTLE_HW_SLOWDOWN
    | THROTTLE_SW_THERMAL
    | THROTTLE_HW_THERMAL
    | THROTTLE_HW_POWER_BRAKE
)

# Thermal slowdown specifically. On a laptop whose embedded controller clamps the GPU,
# these bits stay set at IDLE long after the temperature has dropped — a healthy idle
# card reports only GpuIdle (0x1). That difference is what makes the clamp detectable
# without running a load: SW power cap alone (0x4) is normal for a 50 W part under work.
THERMAL_CLAMP_MASK = THROTTLE_SW_THERMAL | THROTTLE_HW_THERMAL

# How often to say we are still waiting on the clamp. The poll itself is cheap; the
# reporting is what needs rationing, so a long wait stays visible without flooding.
CLAMP_REPORT_INTERVAL_SEC = 60.0

PERCENT = 100.0

# Substrings that mark a recoverable CUDA/GPU failure. ctranslate2 (faster-whisper)
# and torch (pyannote) phrase these differently, so match case-insensitively.
# A RuntimeError WITHOUT any of these must NOT trigger a CPU fallback — that would
# mask the real bug (e.g. a corrupt audio file) and make work ~10x slower.
_CUDA_ERROR_MARKERS = ("cuda", "out of memory", "cublas", "cudnn")

# faster-whisper's ctranslate2 is built against CUDA 12 and dlopens these by
# SONAME. When only CUDA 13 is on the default library path (e.g. torch's bundled
# cu13 libraries), the SONAME isn't found and transcription silently drops to CPU.
# Preloading the cu12 wheels' copies into the process resolves the SONAME.
_CUDA12_PRELOAD = (
    ("nvidia.cublas.lib", "libcublas.so.12"),
    ("nvidia.cudnn.lib", "libcudnn.so.9"),
)


def is_cuda_error(exc: BaseException) -> bool:
    """True if the exception looks like a recoverable CUDA/GPU failure."""
    message = str(exc).lower()
    return any(marker in message for marker in _CUDA_ERROR_MARKERS)


def preload_cuda_libs() -> None:
    """Preload CUDA 12 cuBLAS/cuDNN so ctranslate2 (faster-whisper) finds them.

    Best-effort: if the cu12 wheels (nvidia-cublas-cu12, nvidia-cudnn-cu12) are
    not installed, this is a no-op and transcription still falls back to CPU with
    a clear message. Saves users on CUDA 13 systems from setting LD_LIBRARY_PATH
    by hand.
    """
    for module_name, soname in _CUDA12_PRELOAD:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for entry in module.__path__:
            so_path = Path(entry) / soname
            if so_path.exists():
                with contextlib.suppress(OSError):
                    ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
                break


def query_nvidia_smi(fields: str) -> list[str] | None:
    """Query nvidia-smi for the first GPU. Returns field values, or None if unavailable.

    Single point of contact with nvidia-smi so that a missing driver, a missing
    binary and a hung query all degrade the same way — to None, never to an
    exception that would take down a transcription run.
    """
    try:
        result = subprocess.run(
            [NVIDIA_SMI, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_SEC,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    first_line = result.stdout.strip().split("\n")[0]
    if not first_line:
        return None
    return [value.strip() for value in first_line.split(",")]


def get_free_vram_mib() -> int | None:
    """Free GPU VRAM in MiB via nvidia-smi. Returns None if unavailable."""
    values = query_nvidia_smi("memory.free")
    if not values:
        return None
    try:
        return int(values[0])
    except ValueError:
        return None


def thermal_clamp_active() -> bool | None:
    """True if the GPU currently reports a thermal slowdown. None if unknown.

    Distinct from `sample_gpu`'s throttle accounting, which measures a stage after
    the fact. This is the "is it safe to measure anything right now" question.
    """
    values = query_nvidia_smi("clocks_event_reasons.active")
    if not values:
        return None
    try:
        reasons = int(values[0], 16)
    except ValueError:
        return None
    return bool(reasons & THERMAL_CLAMP_MASK)


def wait_for_clamp_release(
    timeout: float,
    *,
    poll_interval: float = GPU_SAMPLE_INTERVAL_SEC,
    report: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Block until the thermal clamp releases, or ``timeout`` seconds pass.

    A benchmark run on a clamped card measures the clamp, not the configuration:
    the same setting timed 3519 s clamped against 41 s unclamped here. Without this
    wait, a long grid silently produces a table of thermal state.

    Returns True if the card is (or became) clear, False on timeout. Returns True
    when the clamp cannot be read at all, so a machine without nvidia-smi is never
    blocked by a check it cannot perform.

    Progress is reported at most every CLAMP_REPORT_INTERVAL_SEC, with the elapsed
    time: a wait can legitimately run for minutes, and one that prints nothing looks
    exactly like a hang. Reporting every poll would be 180 lines over a 15-minute wait.
    """
    started = clock()
    deadline = started + timeout
    last_report = -CLAMP_REPORT_INTERVAL_SEC
    while True:
        clamped = thermal_clamp_active()
        if clamped is None or not clamped:
            elapsed = clock() - started
            if report and elapsed >= CLAMP_REPORT_INTERVAL_SEC:
                report(f"GPU thermal clamp released after {elapsed:.0f}s.")
            return True
        if clock() >= deadline:
            if report:
                report(
                    f"GPU still thermally clamped after {timeout:.0f}s — proceeding anyway. "
                    "It clears on idle, so a busy machine may never release it."
                )
            return False
        elapsed = clock() - started
        if report and elapsed - last_report >= CLAMP_REPORT_INTERVAL_SEC:
            last_report = elapsed
            report(
                f"GPU thermally clamped, waiting for it to release "
                f"({elapsed:.0f}s of {timeout:.0f}s)..."
            )
        sleep(poll_interval)


@dataclass(frozen=True)
class GpuStats:
    """Aggregated GPU telemetry over one pipeline stage."""

    samples: int
    sm_avg_mhz: float
    sm_min_mhz: int
    max_temp_c: int
    max_vram_mib: int
    throttled_samples: int

    @property
    def throttled_percent(self) -> float:
        """Share of samples where the card was thermally or power limited."""
        if self.samples == 0:
            return 0.0
        return self.throttled_samples / self.samples * PERCENT

    def format(self) -> str:
        """One-line, human-readable GPU record for a stage."""
        return (
            f"GPU: sm {self.sm_avg_mhz:.0f} MHz avg / {self.sm_min_mhz} min, "
            f"max {self.max_temp_c}°C, {self.max_vram_mib} MiB peak, "
            f"throttled {self.throttled_percent:.0f}% of {self.samples} samples"
        )


def _parse_gpu_sample(values: list[str]) -> tuple[int, int, int, int] | None:
    """Parse one nvidia-smi row into (sm MHz, temp °C, VRAM MiB, throttle bitmask)."""
    expected_fields = 4
    if len(values) < expected_fields:
        return None
    try:
        sm_mhz = int(values[0])
        temp_c = int(values[1])
        vram_mib = int(values[2])
        # The bitmask comes back as "0x0000000000000004".
        reasons = int(values[3], 16)
    except ValueError:
        return None
    return sm_mhz, temp_c, vram_mib, reasons


class _GpuSampler:
    """Background nvidia-smi poller. Collects samples until stopped."""

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self._first_poll = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._sm: list[int] = []
        self._temps: list[int] = []
        self._vram: list[int] = []
        self._throttled = 0

    def _poll_once(self) -> None:
        values = query_nvidia_smi(
            "clocks.sm,temperature.gpu,memory.used,clocks_event_reasons.active"
        )
        sample = _parse_gpu_sample(values) if values else None
        if sample is None:
            return
        sm_mhz, temp_c, vram_mib, reasons = sample
        self._sm.append(sm_mhz)
        self._temps.append(temp_c)
        self._vram.append(vram_mib)
        if reasons & THROTTLE_MASK:
            self._throttled += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            self._poll_once()
            self._first_poll.set()
            # Waiting on the event (not sleep) so stop() returns promptly.
            self._stop.wait(self._interval)

    def start(self) -> None:
        """Start sampling, returning only once the first poll has been attempted.

        Without this a stage shorter than the sampling interval would report
        nothing at all, and the wait makes the sampler's output deterministic
        rather than dependent on thread scheduling.
        """
        self._thread.start()
        self._first_poll.wait(timeout=NVIDIA_SMI_TIMEOUT_SEC)

    def stop(self) -> GpuStats | None:
        self._stop.set()
        self._thread.join(timeout=NVIDIA_SMI_TIMEOUT_SEC + self._interval)
        if not self._sm:
            return None
        return GpuStats(
            samples=len(self._sm),
            sm_avg_mhz=sum(self._sm) / len(self._sm),
            sm_min_mhz=min(self._sm),
            max_temp_c=max(self._temps),
            max_vram_mib=max(self._vram),
            throttled_samples=self._throttled,
        )


@contextmanager
def sample_gpu(
    report: Callable[[str], None],
    *,
    enabled: bool = True,
    interval: float = GPU_SAMPLE_INTERVAL_SEC,
) -> Iterator[None]:
    """Sample GPU clocks/temperature for the duration of the block, then report once.

    On this class of laptop the embedded controller clamps the GPU once the chassis
    saturates, and clocks collapse for the rest of the run. Without this the only
    symptom is "transcription got slow", indistinguishable from a bad model choice.

    Reports nothing when nvidia-smi is unavailable — a CPU-only machine should not
    grow a warning it can do nothing about.
    """
    if not enabled:
        yield
        return
    sampler = _GpuSampler(interval)
    sampler.start()
    try:
        yield
    finally:
        stats = sampler.stop()
        if stats is not None:
            report(stats.format())


def free_gpu_memory() -> None:
    """Release Python refs and clear CUDA cache so the next model fits in VRAM."""
    gc.collect()
    try:
        import torch  # noqa: PLC0415 — optional dependency, guarded by try/except

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
