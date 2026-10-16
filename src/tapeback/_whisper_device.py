"""The device and compute type a Whisper model runs on, decided again for every stage."""

import sys

from tapeback._gpu import get_free_vram_mib, wait_for_clamp_release
from tapeback.settings import Settings

# Compute types ctranslate2 cannot run on CPU. Requesting one there raises
# ValueError rather than degrading, so a device fallback has to translate it.
_CUDA_ONLY_COMPUTE_TYPES = frozenset({"float16", "int8_float16", "bfloat16", "int8_bfloat16"})


def resolve_compute_type(compute_type: str, device: str) -> str:
    """Resolve the compute type for the device we ended up on.

    - auto + cuda → int8_float16
    - auto + cpu  → int8
    - an explicit CUDA-only type on CPU → int8, because ctranslate2 raises otherwise
    - any other explicit value passes through.

    The CPU translation matters because the device is now chosen at runtime: a card that
    is thermally clamped or out of VRAM sends us to the CPU carrying whatever
    TAPEBACK_COMPUTE_TYPE was set for the GPU. Without this, that combination died with
    "Requested int8_float16 compute type, but the target device or backend do not
    support efficient int8_float16 computation" — a crash instead of a fallback.

    int8_float16 rather than float16 because it is faster *and* smaller, which is not
    the usual trade-off. Measured on a GTX 1650 Ti with large-v3-turbo, same 90 s clip,
    twice each: float16 3.90x real time and 2139 MiB, int8_float16 **14.16x and
    1115 MiB**. Quality does not pay for it — decoding the same audio both ways gave
    near-identical text with single-word differences in both directions, and across the
    benchmark grid int8_float16 had the lower share of low-confidence words.

    The likely reason is hardware: this is a Turing part without tensor cores, so fp16
    gets no acceleration while int8 uses the integer datapath. That is a hypothesis;
    the measurements are not. ctranslate2 falls back on its own if a device does not
    support the requested type, so this stays safe on other GPUs.
    """
    if compute_type == "auto":
        return "int8_float16" if device == "cuda" else "int8"
    if device != "cuda" and compute_type in _CUDA_ONLY_COMPUTE_TYPES:
        print(
            f"Warning: compute type {compute_type} is GPU-only; using int8 on {device}.",
            file=sys.stderr,
        )
        return "int8"
    return compute_type


def _enough_vram(settings: Settings) -> bool:
    """False if the card plainly cannot hold a model right now.

    This is prevention, not optimisation. A CUDA out-of-memory during model load
    **leaks the allocation**: ctranslate2 builds the model on the C++ side, and when the
    load fails partway the object is never destroyed and never reaches Python, so there
    is no handle to release. Measured, free VRAM went 3674 MiB -> 95 MiB and stayed
    there for the life of the process; neither dropping the exception's traceback nor
    `CT2_CUDA_ALLOCATOR=cuda_malloc_async` recovers it. Everything afterwards — the
    diarizer's own VRAM check included — then finds an empty card.

    So the only reliable fix is to not attempt a load that cannot fit. The floor is
    deliberately crude: the smallest configuration measured here (large-v3-turbo in
    int8_float16) needs ~1115 MiB, so anything under the threshold cannot work at all.
    Sizing per model would need a table that goes stale; this catches the case that
    actually recurs, which is a card already occupied by something else.
    """
    free_mib = get_free_vram_mib()
    if free_mib is None or free_mib >= settings.min_free_vram_mib:
        return True
    print(
        f"Warning: only {free_mib} MiB VRAM free, below the "
        f"{settings.min_free_vram_mib} MiB needed to load a model — using CPU. "
        "A previous CUDA out-of-memory leaks VRAM until the process restarts.",
        file=sys.stderr,
    )
    return False


def resolve_device(settings: Settings) -> str:
    """Pick the device to actually run on, avoiding a thermally clamped GPU.

    On a laptop sharing one heatsink between CPU and GPU, the controller responds to a
    hot *system* by cutting the GPU's power budget — measured here as 50 W dropping to
    5 W with clocks pinned at 300 MHz while the GPU itself sat at 74 C and the CPU
    package at 93 C. It releases only on idle, and after sustained load it stayed
    latched for over 900 s.

    Transcribing on a card in that state is strictly worse than using the CPU: measured
    on the same clip, CPU 2.39x real time against 0.31x clamped, i.e. the CPU is ~8x
    faster. Waiting it out is what turned a fifteen-minute job into a multi-hour one.
    """
    if settings.device != "cuda":
        return settings.device
    if not _enough_vram(settings):
        return "cpu"
    if not settings.thermal_clamp_check:
        return settings.device

    # Checking is always cheap; waiting is the part that has to be justified. With a
    # zero wait this is a single query, which is what makes returning to the GPU work:
    # the decision is retaken for every stage, so a clamp that clears between channels
    # is picked up at the next one instead of stranding the whole run on the CPU.
    #
    # Reported, not silent: a wait can legitimately last minutes, and a process that
    # prints nothing while it does is indistinguishable from one that has hung.
    if wait_for_clamp_release(
        settings.thermal_clamp_wait,
        report=lambda message: print(message, file=sys.stderr),
    ):
        return "cuda"
    if not settings.thermal_clamp_cpu_fallback:
        print(
            "Warning: GPU is thermally clamped; transcription will be very slow. "
            "Set TAPEBACK_THERMAL_CLAMP_CPU_FALLBACK=true to use the CPU instead.",
            file=sys.stderr,
        )
        return "cuda"
    print(
        "Warning: GPU is thermally clamped and did not release — transcribing on CPU, "
        "which is faster in this state. Let the machine idle to clear it.",
        file=sys.stderr,
    )
    return "cpu"
