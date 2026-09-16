from pathlib import Path
from typing import Literal, Self
from warnings import warn

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from tapeback.glossary import DEFAULT_HOTWORDS

# Default models per provider — used when TAPEBACK_LLM_MODEL is not set.
# Update here when providers deprecate models.
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.5-flash",
    "openrouter": "google/gemini-2.5-flash:free",
    "deepseek": "deepseek-chat",
    "qwen": "qwen-turbo",
}

type LLMProvider = Literal[
    "anthropic",
    "openai",
    "groq",
    "gemini",
    "openrouter",
    "deepseek",
    "qwen",
]

# "local" = faster-whisper on this machine; other values are remote STT backends.
# Today the only remote backend is "openai" (Audio Transcriptions API).
# Remote STT uploads meeting audio off-machine — opt-in only.
type SttBackend = Literal["local", "openai"]

DEFAULT_LOCAL_STT_MODEL = "large-v3-turbo"
# OpenAI remote default: whisper-1 keeps timestamps so local pyannote still works.
# Switch to gpt-transcribe or gpt-4o-transcribe-diarize via TAPEBACK_STT_MODEL.
DEFAULT_REMOTE_STT_MODEL = "whisper-1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TAPEBACK_",
        env_file=(Path.home() / ".config" / "tapeback" / ".env", ".env"),
        env_file_encoding="utf-8",
        populate_by_name=True,
    )

    # Output directory (Obsidian vault or any folder)
    vault_path: Path = Path.home() / "tapeback"

    # Subdirectories in vault
    meetings_dir: str = "meetings"
    attachments_dir: str = "attachments/audio"

    # STT backend. Default stays fully local (faster-whisper).
    # Set TAPEBACK_STT_BACKEND=openai for the OpenAI remote backend — requires
    # tapeback[stt] (or tapeback[llm], which also installs the openai SDK) and
    # TAPEBACK_STT_API_KEY (or OPENAI_API_KEY). Meeting audio then leaves the
    # machine; see README privacy notes.
    stt_backend: SttBackend = "local"
    # Model id for the active STT backend. Local default large-v3-turbo; OpenAI
    # remote default whisper-1. Empty resolves in the validator.
    stt_model: str = ""
    # Deprecated: prefer TAPEBACK_STT_MODEL. Still read so old .env files work.
    whisper_model: str = ""
    # Max concurrent remote uploads when a long recording is sliced into chunks.
    stt_concurrency: int = Field(default=4, ge=1)
    # API key for remote STT (TAPEBACK_STT_API_KEY). Falls back to OPENAI_API_KEY
    # then TAPEBACK_LLM_API_KEY when llm_provider=openai.
    stt_api_key: SecretStr = SecretStr("")

    # Decoding / device (local backend; remote ignores most of these)
    language: str = "auto"
    device: str = "cuda"
    compute_type: str = "auto"  # "int8"/"float16"
    beam_size: int = 4
    # Temperature fallback ladder. Decoding starts at 0.0 (deterministic, best
    # quality) and steps up only when a segment decodes poorly. The HIGH steps are
    # what break Whisper out of hallucination loops on noisy/quiet input — keep the
    # full ladder. Shortening it makes the model get STUCK in repeat loops, which is
    # both slower (it generates tokens up to the limit) and worse (repeats in text).
    temperature: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    # Batched inference (faster-whisper BatchedInferencePipeline): processes VAD
    # segments in parallel batches — several times faster on GPU. Off by default
    # (0) since it can OOM small GPUs; set e.g. TAPEBACK_BATCH_SIZE=8 to enable.
    batch_size: int = Field(default=0, ge=0)
    # Comma-separated terms to bias decoding towards (faster-whisper `hotwords`).
    # Default glossary and the reasoning behind it live in glossary.py.
    # Empty disables the bias entirely.
    hotwords: str = DEFAULT_HOTWORDS
    vad_filter: bool = True
    # Seconds of audio consumed per decode window. Whisper's encoder is FIXED at 30 s:
    # faster-whisper zero-pads every window back to 3000 mel frames before encoding
    # (transcribe.py `pad_or_trim`), so a smaller value does not make the encoder pass
    # cheaper — it only makes the run need more of them. At 2 the encoder does 15x the
    # necessary work; measured on a 145 s file, 2 -> 390.6 s and 30 -> 41.4 s.
    # Do not lower this to fight hallucinations on long pauses; that is what
    # vad_filter, no_speech_threshold and gate_mic_silence are for.
    chunk_length: int = 30
    condition_on_previous_text: bool = False
    # Lower = more aggressive silence rejection (helps suppress Whisper training-data
    # hallucinations like "Субтитры DimaTorzok" on long pauses). Default in Whisper is 0.6.
    no_speech_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    # Number of segments probed before deciding the language. faster-whisper's default
    # of 1 picks the language from the first segment, which misfires when a channel
    # starts with silence (e.g. mic while the user only listens — it can guess Japanese
    # and hallucinate). Raise it to probe more speech before committing.
    language_detection_segments: int = Field(default=1, ge=1)
    # Per-segment language detection — allows mixed languages in one recording
    # (code-switching, e.g. Russian speech with English terms). Less stable than a
    # fixed language; opt-in.
    multilingual: bool = False
    # Skip silent gaps longer than this many seconds when a hallucination is detected
    # (uses word timestamps, which are always on). None disables it.
    hallucination_silence_threshold: float | None = Field(default=None, ge=0.0)

    # Write one JSON record per processing run (config + status lines + outcome), so a
    # run that failed or was interrupted can be diagnosed afterwards. Never contains
    # credentials — the recorded fields are an explicit allow-list in _runlog.py.
    run_log: bool = True
    # None → XDG data dir (~/.local/share/tapeback/runs).
    run_log_dir: Path | None = None

    # Sample GPU clocks/temperature during transcription and report a one-line summary
    # per stage. Observation only — tapeback never changes clock or power caps (that
    # needs root). No-op when nvidia-smi is unavailable.
    gpu_telemetry: bool = True

    # Reuse a channel that was already transcribed with the same audio and the same
    # output-affecting settings, so an interrupted run does not redo finished work.
    # Granularity is a whole channel — see _resume.py for why not finer.
    resume_cache: bool = True
    # None → XDG data dir (~/.local/share/tapeback/resume).
    resume_cache_dir: Path | None = None

    # Run transcription in a child process. A CUDA out-of-memory leaks its allocation
    # for the life of the process it happened in, and nothing reachable from Python
    # releases it — but a process that exits gives the memory back. Costs one process
    # start and the model load per run; set false to transcribe in-process.
    isolate_transcription: bool = True

    # Refuse to load a model on CUDA below this much free VRAM, and use the CPU instead.
    # A CUDA out-of-memory during load leaks the allocation on ctranslate2's C++ side
    # for the life of the process (see transcriber._enough_vram), so the only reliable
    # cure is not to trigger it. The smallest configuration measured needs ~1115 MiB.
    min_free_vram_mib: int = Field(default=1200, ge=0)

    # Look for a GPU thermal clamp before each transcription stage. On laptops that
    # share one heatsink between CPU and GPU, the controller can cut the GPU's power
    # budget (measured: 50 W -> 5 W, clocks pinned to 300 MHz) and hold it there while
    # the CPU stays hot. The check is one nvidia-smi query and is retaken per stage, so
    # a clamp that clears between channels returns the run to the GPU by itself.
    thermal_clamp_check: bool = True
    # Seconds to wait for the clamp to release before giving up on the GPU for this
    # stage. Default 0 — do not wait. The clamp clears on system idle, and the shortest
    # release measured was 451 s, so any wait short enough to be tolerable essentially
    # never succeeds while meanwhile the CPU would already be transcribing at ~2.39x
    # real time against a clamped GPU's ~0.31x. Raise it only if the machine will
    # genuinely be idle.
    thermal_clamp_wait: float = Field(default=0.0, ge=0.0)
    # When the clamp has not released by then, transcribe on CPU instead of on a card
    # limited to a tenth of its power. Measured on the same clip: CPU 2.39x real time,
    # clamped GPU 0.31x — the CPU is ~8x faster, so waiting it out is the worse option.
    thermal_clamp_cpu_fallback: bool = True
    # Idle gap after each transcription stage, to shed heat instead of driving the
    # chassis into the clamp in the first place. Off by default: it costs wall-clock on
    # a machine that cools adequately.
    stage_pause_seconds: float = Field(default=0.0, ge=0.0)

    # Audio
    monitor_source: str = "auto"
    mic_source: str = "auto"
    sample_rate: int = 48000

    # HuggingFace (for pyannote). SecretStr prevents leakage in repr/str/model_dump.
    hf_token: SecretStr = SecretStr("")

    # Diarization
    diarize: bool = True
    max_speakers: int | None = None
    clustering_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    # merge only near-identical spectral profiles
    spectral_merge_threshold: float = Field(default=0.96, ge=0.0, le=1.0)

    # Post-processing
    pause_threshold: float = Field(default=1.0, ge=0.0)  # split on word gaps >= this
    # Silence the mic channel where the user is listening (mic quiet or monitor
    # dominant) before transcription, so Whisper doesn't hallucinate loops on the
    # pauses (slow + garbage). Dual-channel (stereo) pipeline only.
    gate_mic_silence: bool = True

    # Live transcription — opt-in. Off by default because mid-recording GPU
    # contention with the post-recording pipeline causes long stalls on small
    # cards (4 GiB).  Enable with TAPEBACK_LIVE=true when VRAM is plentiful.
    live: bool = False
    live_interval: int = Field(default=60, gt=0)  # seconds between transcription cycles
    live_overlap: float = Field(default=2.0, ge=0.0)  # seconds of overlap between chunks
    live_min_chunk: float = Field(default=5.0, gt=0.0)  # min new audio to trigger transcription

    # Summarization
    summarize: bool = True
    # Replace emails and phone numbers with placeholders in the transcript before it is
    # sent to an LLM provider, and restore the real values in the summary written to the
    # vault. Off by default. Only has any effect while summarization runs — masking does
    # not apply to remote STT (audio leaves the machine as-is when a remote backend is on).
    # See _mask.py.
    mask_pii: bool = False
    # Comma-separated literal terms to mask alongside emails and phone numbers — names,
    # company and project names, which is the PII people actually speak aloud. Matching
    # is case-insensitive and word-bounded, and LITERAL: a term is masked only in the
    # exact forms listed, so an inflected language needs each form spelled out. Ignored
    # unless mask_pii is on.
    mask_terms: str = ""
    llm_provider: LLMProvider = "anthropic"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""

    @model_validator(mode="after")
    def _resolve_stt_settings(self) -> Self:
        """Resolve stt_model defaults; accept deprecated TAPEBACK_WHISPER_MODEL."""
        model = self.stt_model.strip() if self.stt_model else ""
        if not model and self.whisper_model:
            warn(
                "TAPEBACK_WHISPER_MODEL is deprecated; use TAPEBACK_STT_MODEL instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            model = self.whisper_model.strip()
        if not model:
            model = (
                DEFAULT_REMOTE_STT_MODEL
                if self.stt_backend == "openai"
                else DEFAULT_LOCAL_STT_MODEL
            )

        object.__setattr__(self, "stt_model", model)
        object.__setattr__(self, "whisper_model", model)
        return self

    @model_validator(mode="after")
    def _validate_live_chunking(self) -> Self:
        """Live chunks must be shorter than the interval or the loop drops audio."""
        if self.live and self.live_min_chunk > self.live_interval:
            raise ValueError(
                f"live_min_chunk ({self.live_min_chunk}s) must be <= "
                f"live_interval ({self.live_interval}s); otherwise cycles starve."
            )
        return self


def get_settings() -> Settings:
    """Load settings from env vars and .env files."""
    return Settings()  # type: ignore[call-arg]
