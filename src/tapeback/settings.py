from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TAPEBACK_",
        env_file=(Path.home() / ".config" / "tapeback" / ".env", ".env"),
        env_file_encoding="utf-8",
    )

    # Output directory (Obsidian vault or any folder)
    vault_path: Path = Path.home() / "tapeback"

    # Subdirectories in vault
    meetings_dir: str = "meetings"
    attachments_dir: str = "attachments/audio"

    # Whisper
    whisper_model: str = "large-v3-turbo"
    language: str = "auto"
    device: str = "cuda"
    compute_type: str = "auto"  # "int8"/"float16"
    beam_size: int = 5
    vad_filter: bool = True
    chunk_length: int = 7  # seconds — max VAD chunk before splitting for Whisper
    condition_on_previous_text: bool = False
    # Lower = more aggressive silence rejection (helps suppress Whisper training-data
    # hallucinations like "Субтитры DimaTorzok" on long pauses). Default in Whisper is 0.6.
    no_speech_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

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

    # Live transcription — opt-in. Off by default because mid-recording GPU
    # contention with the post-recording pipeline causes long stalls on small
    # cards (4 GiB).  Enable with TAPEBACK_LIVE=true when VRAM is plentiful.
    live: bool = False
    live_interval: int = Field(default=60, gt=0)  # seconds between transcription cycles
    live_overlap: float = Field(default=2.0, ge=0.0)  # seconds of overlap between chunks
    live_min_chunk: float = Field(default=5.0, gt=0.0)  # min new audio to trigger transcription

    # Summarization
    summarize: bool = True
    llm_provider: LLMProvider = "anthropic"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""

    @model_validator(mode="after")
    def _validate_live_chunking(self) -> "Settings":
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
