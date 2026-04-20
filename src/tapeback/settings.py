from pathlib import Path
from typing import Literal

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
    no_speech_threshold: float = 0.4

    # Audio
    monitor_source: str = "auto"
    mic_source: str = "auto"
    sample_rate: int = 48000

    # HuggingFace (for pyannote)
    hf_token: str = ""

    # Diarization
    diarize: bool = True
    max_speakers: int | None = None
    clustering_threshold: float | None = None
    spectral_merge_threshold: float = 0.96  # merge only near-identical spectral profiles

    # Post-processing
    pause_threshold: float = 1.0  # seconds — split segments on word gaps >= this

    # Live transcription
    live: bool = True  # enable live transcription during recording
    live_interval: int = 60  # seconds between transcription cycles
    live_overlap: float = 2.0  # seconds of overlap between chunks
    live_min_chunk: float = 5.0  # minimum new audio (seconds) to trigger transcription

    # Summarization
    summarize: bool = True
    llm_provider: LLMProvider = "anthropic"
    llm_api_key: str = ""
    llm_model: str = ""


def get_settings() -> Settings:
    """Load settings from env vars and .env files."""
    return Settings()  # type: ignore[call-arg]
