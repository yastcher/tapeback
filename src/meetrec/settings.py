from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEETREC_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Required — path to Obsidian vault
    vault_path: Path

    # Subdirectories in vault
    meetings_dir: str = "meetings"
    attachments_dir: str = "attachments/audio"

    # Whisper
    whisper_model: str = "large-v3-turbo"
    language: str = "en"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5
    vad_filter: bool = True

    # Audio
    monitor_source: str = "auto"
    mic_source: str = "auto"
    sample_rate: int = 48000

    # HuggingFace (for pyannote)
    hf_token: str = ""

    # Diarization
    diarize: bool = True
    max_speakers: int | None = None


def get_settings() -> Settings:
    """Load settings. Raises clear error if MEETREC_VAULT_PATH is not set."""
    try:
        return Settings()  # type: ignore[call-arg]
    except Exception:
        raise SystemExit(
            "Error: MEETREC_VAULT_PATH is required.\n"
            "Set it via environment variable or .env file:\n"
            "  export MEETREC_VAULT_PATH=/path/to/obsidian/vault\n"
        ) from None
