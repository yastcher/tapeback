"""Shared constants — values used across multiple modules."""

# Speaker labels
SPEAKER_YOU = "You"
SPEAKER_OTHER = "Other"
SPEAKER_LABEL_FMT = "Speaker {}"

# Audio file names (within output/temp directories)
FILE_STEREO = "stereo.wav"
FILE_MONO_16K = "mono_16k.wav"
FILE_MIC_16K = "mic_16k.wav"
FILE_MONITOR_16K = "monitor_16k.wav"
FILE_MIC = "mic.wav"
FILE_MONITOR = "monitor.wav"
FILE_SESSION = "session.json"

# Temp directory
TEMP_DIR = "/tmp/tapeback"

# Sample rates
SAMPLE_RATE_16K = 16000

# Remote STT upload limits (OpenAI documents a 25 MiB hard cap; stay under it).
# Bitrate chosen so ~50 minutes of mono speech fits in one upload at this margin.
STT_REMOTE_MAX_UPLOAD_BYTES = 24 * 1024 * 1024
STT_REMOTE_MP3_BITRATE_K = 64
# Leave headroom under the hard upload cap when sizing time slices from bitrate.
STT_REMOTE_UPLOAD_MARGIN = 0.95
# Truncate ffmpeg stderr in errors so logs stay readable.
STT_REMOTE_FFMPEG_ERROR_TAIL = 500
# OpenAI gpt-4o-transcribe-diarize requires chunking_strategy above this duration.
OPENAI_DIARIZE_CHUNKING_SECONDS = 30.0
# Hard API duration caps (size limit alone is not enough for these models).
OPENAI_DIARIZE_MAX_UPLOAD_SECONDS = 1400.0
OPENAI_GPT_TRANSCRIBE_MAX_UPLOAD_SECONDS = 1500.0

# Frames read per iteration when de-interleaving a stereo WAV. Large enough that the
# per-chunk overhead is irrelevant, small enough that the transient buffer (~4 MB at
# this size) does not matter next to the output arrays.
READ_CHUNK_FRAMES = 1_000_000

# Audio channel layout
STEREO_CHANNELS = 2
# Minimum kept sub-segment duration when splitting on silence without word timings
MIN_SUB_SEGMENT_DURATION_SEC = 0.5

# FFmpeg loudnorm parameters (EBU R128)
LOUDNORM_PARAMS = "I=-16:TP=-1.5:LRA=11"

# Channel energy classification
CHANNEL_ENERGY_RATIO = 2.0
CHANNEL_EPSILON = 1e-10

# Silence detection
SILENCE_WINDOW_SEC = 0.1
SILENCE_ADAPTIVE_FACTOR = 0.4
SILENCE_MONITOR_FACTOR = 0.3
# RMS energy floor (raw int16 scale) below which audio counts as silence
SILENCE_RMS_THRESHOLD = 200.0

# Spectral analysis
SPECTRAL_FFT_SIZE = 2048
SPECTRAL_MIN_FREQ_HZ = 100.0
SPECTRAL_MAX_FREQ_HZ = 4000.0

# Pyannote
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"

# PulseAudio/PipeWire
PA_DEFAULT_MONITOR = "@DEFAULT_MONITOR@"
PA_DEFAULT_SOURCE = "@DEFAULT_SOURCE@"
PA_MONITOR_SUFFIX = ".monitor"

# Duration warning threshold (seconds)
CHANNEL_DURATION_DIFF_WARN = 2.0

# Minimum segment duration for output (seconds)
MIN_SEGMENT_DURATION = 1.0

# Phrases Whisper emits from its subtitle training corpus rather than from the audio,
# typically over long pauses. Matched case-insensitively as substrings. Kept here
# because both the quality benchmark and the transcript filter must agree on the list.
HALLUCINATION_MARKERS = (
    "субтитры",
    "dimatorzok",
    "продолжение следует",
    "редактор субтитров",
    "корректор",
    "субтитлы",
    "amara.org",
    "субтитри",
    "thanks for watching",
    "thank you for watching",
    "подписывайтесь на канал",
)

# Live transcription
FILE_LIVE_SUFFIX = "_live"
WAV_HEADER_FALLBACK = 44
WAV_CHUNK_HEADER_BYTES = 4  # RIFF chunk id / size fields are 4 bytes each
RESAMPLE_FACTOR = 3  # 48000 / 16000

# API base URLs for LLM providers
API_BASE_GROQ = "https://api.groq.com/openai/v1"
API_BASE_GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai/"
API_BASE_OPENROUTER = "https://openrouter.ai/api/v1"
API_BASE_DEEPSEEK = "https://api.deepseek.com"
API_BASE_QWEN = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
