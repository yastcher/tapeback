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

# FFmpeg loudnorm parameters (EBU R128)
LOUDNORM_PARAMS = "I=-16:TP=-1.5:LRA=11"

# Channel energy classification
CHANNEL_ENERGY_RATIO = 2.0
CHANNEL_EPSILON = 1e-10

# Silence detection
SILENCE_WINDOW_SEC = 0.1
SILENCE_ADAPTIVE_FACTOR = 0.4
SILENCE_MONITOR_FACTOR = 0.3

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

# API base URLs for LLM providers
API_BASE_GROQ = "https://api.groq.com/openai/v1"
API_BASE_GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai/"
API_BASE_OPENROUTER = "https://openrouter.ai/api/v1"
API_BASE_DEEPSEEK = "https://api.deepseek.com"
API_BASE_QWEN = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
