import contextlib
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

from tapeback import const
from tapeback.channel import classify_segment_by_channel, load_stereo_channels
from tapeback.models import DiarizationSegment, Segment
from tapeback.settings import Settings

# pyannote segmentation + embedding models need ~1-1.5 GB VRAM during inference
DIARIZATION_VRAM_MIN_MIB = 1500

# Minor speaker absorption: speakers with very little speech are likely
# echo/crosstalk artifacts.  Use a lower merge threshold for them.
MINOR_SPEAKER_MAX_SEC = 15.0  # absolute: speaker with < 15s is potentially minor
MINOR_SPEAKER_RATIO = 0.2  # relative: must have < 20% of the dominant speaker's speech
MINOR_SPEAKER_MERGE_THRESHOLD = 0.92  # lower cosine for absorbing minor speakers


def _get_free_vram_mib() -> int | None:
    """Get free GPU VRAM in MiB via nvidia-smi. Returns None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0])
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def diarization_available() -> bool:
    """Check if pyannote-audio is installed."""
    try:
        import pyannote.audio  # noqa: F401

        return True
    except ImportError:
        return False


def _unwrap_diarization(result: Any) -> Any:
    """Extract Annotation from pyannote output.

    pyannote 4.x returns DiarizeOutput (with .speaker_diarization),
    older versions return Annotation directly (with .itertracks).
    """
    if hasattr(result, "itertracks"):
        return result
    return result.speaker_diarization


class Diarizer:
    def __init__(self, settings: Settings) -> None:
        """Initialize pyannote pipeline.

        Raises RuntimeError if hf_token is empty.
        Falls back to CPU if CUDA is not available.
        """
        if not settings.hf_token:
            raise RuntimeError(
                "HuggingFace token required for diarization. "
                "Set TAPEBACK_HF_TOKEN in your .env file."
            )

        from pyannote.audio import Pipeline

        self._settings = settings
        pipeline = Pipeline.from_pretrained(
            const.PYANNOTE_MODEL,
            token=settings.hf_token,
        )
        if pipeline is None:
            raise RuntimeError("Failed to load pyannote diarization pipeline")
        self._pipeline = pipeline

        if settings.clustering_threshold is not None:
            params = self._pipeline.parameters(instantiated=True)
            params["clustering"]["threshold"] = settings.clustering_threshold
            self._pipeline.instantiate(params)

        if settings.device == "cuda":
            free_mib = _get_free_vram_mib()
            if free_mib is not None and free_mib < DIARIZATION_VRAM_MIN_MIB:
                print(
                    f"Warning: Not enough VRAM for diarization "
                    f"({free_mib} MiB free < {DIARIZATION_VRAM_MIN_MIB} MiB), using CPU",
                    file=sys.stderr,
                )
            else:
                try:
                    import torch

                    self._pipeline.to(torch.device("cuda"))
                except RuntimeError:
                    print(
                        "Warning: CUDA not available for diarization, using CPU",
                        file=sys.stderr,
                    )

    def _run_pipeline(self, audio_path: Path) -> Any:
        """Run pyannote pipeline with optional max_speakers."""
        if self._settings.max_speakers is not None:
            return self._pipeline(audio_path, max_speakers=self._settings.max_speakers)
        return self._pipeline(audio_path)

    def diarize(self, audio_path: Path) -> list[DiarizationSegment]:
        """Run diarization on audio file.

        Returns list of DiarizationSegment sorted by start time.
        Falls back to CPU if CUDA runs out of memory during inference.
        """
        try:
            diarization = self._run_pipeline(audio_path)
        except RuntimeError as exc:
            if "CUDA" not in str(exc) and "out of memory" not in str(exc):
                raise
            print(
                f"Warning: CUDA out of memory during diarization, falling back to CPU: {exc}",
                file=sys.stderr,
            )
            self._fallback_to_cpu()
            diarization = self._run_pipeline(audio_path)

        # pyannote 4.x returns DiarizeOutput wrapping Annotation
        annotation = _unwrap_diarization(diarization)

        segments: list[DiarizationSegment] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(
                DiarizationSegment(
                    speaker=speaker,
                    start=turn.start,
                    end=turn.end,
                )
            )

        return segments

    def _fallback_to_cpu(self) -> None:
        """Move pipeline to CPU and free CUDA memory."""
        import torch

        self._pipeline.to(torch.device("cpu"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def merge_channel_segments(
    mic_segments: list[Segment],
    monitor_segments: list[Segment],
) -> list[Segment]:
    """Merge segments from both channels, sorted by start time.

    After sorting, consecutive segments from the same speaker are consolidated
    into a single segment.  Overlapping segments from different speakers are
    kept separate.
    """
    merged = sorted(mic_segments + monitor_segments, key=lambda s: s.start)
    return consolidate_segments(merged)


def consolidate_segments(segments: list[Segment]) -> list[Segment]:
    """Merge consecutive segments from the same speaker into one.

    Handles both adjacent and overlapping segments.  Preserves word lists
    by concatenation.
    """
    if not segments:
        return []

    result: list[Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = result[-1]
        if prev.speaker and prev.speaker == seg.speaker:
            # Merge: extend previous segment
            words = None
            if prev.words and seg.words:
                words = prev.words + seg.words
            elif prev.words or seg.words:
                words = prev.words or seg.words
            result[-1] = Segment(
                start=prev.start,
                end=max(prev.end, seg.end),
                text=prev.text + " " + seg.text,
                words=words,
                speaker=prev.speaker,
            )
        else:
            result.append(seg)

    return result


def _speaker_spectral_profile(
    monitor_samples: np.ndarray,
    sample_rate: int,
    diarization_segments: list[DiarizationSegment],
    speaker: str,
) -> np.ndarray:
    """Compute average power spectrum for a speaker's segments.

    Focuses on the 100-4000 Hz range which contains voice formant information.
    Uses Hann-windowed FFT with 50 % overlap.
    """
    n_fft = const.SPECTRAL_FFT_SIZE
    freq_per_bin = sample_rate / n_fft
    min_bin = max(1, int(const.SPECTRAL_MIN_FREQ_HZ / freq_per_bin))
    max_bin = min(n_fft // 2 + 1, int(const.SPECTRAL_MAX_FREQ_HZ / freq_per_bin) + 1)
    n_bins = max_bin - min_bin

    if n_bins <= 0:
        return np.zeros(1)

    window = np.hanning(n_fft)
    spectra: list[np.ndarray] = []

    for seg in diarization_segments:
        if seg.speaker != speaker:
            continue
        sf = max(0, int(seg.start * sample_rate))
        ef = min(len(monitor_samples), int(seg.end * sample_rate))
        chunk = monitor_samples[sf:ef]

        hop = n_fft // 2
        for start in range(0, len(chunk) - n_fft, hop):
            frame = chunk[start : start + n_fft].astype(np.float64)
            spectrum = np.abs(np.fft.rfft(frame * window))[min_bin:max_bin]
            spectra.append(spectrum)

    if not spectra:
        return np.zeros(n_bins)
    result: np.ndarray = np.mean(spectra, axis=0)
    return result


def merge_similar_speakers(
    diarization_segments: list[DiarizationSegment],
    monitor_samples: np.ndarray,
    sample_rate: int,
    similarity_threshold: float = 0.95,
) -> list[DiarizationSegment]:
    """Merge pyannote speakers with similar spectral profiles.

    Fixes over-segmentation where a single speaker is incorrectly split into
    multiple speakers.  Uses power-spectrum cosine similarity in the 100-4000 Hz
    voice frequency range.

    Two-tier thresholds:
    - Standard merge (similarity_threshold, default 0.96): merges near-identical
      profiles (over-segmented single speaker, cosine ~0.98-0.99).
    - Minor speaker absorption (MINOR_SPEAKER_MERGE_THRESHOLD = 0.92): when one
      speaker has very little speech (< 15s and < 20% of dominant), they are likely
      echo/crosstalk artifacts with unreliable spectral profiles. A lower threshold
      absorbs them into the dominant speaker.

    Power-spectrum similarity is a weak signal for voice identity — the channel
    frequency response dominates. Set to 0 to disable.
    """
    if similarity_threshold <= 0:
        return diarization_segments

    speakers = sorted({seg.speaker for seg in diarization_segments})
    if len(speakers) <= 1:
        return diarization_segments

    # Total speech per speaker (for minor speaker detection)
    total_speech: dict[str, float] = {}
    for speaker in speakers:
        total_speech[speaker] = sum(
            s.end - s.start for s in diarization_segments if s.speaker == speaker
        )

    profiles: dict[str, np.ndarray] = {}
    for speaker in speakers:
        profiles[speaker] = _speaker_spectral_profile(
            monitor_samples, sample_rate, diarization_segments, speaker
        )

    merge_map: dict[str, str] = {s: s for s in speakers}

    for i, sp_a in enumerate(speakers):
        for sp_b in speakers[i + 1 :]:
            if merge_map[sp_a] == merge_map[sp_b]:
                continue

            a_profile = profiles[sp_a]
            b_profile = profiles[sp_b]

            norm_a = float(np.linalg.norm(a_profile))
            norm_b = float(np.linalg.norm(b_profile))
            if norm_a < const.CHANNEL_EPSILON or norm_b < const.CHANNEL_EPSILON:
                continue

            similarity = float(np.dot(a_profile, b_profile) / (norm_a * norm_b))

            # Use lower threshold when one speaker is a minor artifact
            # (little speech both absolutely and relative to the dominant speaker)
            threshold = similarity_threshold
            minor_total = min(total_speech[sp_a], total_speech[sp_b])
            major_total = max(total_speech[sp_a], total_speech[sp_b])
            if (
                minor_total < MINOR_SPEAKER_MAX_SEC
                and major_total > 0
                and minor_total / major_total < MINOR_SPEAKER_RATIO
            ):
                threshold = MINOR_SPEAKER_MERGE_THRESHOLD

            if similarity >= threshold:
                target = merge_map[sp_b]
                canonical = merge_map[sp_a]
                for s in speakers:
                    if merge_map[s] == target:
                        merge_map[s] = canonical

    if all(merge_map[s] == s for s in speakers):
        return diarization_segments

    return [
        DiarizationSegment(
            speaker=merge_map[seg.speaker],
            start=seg.start,
            end=seg.end,
        )
        for seg in diarization_segments
    ]


def _find_speaker_for_time(
    start: float,
    end: float,
    diarization_segments: list[DiarizationSegment],
) -> str | None:
    """Find the pyannote speaker with most overlap for a time range."""
    best_speaker = None
    best_overlap = 0.0

    for dseg in diarization_segments:
        overlap = max(0.0, min(end, dseg.end) - max(start, dseg.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = dseg.speaker

    # No overlap — find nearest segment
    if best_speaker is None:
        mid = (start + end) / 2
        min_dist = float("inf")
        for dseg in diarization_segments:
            dist = min(abs(mid - dseg.start), abs(mid - dseg.end))
            if dist < min_dist:
                min_dist = dist
                best_speaker = dseg.speaker

    return best_speaker


def _resegment_by_words(
    segment: Segment,
    diarization_segments: list[DiarizationSegment],
) -> list[tuple[Segment, str | None]]:
    """Split a segment into sub-segments at diarization speaker boundaries.

    Each word is assigned to its pyannote speaker.  Consecutive words from the
    same speaker are grouped into one sub-segment.  Returns list of
    (sub_segment, pyannote_speaker) tuples.
    """
    if not segment.words:
        speaker = _find_speaker_for_time(segment.start, segment.end, diarization_segments)
        return [(segment, speaker)]

    # Assign each word to a pyannote speaker
    word_speakers: list[tuple[str | None, int]] = []
    for i, word in enumerate(segment.words):
        speaker = _find_speaker_for_time(word.start, word.end, diarization_segments)
        word_speakers.append((speaker, i))

    # Group consecutive same-speaker words into sub-segments
    result: list[tuple[Segment, str | None]] = []
    group_start = 0

    for i in range(1, len(word_speakers) + 1):
        if i < len(word_speakers) and word_speakers[i][0] == word_speakers[group_start][0]:
            continue

        # Flush group [group_start, i)
        group_words = segment.words[group_start:i]
        speaker = word_speakers[group_start][0]
        text = "".join(w.word for w in group_words).strip()
        if text:
            result.append(
                (
                    Segment(
                        start=group_words[0].start,
                        end=group_words[-1].end,
                        text=text,
                        words=group_words,
                    ),
                    speaker,
                )
            )
        group_start = i

    return result if result else [(segment, None)]


def assign_speakers(
    segments: list[Segment],
    diarization_segments: list[DiarizationSegment],
    user_speaker: str | None = None,
    stereo_wav: Path | None = None,
) -> list[Segment]:
    """Assign speaker labels to each Segment based on channel energy + pyannote.

    For segments with word timestamps, splits at diarization speaker boundaries
    so that words from different speakers become separate segments.

    When stereo_wav is provided, per-segment channel energy determines
    "You" (mic-dominant) vs "Others" (monitor-dominant). Ambiguous segments
    fall back to pyannote-based assignment via user_speaker.

    Speaker naming:
    - mic-dominant or user_speaker -> "You"
    - Others -> "Speaker 1", "Speaker 2", ... (numbered by appearance order)

    Does NOT mutate input segments — creates new ones.
    """
    if not diarization_segments:
        return list(segments)

    # Load stereo data once (not N times)
    stereo_data: tuple[np.ndarray, np.ndarray, int] | None = None
    if stereo_wav is not None:
        with contextlib.suppress(ValueError, wave.Error):
            stereo_data = load_stereo_channels(stereo_wav)

    speaker_order: list[str] = []

    def get_label(pyannote_speaker: str, channel_verdict: str | None = None) -> str:
        """Get display label for a speaker.

        channel_verdict: 'mic' → force "You", 'monitor' → force non-You,
        None → let pyannote + user_speaker decide.
        """
        if channel_verdict == "mic":
            return const.SPEAKER_YOU
        non_user = [s for s in speaker_order if s != user_speaker]
        if channel_verdict is None and user_speaker and pyannote_speaker == user_speaker:
            return const.SPEAKER_YOU
        if pyannote_speaker in non_user:
            idx = non_user.index(pyannote_speaker) + 1
        else:
            idx = len(non_user) + 1
        return const.SPEAKER_LABEL_FMT.format(idx)

    result = []
    for seg in segments:
        # Split segment at diarization boundaries (word-level)
        sub_segments = _resegment_by_words(seg, diarization_segments)

        for sub_seg, pyannote_speaker in sub_segments:
            if pyannote_speaker and pyannote_speaker not in speaker_order:
                speaker_order.append(pyannote_speaker)

            # Channel-based override if stereo data available
            channel = None
            if stereo_data is not None:
                mic, monitor, sr = stereo_data
                channel = classify_segment_by_channel(sub_seg.start, sub_seg.end, mic, monitor, sr)

            if pyannote_speaker:
                label = get_label(pyannote_speaker, channel_verdict=channel)
            elif channel == "mic":
                label = const.SPEAKER_YOU
            elif channel == "monitor":
                label = const.SPEAKER_LABEL_FMT.format(1)
            else:
                label = None

            result.append(
                Segment(
                    start=sub_seg.start,
                    end=sub_seg.end,
                    text=sub_seg.text,
                    words=sub_seg.words,
                    speaker=label,
                )
            )

    return result
