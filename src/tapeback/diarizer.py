import contextlib
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

from tapeback.models import DiarizationSegment, Segment
from tapeback.settings import Settings


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
            "pyannote/speaker-diarization-3.1",
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
            try:
                import torch

                self._pipeline.to(torch.device("cuda"))
            except RuntimeError:
                print(
                    "Warning: CUDA not available for diarization, using CPU",
                    file=sys.stderr,
                )

    def diarize(self, audio_path: Path) -> list[DiarizationSegment]:
        """Run diarization on audio file.

        Returns list of DiarizationSegment sorted by start time.
        Falls back to CPU if CUDA runs out of memory during inference.
        """
        kwargs: dict[str, int] = {}
        if self._settings.max_speakers is not None:
            kwargs["max_speakers"] = self._settings.max_speakers

        try:
            diarization = self._pipeline(audio_path, **kwargs)  # type: ignore[arg-type]
        except RuntimeError as exc:
            if "CUDA" not in str(exc) and "out of memory" not in str(exc):
                raise
            print(
                f"Warning: CUDA out of memory during diarization, falling back to CPU: {exc}",
                file=sys.stderr,
            )
            self._fallback_to_cpu()
            diarization = self._pipeline(audio_path, **kwargs)  # type: ignore[arg-type]

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


def _rms_for_range(
    start: float,
    end: float,
    samples: np.ndarray,
    sample_rate: int,
) -> float:
    """Compute RMS energy for a time range in samples array."""
    sf = max(0, min(int(start * sample_rate), len(samples)))
    ef = max(0, min(int(end * sample_rate), len(samples)))
    if ef <= sf:
        return 0.0
    return float(np.sqrt(np.mean(samples[sf:ef] ** 2)))


def filter_silent_segments(
    segments: list[Segment],
    channel_samples: np.ndarray,
    sample_rate: int,
    rms_threshold: float = 200.0,
) -> list[Segment]:
    """Remove segments (or parts of segments) where channel RMS is below threshold.

    Uses raw (pre-loudnorm) channel samples so that normalization doesn't
    inflate background noise above the threshold.

    When word timestamps are available, filters at word level: drops individual
    words with low RMS (crosstalk from other channel), keeps the rest, and
    rebuilds the segment from surviving words. This prevents crosstalk fragments
    like monitor audio bleeding into mic from contaminating real speech segments.

    Segments without word timestamps are filtered at segment level.
    """
    result = []
    for seg in segments:
        if seg.words:
            kept_words = [
                w
                for w in seg.words
                if _rms_for_range(w.start, w.end, channel_samples, sample_rate) >= rms_threshold
            ]
            if not kept_words:
                continue
            result.append(
                Segment(
                    start=kept_words[0].start,
                    end=kept_words[-1].end,
                    text=" ".join(w.word.strip() for w in kept_words),
                    words=kept_words,
                    speaker=seg.speaker,
                )
            )
        else:
            rms = _rms_for_range(seg.start, seg.end, channel_samples, sample_rate)
            if rms >= rms_threshold:
                result.append(seg)

    return result


def split_on_silence(
    segments: list[Segment],
    mic_samples: np.ndarray,
    sample_rate: int,
    pause_threshold: float = 1.0,
    monitor_samples: np.ndarray | None = None,
) -> list[Segment]:
    """Split segments at silence gaps detected in raw mic audio.

    Uses an adaptive threshold: a window is "silent" when mic RMS is below
    the segment's median RMS * 0.4.  When a monitor channel is provided,
    a window also counts as silent when the monitor is louder than the mic
    (mic_rms < monitor_rms * 0.3) — the user is quiet while the remote
    speaker is active.

    A contiguous silent region >= pause_threshold seconds triggers a split.
    """
    window_dur = 0.1  # 100ms windows
    window_samples = int(window_dur * sample_rate)
    result: list[Segment] = []

    for seg in segments:
        sf = max(0, min(int(seg.start * sample_rate), len(mic_samples)))
        ef = max(0, min(int(seg.end * sample_rate), len(mic_samples)))

        if ef - sf < window_samples:
            result.append(seg)
            continue

        # Compute mic RMS per window
        mic_rms_values: list[tuple[float, float]] = []  # (time, rms)
        monitor_rms_values: list[float] = []

        for i in range(sf, ef, window_samples):
            end_i = min(i + window_samples, ef)
            chunk = mic_samples[i:end_i]
            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            t = i / sample_rate
            mic_rms_values.append((t, rms))

            if monitor_samples is not None:
                ms = max(0, min(i, len(monitor_samples)))
                me = max(0, min(end_i, len(monitor_samples)))
                if me > ms:
                    mon_rms = float(
                        np.sqrt(np.mean(monitor_samples[ms:me].astype(np.float64) ** 2))
                    )
                else:
                    mon_rms = 0.0
                monitor_rms_values.append(mon_rms)

        if not mic_rms_values:
            result.append(seg)
            continue

        # Adaptive threshold: 40% of median mic RMS within this segment
        all_rms = [r for _, r in mic_rms_values]
        median_rms = float(np.median(all_rms))
        adaptive_threshold = median_rms * 0.4

        # Detect silent windows
        silence_start: float | None = None
        split_points: list[float] = []

        for idx, (t, mic_rms) in enumerate(mic_rms_values):
            is_quiet = mic_rms < adaptive_threshold

            # Monitor-relative check: mic is much quieter than monitor
            if not is_quiet and monitor_rms_values:
                mon_rms = monitor_rms_values[idx]
                if mon_rms > 0 and mic_rms < mon_rms * 0.3:
                    is_quiet = True

            if is_quiet:
                if silence_start is None:
                    silence_start = t
            elif silence_start is not None:
                silence_dur = t - silence_start
                if silence_dur >= pause_threshold:
                    split_points.append(silence_start + silence_dur / 2)
                silence_start = None

        # Check trailing silence
        if silence_start is not None:
            silence_dur = seg.end - silence_start
            if silence_dur >= pause_threshold:
                split_points.append(silence_start + silence_dur / 2)

        if not split_points:
            result.append(seg)
            continue

        # Build sub-segments
        boundaries = [seg.start, *split_points, seg.end]
        for i in range(len(boundaries) - 1):
            sub_start = boundaries[i]
            sub_end = boundaries[i + 1]

            if seg.words:
                sub_words = [
                    w for w in seg.words if w.start >= sub_start - 0.05 and w.end <= sub_end + 0.05
                ]
                if not sub_words:
                    continue
                result.append(
                    Segment(
                        start=sub_words[0].start,
                        end=sub_words[-1].end,
                        text=" ".join(w.word.strip() for w in sub_words),
                        words=sub_words,
                        speaker=seg.speaker,
                    )
                )
            # No words — keep the sub-segment with original text proportioned
            elif sub_end - sub_start >= 0.5:
                result.append(
                    Segment(
                        start=sub_start,
                        end=sub_end,
                        text=seg.text,
                        words=None,
                        speaker=seg.speaker,
                    )
                )

    return result


def merge_channel_segments(
    mic_segments: list[Segment],
    monitor_segments: list[Segment],
) -> list[Segment]:
    """Merge segments from both channels, sorted by start time.

    Overlapping segments (simultaneous speech) are kept as-is.
    """
    return sorted(mic_segments + monitor_segments, key=lambda s: s.start)


def load_stereo_channels(stereo_wav: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load stereo WAV and return (mic_channel, monitor_channel, sample_rate).

    mic = left channel, monitor = right channel.
    Returns float32 arrays for RMS calculations.
    """
    with wave.open(str(stereo_wav), "rb") as wf:
        if wf.getnchannels() != 2:
            raise ValueError(f"Expected stereo WAV, got {wf.getnchannels()} channels")
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2).astype(np.float32)
    return samples[:, 0], samples[:, 1], sample_rate


def classify_segment_by_channel(
    start: float,
    end: float,
    mic: np.ndarray,
    monitor: np.ndarray,
    sample_rate: int,
) -> str | None:
    """Classify a time segment as 'mic', 'monitor', or None (ambiguous).

    Compares RMS energy on mic vs monitor channel for the given time range.
    Returns 'mic' if mic_rms > monitor_rms * 2, 'monitor' if vice versa, else None.
    """
    start_frame = max(0, min(int(start * sample_rate), len(mic)))
    end_frame = max(0, min(int(end * sample_rate), len(mic)))

    if end_frame <= start_frame:
        return None

    mic_rms = float(np.sqrt(np.mean(mic[start_frame:end_frame] ** 2)))
    monitor_rms = float(np.sqrt(np.mean(monitor[start_frame:end_frame] ** 2)))

    epsilon = 1e-10
    if mic_rms > (monitor_rms + epsilon) * 2.0:
        return "mic"
    if monitor_rms > (mic_rms + epsilon) * 2.0:
        return "monitor"
    return None


def identify_user_speaker(
    diarization_segments: list[DiarizationSegment],
    stereo_wav: Path,
) -> str | None:
    """Determine which pyannote speaker is the user (mic channel).

    Compares RMS energy on mic (left) vs monitor (right) channel
    for each speaker's segments. The speaker with the highest
    mic/monitor ratio is identified as the user.

    Returns speaker ID (e.g. "SPEAKER_00") or None if ambiguous.
    """
    speakers = {seg.speaker for seg in diarization_segments}

    if len(speakers) <= 1:
        return None

    with wave.open(str(stereo_wav), "rb") as wf:
        if wf.getnchannels() != 2:
            return None
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2).astype(np.float32)
    mic_channel = samples[:, 0]  # left = mic
    monitor_channel = samples[:, 1]  # right = monitor

    epsilon = 1e-10
    ratios: dict[str, float] = {}

    for speaker in speakers:
        speaker_segs = [s for s in diarization_segments if s.speaker == speaker]

        mic_energy = 0.0
        monitor_energy = 0.0
        total_frames = 0

        for seg in speaker_segs:
            start_frame = max(0, min(int(seg.start * sample_rate), len(mic_channel)))
            end_frame = max(0, min(int(seg.end * sample_rate), len(mic_channel)))

            if end_frame <= start_frame:
                continue

            mic_energy += float(np.sum(mic_channel[start_frame:end_frame] ** 2))
            monitor_energy += float(np.sum(monitor_channel[start_frame:end_frame] ** 2))
            total_frames += end_frame - start_frame

        if total_frames > 0:
            mic_rms = (mic_energy / total_frames) ** 0.5
            monitor_rms = (monitor_energy / total_frames) ** 0.5
            ratios[speaker] = mic_rms / (monitor_rms + epsilon)

    if not ratios:
        return None

    best_speaker = max(ratios, key=lambda s: ratios[s])
    best_ratio = ratios[best_speaker]

    # Require at least 2x difference to be confident
    other_ratios = [r for s, r in ratios.items() if s != best_speaker]
    if other_ratios and best_ratio < 2.0 * max(other_ratios):
        return None

    return best_speaker


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
    n_fft = 2048
    freq_per_bin = sample_rate / n_fft
    min_bin = max(1, int(100.0 / freq_per_bin))
    max_bin = min(n_fft // 2 + 1, int(4000.0 / freq_per_bin) + 1)
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
    similarity_threshold: float = 0.92,
) -> list[DiarizationSegment]:
    """Merge pyannote speakers with similar spectral profiles.

    Fixes over-segmentation where a single speaker is incorrectly split into
    multiple speakers.  Uses power-spectrum cosine similarity in the 100-4000 Hz
    voice frequency range.

    Conservative threshold (default 0.92) ensures only very similar speakers
    are merged, avoiding false merges of genuinely different speakers.
    """
    speakers = sorted({seg.speaker for seg in diarization_segments})
    if len(speakers) <= 1:
        return diarization_segments

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
            if norm_a < 1e-10 or norm_b < 1e-10:
                continue

            similarity = float(np.dot(a_profile, b_profile) / (norm_a * norm_b))
            if similarity >= similarity_threshold:
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


def assign_speakers(
    segments: list[Segment],
    diarization_segments: list[DiarizationSegment],
    user_speaker: str | None = None,
    stereo_wav: Path | None = None,
) -> list[Segment]:
    """Assign speaker labels to each Segment based on channel energy + pyannote.

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
            return "You"
        non_user = [s for s in speaker_order if s != user_speaker]
        if channel_verdict is None and user_speaker and pyannote_speaker == user_speaker:
            return "You"
        if pyannote_speaker in non_user:
            idx = non_user.index(pyannote_speaker) + 1
        else:
            idx = len(non_user) + 1
        return f"Speaker {idx}"

    def find_speaker_for_time(start: float, end: float) -> str | None:
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

    result = []
    for seg in segments:
        # Step 1: determine pyannote speaker
        if seg.words:
            speaker_votes: dict[str, int] = {}
            for word in seg.words:
                speaker = find_speaker_for_time(word.start, word.end)
                if speaker:
                    speaker_votes[speaker] = speaker_votes.get(speaker, 0) + 1

            pyannote_speaker = (
                max(speaker_votes, key=lambda s: speaker_votes[s]) if speaker_votes else None
            )
        else:
            pyannote_speaker = find_speaker_for_time(seg.start, seg.end)

        if pyannote_speaker and pyannote_speaker not in speaker_order:
            speaker_order.append(pyannote_speaker)

        # Step 2: channel-based override if stereo data available
        channel = None
        if stereo_data is not None:
            mic, monitor, sr = stereo_data
            channel = classify_segment_by_channel(seg.start, seg.end, mic, monitor, sr)

        if pyannote_speaker:
            label = get_label(pyannote_speaker, channel_verdict=channel)
        elif channel == "mic":
            label = "You"
        elif channel == "monitor":
            label = "Speaker 1"
        else:
            label = None

        result.append(
            Segment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                words=seg.words,
                speaker=label,
            )
        )

    return result
