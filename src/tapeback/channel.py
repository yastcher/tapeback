"""Stereo channel analysis — RMS energy, silence detection, speaker identification."""

import wave
from pathlib import Path

import numpy as np

from tapeback import const
from tapeback.models import DiarizationSegment, Segment


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
    window_dur = const.SILENCE_WINDOW_SEC
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
        adaptive_threshold = median_rms * const.SILENCE_ADAPTIVE_FACTOR

        # Detect silent windows
        silence_start: float | None = None
        split_points: list[float] = []

        for idx, (t, mic_rms) in enumerate(mic_rms_values):
            is_quiet = mic_rms < adaptive_threshold

            # Monitor-relative check: mic is much quieter than monitor
            if not is_quiet and monitor_rms_values:
                mon_rms = monitor_rms_values[idx]
                if mon_rms > 0 and mic_rms < mon_rms * const.SILENCE_MONITOR_FACTOR:
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

    if mic_rms > (monitor_rms + const.CHANNEL_EPSILON) * const.CHANNEL_ENERGY_RATIO:
        return "mic"
    if monitor_rms > (mic_rms + const.CHANNEL_EPSILON) * const.CHANNEL_ENERGY_RATIO:
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
            ratios[speaker] = mic_rms / (monitor_rms + const.CHANNEL_EPSILON)

    if not ratios:
        return None

    best_speaker = max(ratios, key=lambda s: ratios[s])
    best_ratio = ratios[best_speaker]

    # Require at least 2x difference to be confident
    other_ratios = [r for s, r in ratios.items() if s != best_speaker]
    if other_ratios and best_ratio < const.CHANNEL_ENERGY_RATIO * max(other_ratios):
        return None

    return best_speaker
