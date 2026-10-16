"""Stereo channel analysis — RMS energy, silence detection, speaker identification."""

import wave
from itertools import pairwise
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
    """Compute RMS energy for a time range in samples array.

    The slice is widened before squaring. Channels are held as int16 to keep a long
    recording out of swap, and a near-full-scale sample squares to ~1.07e9 — squaring
    in the sample dtype wraps around and reports the loudest audio as the quietest.
    """
    sf = max(0, min(int(start * sample_rate), len(samples)))
    ef = max(0, min(int(end * sample_rate), len(samples)))
    if ef <= sf:
        return 0.0
    return float(np.sqrt(np.mean(samples[sf:ef].astype(np.float64) ** 2)))


def filter_silent_segments(
    segments: list[Segment],
    channel_samples: np.ndarray,
    sample_rate: int,
    rms_threshold: float = const.SILENCE_RMS_THRESHOLD,
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


def gate_inactive_regions(
    target_16k: np.ndarray,
    target_raw: np.ndarray,
    other_raw: np.ndarray,
    raw_sr: int,
    rms_threshold: float = const.SILENCE_RMS_THRESHOLD,
) -> np.ndarray:
    """Zero windows of target_16k where the speaker is inactive.

    A window is inactive when the target channel is quiet (RMS below threshold)
    or the other channel dominates it (target_rms < other_rms * factor) — i.e.
    the user is listening, not speaking. Silencing these regions before Whisper
    stops it hallucinating repeat loops on them (which is both slow and garbage).

    Windows are mapped by time: target_16k is 16 kHz, the raw channels may be a
    different rate. Energy is measured on the raw (pre-loudnorm) channels so
    normalization doesn't inflate background noise. Input is not mutated.
    """
    out = target_16k.copy()
    out_sr = const.SAMPLE_RATE_16K
    window_samples = max(1, int(const.SILENCE_WINDOW_SEC * out_sr))
    for start in range(0, len(out), window_samples):
        t0 = start / out_sr
        t1 = (start + window_samples) / out_sr
        target_rms = _rms_for_range(t0, t1, target_raw, raw_sr)
        other_rms = _rms_for_range(t0, t1, other_raw, raw_sr)
        listening = other_rms > 0 and target_rms < other_rms * const.SILENCE_MONITOR_FACTOR
        if target_rms < rms_threshold or listening:
            out[start : start + window_samples] = 0.0
    return out


def _compute_window_rms(
    mic_samples: np.ndarray,
    monitor_samples: np.ndarray | None,
    sf: int,
    ef: int,
    sample_rate: int,
    window_samples: int,
) -> tuple[list[tuple[float, float]], list[float]]:
    """Compute per-window RMS for mic (with timestamps) and optional monitor channel."""
    mic_rms_values: list[tuple[float, float]] = []
    monitor_rms_values: list[float] = []

    for i in range(sf, ef, window_samples):
        end_i = min(i + window_samples, ef)
        chunk = mic_samples[i:end_i]
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        mic_rms_values.append((i / sample_rate, rms))

        if monitor_samples is not None:
            ms = max(0, min(i, len(monitor_samples)))
            me = max(0, min(end_i, len(monitor_samples)))
            if me > ms:
                mon_rms = float(np.sqrt(np.mean(monitor_samples[ms:me].astype(np.float64) ** 2)))
            else:
                mon_rms = 0.0
            monitor_rms_values.append(mon_rms)

    return mic_rms_values, monitor_rms_values


def _find_split_points(
    mic_rms_values: list[tuple[float, float]],
    monitor_rms_values: list[float],
    pause_threshold: float,
    seg_end: float,
) -> list[float]:
    """Find split points where mic is silent for >= pause_threshold seconds."""
    all_rms = [r for _, r in mic_rms_values]
    adaptive_threshold = float(np.median(all_rms)) * const.SILENCE_ADAPTIVE_FACTOR

    silence_start: float | None = None
    split_points: list[float] = []

    for idx, (t, mic_rms) in enumerate(mic_rms_values):
        is_quiet = mic_rms < adaptive_threshold

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

    if silence_start is not None:
        silence_dur = seg_end - silence_start
        if silence_dur >= pause_threshold:
            split_points.append(silence_start + silence_dur / 2)

    return split_points


def _pause_points(
    seg: Segment,
    mic_samples: np.ndarray,
    sample_rate: int,
    pause_threshold: float,
    monitor_samples: np.ndarray | None,
) -> list[float]:
    """Midpoints of the mic's pauses inside seg that last pause_threshold or longer."""
    window_samples = int(const.SILENCE_WINDOW_SEC * sample_rate)
    sf = max(0, min(int(seg.start * sample_rate), len(mic_samples)))
    ef = max(0, min(int(seg.end * sample_rate), len(mic_samples)))
    if ef - sf < window_samples:
        return []

    mic_rms_values, monitor_rms_values = _compute_window_rms(
        mic_samples, monitor_samples, sf, ef, sample_rate, window_samples
    )
    if not mic_rms_values:
        return []
    return _find_split_points(mic_rms_values, monitor_rms_values, pause_threshold, seg.end)


def _build_sub_segments(seg: Segment, split_points: list[float]) -> list[Segment]:
    """Build sub-segments from a segment's words and its internal split points."""
    boundaries = [seg.start, *split_points, seg.end]
    result: list[Segment] = []

    for i in range(len(boundaries) - 1):
        sub_start = boundaries[i]
        sub_end = boundaries[i + 1]

        sub_words = [
            w for w in seg.words or [] if w.start >= sub_start - 0.05 and w.end <= sub_end + 0.05
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

    A segment without word timestamps (a text-only remote upload) is returned whole:
    nothing says which of its text falls on which side of a pause, and every piece
    used to get all of it, repeating the user's words once per pause.
    """
    result: list[Segment] = []
    for seg in segments:
        split_points = (
            _pause_points(seg, mic_samples, sample_rate, pause_threshold, monitor_samples)
            if seg.words
            else []
        )
        result.extend(_build_sub_segments(seg, split_points) if split_points else [seg])
    return result


def _user_speaks(
    start: float,
    end: float,
    mic_samples: np.ndarray,
    monitor_samples: np.ndarray,
    sample_rate: int,
) -> bool:
    """The mic is loud enough over [start, end], and the monitor does not drown it."""
    if _rms_for_range(start, end, mic_samples, sample_rate) < const.SILENCE_RMS_THRESHOLD:
        return False
    verdict = classify_segment_by_channel(start, end, mic_samples, monitor_samples, sample_rate)
    return verdict != "monitor"


def filter_mic_segments(
    segments: list[Segment],
    mic_samples: np.ndarray,
    monitor_samples: np.ndarray,
    sample_rate: int,
    pause_threshold: float,
) -> list[Segment]:
    """Keep the mic segments that carry the user's voice, dropping silence and the
    monitor heard through the mic (headphone bleed).

    A segment with word timestamps has its silent words dropped, then what is left is
    compared with the monitor once: pause splitting has already cut it at the mic's
    pauses. One without (a text-only remote upload) spans minutes. Judged whole, its
    energy is diluted, and it goes to whichever side talked more: a 600 s chunk of a
    real meeting came out "monitor". It is judged stretch by stretch between the
    pauses, and kept whole if the user speaks in any stretch.
    """
    kept: list[Segment] = []
    for seg in segments:
        if seg.words:
            kept.extend(
                s
                for s in filter_silent_segments([seg], mic_samples, sample_rate)
                if classify_segment_by_channel(
                    s.start, s.end, mic_samples, monitor_samples, sample_rate
                )
                != "monitor"
            )
            continue
        points = _pause_points(seg, mic_samples, sample_rate, pause_threshold, monitor_samples)
        bounds = [seg.start, *points, seg.end]
        if any(
            _user_speaks(a, b, mic_samples, monitor_samples, sample_rate)
            for a, b in pairwise(bounds)
        ):
            kept.append(seg)
    return kept


def load_stereo_channels(stereo_wav: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load stereo WAV and return (mic_channel, monitor_channel, sample_rate).

    mic = left channel, monitor = right channel. Channels come back as **int16**, the
    format they are stored in; every consumer widens the slice it works on.

    Memory is the reason. A 37-minute stereo recording at 48 kHz is 434 MB of int16.
    Converting the whole thing to float32 up front doubled that to 869 MB held for the
    entire run, and because the raw buffer stayed alive during the conversion it peaked
    at 1.3 GB. Measured on a machine with 1.4 GB free, that was the difference between
    running and thrashing swap — which in turn heats the CPU and drives the GPU into a
    thermal clamp.

    The per-channel copy is deliberate. `np.frombuffer(...).reshape(-1, 2)[:, 0]` is a
    stride-2 view and needs no copy at all, but the RMS loop walks these arrays tens of
    thousands of times and non-contiguous access roughly halves that throughput.

    Reading in chunks rather than all at once keeps the peak down to the two output
    arrays. Slurping the file first held the interleaved buffer *and* both channels at
    the same time — 828 MB peak against the 434 MB actually needed.
    """
    with wave.open(str(stereo_wav), "rb") as wf:
        if wf.getnchannels() != const.STEREO_CHANNELS:
            raise ValueError(f"Expected stereo WAV, got {wf.getnchannels()} channels")
        sample_rate = wf.getframerate()
        total = wf.getnframes()

        mic = np.empty(total, dtype=np.int16)
        monitor = np.empty(total, dtype=np.int16)

        written = 0
        while written < total:
            raw = wf.readframes(min(const.READ_CHUNK_FRAMES, total - written))
            if not raw:
                break
            chunk = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
            end = written + len(chunk)
            mic[written:end] = chunk[:, 0]
            monitor[written:end] = chunk[:, 1]
            written = end

    # A truncated file reports more frames than it holds; keep only what was read.
    return mic[:written], monitor[:written], sample_rate


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

    # Widened before squaring — see _rms_for_range; int16 samples overflow otherwise.
    mic_rms = float(np.sqrt(np.mean(mic[start_frame:end_frame].astype(np.float64) ** 2)))
    monitor_rms = float(np.sqrt(np.mean(monitor[start_frame:end_frame].astype(np.float64) ** 2)))

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
        if wf.getnchannels() != const.STEREO_CHANNELS:
            return None
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    # int16, contiguous, for the same memory reason as load_stereo_channels.
    interleaved = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
    mic_channel = np.ascontiguousarray(interleaved[:, 0])  # left = mic
    monitor_channel = np.ascontiguousarray(interleaved[:, 1])  # right = monitor

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

            # Widened before squaring — see _rms_for_range. This one accumulates over a
            # whole speaker's segments, so int16 would not merely wrap but go negative,
            # and the square root of that is a complex number.
            mic_energy += float(np.sum(mic_channel[start_frame:end_frame].astype(np.float64) ** 2))
            monitor_energy += float(
                np.sum(monitor_channel[start_frame:end_frame].astype(np.float64) ** 2)
            )
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
