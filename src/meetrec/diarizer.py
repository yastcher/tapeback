import contextlib
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from meetrec.settings import Settings
from meetrec.transcriber import Segment


@dataclass
class DiarizationSegment:
    """A continuous speech segment from one speaker (from pyannote)."""

    speaker: str  # "SPEAKER_00", "SPEAKER_01", ...
    start: float  # seconds
    end: float  # seconds


def _unwrap_diarization(result):
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
                "Set MEETREC_HF_TOKEN in your .env file."
            )

        from pyannote.audio import Pipeline

        self._settings = settings
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=settings.hf_token,
        )

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
        kwargs = {}
        if self._settings.max_speakers is not None:
            kwargs["max_speakers"] = self._settings.max_speakers

        try:
            diarization = self._pipeline(audio_path, **kwargs)
        except RuntimeError as exc:
            if "CUDA" not in str(exc) and "out of memory" not in str(exc):
                raise
            print(
                f"Warning: CUDA out of memory during diarization, falling back to CPU: {exc}",
                file=sys.stderr,
            )
            self._fallback_to_cpu()
            diarization = self._pipeline(audio_path, **kwargs)

        # pyannote 4.x returns DiarizeOutput wrapping Annotation
        annotation = _unwrap_diarization(diarization)

        segments = []
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
    stereo_data = None
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
