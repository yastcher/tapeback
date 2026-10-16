"""Channel analysis — stereo toolkit integration flow + guard-branch edges."""

import numpy as np
import pytest

from tapeback.channel import (
    classify_segment_by_channel,
    filter_mic_segments,
    filter_silent_segments,
    gate_inactive_regions,
    identify_user_speaker,
    load_stereo_channels,
    split_on_silence,
)
from tapeback.models import DiarizationSegment, Segment, Word
from tests.fixtures import create_mono_wav, create_stereo_wav


def test_stereo_channel_pipeline_flow(stereo_wav):
    """Continuous flow over the channel toolkit, mirroring process_stereo_file:

    load stereo -> split mic on a real pause -> drop crosstalk words ->
    classify regions by channel energy -> identify the user (mic) speaker.
    """
    # mic (left) speaks 0-2s; monitor (right) speaks 2-4s while the mic goes quiet.
    wav = stereo_wav([(2.0, 0.8, 0.0), (2.0, 0.0, 0.8)])
    mic_raw, monitor_raw, sr = load_stereo_channels(wav)
    assert sr == 16000
    assert len(mic_raw) > 0
    assert len(monitor_raw) > 0

    # One mic segment over the whole recording, with a crosstalk word ("bleed")
    # landing in the monitor region where the mic is silent.
    mic_seg = Segment(
        start=0.0,
        end=4.0,
        text="hello world bleed",
        words=[
            Word(start=0.3, end=0.9, word="hello", probability=0.9),
            Word(start=1.3, end=1.9, word="world", probability=0.9),
            Word(start=2.5, end=3.0, word="bleed", probability=0.5),
        ],
        speaker="You",
    )

    # The mic falls silent at 2s while the monitor keeps going -> the segment is cut.
    split = split_on_silence(
        [mic_seg], mic_raw, sr, pause_threshold=1.0, monitor_samples=monitor_raw
    )
    assert any(s.end < 4.0 for s in split)

    # Word-level filtering drops the crosstalk word (mic silent there); real speech stays.
    filtered = filter_silent_segments(split, mic_raw, sr)
    kept_text = " ".join(s.text for s in filtered)
    assert "hello" in kept_text
    assert "world" in kept_text
    assert "bleed" not in kept_text

    # Channel classification: mic region is mic-dominant, monitor region monitor-dominant.
    assert classify_segment_by_channel(0.0, 2.0, mic_raw, monitor_raw, sr) == "mic"
    assert classify_segment_by_channel(2.0, 4.0, mic_raw, monitor_raw, sr) == "monitor"

    # The mic-region speaker is identified as the user.
    dsegs = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.0),
        DiarizationSegment(speaker="SPEAKER_01", start=2.0, end=4.0),
    ]
    assert identify_user_speaker(dsegs, wav) == "SPEAKER_00"


def test_split_on_silence_keeps_a_segment_without_words_whole(stereo_wav):
    """Nothing places a wordless segment's text on either side of a pause."""
    wav = stereo_wav([(1.5, 0.8, 0.0), (1.5, 0.0, 0.0), (1.5, 0.8, 0.0)])
    mic_raw, _monitor_raw, sr = load_stereo_channels(wav)

    seg = Segment(start=0.0, end=4.5, text="no word timings", words=None, speaker="You")

    assert split_on_silence([seg], mic_raw, sr, pause_threshold=1.0) == [seg]


def test_mic_filter_keeps_a_wordless_segment_the_user_spoke_in(stereo_wav):
    """Judged whole, the segment is the other side's: the user spoke two seconds of
    twelve. Judged stretch by stretch between the mic's pauses, it is the user's."""
    wav = stereo_wav([(2.0, 0.8, 0.0), (10.0, 0.0, 0.8)])
    mic_raw, monitor_raw, sr = load_stereo_channels(wav)
    seg = Segment(start=0.0, end=12.0, text="Budget first.", words=None, speaker="You")

    assert classify_segment_by_channel(0.0, 12.0, mic_raw, monitor_raw, sr) == "monitor"
    assert filter_mic_segments([seg], mic_raw, monitor_raw, sr, pause_threshold=1.0) == [seg]


def test_mic_filter_drops_a_wordless_segment_that_is_bleed_throughout(stereo_wav):
    wav = stereo_wav([(6.0, 0.05, 0.8)])
    mic_raw, monitor_raw, sr = load_stereo_channels(wav)
    seg = Segment(start=0.0, end=6.0, text="Heard through the mic.", words=None, speaker="You")

    assert filter_mic_segments([seg], mic_raw, monitor_raw, sr, pause_threshold=1.0) == []


def test_mic_filter_does_not_count_silence_as_the_user(stereo_wav):
    """Bleed, then quiet on both channels: the quiet stretch is nobody's."""
    wav = stereo_wav([(3.0, 0.05, 0.8), (3.0, 0.0, 0.0)])
    mic_raw, monitor_raw, sr = load_stereo_channels(wav)
    seg = Segment(start=0.0, end=6.0, text="Heard through the mic.", words=None, speaker="You")

    assert filter_mic_segments([seg], mic_raw, monitor_raw, sr, pause_threshold=1.0) == []


def test_mic_filter_judges_a_segment_with_words_whole(stereo_wav):
    """Pause splitting already cut it at the pauses: one comparison decides."""
    wav = stereo_wav([(2.0, 0.8, 0.0), (2.0, 0.05, 0.8)])
    mic_raw, monitor_raw, sr = load_stereo_channels(wav)
    own = Segment(
        start=0.0,
        end=2.0,
        text="Hi",
        words=[Word(start=0.0, end=2.0, word=" Hi", probability=0.9)],
        speaker="You",
    )
    bleed = Segment(
        start=2.0,
        end=4.0,
        text="Hi",
        words=[Word(start=2.0, end=4.0, word=" Hi", probability=0.9)],
        speaker="You",
    )

    result = filter_mic_segments([own, bleed], mic_raw, monitor_raw, sr, pause_threshold=1.0)

    assert result == [own]


def test_mic_filter_drops_silent_words(stereo_wav):
    """Quiet on both channels: the word was never said into the mic."""
    wav = stereo_wav([(2.0, 0.0, 0.0)])
    mic_raw, monitor_raw, sr = load_stereo_channels(wav)
    seg = Segment(
        start=0.0,
        end=2.0,
        text="Hi",
        words=[Word(start=0.0, end=2.0, word=" Hi", probability=0.9)],
        speaker="You",
    )

    assert filter_mic_segments([seg], mic_raw, monitor_raw, sr, pause_threshold=1.0) == []


def test_mic_filter_judges_a_wordless_segment_by_stretches_not_its_average(stereo_wav):
    """Two seconds of the user in thirty: the average is below the silence threshold,
    the stretch between the other side's two remarks is not."""
    wav = stereo_wav(
        [(10.0, 0.0, 0.0), (2.0, 0.0, 0.02), (2.0, 0.02, 0.0), (2.0, 0.0, 0.02), (14.0, 0.0, 0.0)]
    )
    mic_raw, monitor_raw, sr = load_stereo_channels(wav)
    seg = Segment(start=0.0, end=30.0, text="Agreed.", words=None, speaker="You")

    assert filter_silent_segments([seg], mic_raw, sr) == []
    assert filter_mic_segments([seg], mic_raw, monitor_raw, sr, pause_threshold=1.0) == [seg]


def test_split_on_silence_segment_shorter_than_window_unchanged():
    """A segment shorter than the analysis window is returned untouched."""
    sr = 16000
    mic = np.zeros(int(0.05 * sr), dtype=np.float32)  # shorter than the 0.1s window
    seg = Segment(start=0.0, end=0.05, text="blip", words=None)

    assert split_on_silence([seg], mic, sr, pause_threshold=1.0) == [seg]


def test_filter_silent_segments_drops_zero_duration_segment():
    """An empty (start == end) wordless segment has zero RMS and is dropped."""
    sr = 16000
    loud = np.ones(sr, dtype=np.float32) * 1000.0
    seg = Segment(start=0.5, end=0.5, text="empty", words=None)

    assert filter_silent_segments([seg], loud, sr) == []


def test_classify_segment_by_channel_empty_range_is_none():
    """An empty time range (end <= start) is ambiguous -> None."""
    sr = 16000
    mic = np.zeros(sr, dtype=np.float32)
    monitor = np.zeros(sr, dtype=np.float32)

    assert classify_segment_by_channel(0.5, 0.5, mic, monitor, sr) is None


def test_gate_inactive_regions_silences_listening_windows():
    """Windows where the user listens (mic quiet / monitor dominant) are zeroed;
    windows where the user speaks are preserved — so Whisper never sees the silence.
    """
    raw_sr = 48000
    out_sr = 16000
    # 0-1s: user speaks (mic loud, monitor quiet). 1-2s: user listens (mic quiet, monitor loud).
    mic_raw = np.concatenate([np.full(raw_sr, 5000.0), np.zeros(raw_sr)]).astype(np.float32)
    monitor_raw = np.concatenate([np.zeros(raw_sr), np.full(raw_sr, 5000.0)]).astype(np.float32)
    target_16k = np.full(2 * out_sr, 1000.0, dtype=np.float32)

    gated = gate_inactive_regions(target_16k, mic_raw, monitor_raw, raw_sr)

    assert np.any(gated[:out_sr] != 0)  # speech region preserved
    assert np.all(gated[out_sr:] == 0)  # listening region silenced
    assert target_16k[out_sr] == 1000.0  # input not mutated


def test_load_stereo_channels_rejects_mono(tmp_path):
    """Mono input is a programming error for the dual-channel path."""
    mono = tmp_path / "mono.wav"
    create_mono_wav(mono, duration=1.0)

    with pytest.raises(ValueError, match="stereo"):
        load_stereo_channels(mono)


def test_identify_user_speaker_returns_none_for_mono(tmp_path):
    """identify_user_speaker needs a stereo file; mono yields no answer."""
    mono = tmp_path / "mono.wav"
    create_mono_wav(mono, duration=1.0)
    dsegs = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=0.5),
        DiarizationSegment(speaker="SPEAKER_01", start=0.5, end=1.0),
    ]

    assert identify_user_speaker(dsegs, mono) is None


def test_identify_user_speaker_none_when_segments_have_no_energy(tmp_path):
    """Zero-length diarization segments contribute no frames -> no ratios -> None."""
    wav = tmp_path / "stereo.wav"
    create_stereo_wav(
        wav, duration=1.0, sample_rate=16000, left_amplitude=0.8, right_amplitude=0.05
    )
    dsegs = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.5, end=0.5),
        DiarizationSegment(speaker="SPEAKER_01", start=0.5, end=0.5),
    ]

    assert identify_user_speaker(dsegs, wav) is None
