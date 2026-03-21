"""Integration tests — verify multi-module pipelines end-to-end."""

import shutil

import pytest

from meetrec.diarizer import (
    DiarizationSegment,
    assign_speakers,
    classify_segment_by_channel,
    filter_silent_segments,
    identify_user_speaker,
    load_stereo_channels,
    merge_channel_segments,
)
from meetrec.formatter import format_markdown
from meetrec.transcriber import Segment
from tests.fixtures import create_stereo_wav_segments


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_split_and_classify_channel(tmp_path):
    """split_channels_16k → load_stereo_channels → classify_segment_by_channel."""
    from meetrec.audio import split_channels_16k

    stereo = tmp_path / "stereo.wav"
    # 0-1s: mic dominant, 1-2s: monitor dominant
    create_stereo_wav_segments(
        stereo,
        sample_rate=48000,
        segments_spec=[
            (1.0, 0.8, 0.05),  # mic
            (1.0, 0.05, 0.8),  # monitor
        ],
    )

    mic_16k, monitor_16k = split_channels_16k(stereo, tmp_path / "output")

    # Load the original stereo (not the loudnorm'd splits) for classification
    mic, monitor, sr = load_stereo_channels(stereo)

    assert classify_segment_by_channel(0.0, 1.0, mic, monitor, sr) == "mic"
    assert classify_segment_by_channel(1.0, 2.0, mic, monitor, sr) == "monitor"

    # Verify split files exist and are usable
    assert mic_16k.exists()
    assert monitor_16k.exists()


def test_filter_silent_segments_with_stereo(stereo_wav):
    """load_stereo_channels → filter_silent_segments on both channels."""
    wav_path = stereo_wav(
        [
            (1.0, 0.8, 0.003),  # mic loud, monitor silent (RMS ~70 < 200)
            (1.0, 0.003, 0.8),  # mic silent, monitor loud
        ]
    )

    mic_raw, monitor_raw, sr = load_stereo_channels(wav_path)

    segments = [
        Segment(start=0.0, end=1.0, text="Mic speech"),
        Segment(start=1.0, end=2.0, text="Monitor speech"),
    ]

    mic_kept = filter_silent_segments(segments, mic_raw, sr)
    monitor_kept = filter_silent_segments(segments, monitor_raw, sr)

    # Mic filter: keeps 0-1s (loud), drops 1-2s (silent)
    assert len(mic_kept) == 1
    assert mic_kept[0].text == "Mic speech"

    # Monitor filter: drops 0-1s (silent), keeps 1-2s (loud)
    assert len(monitor_kept) == 1
    assert monitor_kept[0].text == "Monitor speech"


def test_assign_speakers_then_format():
    """assign_speakers → format_markdown: speaker labels appear in markdown."""
    segments = [
        Segment(start=0.0, end=5.0, text="Hello from user."),
        Segment(start=5.0, end=10.0, text="Hello from remote."),
        Segment(start=10.0, end=15.0, text="User again."),
    ]

    dsegs = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=5.0),
        DiarizationSegment(speaker="SPEAKER_01", start=5.0, end=10.0),
        DiarizationSegment(speaker="SPEAKER_00", start=10.0, end=15.0),
    ]

    labeled = assign_speakers(segments, dsegs, user_speaker="SPEAKER_00")

    assert labeled[0].speaker == "You"
    assert labeled[1].speaker == "Speaker 1"
    assert labeled[2].speaker == "You"

    markdown = format_markdown(
        segments=labeled,
        session_name="2026-03-20_10-00-00",
        audio_rel_path="attachments/audio/2026-03-20_10-00-00.wav",
        duration_seconds=15.0,
        language="en",
    )

    assert "**You:** Hello from user." in markdown
    assert "**Speaker 1:** Hello from remote." in markdown
    assert "**You:** User again." in markdown


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_stereo_pipeline_to_markdown(tmp_path):
    """Full pipeline: stereo WAV → split → filter → merge → format_markdown."""
    from meetrec.audio import split_channels_16k

    stereo = tmp_path / "stereo.wav"
    # 0-1s: mic only, 1-2s: monitor only, 2-3s: both
    create_stereo_wav_segments(
        stereo,
        sample_rate=48000,
        segments_spec=[
            (1.0, 0.8, 0.003),  # mic speech, monitor silent
            (1.0, 0.003, 0.8),  # mic silent, monitor speech
            (1.0, 0.6, 0.6),    # both speaking
        ],
    )

    split_channels_16k(stereo, tmp_path / "output")
    mic_raw, monitor_raw, sr = load_stereo_channels(stereo)

    # Simulate Whisper output (one segment per time region)
    mic_segments = [
        Segment(start=0.0, end=1.0, text="My speech.", speaker="You"),
        Segment(start=1.0, end=2.0, text="Crosstalk from monitor.", speaker="You"),
        Segment(start=2.0, end=3.0, text="Both talking.", speaker="You"),
    ]
    monitor_segments = [
        Segment(start=0.0, end=1.0, text="Crosstalk from mic."),
        Segment(start=1.0, end=2.0, text="Remote speech."),
        Segment(start=2.0, end=3.0, text="Both talking too."),
    ]

    # Filter crosstalk via RMS
    mic_filtered = filter_silent_segments(mic_segments, mic_raw, sr)
    monitor_filtered = filter_silent_segments(monitor_segments, monitor_raw, sr)

    merged = merge_channel_segments(mic_filtered, monitor_filtered)

    markdown = format_markdown(
        segments=merged,
        session_name="2026-03-20_10-00-00",
        audio_rel_path="audio.wav",
        duration_seconds=3.0,
        language="en",
    )

    # Mic speech kept, monitor crosstalk dropped (and vice versa)
    assert "My speech." in markdown
    assert "Remote speech." in markdown
    # Crosstalk should be filtered out
    assert "Crosstalk from monitor" not in markdown
    assert "Crosstalk from mic" not in markdown


def test_identify_and_assign_with_real_stereo(stereo_wav):
    """identify_user_speaker → assign_speakers: full path from WAV to labels."""
    wav_path = stereo_wav(
        [
            (1.0, 0.8, 0.05),  # SPEAKER_00 on mic
            (1.0, 0.05, 0.8),  # SPEAKER_01 on monitor
        ]
    )

    dsegs = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=1.0),
        DiarizationSegment(speaker="SPEAKER_01", start=1.0, end=2.0),
    ]

    user_speaker = identify_user_speaker(dsegs, wav_path)
    assert user_speaker == "SPEAKER_00"

    segments = [
        Segment(start=0.0, end=1.0, text="I am the user."),
        Segment(start=1.0, end=2.0, text="I am remote."),
    ]

    labeled = assign_speakers(
        segments,
        dsegs,
        user_speaker=user_speaker,
        stereo_wav=wav_path,
    )

    assert labeled[0].speaker == "You"
    assert labeled[1].speaker == "Speaker 1"
