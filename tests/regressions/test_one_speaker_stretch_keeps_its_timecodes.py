"""Regression: a call in which only the user was heard became one block of the note.

`merge_channel_segments` consolidated every run of one speaker into one segment, whatever
the gaps between its segments and its length. With the other side silent, the whole call
became one segment, and the note's 60 s block cap, which splits only between segments,
could not cut it: one timecode for the whole call.
"""

from tapeback.diarizer import merge_channel_segments
from tapeback.formatter import TranscriptMeta, format_markdown
from tapeback.models import Segment


def _timecodes(segments: list[Segment], duration: float) -> list[str]:
    markdown = format_markdown(
        segments=merge_channel_segments(segments, [], pause_threshold=1.0),
        meta=TranscriptMeta(
            session_name="2026-03-23_14-30-00",
            audio_rel_path="attachments/audio/x.wav",
            duration_seconds=duration,
            language="ru",
        ),
    )
    return [line.split("]")[0] + "]" for line in markdown.splitlines() if line.startswith("[")]


def test_phrases_apart_by_pauses_stay_apart():
    # A 5 s phrase every 10 s for 10 minutes.
    phrases = [
        Segment(start=i * 10.0, end=i * 10.0 + 5.0, text=f"Phrase {i}.", speaker="You")
        for i in range(60)
    ]

    assert len(_timecodes(phrases, 600.0)) == 60


def test_unbroken_speech_gets_a_timecode_a_minute():
    # 4.8 s of speech every 5 s for 3 minutes: no gap reaches the pause threshold.
    speech = [
        Segment(start=i * 5.0, end=i * 5.0 + 4.8, text=f"Part {i}.", speaker="You")
        for i in range(36)
    ]

    assert _timecodes(speech, 180.0) == ["[00:00:00]", "[00:01:00]", "[00:02:00]"]
