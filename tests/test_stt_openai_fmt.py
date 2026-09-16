"""OpenAI Audio Transcriptions response → Segment mapping."""

from types import SimpleNamespace

from tapeback._stt_openai_fmt import (
    _info_from_response,
    _keywords_from_hotwords,
    _segments_from_diarized,
    _segments_from_text_response,
    _segments_from_verbose,
    _word_from_api,
)


def test_segments_from_verbose_maps_words_and_offset():
    response = SimpleNamespace(
        segments=[
            SimpleNamespace(
                start=0.0,
                end=1.5,
                text=" Hello world",
                words=[
                    SimpleNamespace(word="Hello", start=0.0, end=0.5),
                    SimpleNamespace(word="world", start=0.5, end=1.5),
                ],
            )
        ]
    )
    segments = _segments_from_verbose(response, time_offset=10.0)
    assert len(segments) == 1
    assert segments[0].start == 10.0
    assert segments[0].end == 11.5
    assert segments[0].text == "Hello world"
    assert segments[0].words is not None
    assert len(segments[0].words) == 2
    assert segments[0].words[0].start == 10.0
    assert segments[0].words[0].probability == 1.0


def test_segments_from_text_response_single_segment():
    response = SimpleNamespace(text="  one shot  ")
    segments = _segments_from_text_response(response, time_offset=2.0, duration=5.0)
    assert len(segments) == 1
    assert segments[0].start == 2.0
    assert segments[0].end == 7.0
    assert segments[0].text == "one shot"
    assert segments[0].words is None


def test_segments_from_diarized_maps_speakers():
    response = SimpleNamespace(
        segments=[
            SimpleNamespace(speaker="A", start=0.0, end=1.0, text="one"),
            SimpleNamespace(speaker="B", start=1.0, end=2.0, text="two"),
            SimpleNamespace(speaker="A", start=2.0, end=3.0, text="three"),
        ]
    )
    segments = _segments_from_diarized(response, time_offset=5.0)
    assert [s.text for s in segments] == ["one", "two", "three"]
    assert [s.speaker for s in segments] == ["Speaker 1", "Speaker 2", "Speaker 1"]
    assert segments[0].start == 5.0
    assert segments[1].start == 6.0


def test_word_from_api_dict_and_missing():
    word = _word_from_api({"word": "hi", "start": 0.1, "end": 0.2})
    assert word is not None
    assert word.word == "hi"
    assert word.start == 0.1
    assert word.end == 0.2
    assert word.probability == 1.0
    assert _word_from_api({"word": "hi"}) is None


def test_keywords_from_hotwords():
    assert _keywords_from_hotwords("tapeback, RabbitMQ") == ["tapeback", "RabbitMQ"]
    assert _keywords_from_hotwords("  ") == []


def test_info_from_response_languages_list():
    response = SimpleNamespace(languages=[SimpleNamespace(code="en")], duration=3.0)
    info = _info_from_response(response, 9.0, language_fallback=None)
    assert info["language"] == "en"
    assert info["duration"] == 9.0
    assert info["partial"] is False
