"""Unit tests for transcript-quality metrics."""

import pytest

from tapeback._quality import (
    count_hallucination_markers,
    count_recognised_terms,
    count_repeated_phrases,
    count_repeated_words,
    find_hallucination_markers,
    has_speech,
    low_confidence_rate,
    punctuation_per_1000_words,
    strip_hallucinations,
)


def test_find_hallucination_markers_real_transcript_lines():
    """Real lines from the project's own transcripts, glued to neighbouring words."""
    text = (
        "Ведь решение... Субтитры DimaTorzok штука, куда ты вставляешь summary. "
        "поскольку Корректор .Кулакова общаются мире."
    )
    assert find_hallucination_markers(text) == ["субтитры", "dimatorzok", "корректор"]


def test_count_hallucination_markers_counts_occurrences_not_distinct():
    text = "Продолжение следует... и снова Продолжение следует..."
    assert count_hallucination_markers(text) == 2


def test_no_hallucination_markers_in_clean_text():
    assert find_hallucination_markers("Обычная встреча про RAG и вектора.") == []
    assert count_hallucination_markers("Обычная встреча про RAG и вектора.") == 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Boundary: two in a row is emphasis, three is a loop.
        ("да да ладно", 0),
        ("да да да ладно", 1),
        ("да да да да ладно", 1),
        # A long run counts once, not once per extra repeat.
        ("no " * 10, 1),
        # Two separate runs.
        ("да да да потом нет нет нет", 2),
        ("совершенно обычная фраза", 0),
        ("", 0),
        # A doubled word after another word is still emphasis.
        ("ладно да да", 0),
        # Different words, though they share an upper case.
        ("Maße Masse Maße", 0),
    ],
)
def test_count_repeated_words(text, expected):
    assert count_repeated_words(text) == expected


def test_count_repeated_words_ignores_digits():
    """Timecodes and numbers must not register as repeated words."""
    assert count_repeated_words("00 00 00 15 15 15") == 0


def test_count_repeated_words_is_case_insensitive():
    assert count_repeated_words("Да да ДА ладно") == 1


def test_count_repeated_words_splits_on_apostrophes():
    """Contractions tokenise into two words, so a repeated one alternates.

    "Let's let's let's" becomes let/s/let/s/let/s — no word is adjacent to itself,
    so word-level detection reports nothing. This is why phrase-level detection
    exists alongside it.
    """
    assert count_repeated_words("Let's let's LET'S go") == 0
    assert count_repeated_phrases("Let's let's LET'S go", size=2) == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The failure mode word-level detection misses: no single word repeats.
        ("something happens wrong something happens wrong ok", 1),
        # Boundary: one occurrence is not a loop.
        ("something happens wrong ok", 0),
        # Three back-to-back repeats are still one loop.
        ("a b c a b c a b c", 1),
        ("совсем другая речь без повторов вообще", 0),
        ("", 0),
        # Exactly two phrases' worth of words, nothing around them.
        ("something happens wrong something happens wrong", 1),
        # A loop that starts one word in.
        ("so something happens wrong something happens wrong", 1),
        # Two loops back to back.
        ("we need the we need the rag is broken rag is broken", 2),
    ],
)
def test_count_repeated_phrases(text, expected):
    assert count_repeated_phrases(text) == expected


@pytest.mark.parametrize(
    ("text", "size", "min_repeats", "expected"),
    [
        ("да да ладно", 0, 2, 0),
        ("да да ладно", 1, 2, 1),
        ("да ладно да ладно", 2, 2, 1),
        ("да ладно", 1, 0, 0),
        ("да ладно", 1, 1, 2),
    ],
)
def test_count_repeated_phrases_needs_a_word_and_a_repeat(text, size, min_repeats, expected):
    assert count_repeated_phrases(text, size=size, min_repeats=min_repeats) == expected


def test_count_repeated_phrases_ignores_non_adjacent_recurrence():
    """A phrase recurring later in a meeting is normal speech, not a loop."""
    text = "we need the rag system and then a lot of other words we need the rag"
    assert count_repeated_phrases(text) == 0


def test_strip_hallucinations_cuts_the_phrase_and_keeps_the_speech():
    text = "Ведь решение. Субтитры DimaTorzok штука, куда ты вставляешь summary."
    assert strip_hallucinations(text) == "Ведь решение. штука, куда ты вставляешь summary"


def test_strip_hallucinations_removes_trailing_attribution():
    assert strip_hallucinations("поскольку Корректор .Кулакова общаются") == ("поскольку общаются")


def test_strip_hallucinations_leaves_clean_text_byte_identical():
    """Clean text must come back untouched — an earlier version ate the full stop."""
    text = "Обычная реплика про pipeline."
    assert strip_hallucinations(text) is text


@pytest.mark.parametrize(
    ("text", "cleaned"),
    [
        ("решение Субтитры DimaTorzok, штука", "решение, штука"),
        ("обсудили SpaceX Субтитры DimaTorzok", "обсудили SpaceX"),
    ],
)
def test_strip_hallucinations_tidies_only_the_separators(text, cleaned):
    """No space is left before a comma, and a word's last letter stays."""
    assert strip_hallucinations(text) == cleaned


def test_strip_hallucinations_can_empty_a_segment():
    assert strip_hallucinations("Продолжение следует...") == ""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("слова", True),
        ("words", True),
        ("", False),
        ("   ", False),
        ("... , ;", False),
        ("123", False),
    ],
)
def test_has_speech(text, expected):
    assert has_speech(text) is expected


def test_punctuation_per_1000_words():
    # 10 words, 3 marks -> 300 per 1000.
    assert (
        punctuation_per_1000_words("раз, два, три четыре пять шесть семь восемь девять десять.")
        == 300.0
    )


def test_punctuation_per_1000_words_distinguishes_the_real_failure_mode():
    """The exact pair a hand comparison turned up: same speech, one unpunctuated."""
    unpunctuated = "я слышал что то периодически попадаются в новостях то есть это такая вещь"
    punctuated = (
        "Я слышал что-то, знаешь, периодически попадаются в новостях, то есть это такая вещь."
    )
    assert punctuation_per_1000_words(unpunctuated) == 0.0
    assert punctuation_per_1000_words(punctuated) > 200.0


def test_punctuation_per_1000_words_empty_text():
    assert punctuation_per_1000_words("") == 0.0


@pytest.mark.parametrize(
    ("probabilities", "threshold", "expected"),
    [
        ([0.9, 0.9, 0.9, 0.1], 0.35, 25.0),
        # Boundary: the threshold itself is NOT below it.
        ([0.35], 0.35, 0.0),
        ([0.34], 0.35, 100.0),
        ([], 0.35, 0.0),
    ],
)
def test_low_confidence_rate(probabilities, threshold, expected):
    assert low_confidence_rate(probabilities, threshold) == expected


def test_count_recognised_terms():
    text = "We are building the RAG system with ONNX Runtime and vector search."
    found, missing = count_recognised_terms(text, ["RAG", "ONNX", "OpenVINO", "vector search"])
    assert found == 3
    assert missing == ["OpenVINO"]


def test_count_recognised_terms_is_case_insensitive():
    found, missing = count_recognised_terms("we use onnx runtime", ["ONNX"])
    assert found == 1
    assert missing == []


def test_count_recognised_terms_with_empty_list():
    assert count_recognised_terms("anything", []) == (0, [])
