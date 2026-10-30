"""Pure transcript-quality metrics.

These exist so that "did this configuration get better or worse" is a number rather
than an impression. Transcription tuning on this project has twice been reverted
after a change that felt right measured badly, so every knob is now judged by the
same three questions: did the model hallucinate, did it get stuck repeating itself,
and did it get the technical vocabulary right.

Everything here is pure and text-only — the benchmark harness in `scripts/` and the
transcript filter use the same definitions, so a fix is measured by the metric it
was meant to move.
"""

from __future__ import annotations

import itertools
import re

from tapeback import const

# Word characters for both alphabets in use; \w would also match digits, which
# would let timecodes and numbers register as repeated "words".
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# A repeated run this long is a decoding loop, not natural emphasis. Russian and
# English both allow a doubled word ("да да", "very very"); three in a row is where
# it stops being speech.
MIN_REPEAT_RUN = 3

# Length of the phrase considered for loop detection, and how many times it must
# repeat back-to-back to count. Catches "something wrong something wrong", which
# word-level detection misses because no single word repeats consecutively.
LOOP_NGRAM_SIZE = 3
MIN_NGRAM_REPEATS = 2

PERCENT = 100.0


def _words(text: str) -> list[str]:
    return [match.group().lower() for match in _WORD_RE.finditer(text)]


def find_hallucination_markers(text: str) -> list[str]:
    """Return the subtitle-corpus markers present in the text, in list order.

    Substring matching on a lowercased haystack: the markers turn up glued to
    neighbouring words ("Корректор .Кулакова"), so anchoring on word boundaries
    would miss real occurrences.
    """
    haystack = text.lower()
    return [marker for marker in const.HALLUCINATION_MARKERS if marker in haystack]


def count_hallucination_markers(text: str) -> int:
    """Total occurrences (not distinct markers) of subtitle-corpus phrases."""
    haystack = text.lower()
    return sum(haystack.count(marker) for marker in const.HALLUCINATION_MARKERS)


def strip_hallucinations(text: str) -> str:
    """Remove subtitle-corpus phrases, leaving the surrounding speech intact.

    The markers turn up mid-sentence, splitting real speech ("Ведь решение...
    Субтитры DimaTorzok штука, куда ты вставляешь summary"), so the phrase is cut out
    rather than the segment being discarded — dropping the whole segment would lose
    what was actually said. The trailing name a credit carries ("Корректор .Кулакова",
    "Субтитры DimaTorzok") is removed with it, since it is part of the same artefact.

    Returns the cleaned text, which may be empty when the segment was nothing else.
    """
    cleaned = text
    for marker in const.HALLUCINATION_MARKERS:
        # Optional trailing attribution: ".Кулакова", "DimaTorzok", "Семкин".
        pattern = re.compile(rf"{re.escape(marker)}\s*\.?\s*[^\W\d_]*", re.IGNORECASE | re.UNICODE)
        cleaned = pattern.sub(" ", cleaned)

    # Only tidy up when something was actually removed. Running the cleanup
    # unconditionally rewrote every clean segment too — it stripped the closing full
    # stop off ordinary speech, which the formatter tests caught.
    if cleaned == text:
        return text

    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    return cleaned.strip(" .,;:")


def has_speech(text: str) -> bool:
    """True if anything but punctuation and whitespace remains."""
    return bool(_WORD_RE.search(text))


def count_repeated_words(text: str, min_run: int = MIN_REPEAT_RUN) -> int:
    """Count runs of the same word repeated at least ``min_run`` times in a row.

    A run of any length counts once, so a 10x repeat is one loop rather than eight.

    Known false positives: conversational Russian genuinely repeats short words for
    emphasis or hesitation ("надо надо надо", "да да да да"), and a faithful
    transcript of real speech contains them. Measured on a 31-minute recording, all
    six hits were natural speech, none were decoding loops. Treat a rise here as a
    prompt to look, not as a regression; `count_repeated_phrases` is the reliable
    loop signal because a repeated 3-gram is not something people say.
    """
    words = _words(text)
    loops = 0
    run = 1
    for previous, current in itertools.pairwise(words):
        if current == previous:
            run += 1
            if run == min_run:
                loops += 1
        else:
            run = 1
    return loops


def count_repeated_phrases(
    text: str,
    size: int = LOOP_NGRAM_SIZE,
    min_repeats: int = MIN_NGRAM_REPEATS,
) -> int:
    """Count phrases of ``size`` words repeated back-to-back ``min_repeats`` times.

    Only immediate repetition counts: a phrase recurring later in a meeting is
    normal speech, whereas one repeating on its own heels is a decoding loop.
    """
    words = _words(text)
    # An empty phrase matches itself without end. Too few words need no check here:
    # the loop below does not start.
    if size < 1 or min_repeats < 1:
        return 0

    loops = 0
    index = 0
    while index + size * min_repeats <= len(words):
        phrase = words[index : index + size]
        repeats = 1
        while words[index + repeats * size : index + (repeats + 1) * size] == phrase:
            repeats += 1
        if repeats >= min_repeats:
            loops += 1
            index += repeats * size
        else:
            index += 1
    return loops


def punctuation_per_1000_words(text: str) -> float:
    """Commas and sentence terminators per 1000 words.

    Added after a hand comparison showed the metric suite was blind to the thing that
    most affects reading: one model returned an unpunctuated run of speech where
    another returned the same words as sentences. Density rather than a count, so
    transcripts of different lengths compare.
    """
    words = _words(text)
    if not words:
        return 0.0
    marks = len(re.findall(r"[,.!?;:]", text))
    return marks / len(words) * 1000


def low_confidence_rate(probabilities: list[float], threshold: float) -> float:
    """Share of words Whisper itself scored below ``threshold``, as a percentage.

    The model's own uncertainty is the cheapest honest proxy for recognition
    quality — it needs no reference transcript and no glossary. Measured across
    three configurations of one recording it ordered them exactly as a human
    reading did.
    """
    if not probabilities:
        return 0.0
    below = sum(1 for probability in probabilities if probability < threshold)
    return below / len(probabilities) * PERCENT


def count_recognised_terms(text: str, terms: list[str]) -> tuple[int, list[str]]:
    """Return how many expected terms appear, and which are missing.

    Case-insensitive substring match. The point is whether the term survived
    recognition at all — "RAG" coming back as "Rick" is a miss regardless of how
    the surrounding sentence reads.
    """
    haystack = text.lower()
    missing = [term for term in terms if term.lower() not in haystack]
    return len(terms) - len(missing), missing
