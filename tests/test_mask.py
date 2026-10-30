"""PII masking at the LLM boundary (_mask.py).

The property that matters is the round trip: masking removes raw PII from the text that
crosses to a provider, and unmasking restores the EXACT original, so the vault keeps what
was said. The other half is that masking off means nothing changes at all.
"""

import pytest

from tapeback._mask import Masker

_EMAIL = "ivan.petrov@example.com"
_PHONE = "+7 900 123-45-67"


def test_round_trip_restores_exact_original():
    masker = Masker(enabled=True)
    original = f"Call {_PHONE} or write to {_EMAIL} before Friday."

    masked = masker.mask(original)

    assert "@" not in masked
    assert "+7" not in masked
    assert masked == "Call [PHONE_1] or write to [EMAIL_1] before Friday."
    assert masker.unmask(masked) == original


def test_email_placeholder_and_mapping():
    masker = Masker(enabled=True)

    masked = masker.mask(f"write to {_EMAIL} please")

    assert masked == "write to [EMAIL_1] please"
    assert masker.mapping == {"[EMAIL_1]": "ivan.petrov@example.com"}


@pytest.mark.parametrize(
    "phone",
    ["+7 900 123-45-67", "+79001234567", "8 (900) 123 45 67", "8-900-123-45-67", "+1 415 555 2671"],
)
def test_phone_formats(phone):
    masker = Masker(enabled=True)

    masked = masker.mask(f"call {phone} now")

    assert masked == "call [PHONE_1] now"
    assert masker.mapping == {"[PHONE_1]": phone}


@pytest.mark.parametrize(
    "text",
    ["recorded in 2026", "order 12345678", "8 items left", "build +2 patch", "version 1.2.3"],
)
def test_non_pii_is_left_alone(text):
    masker = Masker(enabled=True)

    assert masker.mask(text) == text
    assert masker.mapping == {}


def test_same_value_gets_one_placeholder():
    masker = Masker(enabled=True)

    masked = masker.mask(f"{_EMAIL} and again {_EMAIL}")

    assert masked == "[EMAIL_1] and again [EMAIL_1]"
    assert masker.mapping == {"[EMAIL_1]": "ivan.petrov@example.com"}


def test_distinct_values_get_distinct_placeholders():
    masker = Masker(enabled=True)

    masked = masker.mask("a@x.com, b@y.com")

    assert masked == "[EMAIL_1], [EMAIL_2]"
    assert masker.mapping == {"[EMAIL_1]": "a@x.com", "[EMAIL_2]": "b@y.com"}


def test_counter_continues_across_texts_in_one_call():
    masker = Masker(enabled=True)

    first = masker.mask(f"note {_EMAIL}")
    second = masker.mask(f"user wrote {_EMAIL} and a@other.com")

    assert first == "note [EMAIL_1]"
    assert second == "user wrote [EMAIL_1] and [EMAIL_2]"
    assert masker.unmask(second) == "user wrote ivan.petrov@example.com and a@other.com"


def test_unmask_prefers_the_longest_placeholder():
    """[EMAIL_1] must not clobber the [EMAIL_11] that contains it."""
    masker = Masker(enabled=True)
    addresses = [f"user{i}@example.com" for i in range(1, 12)]

    masked = masker.mask(", ".join(addresses))

    assert masked.endswith("[EMAIL_11]")
    assert masker.unmask(masked) == ", ".join(addresses)


def test_disabled_masker_changes_nothing():
    masker = Masker(enabled=False)
    original = f"Call {_PHONE} or write to {_EMAIL}."

    masked = masker.mask(original)

    assert masked == original
    assert masker.mapping == {}
    assert masker.unmask(masked) == original


def test_empty_text():
    masker = Masker(enabled=True)

    assert masker.mask("") == ""
    assert masker.unmask("") == ""


def test_unmask_without_any_replacement_is_identity():
    masker = Masker(enabled=True)
    masker.mask("nothing personal here")

    assert masker.unmask("still [EMAIL_1] nothing") == "still [EMAIL_1] nothing"


def test_residual_placeholders_found():
    masker = Masker(enabled=True)
    masker.mask(f"write to {_EMAIL}")

    assert masker.residual_placeholders("reach [EMAIL_7] instead") == ["[EMAIL_7]"]


def test_no_residual_placeholders_after_successful_unmask():
    masker = Masker(enabled=True)
    masked = masker.mask(f"write to {_EMAIL}")

    assert masker.residual_placeholders(masker.unmask(masked)) == []


def test_residual_placeholders_ignored_when_nothing_was_masked():
    """A placeholder-shaped string we did not create is the model's own text."""
    masker = Masker(enabled=True)
    masker.mask("nothing personal here")

    assert masker.residual_placeholders("the model wrote [EMAIL_1]") == []


# ---- user-supplied terms ----------------------------------------------------


def test_term_is_masked_and_restored():
    masker = Masker(enabled=True, terms="Ivan")

    masked = masker.mask("Ivan promised to send it")

    assert masked == "[TERM_1] promised to send it"
    assert masker.unmask(masked) == "Ivan promised to send it"


def test_term_matching_is_case_insensitive_and_restores_what_was_written():
    masker = Masker(enabled=True, terms="Ivan")

    masked = masker.mask("Ivan and ivan")

    # Different casings are different originals, so each restores exactly as written.
    assert masked == "[TERM_1] and [TERM_2]"
    assert masker.unmask(masked) == "Ivan and ivan"


@pytest.mark.parametrize(
    "terms",
    ["Ivan, Ivan Petrov", "ivan, Ivan Petrov"],
    ids=["same case", "a lowercase prefix"],
)
def test_longest_term_wins(terms):
    masker = Masker(enabled=True, terms=terms)

    masked = masker.mask("Ivan Petrov called Ivan")

    assert masked == "[TERM_1] called [TERM_2]"
    assert masker.mapping == {"[TERM_1]": "Ivan Petrov", "[TERM_2]": "Ivan"}


def test_term_is_word_bounded():
    """A term that is a strict prefix of another word must not be masked inside it."""
    masker = Masker(enabled=True, terms="Ann")

    masked = masker.mask("Ann met Anna and Annabel")

    assert masked == "[TERM_1] met Anna and Annabel"


def test_term_ending_in_punctuation():
    masker = Masker(enabled=True, terms="Acme Inc.")

    masked = masker.mask("signed with Acme Inc. today")

    assert masked == "signed with [TERM_1] today"


def test_blank_and_duplicate_entries_are_dropped():
    """A blank entry is skipped, and the terms after it still count."""
    masker = Masker(enabled=True, terms=" Ivan , , Petrov , Ivan ,")

    masked = masker.mask("Ivan and Petrov and Ivan")

    assert masked == "[TERM_1] and [TERM_2] and [TERM_1]"
    assert masker.mapping == {"[TERM_1]": "Ivan", "[TERM_2]": "Petrov"}


def test_no_terms_unless_given():
    masker = Masker(enabled=True)

    assert masker.mask("PIN XXXX") == "PIN XXXX"
    assert masker.mapping == {}


@pytest.mark.parametrize(
    "reserved",
    ["You", "you", "Other", "Speaker 1", "Speaker 12", "speaker 3", "EMAIL", "term"],
)
def test_reserved_terms_are_refused_with_a_warning(reserved, capsys):
    """Masking a transcript label would corrupt structure and protect nothing."""
    masker = Masker(enabled=True, terms=reserved)

    assert masker.mask(f"{reserved} said so") == f"{reserved} said so"
    assert reserved in capsys.readouterr().err


def test_reserved_term_does_not_disable_the_valid_ones(capsys):
    masker = Masker(enabled=True, terms="You, Ivan")

    masked = masker.mask("You heard Ivan")

    assert masked == "You heard [TERM_1]"
    assert "You" in capsys.readouterr().err


def test_terms_do_not_cut_into_an_email():
    """Email is consumed first, so a term matching part of it finds nothing left."""
    masker = Masker(enabled=True, terms="example")

    masked = masker.mask("write to ivan@example.com")

    assert masked == "write to [EMAIL_1]"
    assert masker.unmask(masked) == "write to ivan@example.com"


def test_labels_are_counted_independently():
    masker = Masker(enabled=True, terms="Ivan")

    masked = masker.mask(f"Ivan wrote from {_EMAIL} and Ivan called {_PHONE}")

    assert masked == "[TERM_1] wrote from [EMAIL_1] and [TERM_1] called [PHONE_1]"


def test_disabled_masker_ignores_terms():
    masker = Masker(enabled=False, terms="Ivan")

    assert masker.mask("Ivan promised to send it") == "Ivan promised to send it"
    assert masker.mapping == {}


def test_disabled_masker_does_not_warn_about_reserved_terms(capsys):
    """A stale term list must not nag while the feature is off."""
    Masker(enabled=False, terms="You")

    assert capsys.readouterr().err == ""


def test_no_terms_configured():
    masker = Masker(enabled=True, terms="")

    assert masker.mask("Ivan promised to send it") == "Ivan promised to send it"
