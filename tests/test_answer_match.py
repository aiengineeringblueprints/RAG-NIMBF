"""Deterministic answer matching against labelled answer patterns."""

import pytest

from benchmark.answer_match import answer_matches


@pytest.mark.parametrize(
    "answer, patterns",
    [
        ("The lab is located in **Bergen**.", [["Bergen"]]),
        ("That is 2,925.00 NOK.", [["2925.00"]]),
        ("That is 2 925,00 NOK.", [["2925,00"]]),
        ("Up to 60\u00a0°C.", [["60 °C", "60 degrees"]]),
        ("Max 60 ℃ (NFKC).", [["60 °C"]]),
        ("It shipped with DHL; 96 units remain.", [["shipped"], ["96"]]),
        ("It is CANOPEN.", [["CANopen"]]),
        ("Minimum is −40 °C.", [["-40 °C"]]),
    ],
)
def test_matches(answer, patterns):
    assert answer_matches(answer, patterns) is True


@pytest.mark.parametrize(
    "answer, patterns",
    [
        ("Ingrid Solberg heads the lab.", [["Bergen"]]),
        # Every group must match.
        ("It shipped with DHL.", [["shipped"], ["96"]]),
        # Numbers must not match inside longer numbers.
        ("The extension is 14110.", [["4411"]]),
        ("It costs 1610 EUR.", [["610"]]),
        ("", [["anything"]]),
    ],
)
def test_rejects(answer, patterns):
    assert answer_matches(answer, patterns) is False


def test_regex_alternatives_match_normalized_answer():
    pattern = [[r"re:(?<!\w)96(?:\s+[\w-]+){0,3}\s+sensors"]]

    assert answer_matches("There are **96 Kestrel K7 sensors** in stock.", pattern)
    assert answer_matches("96 sensors", pattern)
    assert not answer_matches("196 sensors", pattern)
    assert not answer_matches("96 of the many listed K7 sensors", pattern)


def test_prenormalized_text_matches_like_raw_text():
    from benchmark.answer_match import answer_matches_normalized, normalize_answer

    haystack = normalize_answer("It costs **2,925.00 NOK** and ships via DHL.")

    assert answer_matches_normalized(haystack, [["2925.00"], ["dhl"]])
    assert not answer_matches_normalized(haystack, [["2925"], ["ups"]])


def test_empty_patterns_never_match():
    assert answer_matches("text", []) is False
