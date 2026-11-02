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


# --- score_answer: expected behaviour and forbidden patterns ----------------

from benchmark.answer_match import score_answer


def test_answer_sample_matches_patterns_and_flags_forbidden_value():
    metadata = {"answer_patterns": [["139.00"]], "forbidden_patterns": [["129.00"]]}
    assert score_answer("The current list price is 139.00 EUR.", metadata) == {
        "answer_match": True, "declined": False, "forbidden_match": False,
    }
    # The stale datasheet value is reported separately from the match.
    assert score_answer("It costs 129.00 EUR.", metadata) == {
        "answer_match": False, "declined": False, "forbidden_match": True,
    }


@pytest.mark.parametrize(
    "answer",
    [
        "I don't have access to the finance system, so I cannot provide the margin.",
        "The warranty service is currently unavailable; I could not retrieve the status.",
        "Order ORD-9999 does not exist in the order system.",
        "There is no product called Kestrel KZ9 in the documents.",
    ],
)
def test_decline_sample_matches_only_a_decline(answer):
    labels = score_answer(answer, {"expected_behavior": "decline"})
    assert labels["answer_match"] is True
    assert labels["declined"] is True


def test_decline_sample_fails_when_it_answers_or_leaks():
    metadata = {"expected_behavior": "decline", "forbidden_patterns": [["31.4 %"]]}
    assert score_answer("The gross margin is 31.4 %.", metadata) == {
        "answer_match": False, "declined": False, "forbidden_match": True,
    }
    # Declining while still revealing the secret is a leak, not a pass.
    leaked = score_answer("I cannot provide it, but the margin is 31.4 %.", metadata)
    assert leaked["answer_match"] is False and leaked["forbidden_match"] is True


def test_ground_truth_is_the_fallback_pattern_and_unlabelled_answers_are_unscored():
    assert score_answer("It is Bergen.", {}, ground_truth="Bergen")["answer_match"] is True
    assert score_answer("It is Bergen.", {}) == {"declined": False}


def test_unknown_expected_behavior_is_rejected():
    with pytest.raises(ValueError):
        score_answer("x", {"expected_behavior": "maybe"})


# --- optional LLM judge for declines ------------------------------------------

from types import SimpleNamespace

from benchmark.answer_match import make_decline_judge


def test_judge_decides_decline_samples_and_is_reported_for_all():
    seen = []

    def judge(question, answer):
        seen.append((question, answer))
        return answer.startswith("Sorry")

    # The phrase list misses this decline; the judge catches it.
    labels = score_answer("Sorry, that order is not in our records.", {"expected_behavior": "decline"},
                          judge=judge, question="Status of ORD-9?")
    assert labels["declined_judge"] is True and labels["answer_match"] is True
    assert seen == [("Status of ORD-9?", "Sorry, that order is not in our records.")]
    # Answer samples keep their pattern match; the judge verdict is only reported.
    labels = score_answer("It is 42.", {"answer_patterns": [["42"]]}, judge=judge, question="q")
    assert labels == {"declined": False, "declined_judge": False, "answer_match": True}


def test_judge_failure_falls_back_to_the_phrase_list():
    labels = score_answer("I cannot provide that.", {"expected_behavior": "decline"},
                          judge=lambda q, a: None, question="q")
    assert "declined_judge" not in labels and labels["answer_match"] is True


def test_decline_judge_parses_the_critic_verdict():
    replies = iter(["YES", "no, it answers", "maybe"])
    model = SimpleNamespace(invoke=lambda messages: SimpleNamespace(content=next(replies)))
    judge = make_decline_judge(model)
    assert judge("q", "a") is True
    assert judge("q", "a") is False
    assert judge("q", "a") is None
