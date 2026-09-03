from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmark.llm_judge import _parse_score, judge_answers


class FakeJudge:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_parse_score_handles_json_and_fallback():
    assert _parse_score('{"score": 7}') == 7.0
    assert _parse_score('noise "score": 8.5 more') == 8.5
    assert _parse_score("no score here") is None


def test_judge_scores_and_order_bias(monkeypatch):
    # rubric order A -> 8, rubric order B (reversed) -> 6 => score 0.7, bias 0.2
    fake = FakeJudge(
        [SimpleNamespace(content='{"score": 8}'), SimpleNamespace(content='{"score": 6}')]
    )
    monkeypatch.setattr("benchmark.llm_judge.get_chat_model", lambda **kw: fake)

    result = judge_answers(
        ["q?"], ["gt"], ["answer"], critic_llm_model="ollama:test"
    )

    assert result.error is None
    assert result.metric_means["llm_judge_score"] == pytest.approx(0.7)
    assert result.metric_means["llm_judge_order_bias"] == pytest.approx(0.2)
    assert result.per_sample[0]["llm_judge_score"] == pytest.approx(0.7)
    assert result.per_sample[0]["llm_judge_order_bias"] == pytest.approx(0.2)
    # the two prompts must present criteria in opposite orders
    assert fake.prompts[0] != fake.prompts[1]


def test_judge_clamps_scores_to_unit_interval(monkeypatch):
    fake = FakeJudge(
        [SimpleNamespace(content='{"score": 42}'), SimpleNamespace(content='{"score": -3}')]
    )
    monkeypatch.setattr("benchmark.llm_judge.get_chat_model", lambda **kw: fake)

    result = judge_answers(["q?"], ["gt"], ["a"], critic_llm_model="ollama:t")

    assert result.metric_means["llm_judge_score"] == 0.5  # (1.0 + 0.0) / 2


def test_judge_unparseable_both_rounds_records_none(monkeypatch):
    fake = FakeJudge(
        [SimpleNamespace(content="garbage"), SimpleNamespace(content="still garbage")]
    )
    monkeypatch.setattr("benchmark.llm_judge.get_chat_model", lambda **kw: fake)

    result = judge_answers(["q?"], ["gt"], ["a"], critic_llm_model="ollama:t")

    assert result.metric_means == {}
    assert result.per_sample[0]["llm_judge_score"] is None
    assert result.error is not None


def test_judge_model_init_failure_returns_error_result(monkeypatch):
    def boom(**kw):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("benchmark.llm_judge.get_chat_model", boom)

    result = judge_answers(["q?"], ["gt"], ["a"], critic_llm_model="ollama:t")

    assert result.error is not None
    assert "connection refused" in result.error
    assert result.metric_means == {}
