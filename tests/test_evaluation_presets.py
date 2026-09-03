from __future__ import annotations

import pytest

import benchmark.evaluation as evaluation


class _FakeResult:
    def __init__(self, scores):
        self.scores = scores


def _patch_critic_stack(monkeypatch, scores, tmp_path=None):
    if tmp_path is not None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    monkeypatch.setattr(
        evaluation,
        "get_chat_model",
        lambda **kw: object(),
    )
    monkeypatch.setattr(
        evaluation, "get_embedding_model", lambda *a, **kw: object()
    )
    monkeypatch.setattr(
        evaluation, "wrap_for_ragas", lambda model: object()
    )
    monkeypatch.setattr(
        evaluation,
        "LangchainLLMWrapper",
        lambda model: object(),
    )
    monkeypatch.setattr(
        evaluation, "LangchainEmbeddingsWrapper", lambda model: object()
    )
    captured: dict = {}

    def fake_evaluate(*, dataset, metrics, llm, embeddings, run_config):
        captured["metrics"] = [type(m).__name__ if not isinstance(m, str) else m for m in metrics]
        names = []
        for m in metrics:
            name = getattr(m, "name", None)
            names.append(name if name else type(m).__name__)
        captured["names"] = names
        return _FakeResult(scores)

    monkeypatch.setattr(evaluation, "evaluate", fake_evaluate)
    return captured


def test_metric_preset_core_keeps_three_metrics(monkeypatch, tmp_path):
    captured = _patch_critic_stack(
        monkeypatch,
        [{"faithfulness": 0.9, "context_recall": 0.8, "semantic_similarity": 0.7}],
        tmp_path=tmp_path,
    )
    result = evaluation.evaluate_results(
        ["q"], ["gt"], ["a"], [["ctx"]],
        metric_preset="core",
    )
    assert result.error is None
    assert result.metric_means["faithfulness"] == 0.9
    assert len(captured["names"]) == 3


def test_metric_preset_full_enables_all_six(monkeypatch, tmp_path):
    captured = _patch_critic_stack(
        monkeypatch,
        [{
            "faithfulness": 0.9,
            "context_recall": 0.8,
            "semantic_similarity": 0.7,
            "answer_relevancy": 0.6,
            "answer_correctness": 0.5,
            "context_precision": 0.4,
        }],
        tmp_path=tmp_path,
    )
    result = evaluation.evaluate_results(
        ["q"], ["gt"], ["a"], [["ctx"]],
        metric_preset="full",
    )
    assert result.error is None
    assert result.metric_means["answer_correctness"] == 0.5
    assert result.metric_means["context_precision"] == 0.4
    assert result.metric_means["answer_relevancy"] == 0.6
    assert len(captured["names"]) == 6


def test_metric_preset_rejects_unknown(monkeypatch, tmp_path):
    _patch_critic_stack(monkeypatch, [], tmp_path=tmp_path)
    with pytest.raises(ValueError, match="Unknown RAGAS metric preset"):
        evaluation.evaluate_results(
            ["q"], ["gt"], ["a"], [["ctx"]],
            metric_preset="bogus",
        )
