"""Calibration report: direct RAGAS vs. MLflow scorers (issue #46)."""

from __future__ import annotations

import pytest

from benchmark.reporting.judge_calibration import pair_stats


class TestPairStats:
    def test_worked_example(self):
        # Sample 3 failed in direct RAGAS: excluded from the paired stats,
        # counted in the direct failure rate.
        stats = pair_stats(direct=[0.0, 0.5, 1.0, None], judge=[0.0, 1.0, 1.0, 0.5])

        assert stats.n_samples == 4
        assert stats.n_paired == 3
        assert stats.direct_mean == pytest.approx(0.5)
        assert stats.judge_mean == pytest.approx(2 / 3)
        assert stats.mean_diff == pytest.approx(1 / 6)  # judge - direct
        assert stats.mean_abs_diff == pytest.approx(1 / 6)
        assert stats.pearson == pytest.approx(0.8660, abs=1e-4)
        assert stats.spearman == pytest.approx(0.8660, abs=1e-4)  # tied ranks averaged
        assert stats.direct_failure_rate == pytest.approx(0.25)
        assert stats.judge_failure_rate == 0.0

    def test_identical_scores_agree_perfectly(self):
        stats = pair_stats(direct=[0.2, 0.4, 0.9], judge=[0.2, 0.4, 0.9])

        assert stats.mean_diff == 0.0
        assert stats.mean_abs_diff == 0.0
        assert stats.pearson == pytest.approx(1.0)
        assert stats.spearman == pytest.approx(1.0)

    def test_correlation_is_undefined_for_constant_scores(self):
        # A judge that says "yes" to everything carries no ranking signal.
        stats = pair_stats(direct=[0.2, 0.4, 0.9], judge=[1.0, 1.0, 1.0])

        assert stats.mean_diff == pytest.approx(0.5)
        assert stats.pearson is None
        assert stats.spearman is None

    def test_no_paired_samples(self):
        stats = pair_stats(direct=[None, 0.5], judge=[1.0, None])

        assert stats.n_paired == 0
        assert stats.mean_diff is None
        assert stats.pearson is None
        assert stats.direct_failure_rate == 0.5
        assert stats.judge_failure_rate == 0.5

    def test_lists_must_cover_the_same_samples(self):
        with pytest.raises(ValueError):
            pair_stats(direct=[0.1], judge=[0.1, 0.2])


# ---------------------------------------------------------------------------
# Per-sample judge scores read back from a scored run's traces
# ---------------------------------------------------------------------------

from benchmark.mlflow_scorers import score_run_traces  # noqa: E402
from benchmark.reporting.judge_calibration import judge_scores  # noqa: E402
from test_mlflow_scorers import ENDPOINT, _finished_config_run  # noqa: E402
from test_mlflow_tracing import (  # noqa: E402,F401 - pytest fixtures
    SAMPLES,
    generator_calls,
    tracking_store,
)

# SAMPLES[0] asks about Paris, SAMPLES[1] about Gustave Eiffel.
PARIS = SAMPLES[0]["ground_truth"]


def _stub_factories():
    """Deterministic stand-ins emitting assessments like the real scorers."""
    from mlflow.entities import Feedback
    from mlflow.genai.scorers import scorer

    @scorer(name="correctness")
    def yes_no(expectations) -> str:  # built-in judges answer yes/no
        return "yes" if expectations["expected_response"] == PARIS else "no"

    @scorer(name="Faithfulness")  # RAGAS wrapper: metric class name
    def ragas_like(expectations) -> float:
        return 0.25 if expectations["expected_response"] == PARIS else 0.75

    @scorer(name="retrieval_relevance")
    def chunk_judge(expectations) -> list[Feedback]:
        # Per-chunk verdicts plus the per-span precision the report uses.
        precision = 0.5 if expectations["expected_response"] == PARIS else 1.0
        return [
            Feedback(name="retrieval_relevance/precision", value=precision),
            Feedback(name="retrieval_relevance", value="yes"),
        ]

    @scorer(name="answer_correctness")
    def failing(expectations) -> Feedback:
        if expectations["expected_response"] == PARIS:
            return Feedback(name="answer_correctness", error=ValueError("bad claim counts"))
        return Feedback(name="answer_correctness", value=0.8)

    return {
        "correctness": lambda e: yes_no,
        "ragas_faithfulness": lambda e: ragas_like,
        "retrieval_relevance": lambda e: chunk_judge,
        "answer_correctness": lambda e: failing,
    }


def test_judge_scores_are_read_per_sample_from_the_traces(
    tracking_store, generator_calls, monkeypatch
):
    run_id = _finished_config_run(monkeypatch)
    factories = _stub_factories()
    score_run_traces(run_id, tuple(factories), ENDPOINT, factories=factories)

    scores = judge_scores(run_id, tuple(factories), n_samples=len(SAMPLES))

    # Indexed by benchmark.sample_index; yes/no -> 1/0; errors -> None.
    assert scores == {
        "correctness": [1.0, 0.0],
        "ragas_faithfulness": [0.25, 0.75],
        "retrieval_relevance": [0.5, 1.0],
        "answer_correctness": [None, 0.8],
    }


def test_samples_without_a_trace_or_assessment_count_as_missing(
    tracking_store, generator_calls, monkeypatch
):
    run_id = _finished_config_run(monkeypatch)
    factories = _stub_factories()
    score_run_traces(
        run_id, ("correctness",), ENDPOINT, factories={"correctness": factories["correctness"]}
    )

    scores = judge_scores(run_id, ("correctness", "ragas_faithfulness"), n_samples=3)

    assert scores == {
        "correctness": [1.0, 0.0, None],
        "ragas_faithfulness": [None, None, None],
    }


# ---------------------------------------------------------------------------
# Calibration report: config JSON (direct RAGAS) + scored MLflow run
# ---------------------------------------------------------------------------

import json  # noqa: E402

import mlflow  # noqa: E402

from benchmark.reporting.judge_calibration import build_report, main  # noqa: E402


def _config_json(tmp_path, per_sample_ragas):
    """Worker-style config JSON with direct per-sample RAGAS scores."""
    path = tmp_path / "configs" / "cfg.json"
    path.parent.mkdir()
    path.write_text(json.dumps({
        "dataset_name": "toy",
        "results": [{
            "config_name": "cfg",
            "per_sample": [{"ragas_scores": scores} for scores in per_sample_ragas],
        }],
    }))
    return path


def _scored_run(monkeypatch):
    run_id = _finished_config_run(monkeypatch)
    factories = _stub_factories()
    score_run_traces(run_id, tuple(factories), ENDPOINT, factories=factories)
    mlflow.MlflowClient().log_metric(run_id, "stage_ragas_eval_seconds", 40.0)
    return run_id


def test_report_pairs_direct_ragas_with_the_judges(
    tracking_store, generator_calls, monkeypatch, tmp_path
):
    run_id = _scored_run(monkeypatch)
    # Direct RAGAS failed on sample 2's answer_correctness (no key).
    results = _config_json(tmp_path, [
        {"faithfulness": 0.25, "context_precision": 1.0, "answer_correctness": 0.6},
        {"faithfulness": 0.5, "context_precision": 0.5},
    ])

    report = build_report(results, run_id)

    pairs = {(p.direct, p.judge): s for p, s in report.pairs}
    # Only pairs where both evaluators ran; correctness pairs with
    # answer_correctness, retrieval_relevance with context_precision.
    assert set(pairs) == {
        ("faithfulness", "ragas_faithfulness"),
        ("context_precision", "retrieval_relevance"),
        ("answer_correctness", "answer_correctness"),
        ("answer_correctness", "correctness"),
    }
    faith = pairs[("faithfulness", "ragas_faithfulness")]
    assert faith.n_paired == 2
    assert faith.mean_diff == pytest.approx(0.125)  # judge 0.25/0.75 vs. 0.25/0.5
    # Both sides failed on different samples -> nothing paired.
    ac = pairs[("answer_correctness", "answer_correctness")]
    assert ac.n_paired == 0
    assert ac.direct_failure_rate == 0.5 and ac.judge_failure_rate == 0.5
    assert report.direct_seconds == 40.0
    assert report.judge_seconds is not None and report.judge_seconds >= 0.0
    assert report.n_samples == 2


def test_a_configured_scorer_that_always_fails_is_still_reported(
    tracking_store, generator_calls, monkeypatch, tmp_path
):
    # A judge that errors on every sample (e.g. critic timeouts) is a
    # calibration finding, not a pair to hide.
    from mlflow.entities import Feedback
    from mlflow.genai.scorers import scorer

    @scorer(name="ContextPrecision")
    def timing_out(outputs) -> Feedback:
        return Feedback(name="ContextPrecision", error=TimeoutError("read timed out"))

    run_id = _finished_config_run(monkeypatch)
    score_run_traces(
        run_id,
        ("ragas_context_precision",),
        ENDPOINT,
        factories={"ragas_context_precision": lambda e: timing_out},
    )
    results = _config_json(tmp_path, [{"context_precision": 1.0}, {"context_precision": 0.5}])

    report = build_report(results, run_id)

    ((pair, stats),) = report.pairs
    assert (pair.direct, pair.judge) == ("context_precision", "ragas_context_precision")
    assert stats.judge_failure_rate == 1.0
    assert stats.n_paired == 0


def test_cli_writes_markdown_and_json(
    tracking_store, generator_calls, monkeypatch, tmp_path
):
    run_id = _scored_run(monkeypatch)
    results = _config_json(tmp_path, [{"faithfulness": 0.25}, {"faithfulness": 0.5}])
    out = tmp_path / "calibration.md"

    main([str(results), "--run-id", run_id, "--out", str(out)])

    markdown = out.read_text()
    assert "| faithfulness | ragas_faithfulness |" in markdown
    assert "stage_ragas_eval_seconds" in markdown
    payload = json.loads(out.with_suffix(".json").read_text())
    assert payload["run_id"] == run_id
    (pair,) = payload["pairs"]
    assert pair["direct"] == "faithfulness" and pair["n_paired"] == 2


@pytest.mark.parametrize(
    ("out", "expected"),
    [
        ("calib.md", ("calib.md", "calib.json")),
        ("calib.json", ("calib.md", "calib.json")),
        ("pass.2", ("pass.2.md", "pass.2.json")),  # dot in the stem is kept
        ("calib", ("calib.md", "calib.json")),
    ],
)
def test_output_paths_keep_dotted_stems(out, expected):
    from pathlib import Path

    from benchmark.reporting.judge_calibration import _output_paths

    md, js = _output_paths(Path(out))

    assert (md.name, js.name) == expected


def test_sample_index_skips_root_spans_without_the_attribute():
    # A stray root span (no sample index) before the sample's root span
    # must not hide the sample.
    from types import SimpleNamespace

    from benchmark.reporting.judge_calibration import _sample_index

    def span(parent, attrs):
        return SimpleNamespace(parent_id=parent, attributes=attrs)

    trace = SimpleNamespace(data=SimpleNamespace(spans=[
        span(None, {}),
        span("x", {"benchmark.sample_index": 9}),  # child spans are ignored
        span(None, {"benchmark.sample_index": 3}),
    ]))

    assert _sample_index(trace) == 3
    assert _sample_index(SimpleNamespace(data=SimpleNamespace(spans=[span(None, {})]))) is None
