"""Fixed regression benchmark for iterative Blueprint evaluation (issue #64)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
import yaml

from benchmark.regression import compare_runs, freeze_question_set, main, sample_metrics, summarize_run


def _sample(question: str, *, answerable: bool = True, **metadata) -> dict:
    return {
        "question": question,
        "ground_truth": f"answer to {question}",
        "context": f"context for {question}",
        "metadata": {
            "id": question,
            "all_relevant_sentence_keys": ["0a"] if answerable else [],
            "answerable": answerable,
            # Stamped by the loader; must not leak into the frozen rows.
            "doc_id": "dataset-doc-0",
            "gold_doc_id": "dataset-doc-0",
            **metadata,
        },
    }


def test_freeze_splits_dev_prefix_and_disjoint_holdout():
    samples = [_sample(f"q{i}") for i in range(6)]
    # A holdout candidate that repeats a dev question must be skipped.
    samples.insert(3, _sample("q0"))

    frozen = freeze_question_set(samples, dev_size=3, holdout_size=2, source={"name": "toy"})

    assert [row["question"] for row in frozen.dev] == ["q0", "q1", "q2"]
    assert [row["question"] for row in frozen.holdout] == ["q3", "q4"]


def test_freeze_labels_categories_and_drops_loader_doc_ids():
    samples = [_sample("a"), _sample("u", answerable=False)]

    frozen = freeze_question_set(samples, dev_size=2, holdout_size=0, source={"name": "toy"})

    assert [row["metadata"]["category"] for row in frozen.dev] == ["answerable", "unanswerable"]
    for row in frozen.dev:
        assert "doc_id" not in row["metadata"]
        assert "gold_doc_id" not in row["metadata"]


def test_freeze_manifest_records_sizes_categories_and_content_hash():
    samples = [_sample("a"), _sample("u", answerable=False), _sample("h")]

    frozen = freeze_question_set(
        samples, dev_size=2, holdout_size=1, source={"name": "toy", "seed": 42}
    )
    again = freeze_question_set(
        samples, dev_size=2, holdout_size=1, source={"name": "toy", "seed": 42}
    )
    changed = freeze_question_set(
        [_sample("a"), _sample("u2", answerable=False), _sample("h")],
        dev_size=2,
        holdout_size=1,
        source={"name": "toy", "seed": 42},
    )

    dev = frozen.manifest["splits"]["dev"]
    assert dev["size"] == 2
    assert dev["categories"] == {"answerable": 1, "unanswerable": 1}
    assert frozen.manifest["splits"]["holdout"]["size"] == 1
    assert frozen.manifest["source"] == {"name": "toy", "seed": 42}
    assert dev["sha256"] == again.manifest["splits"]["dev"]["sha256"]
    assert dev["sha256"] != changed.manifest["splits"]["dev"]["sha256"]


def test_freeze_refuses_when_source_is_too_small():
    with pytest.raises(ValueError, match="holdout"):
        freeze_question_set([_sample("a"), _sample("b")], dev_size=2, holdout_size=1, source={})


REFUSAL = "The provided context does not contain information about this."


def _result(question: str, answer: str = "Restart the service.", **fields) -> dict:
    """One per-sample record as written to ``*.samples.json.gz``."""
    record = {
        "question": question,
        "answer": answer,
        "answer_valid": True,
        "ground_truth_doc_ids": ["dataset-doc-0"],
        "retrieved_doc_ids": ["dataset-doc-0"],
        "ragas_scores": {},
        "custom_scores": {"bert_score_f1": 0.8, "hit@1": 1.0},
        "total_seconds": 2.0,
        "ttft_seconds": 0.5,
        "total_tokens": 0,
        "estimated_cost_usd": None,
    }
    record.update(fields)
    return record


def _rows(categories: dict[str, str]) -> list[dict]:
    return [{"question": q, "metadata": {"category": c}} for q, c in categories.items()]


def test_summary_reports_dimensions_from_available_scores():
    results = [
        _result("a1", ragas_scores={"answer_correctness": 0.6, "faithfulness": 1.0},
                custom_scores={"bert_score_f1": 0.7, "hit@1": 1.0}, total_seconds=1.0),
        _result("a2", ragas_scores={"answer_correctness": 0.8, "faithfulness": 0.5},
                custom_scores={"bert_score_f1": 0.9, "hit@1": 0.0}, total_seconds=3.0),
        _result("u1", answer=REFUSAL, total_seconds=2.0),
        _result("u2", answer="Install fix pack 3.", total_seconds=10.0),
    ]
    question_set = _rows({"a1": "answerable", "a2": "answerable",
                          "u1": "unanswerable", "u2": "unanswerable"})

    summary = summarize_run(results, question_set)
    dims = summary["dimensions"]

    assert summary["n"] == 4
    assert dims["correctness"]["metrics"]["answer_correctness"] == pytest.approx(0.7)
    assert dims["faithfulness"]["metrics"]["faithfulness"] == pytest.approx(0.75)
    assert dims["abstention"]["metrics"]["abstention_rate"] == pytest.approx(0.5)
    assert dims["abstention"]["metrics"]["false_refusal_rate"] == pytest.approx(0.0)
    assert dims["latency"]["metrics"]["total_seconds_median"] == pytest.approx(2.5)
    assert dims["latency"]["metrics"]["total_seconds_p95"] == pytest.approx(10.0)


def test_summary_scores_retrieval_on_answerable_gold_labels_only():
    results = [
        _result("a1", custom_scores={"hit@1": 1.0}),
        _result("a2", custom_scores={"hit@1": 0.0}),
        _result("u1", custom_scores={"hit@1": 0.0}),
    ]
    question_set = _rows({"a1": "answerable", "a2": "answerable", "u1": "unanswerable"})

    retrieval = summarize_run(results, question_set)["dimensions"]["retrieval"]

    assert retrieval["status"] == "ok"
    assert retrieval["metrics"]["hit@1"] == pytest.approx(0.5)


def test_summary_marks_dimensions_unavailable_instead_of_guessing():
    # RAGAS off, no gold IDs, no category labels, no pricing.
    results = [_result("q", ground_truth_doc_ids=[], ragas_scores={})]

    dims = summarize_run(results)["dimensions"]

    for name in ("faithfulness", "retrieval", "abstention", "cost"):
        assert dims[name]["status"] == "unavailable", name
        assert dims[name]["reason"]
    # BERTScore still measures correctness when RAGAS is off.
    assert dims["correctness"]["status"] == "ok"


def test_summary_keeps_duplicate_questions_apart_by_reference_answer():
    # RAGBench TechQA lists every question twice (same documents, two
    # reference answers); the labels can differ, so the question text alone
    # cannot identify a sample.
    question_set = [
        {"question": "q", "ground_truth": "gt-a", "metadata": {"category": "answerable"}},
        {"question": "q", "ground_truth": "gt-u", "metadata": {"category": "unanswerable"}},
    ]
    results = [  # run order differs from the frozen order
        _result("q", answer=REFUSAL, ground_truth="gt-u"),
        _result("q", answer="Restart the service.", ground_truth="gt-a"),
    ]

    abstention = summarize_run(results, question_set)["dimensions"]["abstention"]["metrics"]

    assert abstention["abstention_rate"] == pytest.approx(1.0)
    assert abstention["false_refusal_rate"] == pytest.approx(0.0)


CRITERIA = {
    "primary": ["answer_correctness", "bert_score_f1"],
    "min_improvement": 0.02,
    "guardrails": {
        "faithfulness": {"max_worsening": 0.02},
        "total_seconds": {"max_worsening_pct": 25},
    },
}


def _run(bert: list[float], seconds: float = 2.0, prefix: str = "q") -> list[dict]:
    return [
        _result(f"{prefix}{i}", custom_scores={"bert_score_f1": value}, total_seconds=seconds)
        for i, value in enumerate(bert)
    ]


def test_compare_pairs_questions_regardless_of_order():
    baseline = _run([0.5, 0.6, 0.7])
    candidate = list(reversed(_run([0.6, 0.7, 0.8])))

    comparison = compare_runs(baseline, candidate, criteria=CRITERIA)

    bert = comparison["metrics"]["bert_score_f1"]
    assert bert["n"] == 3
    assert bert["delta"] == pytest.approx(0.1)
    assert bert["ci95"][0] == pytest.approx(0.1)
    assert bert["ci95"][1] == pytest.approx(0.1)



def test_compare_scores_retrieval_only_on_questions_that_have_it():
    # Unanswerable questions carry no retrieval metrics; they must not break
    # the pairing of the answerable ones.
    question_set = _rows({"q0": "answerable", "q1": "unanswerable"})
    baseline = [_result("q0", custom_scores={"hit@1": 0.0}), _result("q1", answer=REFUSAL)]
    candidate = [_result("q0", custom_scores={"hit@1": 1.0}), _result("q1", answer=REFUSAL)]

    comparison = compare_runs(baseline, candidate, criteria=CRITERIA, question_set=question_set)

    assert comparison["metrics"]["hit@1"]["n"] == 1
    assert comparison["metrics"]["hit@1"]["delta"] == pytest.approx(1.0)

def test_compare_refuses_runs_on_different_question_sets():
    with pytest.raises(ValueError, match="question"):
        compare_runs(_run([0.5, 0.5]), _run([0.5, 0.5], prefix="other"), criteria=CRITERIA)


def test_verdict_accepts_significant_improvement_within_trade_offs():
    # RAGAS off: primary falls back to bert_score_f1.
    comparison = compare_runs(_run([0.5] * 20), _run([0.6] * 20, seconds=2.2), criteria=CRITERIA)

    assert comparison["verdict"]["decision"] == "accept"
    assert comparison["verdict"]["primary"] == "bert_score_f1"


def test_verdict_rejects_improvement_that_breaks_a_trade_off():
    # +0.1 correctness, but 50 % slower than the 25 % latency budget allows.
    comparison = compare_runs(_run([0.5] * 20), _run([0.6] * 20, seconds=3.0), criteria=CRITERIA)

    assert comparison["verdict"]["decision"] == "reject"
    assert any("total_seconds" in reason for reason in comparison["verdict"]["reasons"])


def test_verdict_is_inconclusive_below_the_minimum_improvement():
    comparison = compare_runs(_run([0.5] * 20), _run([0.51] * 20), criteria=CRITERIA)

    assert comparison["verdict"]["decision"] == "inconclusive"


def test_verdict_is_inconclusive_when_the_gain_is_noise():
    noisy = [0.9 if i % 2 else 0.2 for i in range(20)]
    # Paired differences alternate +0.7 / -0.6: mean gain 0.05 clears the
    # minimum, but the paired CI spans zero.
    shuffled = [0.3 if i % 2 else 0.9 for i in range(20)]

    comparison = compare_runs(_run(noisy), _run(shuffled), criteria=CRITERIA)

    assert comparison["verdict"]["decision"] == "inconclusive"


def test_binary_metrics_get_an_exact_mcnemar_test():
    baseline = [_result(f"q{i}", custom_scores={"hit@1": 0.0}) for i in range(10)]
    candidate = [_result(f"q{i}", custom_scores={"hit@1": 1.0}) for i in range(10)]

    hit = compare_runs(baseline, candidate, criteria=CRITERIA)["metrics"]["hit@1"]

    assert hit["mcnemar"] == {"only_candidate": 10, "only_baseline": 0, "p_value": pytest.approx(2 / 1024)}


def test_compare_refuses_runs_not_on_the_frozen_question_set():
    with pytest.raises(ValueError, match="frozen"):
        compare_runs(_run([0.5]), _run([0.6]), criteria=CRITERIA, question_set=_rows({"other": "answerable"}))


# --------------------------------------------------------------------------
# CLI: python -m benchmark.regression freeze | verify | compare
# --------------------------------------------------------------------------


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_samples(path, results):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"format": 1, "contexts": [], "results": [results]}, fh)
    return str(path)


def test_cli_freeze_writes_splits_that_verify_until_tampered(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [_sample(f"q{i}", answerable=i != 1) for i in range(5)])
    out = tmp_path / "v1"

    assert main(["freeze", "--dataset", "jsonl", "--dataset-path", str(source),
                 "--dev-size", "3", "--holdout-size", "2", "--out", str(out)]) == 0

    dev = [json.loads(line) for line in (out / "dev.jsonl").read_text().splitlines()]
    assert [row["question"] for row in dev] == ["q0", "q1", "q2"]
    assert dev[1]["metadata"]["category"] == "unanswerable"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["splits"]["holdout"]["size"] == 2
    assert main(["verify", "--question-set", str(out)]) == 0

    (out / "dev.jsonl").write_text((out / "dev.jsonl").read_text().replace("q2", "q9"))
    assert main(["verify", "--question-set", str(out)]) == 1



def test_cli_freeze_rebuilds_a_committed_manifest_only_when_content_matches(tmp_path):
    # Only manifest.json is committed; the split files are regenerated from
    # the source and must reproduce the recorded hashes.
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [_sample(f"q{i}") for i in range(5)])
    out = tmp_path / "v1"
    args = ["freeze", "--dataset", "jsonl", "--dataset-path", str(source),
            "--dev-size", "3", "--holdout-size", "2", "--out", str(out)]
    assert main(args) == 0
    manifest = (out / "manifest.json").read_text()
    for split in ("dev", "holdout"):
        (out / f"{split}.jsonl").unlink()

    assert main(args) == 0
    assert main(["verify", "--question-set", str(out)]) == 0
    assert (out / "manifest.json").read_text() == manifest

    for split in ("dev", "holdout"):
        (out / f"{split}.jsonl").unlink()
    _write_jsonl(source, [_sample(f"changed{i}") for i in range(5)])
    assert main(args) == 1
    assert not (out / "dev.jsonl").exists()

def test_cli_compare_reports_verdicts_against_baseline_and_best(tmp_path):
    question_set = tmp_path / "v1"
    question_set.mkdir()
    _write_jsonl(question_set / "dev.jsonl", _rows({f"q{i}": "answerable" for i in range(20)}))
    criteria = tmp_path / "criteria.yaml"
    criteria.write_text(yaml.safe_dump(CRITERIA))
    baseline = _write_samples(tmp_path / "base.samples.json.gz", _run([0.5] * 20))
    best = _write_samples(tmp_path / "best.samples.json.gz", _run([0.7] * 20))
    candidate = _write_samples(tmp_path / "cand.samples.json.gz", _run([0.6] * 20))
    out = tmp_path / "report"

    assert main(["compare", "--question-set", str(question_set), "--criteria", str(criteria),
                 "--baseline", baseline, "--best", best, "--candidate", candidate,
                 "--out", str(out)]) == 0

    report = json.loads((out / "comparison.json").read_text())
    assert report["comparisons"]["baseline"]["verdict"]["decision"] == "accept"
    assert report["comparisons"]["best"]["verdict"]["decision"] == "reject"
    assert set(report["summaries"]) == {"baseline", "best", "candidate"}
    markdown = (out / "report.md").read_text()
    assert "accept" in markdown and "reject" in markdown


REGRESSION_DIR = Path(__file__).resolve().parents[1] / "experiments" / "blueprint-regression"


def _manifest(name):
    return yaml.safe_load((REGRESSION_DIR / name).read_text())


def _flat(manifest):
    return {**manifest["dataset"], **manifest["settings"], **manifest["matrix"]}


def test_candidate_manifest_changes_one_component_of_the_baseline():
    baseline, candidate = _flat(_manifest("baseline.yaml")), _flat(_manifest("candidate-hybrid.yaml"))

    changed = {key for key in baseline.keys() | candidate.keys() if baseline.get(key) != candidate.get(key)}

    assert changed == {"retrieval_strategy", "retrieval_hybrid_alpha"}


def test_criteria_only_name_metrics_the_comparison_produces():
    criteria = yaml.safe_load((REGRESSION_DIR / "criteria.yaml").read_text())
    full = _result("q", answer="x", ragas_scores={"answer_correctness": 1.0, "faithfulness": 1.0},
                   custom_scores={"bert_score_f1": 1.0, "hit@5": 1.0}, total_seconds=1.0,
                   estimated_cost_usd=0.1, total_tokens=5)
    known = set(sample_metrics(full, "answerable")) | set(sample_metrics(full, "unanswerable"))

    assert set(criteria["primary"]) <= known
    assert set(criteria["guardrails"]) <= known


# --------------------------------------------------------------------------
# Review fixes
# --------------------------------------------------------------------------


def test_guardrail_breach_rejects_even_without_a_primary_metric():
    baseline = [_result(f"q{i}", custom_scores={}, total_seconds=1.0) for i in range(3)]
    candidate = [_result(f"q{i}", custom_scores={}, total_seconds=2.0) for i in range(3)]

    verdict = compare_runs(baseline, candidate, criteria=CRITERIA)["verdict"]

    assert verdict["decision"] == "reject"


def test_summary_reports_false_refusals_without_unanswerable_questions():
    question_set = _rows({"q0": "answerable", "q1": "answerable"})
    results = [_result("q0", answer=REFUSAL), _result("q1")]

    abstention = summarize_run(results, question_set)["dimensions"]["abstention"]

    assert abstention["status"] == "ok"
    assert abstention["metrics"]["false_refusal_rate"] == pytest.approx(0.5)
    assert "abstention_rate" not in abstention["metrics"]


def test_bootstrap_resamples_repeated_questions_together():
    # TechQA repeats each question with a second reference answer. Copies of
    # one question are not independent evidence: two copies of the same
    # (question, delta) must give the same CI as the question alone.
    deltas = [0.0, 0.1, 0.4, -0.2, 0.3]
    baseline = _run([0.5] * 5)
    candidate = _run([0.5 + d for d in deltas])

    def doubled(results):
        return results + [{**r, "ground_truth": "second reference"} for r in results]

    single = compare_runs(baseline, candidate, criteria=CRITERIA)["metrics"]["bert_score_f1"]
    double = compare_runs(doubled(baseline), doubled(candidate), criteria=CRITERIA)["metrics"]["bert_score_f1"]

    assert double["n"] == 10
    assert double["ci95"] == pytest.approx(single["ci95"])


def test_verdict_flags_a_fallback_primary_metric():
    baseline, candidate = _run([0.5, 0.6, 0.7]), _run([0.6, 0.7, 0.8])

    verdict = compare_runs(baseline, candidate, criteria=CRITERIA)["verdict"]

    assert verdict["primary"] == "bert_score_f1"
    assert verdict["primary_fallback"] is True


def test_cli_verify_explains_missing_split_files(tmp_path, capsys):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"splits": {"dev": {"sha256": "x"}, "holdout": {"sha256": "y"}}})
    )

    assert main(["verify", "--question-set", str(tmp_path)]) == 1
    assert "freeze" in capsys.readouterr().out
