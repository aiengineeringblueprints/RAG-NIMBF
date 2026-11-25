"""Paired comparison of two runs by lineage (#65)."""

from __future__ import annotations

import json

import pytest

from benchmark.lineage.compare import compare_runs
from benchmark.lineage.report import write_comparison
from benchmark.lineage.experiment import scorer_identity
from benchmark.lineage.runs import RunRecord, load_run_record, scorer_fingerprint

SCORER = {"eval_critic_llm": "judge-a", "ragas_enabled": True}


def _record(
    ref, scores, *, scorer=SCORER, corpus="c1", system=None, extra=None, questions=None,
    invalid=(), run_errors=None,
):
    """scores: {sample_id: (answer_correctness, total_seconds)}"""
    samples = [
        {
            "sample_id": sid,
            "question_id": (questions or {}).get(sid, f"question-{sid}"),
            "answer_valid": sid not in invalid,
            "category": "answerable" if i % 2 == 0 else "unanswerable",
            "metrics": {
                "answer_correctness": ac,
                "total_seconds": secs,
                **((extra or {}).get(sid, {})),
            },
        }
        for i, (sid, (ac, secs)) in enumerate(scores.items())
    ]
    return RunRecord(
        ref=ref,
        config_name="cfg",
        lineage={
            "system": system or {"system_id": "acme", "status": "captured", "version": ref},
            "dataset": {"name": "ds", "corpus_fingerprint": corpus},
            "scorer": scorer,
        },
        samples=samples,
        run_errors=run_errors or {},
    )


BASE = {"q1": (0.5, 2.0), "q2": (0.5, 2.0), "q3": (0.5, 2.0), "q4": (0.5, 2.0)}
CAND = {"q1": (0.9, 1.0), "q2": (0.7, 1.0), "q3": (0.9, 1.0), "q4": (0.7, 1.0)}


def test_matching_question_sets_are_paired_with_deltas_and_ci():
    result = compare_runs(_record("base", BASE), _record("cand", CAND))

    assert result["status"] == "paired"
    assert result["reasons"] == []
    ac = result["metrics"]["answer_correctness"]
    assert ac["baseline"] == pytest.approx(0.5)
    assert ac["candidate"] == pytest.approx(0.8)
    assert ac["delta"] == pytest.approx(0.3)
    assert ac["direction"] == "higher_is_better"
    assert ac["n"] == 4
    low, high = ac["ci95"]
    assert 0.19 < low <= 0.3 <= high < 0.41
    secs = result["metrics"]["total_seconds"]
    assert secs["direction"] == "lower_is_better"
    assert secs["delta"] == pytest.approx(-1.0)


def test_per_question_and_category_breakdown():
    result = compare_runs(_record("base", BASE), _record("cand", CAND))

    q1 = next(q for q in result["questions"] if q["sample_id"] == "q1")
    assert q1["deltas"]["answer_correctness"] == pytest.approx(0.4)
    categories = result["categories"]
    assert categories["answerable"]["answer_correctness"]["delta"] == pytest.approx(0.4)
    assert categories["unanswerable"]["answer_correctness"]["delta"] == pytest.approx(0.2)


def test_missing_metric_is_counted_never_zero():
    base = _record("base", BASE, extra={"q1": {"faithfulness": 0.8}, "q2": {"faithfulness": 0.6}})
    cand = _record("cand", CAND, extra={"q1": {"faithfulness": 0.9}, "q2": {"faithfulness": None}})

    faith = compare_runs(base, cand)["metrics"]["faithfulness"]

    assert faith["n"] == 1
    assert faith["delta"] == pytest.approx(0.1)
    assert faith["missing_baseline"] == 2
    assert faith["missing_candidate"] == 3


def test_partial_overlap_is_partially_paired():
    cand = {k: v for k, v in CAND.items() if k != "q4"} | {"q9": (1.0, 1.0)}

    result = compare_runs(_record("base", BASE), _record("cand", cand))

    assert result["status"] == "partially_paired"
    assert result["pairing"] == {"paired": 3, "only_baseline": 1, "only_candidate": 1}
    assert result["metrics"]["answer_correctness"]["n"] == 3


def test_disjoint_question_sets_are_not_comparable():
    other = {"x1": (1.0, 1.0), "x2": (1.0, 1.0)}

    result = compare_runs(_record("base", BASE), _record("cand", other))

    assert result["status"] == "not_comparable"
    assert any("no shared samples" in r for r in result["reasons"])
    assert result["metrics"] == {}


def test_different_scorer_is_not_comparable():
    cand = _record("cand", CAND, scorer={**SCORER, "eval_critic_llm": "judge-b"})

    result = compare_runs(_record("base", BASE), cand)

    assert result["status"] == "not_comparable"
    assert any("scorer" in r for r in result["reasons"])


def test_corpus_mismatch_is_not_comparable():
    result = compare_runs(_record("base", BASE), _record("cand", CAND, corpus="c2"))

    assert result["status"] == "not_comparable"
    assert any("corpus" in r for r in result["reasons"])


def test_candidate_pointing_at_another_baseline_is_flagged():
    cand = _record("cand", CAND)
    cand.lineage["experiment"] = {"baseline_run_id": "some-other-run"}

    result = compare_runs(_record("base", BASE), cand)

    assert any("baseline_run_id some-other-run" in w for w in result["warnings"])


def test_markdown_shows_signed_deltas(tmp_path):
    _, md_path = write_comparison(
        compare_runs(_record("base", BASE), _record("cand", CAND)), tmp_path
    )

    assert "+0.3000" in md_path.read_text()


def test_missing_report_path_is_reported_as_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run_record(str(tmp_path / "nope.json#cfg"))


def test_unknown_lineage_is_unverified_not_paired():
    old = RunRecord(ref="old", config_name=None, lineage={}, samples=_record("b", BASE).samples)

    result = compare_runs(old, _record("cand", CAND))

    assert result["status"] == "unverified"
    assert result["baseline"]["system"]["status"] == "not_available"
    assert result["baseline"]["scorer_fingerprint"] is None
    assert any("scorer" in r and "unknown" in r for r in result["reasons"])
    assert result["metrics"]["answer_correctness"]["n"] == 4


def test_unknown_corpus_on_one_side_is_unverified():
    result = compare_runs(_record("base", BASE, corpus=None), _record("cand", CAND))

    assert result["status"] == "unverified"
    assert any("corpus" in r for r in result["reasons"])


def test_repeated_question_is_one_bootstrap_cluster():
    same_question = {"q1": "Q-A", "q2": "Q-A", "q3": "Q-B", "q4": "Q-C"}

    result = compare_runs(
        _record("base", BASE, questions=same_question),
        _record("cand", CAND, questions=same_question),
    )

    ac = result["metrics"]["answer_correctness"]
    assert ac["n"] == 4
    assert ac["clusters"] == 3


def test_error_counts_are_reported_per_side():
    base = _record("base", BASE, run_errors={"evaluation_error": "critic timeout"})
    cand = _record("cand", CAND, invalid={"q2", "q3"})

    errors = compare_runs(base, cand)["errors"]

    assert errors["baseline"] == {"invalid_answers": 0, "evaluation_error": "critic timeout"}
    assert errors["candidate"] == {"invalid_answers": 2, "evaluation_error": None}


def test_candidate_without_declared_baseline_is_flagged():
    cand = _record("cand", CAND)
    cand.lineage["experiment"] = {"candidate_id": "k8"}

    result = compare_runs(_record("base", BASE), cand)

    assert any("does not declare baseline_run_id" in w for w in result["warnings"])


def test_write_comparison_emits_json_and_markdown(tmp_path):
    result = compare_runs(_record("base", BASE), _record("cand", CAND))

    json_path, md_path = write_comparison(result, tmp_path / "cmp")

    assert json.loads(json_path.read_text())["status"] == "paired"
    markdown = md_path.read_text()
    assert "answer_correctness" in markdown
    assert "paired" in markdown
    assert "acme" in markdown


def test_old_report_without_lineage_loads_with_derived_ids(tmp_path):
    sample = {
        "question": "Q?", "ground_truth": "A", "answer": "A", "contexts": [],
        "ttft_seconds": 0.1, "total_seconds": 1.0, "token_count": 3,
        "tokens_per_second": 3.0, "gpu_usage": None, "ragas_scores": {},
        "custom_scores": {},
    }
    report = tmp_path / "old.json"
    report.write_text(json.dumps({
        "dataset_name": "legacy",
        "results": [{"config_name": "c", "per_sample": [sample]}],
    }))

    record = load_run_record(str(report))

    assert record.lineage["dataset"]["name"] == "legacy"
    assert len(record.samples[0]["sample_id"]) == 16
    assert record.samples[0]["metrics"]["total_seconds"] == 1.0


def test_cli_compares_two_reports_end_to_end(tmp_path):
    from benchmark.lineage.__main__ import main as cli

    def report(name, answers):
        path = tmp_path / f"{name}.json"
        samples = [
            {"question": q, "ground_truth": "A", "answer": "A", "contexts": [],
             "total_seconds": secs, "ragas_scores": {"answer_correctness": ac},
             "custom_scores": {}}
            for q, (ac, secs) in answers.items()
        ]
        path.write_text(json.dumps({
            "dataset_name": "ds",
            "results": [{"config_name": name, "per_sample": samples,
                         "lineage": {"scorer": SCORER, "dataset": {"name": "ds"}}}],
        }))
        return str(path)

    base = report("base", BASE)
    cand = report("cand", CAND)

    assert cli(["compare", "--baseline", base, "--candidate", cand, "--out", str(tmp_path / "o")]) == 0
    written = json.loads((tmp_path / "o" / "comparison.json").read_text())
    assert written["status"] == "paired"
    assert written["metrics"]["answer_correctness"]["delta"] == pytest.approx(0.3)


def test_scorer_identity_covers_critic_limits_and_implementation():
    from types import SimpleNamespace

    scorer = scorer_identity(
        SimpleNamespace(eval_critic_max_tokens=512, eval_critic_thinking_control="none")
    )

    assert scorer["eval_critic_max_tokens"] == 512
    assert scorer["eval_critic_thinking_control"] == "none"
    implementation = scorer["implementation"]
    assert len(implementation["evaluator_source_sha256"]) == 64
    assert "ragas" in implementation["packages"]


def test_changed_evaluator_implementation_changes_fingerprint():
    old = {**SCORER, "implementation": {"evaluator_source_sha256": "a"}}
    new = {**SCORER, "implementation": {"evaluator_source_sha256": "b"}}

    assert scorer_fingerprint(old) != scorer_fingerprint(new)
    result = compare_runs(_record("base", BASE, scorer=old), _record("cand", CAND, scorer=new))
    assert result["status"] == "not_comparable"
