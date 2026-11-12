"""LLM-judged answer correctness per question category and run variant, from stored runs."""

import gzip
import json

import pytest

from benchmark.reporting.answer_judging import judge_runs

SIMILARITY = "recursive_cs512_co64_emb_llm_concise_k10_temp0_enterprise_rag_mcp_managed-aaaaaaaaaa"
HYBRID = "recursive_cs512_co64_emb_llm_concise_k10_hybrid-a0.5_temp0_enterprise_rag_mcp_managed-aaaaaaaaaa"


def _sample(question, answer, match):
    return {"question": question, "answer": answer, "adapter_diagnostics": {"answer_match": match}}


def _write_run(run_dir, results):
    run_dir.mkdir()
    samples_file = "benchmark_20261111_000000.samples.json.gz"
    report = {
        "results": [{"config_name": name} for name, _ in results],
        "samples_file": samples_file,
    }
    (run_dir / "benchmark_20261111_000000.json").write_text(json.dumps(report))
    with gzip.open(run_dir / samples_file, "wt", encoding="utf-8") as fh:
        json.dump({"format": 1, "contexts": [], "results": [s for _, s in results]}, fh)


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "questions.jsonl"
    rows = [
        {"question": "c1", "ground_truth": "Yes", "metadata": {"question_type": "comparison_query"}},
        {"question": "c2", "ground_truth": "no", "metadata": {"question_type": "comparison_query"}},
        {
            "question": "n1",
            "ground_truth": "Insufficient information.",
            "metadata": {"question_type": "null_query", "expected_behavior": "decline"},
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def _verdicts(table):
    """Fake judge: verdict by answer text; records every call."""
    calls = []

    def judge(question, reference, answer):
        calls.append((question, reference, answer))
        return table[answer]

    judge.calls = calls
    return judge


def test_rows_per_variant_and_category(tmp_path, dataset):
    run_dir = tmp_path / "run1"
    _write_run(
        run_dir,
        [
            (SIMILARITY, [
                _sample("c1", "Both affirm it.", False),
                _sample("c2", "They agree.", False),
                _sample("n1", "Not in the context.", True),
            ]),
            (HYBRID, [
                _sample("c1", "Both affirm it.", False),
                _sample("c2", "The context lacks it.", False),
                _sample("n1", "TomTom is Dutch.", False),
            ]),
        ],
    )
    judge = _verdicts({
        "Both affirm it.": "correct",
        "They agree.": "incorrect",
        "Not in the context.": "declined",
        "The context lacks it.": "declined",
        "TomTom is Dutch.": "incorrect",
    })

    rows = judge_runs([run_dir], dataset, judge)
    by_key = {(r["variant"], r["category"]): r for r in rows}

    sim_comparison = by_key[("similarity k10", "comparison_query")]
    assert sim_comparison["n"] == 2
    assert sim_comparison["judge_correct"] == pytest.approx(0.5)
    assert sim_comparison["judge_incorrect"] == pytest.approx(0.5)
    assert sim_comparison["judge_declined"] == pytest.approx(0.0)
    assert sim_comparison["answer_match"] == pytest.approx(0.0)

    hyb_comparison = by_key[("hybrid-a0.5 k10", "comparison_query")]
    assert hyb_comparison["judge_correct"] == pytest.approx(0.5)
    assert hyb_comparison["judge_declined"] == pytest.approx(0.5)

    # A decline sample is correct exactly when the judge sees a decline.
    assert by_key[("similarity k10", "null_query")]["judge_correct"] == pytest.approx(1.0)
    hyb_null = by_key[("hybrid-a0.5 k10", "null_query")]
    assert hyb_null["judge_correct"] == pytest.approx(0.0)
    assert hyb_null["judge_incorrect"] == pytest.approx(1.0)

    sim_all = by_key[("similarity k10", "ALL")]
    assert sim_all["n"] == 3
    assert sim_all["judge_correct"] == pytest.approx(2 / 3)
    assert ("c1", "Yes", "Both affirm it.") in judge.calls


def test_identical_answers_are_judged_once_and_cached_across_calls(tmp_path, dataset):
    run_dir = tmp_path / "run1"
    same = [_sample("c1", "Both affirm it.", False)]
    _write_run(run_dir, [(SIMILARITY, same), (HYBRID, same)])
    cache = tmp_path / "verdicts.jsonl"

    first = _verdicts({"Both affirm it.": "correct"})
    judge_runs([run_dir], dataset, first, cache_path=cache)
    assert len(first.calls) == 1

    # A resumed run reads the cache instead of calling the critic again.
    second = _verdicts({"Both affirm it.": "incorrect"})
    rows = judge_runs([run_dir], dataset, second, cache_path=cache)
    assert second.calls == []
    assert all(r["judge_correct"] == pytest.approx(1.0) for r in rows)


def test_unparseable_verdicts_are_left_out_and_retried(tmp_path, dataset):
    run_dir = tmp_path / "run1"
    _write_run(run_dir, [(SIMILARITY, [_sample("c1", "Hmm.", False), _sample("c2", "No.", True)])])
    cache = tmp_path / "verdicts.jsonl"

    rows = judge_runs([run_dir], dataset, _verdicts({"Hmm.": None, "No.": "correct"}), cache_path=cache)
    row = next(r for r in rows if r["category"] == "comparison_query")
    assert row["n"] == 2
    assert row["judged"] == 1
    assert row["judge_correct"] == pytest.approx(1.0)
    assert row["answer_match"] == pytest.approx(0.5)

    retry = _verdicts({"Hmm.": "incorrect", "No.": "correct"})
    judge_runs([run_dir], dataset, retry, cache_path=cache)
    assert retry.calls == [("c1", "Yes", "Hmm.")]
