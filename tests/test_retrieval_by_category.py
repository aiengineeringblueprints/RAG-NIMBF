"""Gold-doc retrieval and answer match per question category and run variant."""

import gzip
import json

import pytest

from benchmark.reporting.retrieval_by_category import summarize_by_category

SIMILARITY = "recursive_cs512_co64_emb_llm_concise_k10_temp0_enterprise_rag_mcp_managed-aaaaaaaaaa"
HYBRID = "recursive_cs512_co64_emb_llm_concise_k10_hybrid-a0.5_temp0_enterprise_rag_mcp_managed-aaaaaaaaaa"


def _sample(question, retrieved, gold, match):
    return {
        "question": question,
        "retrieved_doc_ids": retrieved,
        "ground_truth_doc_ids": gold,
        "adapter_diagnostics": {"answer_match": match},
    }


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
        {"question": "q1", "metadata": {"question_type": "factoid"}},
        {"question": "q2", "metadata": {"question_type": "factoid"}},
        {"question": "q3", "metadata": {"question_type": "multi-hop"}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_rows_per_variant_and_category(tmp_path, dataset):
    run_dir = tmp_path / "run1"
    _write_run(
        run_dir,
        [
            (
                SIMILARITY,
                [
                    _sample("q1", ["A", "B", "C"], ["A"], True),
                    _sample("q2", ["C", "C", "B"], ["B"], False),
                    # Second hop only at rank 4; duplicate chunks of one doc.
                    _sample("q3", ["D", "D", "X", "E"], ["D", "E"], False),
                ],
            ),
            (
                HYBRID,
                [
                    _sample("q1", ["A"], ["A"], True),
                    _sample("q2", ["B"], ["B"], True),
                    _sample("q3", ["E", "D"], ["D", "E"], True),
                ],
            ),
        ],
    )

    rows = summarize_by_category([run_dir], dataset, ks=[1, 3])
    by_key = {(r["variant"], r["category"]): r for r in rows}

    sim_factoid = by_key[("similarity k10", "factoid")]
    assert sim_factoid["n"] == 2
    assert sim_factoid["recall@1"] == pytest.approx(0.5)
    assert sim_factoid["recall@3"] == pytest.approx(1.0)
    assert sim_factoid["answer_match"] == pytest.approx(0.5)

    sim_multihop = by_key[("similarity k10", "multi-hop")]
    assert sim_multihop["recall@3"] == pytest.approx(0.5)
    assert sim_multihop["complete@3"] == pytest.approx(0.0)

    hyb_multihop = by_key[("hybrid-a0.5 k10", "multi-hop")]
    assert hyb_multihop["recall@1"] == pytest.approx(0.5)
    assert hyb_multihop["complete@1"] == pytest.approx(0.0)
    assert hyb_multihop["complete@3"] == pytest.approx(1.0)

    sim_all = by_key[("similarity k10", "ALL")]
    assert sim_all["n"] == 3
    assert sim_all["recall@3"] == pytest.approx((1 + 1 + 0.5) / 3)


def test_gold_falls_back_to_dataset_metadata(tmp_path):
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(
        json.dumps({"question": "q1", "metadata": {"question_type": "multi-hop", "gold_doc_ids": ["A", "B"]}})
    )
    run_dir = tmp_path / "run1"
    _write_run(run_dir, [(SIMILARITY, [_sample("q1", ["A", "Z"], [], True)])])

    (row,) = [r for r in summarize_by_category([run_dir], dataset, ks=[2]) if r["category"] != "ALL"]

    assert row["recall@2"] == pytest.approx(0.5)


def test_unknown_question_is_rejected(tmp_path, dataset):
    run_dir = tmp_path / "run1"
    _write_run(run_dir, [(SIMILARITY, [_sample("not in dataset", ["A"], ["A"], True)])])

    with pytest.raises(ValueError, match="not in dataset"):
        summarize_by_category([run_dir], dataset, ks=[1])
