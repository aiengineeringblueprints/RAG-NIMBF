"""Conversion of MultiHop-RAG (HF yixuantt/MultiHopRAG) into a jsonl-shared dataset."""

import importlib.util
import json
from pathlib import Path

from benchmark.answer_match import score_answer

ROOT = Path(__file__).resolve().parents[1] / "datasets" / "multihop_rag"

# Loaded by path: the top-level ``datasets`` name belongs to HuggingFace.
_spec = importlib.util.spec_from_file_location("mhr_prepare", ROOT / "prepare.py")
prepare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare)


def _article(n, title, source="The Verge"):
    return {
        "title": title,
        "author": "A. Writer",
        "source": source,
        "published_at": f"2023-10-0{n}T12:00:00+00:00",
        "category": "technology",
        "url": f"https://example.com/{n}",
        "body": f"Body of article {n}.",
    }


def _evidence(article):
    return {**{k: article[k] for k in ("title", "url", "source")}, "fact": "a fact"}


CORPUS = [_article(1, "FTX trial"), _article(2, "SBF timeline", "TechCrunch"), _article(3, "Pixel 8")]
QUERIES = [
    {
        "query": "Who faces trial, as reported by The Verge and TechCrunch?",
        "answer": "Sam Bankman-Fried",
        "question_type": "inference_query",
        "evidence_list": [_evidence(CORPUS[0]), _evidence(CORPUS[1]), _evidence(CORPUS[1])],
    },
    {
        "query": "What is the first letter of the CEO's name?",
        "answer": "Insufficient information.",
        "question_type": "null_query",
        "evidence_list": [],
    },
    {
        "query": "Did both articles report the same launch date?",
        "answer": "Yes",
        "question_type": "comparison_query",
        "evidence_list": [_evidence(CORPUS[2]), _evidence(CORPUS[0])],
    },
]


def test_documents_carry_metadata_the_queries_refer_to():
    documents, _ = prepare.build(CORPUS, QUERIES)

    assert sorted(documents) == ["MHR-001", "MHR-002", "MHR-003"]
    text = documents["MHR-002"]
    assert text.startswith("# SBF timeline")
    assert "TechCrunch" in text
    assert "2023-10-02" in text
    assert "Body of article 2." in text


def test_questions_resolve_evidence_to_unique_gold_doc_ids():
    _, questions = prepare.build(CORPUS, QUERIES, seed=None)
    by_type = {q["metadata"]["question_type"]: q for q in questions}

    inference = by_type["inference_query"]
    assert inference["question"] == QUERIES[0]["query"]
    assert inference["ground_truth"] == "Sam Bankman-Fried"
    assert inference["metadata"]["gold_doc_ids"] == ["MHR-001", "MHR-002"]
    assert inference["metadata"]["category"] == "inference_query"


def test_null_queries_expect_a_decline_without_gold():
    _, questions = prepare.build(CORPUS, QUERIES, seed=None)
    (null,) = [q for q in questions if q["metadata"]["question_type"] == "null_query"]

    assert null["metadata"]["gold_doc_ids"] == []
    assert null["metadata"]["expected_behavior"] == "decline"
    assert score_answer("The context does not contain this information.", null["metadata"])["answer_match"]
    assert not score_answer("It is S.", null["metadata"])["answer_match"]


def test_yes_no_answers_do_not_match_when_both_words_appear():
    _, questions = prepare.build(CORPUS, QUERIES, seed=None)
    (comparison,) = [q for q in questions if q["metadata"]["question_type"] == "comparison_query"]
    metadata, truth = comparison["metadata"], comparison["ground_truth"]

    assert score_answer("Yes, both report October 5.", metadata, truth)["answer_match"]
    assert not score_answer("No.", metadata, truth)["answer_match"]
    assert not score_answer("Yes or no cannot be said.", metadata, truth)["answer_match"]


def test_seeded_order_is_stable_and_mixes_rows():
    _, first = prepare.build(CORPUS, QUERIES, seed=7)
    _, second = prepare.build(CORPUS, QUERIES, seed=7)

    assert [q["question"] for q in first] == [q["question"] for q in second]
    assert sorted(q["question"] for q in first) == sorted(q["query"] for q in QUERIES)


def test_write_creates_jsonl_shared_layout(tmp_path):
    prepare.write(CORPUS, QUERIES, tmp_path)

    assert sorted(p.name for p in (tmp_path / "corpus").iterdir()) == ["MHR-001.md", "MHR-002.md", "MHR-003.md"]
    rows = [json.loads(line) for line in (tmp_path / "questions.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert {"question", "ground_truth", "context", "metadata"} <= set(rows[0])
