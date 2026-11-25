"""Dataset identity for experiment lineage (issue #65)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from benchmark.dataset import SAMPLING_SEED
from benchmark.lineage.dataset import (
    assign_sample_identities,
    assign_sample_ids,
    corpus_fingerprint,
    dataset_identity,
    question_set_hash,
    sample_category,
)


def _sample(question, ground_truth="gt", **metadata):
    return {"question": question, "ground_truth": ground_truth, "context": "", "metadata": metadata}


def _config(**overrides):
    values = dict(
        dataset_name="jsonl-shared",
        dataset_subset=None,
        dataset_split=None,
        dataset_sample_size=0,
        dataset_path=None,
        dataset_corpus_path=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_sample_ids_are_stable_across_loads():
    data = [_sample("What is X?"), _sample("What is Y?")]

    assert assign_sample_ids(data) == assign_sample_ids([dict(s) for s in data])


def test_sample_ids_do_not_depend_on_position():
    a, b = _sample("What is X?"), _sample("What is Y?")

    forward = assign_sample_ids([a, b])
    backward = assign_sample_ids([b, a])

    assert forward == list(reversed(backward))


def test_duplicate_questions_get_distinct_ids():
    """TechQA lists questions twice; each copy must stay addressable."""
    ids = assign_sample_ids([_sample("Q"), _sample("Q"), _sample("Q", "other")])

    assert len(set(ids)) == 3


def test_sample_ids_contain_no_question_text():
    (sid,) = assign_sample_ids([_sample("Secret customer question")])

    assert "Secret" not in sid and len(sid) == 16


def test_question_set_hash_is_order_sensitive_and_deterministic():
    ids = assign_sample_ids([_sample("A"), _sample("B")])

    assert question_set_hash(ids) == question_set_hash(list(ids))
    assert question_set_hash(ids) != question_set_hash(list(reversed(ids)))


def test_category_from_explicit_label_or_answerable_flag():
    assert sample_category(_sample("Q", category="billing")) == "billing"
    assert sample_category(_sample("Q", answerable=True)) == "answerable"
    assert sample_category(_sample("Q", answerable=False)) == "unanswerable"
    assert sample_category(_sample("Q")) is None


def test_corpus_fingerprint_ignores_document_order_but_not_content():
    docs = [{"context": "alpha", "metadata": {}}, {"context": "beta", "metadata": {}}]

    assert corpus_fingerprint(docs) == corpus_fingerprint(list(reversed(docs)))
    assert corpus_fingerprint(docs) != corpus_fingerprint(
        [{"context": "alpha", "metadata": {}}, {"context": "beta!", "metadata": {}}]
    )
    assert corpus_fingerprint(None) is None


def test_dataset_identity_records_sampling_and_hashes(tmp_path):
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"question": "Q", "ground_truth": "A"}) + "\n")
    data = [_sample("Q", "A")]
    corpus = [{"context": "doc", "metadata": {}}]

    identity = dataset_identity(
        _config(dataset_path=str(questions), dataset_sample_size=5), data, corpus
    )

    assert identity["name"] == "jsonl-shared"
    assert identity["sample_size"] == 5
    assert identity["adapter"] == "jsonl-shared"
    assert identity["sampling_method"] == "none"
    assert identity["sampling_seed"] is None
    assert identity["version"] == identity["source_sha256"]
    assert identity["num_samples"] == 1
    assert identity["question_set_hash"] == question_set_hash(assign_sample_ids(data))
    assert identity["corpus_fingerprint"] == corpus_fingerprint(corpus)
    assert identity["corpus_documents"] == 1
    assert len(identity["source_sha256"]) == 64
    assert "Q" not in json.dumps(identity)


def test_local_dataset_truncation_is_head_sampling():
    identity = dataset_identity(_config(dataset_sample_size=1), [_sample("Q", dataset_rows=3)])

    assert identity["sampling_method"] == "head"
    assert identity["sampling_seed"] is None


def test_huggingface_subsample_is_seeded_shuffle():
    identity = dataset_identity(
        _config(dataset_name="squad", dataset_sample_size=1), [_sample("Q", dataset_rows=10)]
    )

    assert identity["adapter"] == "squad"
    assert identity["source"] == "rajpurkar/squad"
    assert identity["sampling_method"] == "seeded_shuffle"
    assert identity["sampling_seed"] == SAMPLING_SEED


def test_sample_identities_share_question_id_across_references():
    first, second, other = assign_sample_identities(
        [_sample("Q", "gt-1"), _sample("Q", "gt-2", category="x"), _sample("R")]
    )

    assert first.sample_id != second.sample_id
    assert first.question_id == second.question_id != other.question_id
    assert second.category == "x"
    assert "Q" not in first.question_id


def test_sample_size_equal_to_source_is_not_a_shuffle():
    identity = dataset_identity(
        _config(dataset_name="squad", dataset_sample_size=1), [_sample("Q", dataset_rows=1)]
    )

    assert identity["sampling_method"] == "none"
    assert identity["sampling_seed"] is None


def test_unknown_source_size_does_not_claim_a_seed():
    identity = dataset_identity(
        _config(dataset_name="squad", dataset_sample_size=1), [_sample("Q")]
    )

    assert identity["sampling_method"] == "unknown"
    assert identity["sampling_seed"] is None


def test_default_resolved_split_is_recorded():
    identity = dataset_identity(
        _config(dataset_name="squad"), [_sample("Q", dataset_rows=1, dataset_split="validation")]
    )

    assert identity["split"] == "validation"


def test_local_loader_records_source_rows(tmp_path):
    from benchmark.dataset import load_benchmark_data

    path = tmp_path / "q.jsonl"
    path.write_text("\n".join(
        json.dumps({"question": f"Q{i}", "ground_truth": "A", "context": "c"}) for i in range(3)
    ))

    data = load_benchmark_data("jsonl", sample_size=3, dataset_path=str(path))

    assert [s["metadata"]["dataset_rows"] for s in data] == [3, 3, 3]
    assert dataset_identity(_config(dataset_name="jsonl", dataset_sample_size=3), data)[
        "sampling_method"
    ] == "none"
