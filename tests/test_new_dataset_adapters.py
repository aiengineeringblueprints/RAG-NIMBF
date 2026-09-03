from __future__ import annotations

from benchmark.dataset_adapters import DatasetAdapter, get_adapter


def test_hotpotqa_adapter_registered_with_correct_shape():
    adapter = get_adapter("hotpotqa")
    assert adapter.hf_id == "hotpot_qa"
    assert adapter.requires_subset is True
    assert adapter.default_subset == "distractor"
    assert adapter.has_shared_corpus is True
    assert "supporting_facts" in adapter.metadata_keys

    row = {
        "question": "Who is bigger?",
        "answer": "Germany",
        "context": {
            "title": ["Germany", "France"],
            "sentences": [
                ["Germany is big."],
                ["France is smaller."],
            ],
        },
    }
    context = adapter.build_context(row)
    assert "Germany" in context
    assert "France is smaller." in context


def test_nq_open_adapter_handles_list_answers_and_empty_context():
    adapter = get_adapter("nq_open")
    assert adapter.hf_id == "google-research-datasets/nq_open"
    assert adapter.has_shared_corpus is False
    assert adapter.build_context({"question": "q"}) == ""
    assert (
        adapter.ground_truth_transform(["Paris", "paris france"]) == "Paris | paris france"
    )
    assert adapter.ground_truth_transform("Paris") == "Paris"


def test_default_subset_used_when_requires_subset_and_unset():
    adapter = DatasetAdapter(
        name="x",
        hf_id="some/dataset",
        question_key="question",
        ground_truth_key="answer",
        build_context=lambda row: "",
        requires_subset=True,
        default_subset="cfg",
    )
    assert adapter.default_subset == "cfg"
    plain = DatasetAdapter(
        name="y",
        hf_id="other",
        question_key="q",
        ground_truth_key="a",
        build_context=lambda row: "",
    )
    assert plain.default_subset is None
