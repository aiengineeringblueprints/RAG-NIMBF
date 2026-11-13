"""Tests for the shared canonical document ID resolution (benchmark.doc_ids)."""

import pytest

from benchmark.doc_ids import canonical_doc_id, resolve_dataset_doc_id


@pytest.mark.parametrize(
    "value",
    [
        "dataset-doc-0",
        "Report.v1",
        "release.v1",
        "report.1",
        "manual",
        "/docs/manual.pdf",
        "/a/manual.pdf",
        "/b/manual.pdf",
        "docs/manual.pdf",
        "C:\\docs\\manual.pdf",
        "manual.pdf",
    ],
)
def test_canonical_doc_id_is_idempotent(value):
    """canonical_doc_id(canonical_doc_id(x)) == canonical_doc_id(x): gold
    stamping, indexing and retrieval may apply it repeatedly in sequence."""
    once = canonical_doc_id(value)
    assert canonical_doc_id(once) == once


def test_canonical_doc_id_handles_missing_values():
    assert canonical_doc_id(None) is None
    assert canonical_doc_id("") is None
    assert canonical_doc_id("   ") is None


def test_path_sources_keep_full_path_no_collisions():
    """Same-named files in different directories must stay distinct."""
    assert canonical_doc_id("/a/manual.pdf") == "/a/manual.pdf"
    assert canonical_doc_id("/b/manual.pdf") == "/b/manual.pdf"
    assert canonical_doc_id("/a/manual.pdf") != canonical_doc_id("/b/manual.pdf")


def test_dotted_versioned_names_keep_identity():
    """Only known document suffixes are stripped; versioned dotted names are
    document identities and must not collapse."""
    assert canonical_doc_id("report.1") == "report.1"
    assert canonical_doc_id("report.2") == "report.2"
    assert canonical_doc_id("report.1") != canonical_doc_id("report.2")
    assert canonical_doc_id("release.v1.pdf") == "release.v1"
    assert canonical_doc_id("release.v1") == "release.v1"


def test_known_document_suffix_is_stripped():
    assert canonical_doc_id("manual.pdf") == "manual"
    assert canonical_doc_id("notes.md") == "notes"
    assert canonical_doc_id("scan.PNG") == "scan"


def test_resolve_dataset_doc_id_precedence_and_fallback():
    assert resolve_dataset_doc_id({"source": "A", "doc_id": "B"}, 0) == "A"
    assert resolve_dataset_doc_id({"source": "", "doc_id": "B"}, 1) == "B"
    assert resolve_dataset_doc_id({"source": None}, 2) == "dataset-doc-2"
    assert resolve_dataset_doc_id({}, 3) == "dataset-doc-3"
    assert resolve_dataset_doc_id(None, 4) == "dataset-doc-4"
    assert resolve_dataset_doc_id({}) is None


def test_distinct_documents_never_merge_through_full_pipeline():
    """End-to-end guard: two distinct documents keep distinct gold IDs."""
    from benchmark.dataset import assign_gold_doc_ids

    samples = assign_gold_doc_ids(
        [
            {"context": "one", "metadata": {"source": "report.1"}},
            {"context": "two", "metadata": {"source": "report.2"}},
        ]
    )
    golds = {s["metadata"]["gold_doc_id"] for s in samples}
    assert len(golds) == 2
