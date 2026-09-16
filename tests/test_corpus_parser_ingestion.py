"""Corpus ingestion through a parser adapter (OCR-07).

The ``corpus_parser`` knob routes corpus construction through a registered
DocumentParser: raw documents on disk are parsed to Markdown and the corpus
docs keep the exact shape of a directly loaded text corpus, so chunking,
indexing, and retrieval run unchanged downstream.
"""

from __future__ import annotations

import base64
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from benchmark.adapters import (
    AdapterCapabilities,
    AdapterGenerationResult,
    PreparedTarget,
    RetrievalResult,
    RetrievedChunk,
    register_rag_adapter,
)
from benchmark.dataset import (
    _load_local_corpus,
    load_corpus_and_questions,
    load_corpus_documents,
    load_corpus_via_parser,
)
from benchmark.orchestration.matrix import (
    ExperimentSpec,
    build_configs_from_spec,
)
from benchmark.orchestration.worker import ExperimentWorker, WorkerOptions
from benchmark.parsing import ParsedPage, ParseResult, register_parser_adapter


class StubParser:
    """Deterministic in-process parser: Markdown-wraps the page text."""

    name = "stub-corpus-parser"
    parser_version = "1.2.3"

    def parse(self, document: dict, config=None) -> ParseResult:
        pages = tuple(
            ParsedPage(
                page_number=page["page_number"],
                markdown=f"# parsed {document['document_id']}\n\n{page.get('text', '')}",
            )
            for page in document.get("pages", [])
        )
        return ParseResult(
            document_id=document["document_id"],
            pages=pages,
            parser_name=self.name,
            parser_version=self.parser_version,
        )


@pytest.fixture
def raw_docs(tmp_path: Path) -> Path:
    root = tmp_path / "raw_docs"
    root.mkdir()
    (root / "b_doc.txt").write_text("alpha content", encoding="utf-8")
    (root / "a_doc.md").write_text("markdown source", encoding="utf-8")
    return root


def test_load_corpus_documents_discovers_sorted_text_documents(raw_docs: Path):
    documents = load_corpus_documents(raw_docs)

    assert [doc["document_id"] for doc in documents] == ["a_doc", "b_doc"]
    assert documents[0]["pages"] == [
        {
            "page_number": 1,
            "image_base64": None,
            "image_media_type": "",
            "text": "markdown source",
        }
    ]
    assert documents[1]["pages"] == [
        {
            "page_number": 1,
            "image_base64": None,
            "image_media_type": "",
            "text": "alpha content",
        }
    ]
    assert documents[0]["metadata"]["source_path"] == "a_doc.md"


def test_load_corpus_documents_rejects_symlinks(raw_docs: Path, tmp_path: Path):
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    (raw_docs / "link.txt").symlink_to(outside)

    with pytest.raises(ValueError, match="Symbolic links"):
        load_corpus_documents(raw_docs)


def test_load_corpus_documents_supports_image_pages(tmp_path: Path):
    root = tmp_path / "raw_docs"
    root.mkdir()
    payload = base64.b64encode(b"fake-png-bytes").decode("ascii")
    (root / "scan.png").write_bytes(b"fake-png-bytes")

    documents = load_corpus_documents(root)

    page = documents[0]["pages"][0]
    assert page["image_base64"] == payload
    assert page["image_media_type"] == "image/png"
    assert documents[0]["metadata"]["source_name"] == "scan.png"


def test_load_corpus_documents_rejects_empty_directory(tmp_path: Path):
    root = tmp_path / "empty"
    root.mkdir()

    with pytest.raises(ValueError, match="No documents"):
        load_corpus_documents(root)


def test_load_corpus_via_parser_builds_markdown_corpus(raw_docs: Path):
    corpus = load_corpus_via_parser(raw_docs, StubParser())

    assert [doc["metadata"]["doc_id"] for doc in corpus] == ["a_doc", "b_doc"]
    assert corpus[0]["context"] == "# parsed a_doc\n\nmarkdown source"
    metadata = corpus[0]["metadata"]
    assert metadata["parser_name"] == "stub-corpus-parser"
    assert metadata["parser_version"] == "1.2.3"
    assert metadata["source_path"] == "a_doc.md"


def test_parsed_corpus_matches_direct_text_corpus_shape(tmp_path: Path):
    """Same on-disk docs: parsed corpus keeps the text-corpus contract."""
    root = tmp_path / "raw_docs"
    root.mkdir()
    (root / "only.md").write_text("body", encoding="utf-8")

    parsed = load_corpus_via_parser(root, StubParser())
    direct = _load_local_corpus(str(root))

    assert set(parsed[0]["metadata"]) >= {
        "doc_id",
        "source_id",
        "source_name",
        "source_path",
    }
    assert parsed[0]["metadata"]["doc_id"] == direct[0]["metadata"]["doc_id"]
    assert set(parsed[0]) == set(direct[0]) == {"context", "metadata"}


def test_load_corpus_and_questions_uses_parser_for_jsonl_shared(
    tmp_path: Path,
):
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {
                "question": "What is parsed?",
                "ground_truth": "markdown source",
                "context": "markdown source",
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    corpus_dir = tmp_path / "raw_docs"
    corpus_dir.mkdir()
    (corpus_dir / "a_doc.md").write_text("markdown source", encoding="utf-8")

    corpus, questions = load_corpus_and_questions(
        dataset_name="jsonl-shared",
        dataset_path=str(questions_path),
        corpus_path=str(corpus_dir),
        sample_size=5,
        corpus_parser=StubParser(),
    )

    assert corpus[0]["context"] == "# parsed a_doc\n\nmarkdown source"
    assert corpus[0]["metadata"]["parser_name"] == "stub-corpus-parser"
    assert len(questions) == 1


def test_load_corpus_and_questions_rejects_parser_for_plain_dataset(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="corpus_parser"):
        load_corpus_and_questions(
            dataset_name="squad",
            sample_size=1,
            corpus_parser=StubParser(),
        )


# --- RAG-level: parsed corpus flows through the full worker pipeline ---

CAPTURED: dict[str, Any] = {}

RAG_STUB_NAME = "stub-ingest-rag"
PARSER_STUB_NAME = "stub-ingest-parser"


class _CapturingAdapter:
    """Minimal managed adapter that records the corpus it was handed."""

    name = RAG_STUB_NAME

    def __init__(self, config: Any) -> None:
        pass

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            ingestion=True,
            retrieval=True,
            generation=True,
            token_usage=True,
            cleanup=True,
        )

    def prepare(self, config, data, corpus=None) -> PreparedTarget:
        CAPTURED["corpus"] = corpus
        CAPTURED["data"] = data
        return PreparedTarget(target_id="stub-target", metadata={"chunk_count": 1})

    def retrieve(self, target, sample, config) -> RetrievalResult:
        return RetrievalResult(
            chunks=(RetrievedChunk(text="parsed", rank=1),),
            total_seconds=0.01,
        )

    def generate(self, target, sample, config, retrieval=None):
        return AdapterGenerationResult(
            answer="stub answer",
            retrieval=retrieval or self.retrieve(target, sample, config),
            ttft_seconds=0.01,
            total_seconds=0.02,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )

    def cleanup(self, target, config) -> None:
        pass


register_rag_adapter(RAG_STUB_NAME, _CapturingAdapter)
register_parser_adapter(PARSER_STUB_NAME, lambda config: StubParser())


def test_worker_pipeline_runs_parsed_corpus_unchanged(tmp_path, monkeypatch):
    """Stub parser + tiny document set: the full worker pipeline runs and
    evaluates with the parsed Markdown corpus, with parser provenance
    pinned into the run metadata."""
    import mlflow

    import benchmark.orchestration.worker as worker_module

    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps(
            {
                "question": "What is parsed?",
                "ground_truth": "markdown source",
                "context": "markdown source",
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    corpus_dir = tmp_path / "raw_docs"
    corpus_dir.mkdir()
    (corpus_dir / "a_doc.md").write_text("markdown source", encoding="utf-8")

    spec = ExperimentSpec(
        name="corpus-parser-e2e",
        dataset={
            "name": "jsonl-shared",
            "path": str(questions),
            "corpus_path": str(corpus_dir),
            "sample_size": 1,
            "license": "apache-2.0",
        },
        settings={
            "rag_system_adapter": RAG_STUB_NAME,
            "corpus_parser": PARSER_STUB_NAME,
            "dataset_question_field": "question",
            "dataset_ground_truth_field": "ground_truth",
            "dataset_context_field": "context",
            "dataset_metadata_field": "metadata",
            "ragas_enabled": False,
            "custom_metrics_enabled": False,
        },
        matrix={},
    )

    def fake_log_run(result, reproducibility_dir=None, nested=None):
        CAPTURED["result"] = result

    @contextmanager
    def fake_start_run(*args, **kwargs):
        yield None

    monkeypatch.setattr(worker_module, "log_benchmark_run", fake_log_run)
    monkeypatch.setattr(
        worker_module, "log_aggregate_artifacts_to_mlflow", lambda *a, **k: None
    )
    monkeypatch.setattr(mlflow, "start_run", fake_start_run)

    configs = build_configs_from_spec(spec)
    worker = ExperimentWorker(
        configs,
        WorkerOptions(
            run_dir=tmp_path / "run1",
            experiment_name="corpus-parser-e2e",
            write_reports=False,
        ),
    )
    results = worker.run()

    assert len(results) == 1
    corpus = CAPTURED["corpus"]
    assert corpus[0]["context"] == "# parsed a_doc\n\nmarkdown source"
    assert corpus[0]["metadata"]["parser_name"] == StubParser.name
    assert corpus[0]["metadata"]["parser_version"] == "1.2.3"

    result = CAPTURED["result"]
    assert result.corpus_parser == PARSER_STUB_NAME
    assert result.dataset_license == "apache-2.0"

    progress = json.loads(
        ((tmp_path / "run1") / "progress.json").read_text(encoding="utf-8")
    )
    assert progress["configs"][configs[0].name]["status"] == "completed"
