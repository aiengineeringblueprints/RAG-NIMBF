from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from rich.console import Console

from benchmark.dataset_adapters import get_adapter

console = Console()

RAGPERF_WIKIPEDIA_DATASET = "ragperf-wikipedia-nq"
RAGPERF_WIKIPEDIA_HF_ID = "wikimedia/wikipedia"
RAGPERF_WIKIPEDIA_CONFIG = "20231101.en"
RAGPERF_NQ_HF_ID = "sentence-transformers/natural-questions"


@dataclass(frozen=True)
class BenchmarkSample:
    """Canonical question record consumed by the benchmark runner."""

    question: str
    ground_truth: str | list[str] | None = None
    context: str | list[str] | None = None
    id: str | None = None
    relevant_context_ids: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_legacy_dict(self) -> dict[str, Any]:
        metadata = dict(self.metadata)
        if self.id is not None:
            metadata.setdefault("id", self.id)
        if self.relevant_context_ids is not None:
            metadata.setdefault("relevant_context_ids", self.relevant_context_ids)
        return {
            "id": self.id,
            "question": self.question,
            "ground_truth": _stringify_ground_truth(self.ground_truth),
            "context": _normalise_context(self.context),
            "metadata": metadata,
        }


@dataclass(frozen=True)
class CorpusDocument:
    """Canonical searchable document record used for corpus-based RAG tests."""

    text: str
    id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_legacy_dict(self) -> dict[str, Any]:
        metadata = dict(self.metadata)
        if self.id is not None:
            metadata.setdefault("id", self.id)
        return {
            "id": self.id,
            "context": self.text,
            "metadata": metadata,
        }


@dataclass(frozen=True)
class DatasetMapping:
    """Maps arbitrary source columns into the canonical benchmark schema."""

    question: str = "question"
    ground_truth: str | None = "ground_truth"
    context: str | None = "context"
    id: str | None = "id"
    relevant_context_ids: str | None = "relevant_context_ids"
    metadata: tuple[str, ...] = ()
    text: str = "text"

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "DatasetMapping":
        if not raw:
            return cls()
        metadata = raw.get("metadata", ())
        if isinstance(metadata, str):
            metadata = tuple(v.strip() for v in metadata.split(",") if v.strip())
        return cls(
            question=raw.get("question", cls.question),
            ground_truth=raw.get("ground_truth", cls.ground_truth),
            context=raw.get("context", cls.context),
            id=raw.get("id", cls.id),
            relevant_context_ids=raw.get(
                "relevant_context_ids", cls.relevant_context_ids
            ),
            metadata=tuple(metadata or ()),
            text=raw.get("text", cls.text),
        )


def parse_mapping(value: str | None) -> DatasetMapping:
    """Parse DATASET_MAPPING JSON into a :class:`DatasetMapping`."""
    if not value:
        return DatasetMapping()
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "DATASET_MAPPING must be a JSON object, for example "
            """{"question": "query", "ground_truth": "answer"}"""
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError("DATASET_MAPPING must be a JSON object.")
    return DatasetMapping.from_dict(raw)


def _stringify_ground_truth(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    return str(value)


def _normalise_context(value: Any) -> str | list[str]:
    if value is None:
        return ""
    if isinstance(value, list):
        return [str(v) for v in value]
    return str(value)


def _coerce_list(value: Any) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, tuple):
        return [str(v) for v in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except json.JSONDecodeError:
                pass
        return [v.strip() for v in stripped.split(",") if v.strip()]
    return [str(value)]


def _row_get(row: dict[str, Any], key: str | None, default: Any = None) -> Any:
    if key is None:
        return default
    return row.get(key, default)


def _sample_from_row(row: dict[str, Any], mapping: DatasetMapping) -> BenchmarkSample:
    question = _row_get(row, mapping.question)
    if question is None or str(question).strip() == "":
        raise ValueError(f"Dataset row is missing question field '{mapping.question}'.")
    metadata = {
        key: row.get(key)
        for key in mapping.metadata
        if key in row
    }
    return BenchmarkSample(
        id=str(_row_get(row, mapping.id)) if _row_get(row, mapping.id) is not None else None,
        question=str(question),
        ground_truth=_row_get(row, mapping.ground_truth),
        context=_row_get(row, mapping.context),
        relevant_context_ids=_coerce_list(_row_get(row, mapping.relevant_context_ids)),
        metadata=metadata,
    )


def _document_from_row(row: dict[str, Any], mapping: DatasetMapping) -> CorpusDocument:
    text = _row_get(row, mapping.text)
    if text is None and mapping.context:
        text = _row_get(row, mapping.context)
    if text is None or str(text).strip() == "":
        raise ValueError(f"Corpus row is missing text field '{mapping.text}'.")
    metadata = {
        key: row.get(key)
        for key in mapping.metadata
        if key in row
    }
    return CorpusDocument(
        id=str(_row_get(row, mapping.id)) if _row_get(row, mapping.id) is not None else None,
        text=str(text),
        metadata=metadata,
    )


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_no}.")
        rows.append(row)
    return rows


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _limit_rows(rows: list[Any], sample_size: int | None) -> list[Any]:
    if sample_size and sample_size < len(rows):
        return rows[:sample_size]
    return rows


def samples_from_records(
    records: Iterable[dict[str, Any]],
    mapping: DatasetMapping | dict[str, Any] | None = None,
    sample_size: int | None = None,
) -> list[dict[str, Any]]:
    """Normalise arbitrary in-memory rows into benchmark sample dicts."""
    effective_mapping = (
        DatasetMapping.from_dict(mapping) if isinstance(mapping, dict) else mapping
    ) or DatasetMapping()
    samples = [_sample_from_row(dict(row), effective_mapping) for row in records]
    return [sample.to_legacy_dict() for sample in _limit_rows(samples, sample_size)]


def corpus_from_records(
    records: Iterable[dict[str, Any]],
    mapping: DatasetMapping | dict[str, Any] | None = None,
    sample_size: int | None = None,
) -> list[dict[str, Any]]:
    """Normalise arbitrary in-memory rows into benchmark corpus dicts."""
    effective_mapping = (
        DatasetMapping.from_dict(mapping) if isinstance(mapping, dict) else mapping
    ) or DatasetMapping()
    docs = [_document_from_row(dict(row), effective_mapping) for row in records]
    return [doc.to_legacy_dict() for doc in _limit_rows(docs, sample_size)]


def load_jsonl_dataset(
    path: str | Path,
    mapping: DatasetMapping | dict[str, Any] | None = None,
    sample_size: int | None = None,
) -> list[dict[str, Any]]:
    return samples_from_records(_read_jsonl(path), mapping=mapping, sample_size=sample_size)


def load_csv_dataset(
    path: str | Path,
    mapping: DatasetMapping | dict[str, Any] | None = None,
    sample_size: int | None = None,
) -> list[dict[str, Any]]:
    return samples_from_records(_read_csv(path), mapping=mapping, sample_size=sample_size)


def load_corpus_and_questions_from_jsonl(
    corpus_path: str | Path,
    questions_path: str | Path,
    mapping: DatasetMapping | dict[str, Any] | None = None,
    sample_size: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    effective_mapping = (
        DatasetMapping.from_dict(mapping) if isinstance(mapping, dict) else mapping
    ) or DatasetMapping()
    corpus = corpus_from_records(_read_jsonl(corpus_path), mapping=effective_mapping)
    questions = samples_from_records(
        _read_jsonl(questions_path),
        mapping=effective_mapping,
        sample_size=sample_size,
    )
    return corpus, questions


def load_huggingface_dataset(
    hf_id: str,
    *,
    subset: str | None = None,
    split: str | None = None,
    mapping: DatasetMapping | dict[str, Any] | None = None,
    sample_size: int | None = None,
) -> list[dict[str, Any]]:
    """Load any Hugging Face dataset when a column mapping is supplied."""
    effective_mapping = (
        DatasetMapping.from_dict(mapping) if isinstance(mapping, dict) else mapping
    ) or DatasetMapping()
    kwargs: dict[str, Any] = {}
    if subset:
        kwargs["name"] = subset
    ds = load_dataset(hf_id, **kwargs)
    selected_split = split
    if selected_split is None or selected_split not in ds:
        selected_split = "test" if "test" in ds else list(ds.keys())[0]
    rows = ds[selected_split]
    if sample_size and sample_size < len(rows):
        rows = rows.shuffle(seed=42).select(range(sample_size))
    return samples_from_records(rows, mapping=effective_mapping)


def load_dataset_for_config(config: Any) -> tuple[list[dict], list[dict] | None]:
    """Load benchmark data from the configured source.

    Returns ``(questions, corpus)`` to match the existing runner shape.  Built-in
    datasets preserve their legacy adapter behavior; custom sources use
    ``DATASET_MAPPING`` to map arbitrary columns into the canonical schema.
    """
    source = getattr(config, "dataset_source", "builtin")
    mapping = parse_mapping(getattr(config, "dataset_mapping", None))
    sample_size = getattr(config, "dataset_sample_size", None)

    if source == "builtin":
        adapter = get_adapter(config.dataset_name)
        if adapter.has_shared_corpus:
            corpus, questions = load_corpus_and_questions(
                dataset_name=config.dataset_name,
                subset=config.dataset_subset or None,
                sample_size=sample_size,
            )
            return questions, corpus
        return (
            load_benchmark_data(
                dataset_name=config.dataset_name,
                subset=config.dataset_subset or None,
                sample_size=sample_size,
            ),
            None,
        )

    if source == "huggingface":
        return (
            load_huggingface_dataset(
                config.dataset_name,
                subset=config.dataset_subset or None,
                split=getattr(config, "dataset_split", None),
                mapping=mapping,
                sample_size=sample_size,
            ),
            None,
        )

    if source == "jsonl":
        return (
            load_jsonl_dataset(config.dataset_path, mapping=mapping, sample_size=sample_size),
            None,
        )

    if source == "csv":
        return (
            load_csv_dataset(config.dataset_path, mapping=mapping, sample_size=sample_size),
            None,
        )

    if source == "corpus_jsonl":
        corpus, questions = load_corpus_and_questions_from_jsonl(
            config.dataset_corpus_path,
            config.dataset_questions_path,
            mapping=mapping,
            sample_size=sample_size,
        )
        return questions, corpus

    raise ValueError(f"Unsupported dataset source: {source}")


def load_benchmark_data(
    dataset_name: str = "t2-ragbench",
    subset: str | None = None,
    sample_size: int = 50,
) -> list[dict]:
    adapter = get_adapter(dataset_name)

    label = subset or "default"
    console.print(f"[bold blue]Loading {adapter.hf_id} ({label})...[/bold blue]")

    kwargs: dict = {}
    if adapter.requires_subset and subset:
        kwargs["name"] = subset
    ds = load_dataset(adapter.hf_id, **kwargs)

    split = adapter.preferred_split
    if split not in ds:
        split = list(ds.keys())[0]
    data = ds[split]

    if sample_size and sample_size < len(data):
        data = data.shuffle(seed=42).select(range(sample_size))

    samples = []
    for row in data:
        gt_raw = row.get(adapter.ground_truth_key, "")
        if adapter.ground_truth_transform:
            gt = adapter.ground_truth_transform(gt_raw)
        else:
            gt = str(gt_raw)

        samples.append({
            "question": row[adapter.question_key],
            "ground_truth": gt,
            "context": adapter.build_context(row),
            "metadata": {
                k: row.get(k)
                for k in adapter.metadata_keys
                if k in row
            },
        })

    console.print(
        f"[green]Loaded {len(samples)} samples from {adapter.hf_id} "
        f"({split} split)[/green]"
    )
    return samples


def load_corpus_and_questions(
    dataset_name: str = "squad",
    subset: str | None = None,
    sample_size: int = 50,
) -> tuple[list[dict], list[dict]]:
    """Load data and split into a deduplicated corpus and per-question entries.

    Returns (corpus, questions) where:
      - corpus: list of {context, metadata} dicts — all unique contexts
      - questions: list of {question, ground_truth, context, metadata} dicts
    """
    if dataset_name == RAGPERF_WIKIPEDIA_DATASET:
        return _load_ragperf_wikipedia_nq(sample_size=sample_size)

    samples = load_benchmark_data(dataset_name, subset, sample_size)

    seen: dict[str, int] = {}  # context text → index in corpus
    corpus: list[dict] = []

    for sample in samples:
        ctx = sample["context"]
        if ctx not in seen:
            seen[ctx] = len(corpus)
            corpus.append({
                "context": ctx,
                "metadata": sample.get("metadata", {}),
            })

    console.print(
        f"[green]Deduplicated corpus: {len(corpus)} unique documents "
        f"for {len(samples)} questions[/green]"
    )
    return corpus, samples


def _load_ragperf_wikipedia_nq(sample_size: int) -> tuple[list[dict], list[dict]]:
    """Load RAGPerf-style Wikipedia corpus with Natural Questions QA pairs.

    RAGPerf indexes ``wikimedia/wikipedia`` and evaluates with
    ``sentence-transformers/natural-questions``. Natural Questions provides
    answers, but RAGPerf does not provide gold Wikipedia document or chunk IDs.
    """
    corpus_size = int(
        os.getenv("RAGPERF_WIKIPEDIA_CORPUS_SIZE", str(max(sample_size * 20, 1000)))
    )

    console.print(
        "[bold blue]Loading RAGPerf-style Wikipedia corpus "
        f"({RAGPERF_WIKIPEDIA_CONFIG}, {corpus_size} docs)...[/bold blue]"
    )
    wiki = load_dataset(RAGPERF_WIKIPEDIA_HF_ID, RAGPERF_WIKIPEDIA_CONFIG, split="train")
    if corpus_size and corpus_size < len(wiki):
        wiki = wiki.shuffle(seed=42).select(range(corpus_size))

    corpus = [
        {
            "context": str(row.get("text", "")),
            "metadata": {
                "id": row.get("id"),
                "title": row.get("title"),
                "source_dataset": RAGPERF_WIKIPEDIA_HF_ID,
                "source_config": RAGPERF_WIKIPEDIA_CONFIG,
            },
        }
        for row in wiki
        if row.get("text")
    ]

    console.print(
        "[bold blue]Loading Natural Questions QA pairs "
        f"({sample_size} questions)...[/bold blue]"
    )
    nq = load_dataset(RAGPERF_NQ_HF_ID, split="train")
    if sample_size and sample_size < len(nq):
        nq = nq.shuffle(seed=42).select(range(sample_size))

    questions = [
        {
            "question": row["query"],
            "ground_truth": str(row["answer"]),
            "context": "",
            "metadata": {
                "source_dataset": RAGPERF_NQ_HF_ID,
                "retrieval_ground_truth": "unavailable",
            },
        }
        for row in nq
    ]

    console.print(
        f"[green]Loaded {len(corpus)} Wikipedia documents and "
        f"{len(questions)} Natural Questions QA pairs[/green]"
    )
    return corpus, questions
