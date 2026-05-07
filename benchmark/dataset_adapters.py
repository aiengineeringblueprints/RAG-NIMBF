"""Dataset adapter registry — maps short names to HuggingFace dataset loaders.

Each adapter describes how to load a specific HuggingFace dataset and normalise
its columns into the standard ``{question, ground_truth, context, metadata}``
format consumed by the rest of the framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class DatasetAdapter:
    """Describes how to load and normalise one HuggingFace dataset."""

    name: str  # short registry key, e.g. "t2-ragbench"
    hf_id: str  # HuggingFace dataset ID, e.g. "G4KMU/t2-ragbench"
    question_key: str  # column name for question text
    ground_truth_key: str  # column name for ground truth answer
    build_context: Callable[[dict], str]  # row -> context string
    preferred_split: str = "test"  # which split to prefer
    metadata_keys: tuple[str, ...] = ()  # extra columns to include in metadata
    requires_subset: bool = False  # True if the HF dataset needs a config/subset
    ground_truth_transform: Callable[[Any], str] | None = None  # for complex fields
    has_shared_corpus: bool = False  # True if contexts should be deduplicated into a shared corpus


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, DatasetAdapter] = {}


def register(adapter: DatasetAdapter) -> None:
    REGISTRY[adapter.name] = adapter


def get_adapter(name: str) -> DatasetAdapter:
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {', '.join(sorted(REGISTRY))}"
        )
    return REGISTRY[name]


# ---------------------------------------------------------------------------
# Built-in adapters
# ---------------------------------------------------------------------------


def _t2_ragbench_context(row: dict) -> str:
    parts = []
    if row.get("pre_text"):
        parts.append(row["pre_text"])
    if row.get("table"):
        parts.append(row["table"])
    if row.get("post_text"):
        parts.append(row["post_text"])
    if not parts and row.get("context"):
        parts.append(row["context"])
    return "\n\n".join(parts)


register(DatasetAdapter(
    name="t2-ragbench",
    hf_id="G4KMU/t2-ragbench",
    question_key="question",
    ground_truth_key="program_answer",
    build_context=_t2_ragbench_context,
    preferred_split="test",
    metadata_keys=(
        "file_name", "company_name", "company_symbol",
        "report_year", "page_number", "context_id",
    ),
    requires_subset=True,
))


def _ragbench_context(row: dict) -> str:
    parts: list[str] = []
    docs = row.get("documents")
    if docs:
        if isinstance(docs, list):
            parts.extend(str(d) for d in docs)
        else:
            parts.append(str(docs))
    ctx = row.get("context")
    if ctx and not parts:
        if isinstance(ctx, list):
            parts.extend(str(c) for c in ctx)
        else:
            parts.append(str(ctx))
    return "\n\n".join(parts)


register(DatasetAdapter(
    name="ragbench",
    hf_id="rungalileo/ragbench",
    question_key="question",
    ground_truth_key="response",
    build_context=_ragbench_context,
    preferred_split="test",
    metadata_keys=("id", "dataset_name"),
    requires_subset=True,
))


def _squad_ground_truth(raw: Any) -> str:
    if isinstance(raw, dict) and raw.get("text"):
        return raw["text"][0]
    return str(raw)


register(DatasetAdapter(
    name="squad",
    hf_id="rajpurkar/squad",
    question_key="question",
    ground_truth_key="answers",
    build_context=lambda row: row.get("context", ""),
    ground_truth_transform=_squad_ground_truth,
    preferred_split="validation",
    metadata_keys=("id", "title"),
    has_shared_corpus=True,
))


def _ragas_wikiqa_context(row: dict) -> str:
    """Build context from the ``context`` column (list of chunk strings)."""
    ctx = row.get("context")
    if isinstance(ctx, list):
        return "\n\n".join(str(c) for c in ctx)
    return str(ctx) if ctx else ""


register(DatasetAdapter(
    name="ragas-wikiqa",
    hf_id="vibrantlabsai/ragas-wikiqa",
    question_key="question",
    ground_truth_key="correct_answer",
    build_context=_ragas_wikiqa_context,
    preferred_split="train",
    metadata_keys=(),
))


def _ragperf_wikipedia_nq_context(row: dict) -> str:
    return str(row.get("text", ""))


register(DatasetAdapter(
    name="ragperf-wikipedia-nq",
    hf_id="sentence-transformers/natural-questions",
    question_key="query",
    ground_truth_key="answer",
    build_context=_ragperf_wikipedia_nq_context,
    preferred_split="train",
    metadata_keys=(),
    has_shared_corpus=True,
))
