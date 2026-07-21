"""Dataset adapter registry — maps short names to HuggingFace dataset loaders.

Each adapter describes how to load a specific HuggingFace dataset and normalise
its columns into the standard ``{question, ground_truth, context, metadata}``
format consumed by the rest of the framework.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Callable


# Sentinel hf_id / key values that mark an adapter as env-driven. The loader
# resolves these at call time from os.getenv so users can point at arbitrary
# HuggingFace datasets without registering a dedicated adapter.
ENV_HF_ID_SENTINEL = "__env__"
ENV_KEY_SENTINEL = "__env_field__"


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


def resolve_adapter(name: str) -> DatasetAdapter:
    """Return the adapter for ``name``, resolving env-driven fields at call time.

    Adapters whose ``hf_id`` is the ``ENV_HF_ID_SENTINEL`` mark themselves as
    env-driven: their hf_id / question_key / ground_truth_key / split / subset
    are read from environment variables so users can target arbitrary
    HuggingFace datasets without registering a dedicated adapter.
    """
    adapter = get_adapter(name)
    if adapter.hf_id != ENV_HF_ID_SENTINEL:
        return adapter

    hf_id = os.getenv("DATASET_HF_ID")
    if not hf_id:
        raise ValueError(
            f"Adapter '{name}' is env-driven but DATASET_HF_ID is not set. "
            f"Set it to a HuggingFace dataset id (e.g. 'hotpotqa/hotpot_qa')."
        )

    subset = os.getenv("DATASET_SUBSET", "").strip()
    return replace(
        adapter,
        hf_id=hf_id,
        question_key=os.getenv("DATASET_QUESTION_FIELD", adapter.question_key),
        ground_truth_key=os.getenv(
            "DATASET_GROUND_TRUTH_FIELD", adapter.ground_truth_key
        ),
        preferred_split=os.getenv("DATASET_SPLIT", adapter.preferred_split),
        requires_subset=bool(subset),
    )


# ---------------------------------------------------------------------------
# Built-in adapters
# ---------------------------------------------------------------------------


register(DatasetAdapter(
    name="jsonl",
    hf_id="local-jsonl",
    question_key="question",
    ground_truth_key="ground_truth",
    build_context=lambda row: str(row.get("context", "")),
    preferred_split="local",
))


register(DatasetAdapter(
    name="csv",
    hf_id="local-csv",
    question_key="question",
    ground_truth_key="ground_truth",
    build_context=lambda row: str(row.get("context", "")),
    preferred_split="local",
))


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


# ---------------------------------------------------------------------------
# Env-driven generic adapters
# ---------------------------------------------------------------------------
# These let users target arbitrary HuggingFace datasets purely via .env vars
# without registering a dedicated adapter. Set DATASET_NAME=hf-generic (or
# multihop-generic) plus the DATASET_HF_ID / DATASET_*_FIELD / DATASET_SPLIT
# variables documented in .env.example.


def _hf_generic_context(row: dict) -> str:
    """Flatten the column named by DATASET_CONTEXT_FIELD into a string.

    Accepts str, list[str], or list[list[str]] (some HF datasets nest rows).
    """
    field = os.getenv("DATASET_CONTEXT_FIELD", "context")
    raw = row.get(field, "")
    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, list):
                parts.append(" ".join(str(s) for s in item))
            else:
                parts.append(str(item))
        return "\n\n".join(parts)
    return str(raw) if raw else ""


register(DatasetAdapter(
    name="hf-generic",
    hf_id=ENV_HF_ID_SENTINEL,
    question_key=ENV_KEY_SENTINEL,
    ground_truth_key=ENV_KEY_SENTINEL,
    build_context=_hf_generic_context,
    preferred_split="test",
    metadata_keys=("id",),
))


def _multihop_context(row: dict) -> str:
    """Flatten multi-hop context into a single string.

    Handles three HF schemas:
      - ``list[list[str]]``  — list of paragraphs, each a sentence list
      - ``list[dict]``       — each dict has ``title`` + ``sentences``/``paragraph_text``
      - ``dict``             — HotpotQA form: ``{'title': [...], 'sentences': [[...]]}``
    """
    ctx = row.get("context", "")
    if isinstance(ctx, dict):
        titles = ctx.get("title") or []
        sents = ctx.get("sentences") or []
        if isinstance(titles, list) and isinstance(sents, list):
            parts = []
            for title, paragraph in zip(titles, sents):
                if isinstance(paragraph, list):
                    paragraph = " ".join(str(s) for s in paragraph)
                parts.append(f"{title}\n{paragraph}" if title else str(paragraph))
            return "\n\n".join(parts)
        return str(ctx) if ctx else ""
    if isinstance(ctx, list):
        parts = []
        for item in ctx:
            if isinstance(item, list):
                parts.append(" ".join(str(s) for s in item))
            elif isinstance(item, dict):
                title = item.get("title") or item.get("paragraph_id") or ""
                text = item.get("paragraph_text") or item.get("sentences") or ""
                if isinstance(text, list):
                    text = " ".join(str(s) for s in text)
                parts.append(f"{title}\n{text}" if title else str(text))
            else:
                parts.append(str(item))
        return "\n\n".join(parts)
    return str(ctx) if ctx else ""


def _multihop_ground_truth(raw: Any) -> str:
    if isinstance(raw, list):
        return " | ".join(str(x) for x in raw)
    return str(raw)


register(DatasetAdapter(
    name="multihop-generic",
    hf_id=ENV_HF_ID_SENTINEL,
    question_key=ENV_KEY_SENTINEL,
    ground_truth_key=ENV_KEY_SENTINEL,
    build_context=_multihop_context,
    ground_truth_transform=_multihop_ground_truth,
    preferred_split="validation",
    metadata_keys=(
        "id",
        "type",
        "level",
        "context",
        "supporting_facts",
        "supporting_contexts",
    ),
    has_shared_corpus=False,
))
