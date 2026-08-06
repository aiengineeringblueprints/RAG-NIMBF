"""RAGBench component dataset adapters.

RAGBench (`rungalileo/ragbench`, arXiv:2407.11005) packages 12 component
datasets across five domains under a single HuggingFace dataset, each example
carrying sentence-level relevance/utilization annotations used by the TRACe
metric. This module registers one adapter per component (``ragbench_cuad``,
``ragbench_pubmedqa``, ...) so YAML manifests can select a component directly:

    dataset:
      name: ragbench_cuad
      split: test
      max_examples: 100

Importing this module is a side-effecting registration: it must be imported
from ``benchmark.dataset_adapters`` so the registry is populated before any
``resolve_adapter`` lookup.

The sentence-level annotations (``documents_sentences``, ``response_sentences``,
``all_relevant_sentence_keys``, ``all_utilized_sentence_keys``,
``sentence_support_information``, ``relevance_score``, ``utilization_score``,
``completeness_score``, ``adherence_score``) are written to a sidecar JSON
next to the cache directory so downstream TRACe scoring (separate worktree)
can consume them as gold labels without re-downloading.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.dataset_adapters import DatasetAdapter, register


RAGBENCH_HF_ID = "rungalileo/ragbench"

# 12 RAGBench components grouped by domain (arXiv:2407.11005, Table 1).
RAGBENCH_COMPONENTS: dict[str, str] = {
    # Bio-medical
    "pubmedqa": "PubMedQA",
    "covidqa": "CovidQA",
    # General
    "hotpotqa": "HotpotQA",
    "msmarco": "MS Marco",
    "hagrid": "HAGRID",
    "expertqa": "ExperQA",
    # Legal
    "cuad": "CUAD",
    # Technical
    "emanual": "EManual",
    "techqa": "TechQA",
    # Financial
    "finqa": "FinQA",
    "tatqa": "TAT-QA",
    # Conversation
    "delucionqa": "DelucionQA",
}

# Sentence-level annotation fields preserved on every RAGBench question as
# gold for TRACe / utilization / completeness scoring. Stored under
# ``metadata['ragbench_trace']`` (JSON-serialised by the Chroma coercion when
# the sample flows through load_corpus_and_questions).
RAGBENCH_TRACE_KEYS: tuple[str, ...] = (
    "documents_sentences",
    "response_sentences",
    "sentence_support_information",
    "unsupported_response_sentence_keys",
    "all_relevant_sentence_keys",
    "all_utilized_sentence_keys",
    "relevance_score",
    "utilization_score",
    "completeness_score",
    "adherence_score",
)

RAGBENCH_METADATA_KEYS: tuple[str, ...] = (
    "id",
    "dataset_name",
    "generation_model_name",
    "annotating_model_name",
)


def _ragbench_component_context(row: dict[str, Any]) -> str:
    """Join the ``documents`` array into a single context string.

    RAGBench stores context as ``documents: list[str]``. Some legacy uploads
    expose ``unannotated_context`` or ``gpt3_context`` instead; fall back to
    those if ``documents`` is missing so the adapter tolerates schema drift
    across component uploads.
    """
    docs = row.get("documents")
    if isinstance(docs, list) and docs:
        return "\n\n".join(str(d) for d in docs)
    if isinstance(docs, str) and docs:
        return docs
    for fallback in ("unannotated_context", "gpt3_context", "context"):
        alt = row.get(fallback)
        if isinstance(alt, list) and alt:
            return "\n\n".join(str(a) for a in alt)
        if isinstance(alt, str) and alt:
            return alt
    return ""


def _ragbench_trace_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Extract sentence-level TRACe annotations from a RAGBench row."""
    trace: dict[str, Any] = {}
    for key in RAGBENCH_TRACE_KEYS:
        if key in row:
            trace[key] = row[key]
    return trace


def trace_sidecar_path(
    cache_root: str | Path, component: str, split: str
) -> Path:
    """Return the sidecar JSON path for a component/split cache directory."""
    return Path(cache_root) / "ragbench" / component / split / "trace_gold.json"


def write_trace_sidecar(
    cache_root: str | Path,
    component: str,
    split: str,
    rows: list[dict[str, Any]],
) -> Path:
    """Persist TRACe gold annotations for ``rows`` to a sidecar JSON file.

    Each row is keyed by its ``id`` (falling back to its index) so downstream
    TRACe scoring can join annotations back to question metadata. The sidecar
    is written next to the dataset cache directory so a re-run finds it
    without re-downloading.
    """
    path = trace_sidecar_path(cache_root, component, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        key = str(row.get("id") or f"{component}_{split}_{index}")
        payload[key] = _ragbench_trace_metadata(row)
    path.write_text(
        json.dumps(
            {"component": component, "split": split, "annotations": payload},
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _register_ragbench_components() -> None:
    for component, display in RAGBENCH_COMPONENTS.items():
        register(
            DatasetAdapter(
                name=f"ragbench_{component}",
                hf_id=RAGBENCH_HF_ID,
                question_key="question",
                ground_truth_key="response",
                build_context=_ragbench_component_context,
                preferred_split="test",
                metadata_keys=RAGBENCH_METADATA_KEYS,
                requires_subset=True,
            )
        )


_register_ragbench_components()
