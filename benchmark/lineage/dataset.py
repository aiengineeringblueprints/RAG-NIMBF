"""Dataset identity: which questions and which corpus a run saw (issue #65).

Everything here is a hash or a count — no question text leaves this module,
so the identity can go into MLflow tags without leaking dataset content.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmark.regression.question_set import sample_keys

# Seed of every ``shuffle().select()`` subsample in ``benchmark.dataset``.
SAMPLING_SEED = 42

_DOC_TEXT_KEYS = ("context", "text", "page_content", "content")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assign_sample_ids(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Stable ID per sample: hash of question, reference answer, occurrence.

    Derived from content rather than position, so the same question keeps its
    ID when sampling or ordering changes, and reports written before sample
    IDs existed get the same IDs when they are re-read.
    """
    return [
        _sha256(json.dumps(list(key), ensure_ascii=False))[:16]
        for key in sample_keys(records)
    ]


def sample_category(sample: Mapping[str, Any]) -> str | None:
    """Optional category: explicit ``metadata.category`` or the answerable flag."""
    metadata = sample.get("metadata") or {}
    category = metadata.get("category")
    if category not in (None, ""):
        return str(category)
    answerable = metadata.get("answerable")
    if answerable is None:
        return None
    return "answerable" if answerable else "unanswerable"


def question_set_hash(sample_ids: Sequence[str]) -> str:
    """Hash of the ordered sample IDs: same questions in the same order."""
    return _sha256("\n".join(sample_ids))


def _document_text(document: Any) -> str:
    if isinstance(document, Mapping):
        for key in _DOC_TEXT_KEYS:
            if key in document and document[key] is not None:
                value = document[key]
                return value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        return json.dumps(document, sort_keys=True, default=str)
    return str(getattr(document, "page_content", document))


def corpus_fingerprint(corpus: Sequence[Any] | None) -> str | None:
    """Order-independent hash over the corpus document texts."""
    if corpus is None:
        return None
    digests = sorted(_sha256(_document_text(doc)) for doc in corpus)
    return _sha256("\n".join(digests))


def _file_sha256(path: str | None) -> str | None:
    if not path:
        return None
    file = Path(path).expanduser()
    if not file.is_file():
        return None
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_identity(
    config: Any,
    data: Sequence[Mapping[str, Any]],
    corpus: Sequence[Any] | None = None,
    *,
    sample_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Identity of the evaluated dataset slice; hashes and counts only."""
    ids = list(sample_ids) if sample_ids is not None else assign_sample_ids(data)
    return {
        "name": getattr(config, "dataset_name", None),
        "subset": getattr(config, "dataset_subset", None) or None,
        "split": getattr(config, "dataset_split", None) or None,
        "sample_size": getattr(config, "dataset_sample_size", None),
        "sampling_seed": SAMPLING_SEED,
        "num_samples": len(ids),
        "question_set_hash": question_set_hash(ids),
        "source_sha256": _file_sha256(getattr(config, "dataset_path", None)),
        "corpus_fingerprint": corpus_fingerprint(corpus),
        "corpus_documents": None if corpus is None else len(corpus),
    }
